import json
import time
import random
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from tqdm.asyncio import tqdm_asyncio

from config import (
    GraphType, AgentType, TaskType, DATASET_CONFIG,
    bclient,
    get_model, DEFAULT_AGENT_MODEL, DEFAULT_OPTIMIZER_MODEL,
    DEFAULT_EVALUATOR_MODEL, MODEL_REGISTRY,
    AVAILABLE_MODELS, DEFAULT_LAMBDA_COST,
    DEFAULT_SEED,
    compute_mas_cost,
)
from logger import setup_logger, get_logger
from prompts import OPTIMIZATION_REQUIREMENTS, get_dataset_graph_prompt_map
from dynamic_prompts import get_dynamic_role_prompt_map
from utils import normalize_answer, normalize_qa_answer, extract_output, extract_function_name_from_tests
from data_loaders import load_test_data, load_train_for_opt, get_task_type, get_default_use_judge
from agent import MAS
from judges import HotpotQAJudge, BFCLJudge, LCBJudge
from cs_evaluator import CSEvaluator
from optimizers import MAPromptOptimizer
from dynamic_optimizer import DynamicMAPromptOptimizer
from co_evolver import CoEvolver
from dynamic_co_evolver import DynamicCoEvolver
from trajectory_memory import TrajectoryMemory, make_run_dir
from cost_tracker import CostTracker

GRAPH_TYPE_ALIASES = {
    "aggregate": GraphType.AGGREGATE,
    "hierarchical": GraphType.HIERARCHICAL,
    "reflect": GraphType.REFLECT,
    "council": GraphType.COUNCIL,
    "selector_dynamic": GraphType.SELECTOR_DYNAMIC,
    "single_agent": GraphType.SINGLE_AGENT,
    "single_agent_cot": GraphType.SINGLE_AGENT_COT,
    "single_agent_tpe": GraphType.SINGLE_AGENT_TPE,
    "single_agent_spo": GraphType.SINGLE_AGENT_SPO,
}

def parse_graph_type(value: str) -> GraphType:
    """Parse public topology names and legacy aliases."""
    try:
        return GraphType(value)
    except ValueError:
        try:
            return GRAPH_TYPE_ALIASES[value.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown graph type: {value}") from exc


def parse_flexible_round_depths(value: str) -> List[int]:
    """Parse a per-round optimization schedule such as ``3,2,1``."""
    try:
        depths = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of positive integers, e.g. 3,2,1"
        ) from exc
    if not depths or any(depth <= 0 for depth in depths):
        raise argparse.ArgumentTypeError(
            "must contain at least one positive integer, e.g. 3,2,1"
        )
    return depths

PUBLIC_GRAPH_TYPES = [gt.value for gt in GraphType]
GRAPH_CHOICES = PUBLIC_GRAPH_TYPES + list(GRAPH_TYPE_ALIASES)



async def aprocess_single(item: Dict[str, Any], graph_type: GraphType,
                          task_type: TaskType, prompt_map: Optional[Dict[int, str]] = None,
                          use_judge: bool = False, nr: int = 1,
                          agent_model: str = DEFAULT_AGENT_MODEL,
                          agent_models: Optional[Dict[int, str]] = None,
                          lcb_test_limit: Optional[int] = None,
                          dataset: str = "bfcl_v3",
                          evaluate_cs: bool = False) -> Dict[str, Any]:
    problem = item['problem']
    unique_id = item['unique_id']

    if task_type == TaskType.CODE and 'test_list' in item:
        func_name = extract_function_name_from_tests(item['test_list'])
        if func_name:
            problem = f"{problem}\n\nNote: The function should be named '{func_name}'."
    try:
        mas = MAS(graph_type, task_type, use_judge=use_judge, judge_client=bclient, Nr=nr,
                  agent_model=agent_model, agent_models=agent_models,
                  dataset_name=dataset)
        if prompt_map:
            mas.inject_prompt_map(prompt_map)
        
        start = time.time()
        functions = item.get('functions') if task_type == TaskType.FUNCTION_CALLING else None
        result, cache = await mas.arun_with_cache(problem, functions=functions)
        end = time.time()
        final_output = result['final']
        terminal_id = mas.get_terminal_id()
        final_raw = result["raw_trace"][terminal_id]
        execution_error = next(
            (getattr(agent, "last_error", None) for agent in mas.agents
             if getattr(agent, "last_error", None)),
            None,
        )
        total_cost = compute_mas_cost(cache)
        per_agent_tokens = {}
        for aid in cache.node_input_tokens:
            agent_record = {
                "input_tokens": cache.node_input_tokens.get(aid, 0),
                "output_tokens": cache.node_output_tokens.get(aid, 0),
                "model": cache.node_model.get(aid, "unknown"),
                "input": cache.node_prompts.get(aid, ""),
                "output": cache.node_outputs_raw.get(aid, ""),
            }
            per_agent_tokens[aid] = agent_record
        cs_result = None
        if evaluate_cs:
            try:
                # Keep the original per_agent_tokens untouched. The intermediate
                # context-only view exists only for CS evaluation.
                agents_by_id = {
                    str(agent.id): agent for agent in getattr(mas, "agents", [])
                }
                cs_per_agent_tokens = {}
                for aid, record in per_agent_tokens.items():
                    cs_record = dict(record)
                    agent = agents_by_id.get(str(aid))
                    cs_record["context_only_input"] = (
                        getattr(agent, "last_context_only_prompt", "")
                        if agent is not None else record.get("input", "")
                    )
                    if not cs_record["context_only_input"]:
                        cs_record["context_only_input"] = record.get("input", "")
                    cs_per_agent_tokens[aid] = cs_record

                cs_agent_roles = {}
                for agent in getattr(mas, "agents", []):
                    role_name = getattr(agent, "role_name", "") or getattr(agent, "type", "")
                    if hasattr(role_name, "value"):
                        role_name = role_name.value
                    cs_agent_roles[agent.id] = {
                        "name": str(role_name),
                    }
                cs_result = await CSEvaluator().evaluate(
                    problem=problem,
                    per_agent_tokens=cs_per_agent_tokens,
                    final_output=final_output,
                    agent_roles=cs_agent_roles,
                    coordination_mode=graph_type.value,
                )
            except Exception as cs_exc:
                # CS is strictly auxiliary: a Judge failure must not change
                # the original MAS output or task correctness.
                cs_result = {"error": str(cs_exc)}
        judge_result = None
        
        if task_type == TaskType.CODE:
            if use_judge and mas.judge and item.get('test_list'):
                is_correct = await mas.judge.ajudge(final_output, item['test_list'])
            else:
                is_correct = "def " in final_output
        elif task_type == TaskType.FUNCTION_CALLING:
            bfcl_judge = BFCLJudge()
            judge_result = await bfcl_judge.ajudge(
                final_output,
                function_schemas=item.get('function_schemas', item.get('functions', [])),
                ground_truth=item.get('ground_truth_raw', []),
                category=item.get('category', 'simple'),
            )
            judge_result["evaluated_source"] = "final_output"
            if (not judge_result.get("correct")) and final_raw != final_output:
                fallback_judge_result = await bfcl_judge.ajudge(
                    final_raw,
                    function_schemas=item.get('function_schemas', item.get('functions', [])),
                    ground_truth=item.get('ground_truth_raw', []),
                    category=item.get('category', 'simple'),
                )
                fallback_judge_result["evaluated_source"] = "final_raw"
                if fallback_judge_result.get("correct"):
                    judge_result = fallback_judge_result
                else:
                    judge_result["fallback_final_raw"] = fallback_judge_result
            is_correct = judge_result['correct']
        elif task_type == TaskType.CODE_GENERATION:
            lcb_judge = LCBJudge(test_limit=lcb_test_limit)
            judge_result = await lcb_judge.ajudge(final_output, tests=item.get('tests', []), test_limit=lcb_test_limit)
            is_correct = judge_result['correct']
        elif task_type == TaskType.QA:
            hotpotqa_judge = HotpotQAJudge()
            judge_result = await hotpotqa_judge.ajudge(final_output, item['answer'])
            is_correct = judge_result['correct']
        else:
            if use_judge and mas.judge:
                is_correct = await mas.judge.ajudge(problem, item['answer'], final_raw)
            else:
                correct_answer = normalize_answer(item['answer'])
                model_answer = normalize_answer(final_output)
                is_correct = model_answer == correct_answer
                if task_type in [TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE]:
                    is_correct = is_correct or correct_answer in model_answer
        return {
            "unique_id": unique_id,
            "graph_type": graph_type.value,
            "output": final_output,
            "final_raw": final_raw,
            "correct": is_correct,
            "response_time": end - start,
            "full_output": result,
            "trajectory": result.get("trajectory", []),
            "selected_path": result.get("selected_path", []),
            "stop_reason": result.get("stop_reason"),
            "judge_result": judge_result,
            "error": execution_error,
            "cost": total_cost,
            "per_agent_tokens": per_agent_tokens,
            "cs_result": cs_result,
        }
    except Exception as e:
        return {
            "unique_id": unique_id,
            "graph_type": graph_type.value,
            "output": None,
            "final_raw": None,
            "correct": False,
            "response_time": 0,
            "full_output": None,
            "trajectory": [],
            "selected_path": [],
            "stop_reason": None,
            "judge_result": None,
            "error": str(e),
            "cost": 0.0,
            "per_agent_tokens": {},
            "cs_result": None,
        }
async def aprocess_task(item: Dict[str, Any], graph_types: List[GraphType],
                        task_type: TaskType, prompt_map: Optional[Dict[int, str]] = None,
                        use_judge: bool = False, nr: int = 1,
                        agent_model: str = DEFAULT_AGENT_MODEL,
                        agent_models: Optional[Dict[int, str]] = None,
                        lcb_test_limit: Optional[int] = None,
                        dataset: str = "bfcl_v3",
                        evaluate_cs: bool = False) -> List[Dict[str, Any]]:

    return [await aprocess_single(item, gt, task_type, prompt_map, use_judge, nr, agent_model, agent_models, lcb_test_limit, dataset, evaluate_cs)
            for gt in graph_types]
async def limited_aprocess_task(item: Dict[str, Any], graph_types: List[GraphType],
                                 task_type: TaskType, sem: asyncio.Semaphore,
                                 prompt_map: Optional[Dict[int, str]] = None,
                                 use_judge: bool = False, nr: int = 1,
                                 agent_model: str = DEFAULT_AGENT_MODEL,
                                 agent_models: Optional[Dict[int, str]] = None,
                                 lcb_test_limit: Optional[int] = None,
                                 dataset: str = "bfcl_v3",
                                 evaluate_cs: bool = False):
    async with sem:
        return await aprocess_task(item, graph_types, task_type, prompt_map, use_judge, nr, agent_model, agent_models, lcb_test_limit, dataset, evaluate_cs)
async def arun_test_suite(data: List[Dict[str, Any]],
                          task_type: TaskType,
                          graph_types: List[GraphType] = None,
                          sample_size: int = None,
                          output_file: str = "test_results.json",
                          max_concurrent: int = 20,
                          prompt_map: Optional[Dict[int, str]] = None,
                          use_judge: bool = False,
                          nr: int = 1,
                          agent_model: str = DEFAULT_AGENT_MODEL,
                          agent_models: Optional[Dict[int, str]] = None,
                          lcb_test_limit: Optional[int] = None,
                          dataset: str = "bfcl_v3",
                          evaluate_cs: bool = False):
    if graph_types is None:
        graph_types = list(GraphType)

    original_len = len(data)
    if sample_size and sample_size < original_len:
        data = random.sample(data, sample_size)
        print(f"{original_len} samples {sample_size}")
    else:
        print(f"use {original_len} samples")
    task_sem = asyncio.Semaphore(max_concurrent)
    results = {
        "total": len(data),
        "task_type": task_type.value,
        "graph_types": {gt.value: {"correct": 0, "total": 0, "accuracy": 0.0, "avg_response_time": 0.0,
                                   "total_cost": 0.0, "avg_cost": 0.0, "cost_of_pass": None}
                       for gt in graph_types},
        "detailed": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "use_judge": use_judge,
        "total_cost": 0.0,
    }
    if evaluate_cs:
        results["cs_enabled"] = True
        results["cs_judge_model"] = "deepseek-v4-flash"
        results["cs_judge_temperature"] = 0.0
    start_total = time.time()
    tasks = [
        limited_aprocess_task(item, graph_types, task_type, task_sem, prompt_map, use_judge, nr, agent_model, agent_models, lcb_test_limit, dataset, evaluate_cs)
        for item in data
    ]
    results_list = await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="")
    flat_results = [r for sub in results_list for r in sub]
    detailed_map = {}
    for r in flat_results:
        uid, gt_val = r["unique_id"], r["graph_type"]
        if uid not in detailed_map:
            orig = next(item for item in data if item["unique_id"] == uid)
            detailed_map[uid] = {
                "unique_id": uid,
                "problem": orig["problem"],
                "models": {},
            }
            if task_type == TaskType.CODE:
                detailed_map[uid]["test_list"] = orig.get("test_list", [])
            elif task_type == TaskType.CODE_GENERATION:
                detailed_map[uid]["test_cases"] = orig.get("answer", "")
            else:
                detailed_map[uid]["correct_answer"] = normalize_answer(orig["answer"])
        
        model_record = {
            "output": r["output"],
            "raw_output": r["final_raw"],
            "correct": r["correct"],
            "judge_result": r.get("judge_result"),
            "response_time": r["response_time"],
            "error": r["error"],
            "cost": r.get("cost", 0.0),
            "per_agent_tokens": r.get("per_agent_tokens", {}),
            "trajectory": r.get("trajectory", []),
            "selected_path": r.get("selected_path", []),
            "stop_reason": r.get("stop_reason"),
        }
        if evaluate_cs:
            cs_result = r.get("cs_result")
            if cs_result and "coordination_score" in cs_result:
                model_record.update({
                    "communication_score": cs_result["communication_score"],
                    "planning_score": cs_result["planning_score"],
                    "coordination_score": cs_result["coordination_score"],
                    "cs_judge_model": cs_result["judge_model"],
                    "cs_judge_temperature": cs_result["judge_temperature"],
                    "cs_judge_cost": cs_result["judge_cost"],
                    "cs_judge_usage": cs_result["judge_usage"],
                    "cs_judge_raw": cs_result["judge_raw"],
                })
            else:
                model_record.update({
                    "communication_score": None,
                    "planning_score": None,
                    "coordination_score": None,
                    "cs_status": "failed",
                    "cs_error": (cs_result or {}).get("error", "unknown CS evaluation error"),
                })
        detailed_map[uid]["models"][gt_val] = model_record

        st = results["graph_types"][gt_val]
        st["total"] += 1
        if r["correct"]:
            st["correct"] += 1
        st["avg_response_time"] += r["response_time"]
        st["total_cost"] += r.get("cost", 0.0)
        if evaluate_cs and r.get("cs_result") and "coordination_score" in r["cs_result"]:
            cs_result = r["cs_result"]
            st.setdefault("cs_samples", 0)
            st.setdefault("communication_score_sum", 0.0)
            st.setdefault("planning_score_sum", 0.0)
            st.setdefault("coordination_score_sum", 0.0)
            st.setdefault("cs_judge_cost", 0.0)
            st["cs_samples"] += 1
            st["communication_score_sum"] += cs_result["communication_score"]
            st["planning_score_sum"] += cs_result["planning_score"]
            st["coordination_score_sum"] += cs_result["coordination_score"]
            st["cs_judge_cost"] += cs_result.get("judge_cost", 0.0)

    for gt in graph_types:
        st = results["graph_types"][gt.value]
        if st["total"]:
            st["accuracy"] = st["correct"] / st["total"]
            st["avg_response_time"] /= st["total"]
            st["avg_cost"] = st["total_cost"] / st["total"]
        st["cost_of_pass"] = (
            st["total_cost"] / st["correct"] if st["correct"] else None
        )
        if evaluate_cs:
            cs_samples = st.get("cs_samples", 0)
            st["cs_evaluated_samples"] = cs_samples
            st["avg_communication_score"] = (
                st["communication_score_sum"] / cs_samples if cs_samples else None
            )
            st["avg_planning_score"] = (
                st["planning_score_sum"] / cs_samples if cs_samples else None
            )
            st["avg_coordination_score"] = (
                st["coordination_score_sum"] / cs_samples if cs_samples else None
            )
            st["avg_coordination_score_100"] = (
                st["avg_coordination_score"] * 20
                if st["avg_coordination_score"] is not None else None
            )
            st["avg_communication_score_100"] = (
                st["avg_communication_score"] * 20
                if st["avg_communication_score"] is not None else None
            )
            st["avg_planning_score_100"] = (
                st["avg_planning_score"] * 20
                if st["avg_planning_score"] is not None else None
            )
            st["cs_judge_cost"] = st.get("cs_judge_cost", 0.0)
            for key in ("communication_score_sum", "planning_score_sum", "coordination_score_sum"):
                st.pop(key, None)
    results["total_cost"] = sum(st["total_cost"] for st in results["graph_types"].values())
    results["detailed"] = list(detailed_map.values())
    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["duration"] = time.time() - start_total
    if agent_models:
        results["agent_models"] = agent_models
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    if agent_models:
        print("  Models: {}".format(agent_models))
    for gt_val, st in results["graph_types"].items():
        print("  {}: acc={:.4f} ({}/{}), total_cost={:,.0f}, avg_cost={:,.0f}".format(
            gt_val, st["accuracy"], st["correct"], st["total"], st["total_cost"], st["avg_cost"]))
    print("  duration: {:.0f}s".format(results["duration"]))
    print("Saved to {}".format(output_file))
    return results

def main():
    parser = argparse.ArgumentParser(description="Run MAS evaluation with configurable dataset and prompt mode.")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=list(DATASET_CONFIG.keys()),
                        help=f"Dataset to use: {', '.join(DATASET_CONFIG.keys())}")
    parser.add_argument("--graph", type=str, default=GraphType.REFLECT.value,
                        choices=GRAPH_CHOICES,
                        help="Graph type to use")
    parser.add_argument("--optimize", action="store_true",
                        help="Enable prompt optimization. If not set, use original prompts.")
    parser.add_argument("--use-llm-judge", action="store_true",
                        help="Use LLM judge for non-code tasks (code tasks always use code judge)")
    parser.add_argument("--evaluate-cs", action="store_true",
                        help="Evaluate per-sample MARBLE-style communication, planning, and coordination scores with DeepSeek Flash")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of samples to use (default: all)")
    parser.add_argument("--lcb-test-limit", type=int, default=None,
                        help="Maximum LCB tests per sample. Default follows LCB_TEST_LIMIT env or 3; use 0 for all tests.")
    parser.add_argument("--lcb-full-tests", action="store_true",
                        help="Run all LCB tests per sample (equivalent to --lcb-test-limit 0).")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Maximum concurrent tasks")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Path to pre-optimized prompt file (JSON)")
    
    parser.add_argument("--round-robin", action="store_true",
                        help="Use round-robin optimization instead of sequential optimization")
    parser.add_argument("--depth", type=int, default=10,
                        help="Maximum optimization rounds per agent (for round-robin mode)")
    parser.add_argument("--rounds-per-turn", type=int, default=1,
                        help="Number of depth steps processed in each fixed-round turn.")
    parser.add_argument("--flexible-round-depths", type=parse_flexible_round_depths, default=None,
                        help="Per-round optimization steps, e.g. 3,2,1; overrides --depth/--rounds-per-turn")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience: pause agent after N rounds without improvement")
    parser.add_argument("--fixed-rounds", action="store_true",
                    help="Use fixed rounds per turn optimization (2 rounds per agent each turn)")
    parser.add_argument("--dynamic-switching", action="store_true",
                    help="Enable Dynamic Anchor Switching for joint optimization (zero-cost)")
    parser.add_argument("--stochastic-sampling", action="store_true",
                    help="Enable Stochastic Beam Context Sampling for robust joint optimization")
    parser.add_argument("--beam-refresh", action="store_true",
                    help="Enable Beam Refresh strategy to re-score nodes based on updated partners")
    parser.add_argument("--nr", type=int, default=1,
                    help="Number of reflect rounds (for reflect topology)")
    parser.add_argument("--feedback", action="store_true",
                    help="Enable multi-agent collaborative feedback (pass bad cases from downstream to upstream).")
    parser.add_argument("--misleading-sampling", action="store_true",
                    help="Enable sampling injection of upstream 'misleading cases' (Local Win / Global Lose).")
    parser.add_argument("--lookahead-score", action="store_true",
                    help="Enable Lookahead Scoring: 0.5*Local + 0.3*Next_Local + 0.2*Global")
    parser.add_argument("--lookahead-weights", type=str, default="4:4:2",
                        help="Weights for lookahead scoring (Local:Next:Global), e.g., '4:4:2'. Default is 4:4:2 (0.4, 0.4, 0.2)")
    parser.add_argument("--agent-model", type=str, default=DEFAULT_AGENT_MODEL,
                        help=f"Model for agent inference (default: {DEFAULT_AGENT_MODEL})")
    parser.add_argument("--optimizer-model", type=str, default=DEFAULT_OPTIMIZER_MODEL,
                        help=f"Model for prompt optimization (default: {DEFAULT_OPTIMIZER_MODEL})")
    parser.add_argument("--evaluator-model", type=str, default=DEFAULT_EVALUATOR_MODEL,
                        help=f"Model for evaluation (default: {DEFAULT_EVALUATOR_MODEL})")
    parser.add_argument("--lambda-cost", type=float, default=None,
                        help=f"Cost penalty weight for quality-cost trade-off (default: {DEFAULT_LAMBDA_COST})")
    parser.add_argument("--hetero", action="store_true",
                        help="Enable heterogeneous model optimization (co-optimize prompts AND model assignments). "
                             "When disabled, reverts to original UniMASO prompt-only optimization.")
    parser.add_argument("--trajectory-memory", nargs="?", const=1, default=None, type=int, metavar="N",
                        help="Enable trajectory memory; retrieve up to N items per memory category (default: 1).")
    parser.add_argument("--transfer-memory", type=str, default=None, metavar="PATH",
                        help="Read an existing trajectory-memory run from PATH without storing new memory.")
    parser.add_argument("--co-evolver", nargs="?", const=DEFAULT_OPTIMIZER_MODEL, default=None,
                        metavar="MODEL",
                        help="Run Co-Evolver at round boundaries; optionally specify a model from --model-pool (default: deepseek-v4-flash).")
    parser.add_argument("--voter", type=str, choices=["majority", "trace_entropy"],
                        default="majority",
                        help="Co-Evolver model-assignment voter: majority (default) or trace_entropy.")
    parser.add_argument("--memory-root", type=str, default="memory",
                        help="Root directory for per-optimization trajectory memory runs.")
    parser.add_argument("--memory-embedding-url", type=str,
                        default="http://127.0.0.1:8005/v1",
                        help="OpenAI-compatible embedding endpoint for trajectory retrieval.")
    parser.add_argument("--memory-embedding-model", type=str,
                        default="qwen3.7-text-embedding",
                        help="Embedding model name for trajectory retrieval.")
    parser.add_argument("--model-pool", type=str, nargs="+", default=None,
                        help="Pool of models for model-switch optimization. Space-separated model names.")
    parser.add_argument("--agent-models", type=str, default=None,
                        help='JSON dict mapping agent_id to initial model, e.g. \'{"0":"qwen35-9b","1":"qwen35-4b"}\'')
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Random seed for initial model assignment (default: {DEFAULT_SEED}). Use 0 for truly random.")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")

    
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, mc in MODEL_REGISTRY.items():
            urls = [c.base_url for c in mc._clients]
            print(f"  {name}: path={mc.model_path}, endpoints={urls}")
        return

    # Resolve model names (validate they exist in registry)
    agent_model = args.agent_model
    optimizer_model = args.optimizer_model
    evaluator_model = args.evaluator_model
    for m in [agent_model, optimizer_model, evaluator_model]:
        get_model(m)  # raises KeyError if not found

    # Parse cost-aware optimization args
    agent_models = None
    if args.agent_models:
        agent_models = {int(k): v for k, v in json.loads(args.agent_models).items()}

    model_pool = args.model_pool if args.model_pool else AVAILABLE_MODELS
    # Only enable cost-aware scoring when explicitly requested
    lambda_cost = args.lambda_cost if args.lambda_cost is not None else 0.0

    dataset = args.dataset
    graph_type = parse_graph_type(args.graph)
    task_type = get_task_type(dataset)
    lcb_test_limit = 0 if args.lcb_full_tests else args.lcb_test_limit

    # Determine run mode for log naming
    if args.optimize:
        if args.round_robin:
            mode = "optimize_rr"
        elif args.fixed_rounds:
            flags = []
            if args.beam_refresh: flags.append("br")
            if args.misleading_sampling: flags.append("ms")
            if args.lookahead_score: flags.append("ls")
            mode = "optimize_fr_" + ("_".join(flags) if flags else "base")
        else:
            mode = "optimize_seq"
    elif args.prompt_file:
        mode = "eval_custom"
    else:
        mode = "eval_original"

    log = setup_logger(dataset, graph_type.value, mode)

    log.info(f"Dataset: {dataset} | Task: {task_type.value} | Graph: {graph_type.value} | Mode: {mode}")
    log.info(f"Agent model     : {agent_model}")
    log.info(f"Optimizer model : {optimizer_model}")
    log.info(f"Evaluator model : {evaluator_model}")
    if agent_models:
        log.info(f"Per-agent models: {agent_models}")
    try:
        w_parts = [float(x) for x in args.lookahead_weights.split(':')]
        if len(w_parts) != 3:
            raise ValueError("Must provide exactly 3 numbers separated by colon.")
        total_w = sum(w_parts)
        if total_w == 0:
            raise ValueError("Sum of weights cannot be 0.")
        lookahead_weights = tuple(w / total_w for w in w_parts)
    except Exception as e:
        print(f"[Warning] Invalid lookahead-weights format ({e}), using default 0.4:0.4:0.2")
        lookahead_weights = (0.4, 0.4, 0.2)
    print(f"Lookahead Weights: {args.lookahead_weights} -> Local:{lookahead_weights[0]:.2f}, Next:{lookahead_weights[1]:.2f}, Global:{lookahead_weights[2]:.2f}")

    if args.trajectory_memory is not None and args.trajectory_memory < 1:
        parser.error("--trajectory-memory must be a positive integer")
    if args.trajectory_memory and not args.optimize:
        print("[Warning] --trajectory-memory only applies to --optimize; memory branch disabled.")
        args.trajectory_memory = False
    if args.transfer_memory and not args.optimize:
        print("[Warning] --transfer-memory only applies to --optimize; transfer branch disabled.")
        args.transfer_memory = None
    if args.trajectory_memory and args.transfer_memory:
        parser.error("--trajectory-memory and --transfer-memory are mutually exclusive")
    if args.transfer_memory and args.prompt_file:
        parser.error("--transfer-memory requires optimization without --prompt-file")

    if args.optimize and args.hetero:
        log.info(f"Heterogeneous optimization: lambda_cost={lambda_cost}, model_pool={model_pool}")
    elif args.optimize:
        log.info(f"Prompt-only optimization (heterogeneous=OFF), agent_model={agent_model}")
    if args.optimize:
        log.info(f"Optimization args: round_robin={args.round_robin}, fixed_rounds={args.fixed_rounds}, "
                 f"beam_refresh={args.beam_refresh}, mislead_sampling={args.misleading_sampling}, "
                 f"lookahead={args.lookahead_score}, lookahead_weights={lookahead_weights}, "
                 f"feedback={args.feedback}, depth={args.depth}, nr={args.nr}")

    default_use_judge = get_default_use_judge(dataset)
    if task_type == TaskType.CODE:
        use_judge = True
    else:
        use_judge = args.use_llm_judge

    print(f"Dataset: {dataset}")
    print(f"Task Type: {task_type.value}")
    print(f"Graph Type: {graph_type.value}")
    print(f"Optimize: {args.optimize}")
    print(f"Round-Robin Mode: {args.round_robin}")
    if task_type == TaskType.CODE_GENERATION:
        effective_lcb_limit = lcb_test_limit if lcb_test_limit is not None else 3
        print(f"LCB Test Limit: {'all' if effective_lcb_limit == 0 else effective_lcb_limit}")
    if args.round_robin:
        print(f"  - Depth per Agent: {args.depth}")
        print(f"  - Patience: {args.patience}")
    if args.flexible_round_depths:
        print(f"  - Flexible round depths: {args.flexible_round_depths} (total={sum(args.flexible_round_depths)})")
    print(f"Use Judge: {use_judge}" + (" (code judge)" if task_type == TaskType.CODE else " (LLM judge)" if use_judge else " (string match)"))
    prompt_map = None
    best_models = agent_models or {}
    trajectory_memory = None
    co_evolver = None
    cost_tracker = CostTracker() if args.optimize else None
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        prompt_map = {int(k): v for k, v in data.get("prompts", {}).items()}
        loaded_models = data.get("models", {})
        if loaded_models:
            best_models = {int(k): v for k, v in loaded_models.items()}
        print(f"Loaded {len(prompt_map)} prompts and {len(best_models)} model assignments from {args.prompt_file}")
    else:
        if graph_type is GraphType.SELECTOR_DYNAMIC:
            prompt_map = get_dynamic_role_prompt_map(dataset)
        else:
            prompt_map = get_dataset_graph_prompt_map(dataset, graph_type)
        if prompt_map:
            print(f"Loaded dataset-graph initial prompts: {dataset}/{graph_type.value} ({len(prompt_map)} agents)")
    # In heterogeneous optimization, --agent-model is the initial assignment
    # for every Agent unless an explicit per-agent map was supplied.
    # Previously MAPromptOptimizer fell back to qwen35-4b here, silently
    # ignoring --agent-model for the initial model state.
    if args.optimize and args.hetero and not best_models and prompt_map:
        best_models = {int(aid): agent_model for aid in prompt_map}
        log.info(f"Initial heterogeneous model assignments from --agent-model: {best_models}")
    if args.optimize and not args.prompt_file:
        
        train_k = 50
        seed_value = DEFAULT_SEED if args.seed is None else args.seed
        pool_rng = random.Random(None if seed_value == 0 else seed_value)
        eval_rng = random.Random(None if seed_value == 0 else seed_value + 1)
        train_problems = load_train_for_opt(dataset, k=train_k, rng=pool_rng)
        mas = MAS(graph_type, task_type, Nr=args.nr, agent_model=agent_model,
                  agent_models=agent_models, dataset_name=dataset)
        if prompt_map:
            mas.inject_prompt_map(prompt_map)
        if args.hetero and not best_models:
            best_models = {
                agent.id: agent_model
                for agent in mas.agents
                if agent.type != AgentType.AGGREGATOR
            }
            log.info(f"Initial heterogeneous model assignments from --agent-model: {best_models}")

        seed_map = {
            i: mas.agents[i].template
            for i in range(len(mas.agents))
            if mas.agents[i].type != AgentType.AGGREGATOR and mas.agents[i].template
        }

        requirement = OPTIMIZATION_REQUIREMENTS.get(task_type, "")
        optimizer_client = get_model(optimizer_model)

        if args.trajectory_memory or args.transfer_memory:
            memory_mode = "hetero_memory" if args.hetero else "prompt_memory"
            if args.transfer_memory:
                memory_dir = str(Path(args.transfer_memory).expanduser())
            else:
                memory_dir = make_run_dir(args.memory_root, dataset, graph_type.value, memory_mode)
            trajectory_memory = TrajectoryMemory(
                memory_dir,
                run_meta={
                    "dataset": dataset,
                    "task_type": task_type.value,
                    "graph_type": graph_type.value,
                    "optimization_mode": mode,
                    "heterogeneous": args.hetero,
                    "optimizer_model": optimizer_model,
                    "agent_model": agent_model,
                    "lambda_cost": lambda_cost,
                    "seed": args.seed,
                    "sample_pool_size": len(train_problems),
                },
                embedding_url=args.memory_embedding_url,
                embedding_model=args.memory_embedding_model,
                read_only=bool(args.transfer_memory),
            )
            if args.transfer_memory:
                log.info(
                    "Transferred trajectory memory enabled (read-only): %s; records=%d",
                    memory_dir,
                    len(trajectory_memory.records),
                )
            else:
                log.info(f"Trajectory memory enabled: {memory_dir}")
            log.info(f"Trajectory memory enabled: {memory_dir}")

        if args.co_evolver:
            if args.co_evolver not in model_pool:
                raise ValueError(
                    f"Co-Evolver model '{args.co_evolver}' is not in the active model pool: {model_pool}"
                )
            if trajectory_memory is not None and not trajectory_memory.read_only:
                co_evolver_trace_dir = trajectory_memory.run_dir
            elif args.transfer_memory:
                co_evolver_trace_dir = make_run_dir(
                    "co_evolver_traces", dataset, graph_type.value, "transfer"
                )
            else:
                co_evolver_trace_dir = make_run_dir(
                    "co_evolver_traces", dataset, graph_type.value, "co_evolver"
                )
            coevolver_cls = DynamicCoEvolver if graph_type is GraphType.SELECTOR_DYNAMIC else CoEvolver
            co_evolver = coevolver_cls(
                optimizer_model=args.co_evolver,
                model_pool=model_pool,
                trace_dir=str(co_evolver_trace_dir),
                cost_tracker=cost_tracker,
            )
            log.info(
                "Co-Evolver enabled: model=%s, pool=%s, memory_injection=disabled, "
                "voter=%s, trace=%s",
                args.co_evolver, model_pool, args.voter, co_evolver_trace_dir,
            )

        optimizer_cls = (
            DynamicMAPromptOptimizer
            if graph_type is GraphType.SELECTOR_DYNAMIC
            else MAPromptOptimizer
        )
        optimizer = optimizer_cls(
            mas, train_problems, seed_map, optimizer_client,
            model_pool=model_pool,
            lambda_cost=lambda_cost,
            seed_model_map=best_models,
            seed=args.seed,
            heterogeneous=args.hetero,
            trajectory_memory=trajectory_memory,
            memory_per_category=args.trajectory_memory or 1,
            dataset_name=dataset,
            co_evolver=co_evolver,
            voter=args.voter,
            eval_rng=eval_rng,
            cost_tracker=cost_tracker,
            cost_model_name=optimizer_model,
        )
        statistics = None
        
        statistics = None

        if args.round_robin:
            prompt_map, best_models = asyncio.run(optimizer.optimize_all_round_robin(
                requirement=requirement,
                max_total_depth=args.depth,
                patience=args.patience,
                beam_width=2
            ))
        elif args.fixed_rounds:
            prompt_map, best_models, statistics = asyncio.run(optimizer.optimize_all_fixed_rounds(
                requirement=requirement,
                max_total_depth=(sum(args.flexible_round_depths)
                                 if args.flexible_round_depths else args.depth),
                rounds_per_turn=args.rounds_per_turn,
                flexible_round_depths=args.flexible_round_depths,
                beam_width=2,
                use_dynamic_switching=args.dynamic_switching,
                use_stochastic_sampling=args.stochastic_sampling,
                use_beam_refresh=args.beam_refresh,
                use_feedback=args.feedback,
                use_misleading_sampling=args.misleading_sampling,
                use_lookahead_score=args.lookahead_score,
                lookahead_weights=lookahead_weights
            ))
        else:
            prompt_map, best_models = asyncio.run(optimizer.optimize_all(requirement=requirement))
        
        if graph_type is GraphType.SELECTOR_DYNAMIC:
            mode_suffix = "dynamic"
        elif args.round_robin:
            mode_suffix = "rr"
        elif args.lookahead_score and args.misleading_sampling:
            mode_suffix = "ms_ls"
        elif args.misleading_sampling:
            mode_suffix = "ms"
        elif args.lookahead_score:
            mode_suffix = "ls"
        else:
            mode_suffix = "topo"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_output_file = f"prompt/tbdspo_{dataset}_{graph_type.value}_{mode_suffix}_{agent_model}_{timestamp}.json"
        with open(prompt_output_file, "w", encoding="utf-8") as f:
            json.dump({"prompts": prompt_map, "models": best_models}, f, indent=2, ensure_ascii=False)
        print(f"Optimized prompts saved to {prompt_output_file}")

        if best_models:
            model_output_file = f"prompt/tbdspo_{dataset}_{graph_type.value}_{mode_suffix}_models_{agent_model}_{timestamp}.json"
            with open(model_output_file, "w", encoding="utf-8") as f:
                json.dump(best_models, f, indent=2, ensure_ascii=False)
            print(f"Optimized model assignments saved to {model_output_file}")

        if statistics:
            stats_output_file = f"stats/tbdspo_{dataset}_{graph_type.value}_{mode_suffix}_stats_{agent_model}_{timestamp}.json"

            import os
            os.makedirs("stats", exist_ok=True)
            
            with open(stats_output_file, "w", encoding="utf-8") as f:
                json.dump(statistics, f, indent=2, ensure_ascii=False)
            print(f"Optimization statistics saved to {stats_output_file}")

        if trajectory_memory is not None:
            asyncio.run(trajectory_memory.close())
            if trajectory_memory.read_only:
                print(f"Transferred trajectory memory closed (read-only): {trajectory_memory.run_dir}")
            else:
                print(f"Trajectory memory saved to {trajectory_memory.run_dir}")
            
            
    
    load_limit = None
    if task_type == TaskType.CODE_GENERATION and args.sample_size and not args.optimize:
        load_limit = args.sample_size
        print(f"Fast loading first {load_limit} code-generation samples before full-file parse.")
    test_data = load_test_data(dataset, max_items=load_limit)
    
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "tbdspo" if (args.optimize or args.prompt_file) else "original"
    if args.round_robin and args.optimize:
        suffix = "tbdspo_rr"
    output_file = f"result/{dataset}_{graph_type.value}_{suffix}_{agent_model}_{eval_ts}.json"
    
    evaluation_results = asyncio.run(arun_test_suite(
        test_data,
        task_type=task_type,
        graph_types=[graph_type],
        sample_size=args.sample_size,
        output_file=output_file,
        max_concurrent=args.max_concurrent,
        prompt_map=prompt_map,
        use_judge=use_judge,
        nr=args.nr,
        agent_model=agent_model,
        agent_models=best_models,
        lcb_test_limit=lcb_test_limit,
        dataset=dataset,
        evaluate_cs=args.evaluate_cs,
    ))
    if cost_tracker is not None:
        cost_tracker.record_external_cost(
            "evaluation_inference",
            evaluation_results.get("total_cost", 0.0),
            substage="final_test_suite",
        )
        cost_dir = Path("cost")
        cost_stem = output_file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        cost_tracker.save(
            cost_dir / f"{cost_stem}_cost.json",
            cost_dir / f"{cost_stem}_calls.jsonl",
        )
        print(f"Cost summary saved to {cost_dir / (cost_stem + '_cost.json')}")

if __name__ == "__main__":
    main()
