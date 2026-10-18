
import random
import asyncio
import re
import time
import json
import hashlib
import os
from typing import List, Dict, Optional, Tuple, Set, Any
from tqdm import tqdm
from dataclasses import dataclass, field

from config import (
    GraphType, AgentType, TaskType,
    get_model_cost, compute_mas_cost, get_model_price_ratio,
    AVAILABLE_MODELS, DEFAULT_LAMBDA_COST, DEFAULT_AGENT_MODEL,
    DEFAULT_SEED,
    MODEL_REGISTRY,
    get_model_profile,
)
from prompts import (
    get_role_description, PROMPT_OPTIMIZE_TEMPLATE,
    MEMORY_PROMPT_OPTIMIZE_TEMPLATE,
    ANSWER_EVALUATE_TEMPLATE, INTERMEDIATE_COMPARE_TEMPLATE,
    FINAL_ANSWER_COMPARE_TEMPLATE
)
from utils import async_call_llm, parse_comparison_result, extract_output
from agent import MAS, InferenceCache
from cost_tracker import CostTracker
from logger import get_logger
from trajectory_memory import TrajectoryMemory
from co_evolver import (
    CoEvolver,
    load_topology_profile,
    expand_reflect_profile,
)

@dataclass
class AgentOptState:
    agent_id: int
    current_beam: List[Dict]  
    best_overall_node: Dict   
    total_layers_explored: int = 0 
    current_patience_count: int = 0  
    misleading_cases: List[Dict] = field(default_factory=list)

    recent_bad_cases: List[Dict] = field(default_factory=list)
    misalignment_rates_per_depth: List[float] = field(default_factory=list)  
    beam_refresh_kendall_scores: List[Dict] = field(default_factory=list)
    def __init__(self, agent_id: int, initial_prompt: str, initial_model: str = None):
        self.agent_id = agent_id
        initial_node = {
            "prompt": initial_prompt,
            "model": initial_model or DEFAULT_AGENT_MODEL,
            "cumulative_score": 0.0,
            "quality": 0.0,
            "cost": 0.0,
            "path": [initial_prompt]
        }
        self.current_beam = [initial_node]
        self.best_overall_node = initial_node
        self.total_layers_explored = 0
        self.current_patience_count = 0
        self.recent_bad_cases = []
        self.misleading_cases = []
        self.misalignment_rates_per_depth = []
        self.beam_refresh_kendall_scores = []


class MAPromptOptimizer:

    def __init__(self, mas: MAS, train_questions: List[str],
                 seed_prompt_map: Dict[int, str], evaluator_client,
                 model_pool: List[str] = None,
                 lambda_cost: float = DEFAULT_LAMBDA_COST,
                 seed_model_map: Dict[int, str] = None,
                 seed: int = None,
                 heterogeneous: bool = False,
                 trajectory_memory: Optional[TrajectoryMemory] = None,
                 dataset_name: str = "",
                 co_evolver: Optional[CoEvolver] = None,
                 voter: str = "majority",
                 eval_rng=None,
                 memory_per_category: int = 1,
                 cost_tracker: Optional[CostTracker] = None,
                 cost_model_name: str = None):
        self.mas = mas
        self.train_q = train_questions
        self.seed = seed_prompt_map
        self.best_prompt = seed_prompt_map.copy()
        # Council 中 0/3、1/4、2/5 是同一组 prompt 的两轮实例：
        # 第二轮只是在同一提示词上增加其他 Agent 的上下文，不能拆成六个优化变量。
        self.shared_prompt_groups = {}
        if self.mas.gtype == GraphType.COUNCIL:
            self.shared_prompt_groups = {
                0: (0, 3),
                1: (1, 4),
                2: (2, 5),
            }
        # Council's two rounds represent one logical agent and share a model variable.
        # Other MAS structures keep the original per-agent model assignment behavior.
        self.shared_model_groups = dict(self.shared_prompt_groups)
        self.shared_prompt_leader = {
            member: leader
            for leader, members in self.shared_prompt_groups.items()
            for member in members
        }
        self._normalize_shared_prompts()
        self.evaluator = evaluator_client
        self.cost_tracker = cost_tracker
        self.cost_model_name = cost_model_name or getattr(evaluator_client, "model_path", "unknown")
        self.task_type = mas.task_type
        self.trajectory_memory = trajectory_memory
        self.dataset_name = dataset_name
        self.co_evolver = co_evolver
        self.voter = voter
        # Historical trajectories are retained and retrieved only by the
        # round-boundary Co-Evolver; depth optimizers no longer inject memory.
        self.enable_legacy_memory_injection = False
        self.heterogeneous = heterogeneous
        self.model_pool = model_pool or AVAILABLE_MODELS
        if not self.heterogeneous:
            # Non-hetero: all agents use the same model, no switching possible.
            # Override model_pool to a single entry to prevent LLM from proposing switches.
            self.model_pool = [mas.agent_model]
        self.lambda_cost = lambda_cost if self.heterogeneous else 0.0
        self.opt_seed = seed if seed is not None else DEFAULT_SEED
        self.eval_rng = eval_rng or random.Random(self.opt_seed + 1)
        self.eval_samples_by_depth: Dict[int, List[str]] = {}
        self.memory_per_category = max(1, int(memory_per_category))
        # Per-agent best model assignment (co-optimized with prompts)
        self.best_model = seed_model_map.copy() if seed_model_map else {}
        if not self.best_model:
            default_model = mas.agent_model if not self.heterogeneous else "qwen35-4b"
            for aid in seed_prompt_map:
                self.best_model[aid] = default_model  # always start from smallest for hetero
        self._normalize_shared_models()
        # Optimization trajectory per agent: List[Dict] for LLM context
        self.trajectory_log: Dict[int, List[Dict]] = {}
        self.beam_width = 2
        # In fixed-rounds mode, keep the final Co-Evolver output separate from
        # the Depth state used to initialize the next round.
        self.final_coevolver_prompt: Dict[int, str] = {}
        self.final_coevolver_model: Dict[int, str] = {}

    def get_eval_samples(self, depth_id: int, sample_size: int = 10) -> List[str]:
        """Return one reproducible sample set shared by all candidates at a depth."""
        if depth_id not in self.eval_samples_by_depth:
            self.eval_samples_by_depth[depth_id] = self.eval_rng.sample(
                self.train_q,
                min(sample_size, len(self.train_q)),
            )
        return self.eval_samples_by_depth[depth_id]

    def _normalize_shared_prompts(self) -> None:
        """Make all members of each shared group point to the leader prompt."""
        for leader, members in self.shared_prompt_groups.items():
            prompt = self.best_prompt.get(leader)
            if prompt is None:
                prompt = next((self.best_prompt[m] for m in members if m in self.best_prompt), None)
            if prompt is not None:
                for member in members:
                    self.best_prompt[member] = prompt

    def _prompt_leader(self, agent_id: int) -> int:
        return self.shared_prompt_leader.get(agent_id, agent_id)

    def _model_leader(self, agent_id: int) -> int:
        """Return the shared model variable leader for an agent."""
        for leader, members in self.shared_model_groups.items():
            if agent_id in members:
                return leader
        return agent_id

    def _normalize_shared_models(self) -> None:
        """Make Council model-group members use the leader's model."""
        for leader, members in self.shared_model_groups.items():
            model = self.best_model.get(leader)
            if model is None:
                model = next((self.best_model[m] for m in members if m in self.best_model), None)
            if model is not None:
                for member in members:
                    self.best_model[member] = model

    def _set_best_model(self, agent_id: int, model: str) -> None:
        """Update one model variable and synchronize its Council group."""
        leader = self._model_leader(agent_id)
        for member in self.shared_model_groups.get(leader, (leader,)):
            self.best_model[member] = model

    def _expand_model_map(self, model_map: Dict[int, str]) -> Dict[int, str]:
        """Expand shared Council model variables to all runtime nodes."""
        expanded = model_map.copy()
        for leader, members in self.shared_model_groups.items():
            model = expanded.get(leader, self.best_model.get(leader))
            if model is not None:
                for member in members:
                    expanded[member] = model
        return expanded

    def _optimizable_ids(self, agent_ids: List[int]) -> List[int]:
        """Deduplicate shared Council nodes while preserving topological order."""
        result = []
        seen = set()
        for agent_id in agent_ids:
            leader = self._prompt_leader(agent_id)
            if leader not in seen:
                result.append(leader)
                seen.add(leader)
        return result

    def _expand_prompt_map(self, prompt_map: Dict[int, str]) -> Dict[int, str]:
        """Expand one shared variable back to every runtime node."""
        expanded = prompt_map.copy()
        for leader, members in self.shared_prompt_groups.items():
            prompt = expanded.get(leader, self.best_prompt.get(leader))
            if prompt is not None:
                for member in members:
                    expanded[member] = prompt
        return expanded

    def _set_best_prompt(self, agent_id: int, prompt: str) -> None:
        """Update a prompt variable and synchronize all of its shared nodes."""
        leader = self._prompt_leader(agent_id)
        for member in self.shared_prompt_groups.get(leader, (leader,)):
            self.best_prompt[member] = prompt

    def _beam_select(self, nodes: List[Dict], beam_width: int) -> List[Dict]:
        """Select beam_width nodes.

        In heterogeneous mode: Pareto non-dominated sorting on (quality, -cost).
        In non-hetero mode: simple cumulative_score sorting.
        """
        if len(nodes) <= beam_width:
            return nodes

        if not self.heterogeneous:
            # Original UniMASO: sort by cumulative_score descending
            nodes.sort(key=lambda x: x.get("cumulative_score", 0), reverse=True)
            return nodes[:beam_width]

        # Heterogeneous mode: Pareto front selection

        # Compute Pareto fronts
        remaining = list(nodes)
        selected = []

        while remaining and len(selected) < beam_width:
            # Find non-dominated nodes in remaining set
            front = []
            non_dominated = []
            for i, a in enumerate(remaining):
                dominated = False
                for j, b in enumerate(remaining):
                    if i == j:
                        continue
                    # b dominates a if: quality_b >= quality_a AND cost_b <= cost_a
                    # AND at least one is strict
                    q_better = b.get("quality", 0) >= a.get("quality", 0)
                    c_better = b.get("cost", a.get("norm_cost", 1.0)) <= a.get("cost", a.get("norm_cost", 1.0))
                    q_strict = b.get("quality", 0) > a.get("quality", 0)
                    c_strict = b.get("cost", a.get("norm_cost", 1.0)) < a.get("cost", a.get("norm_cost", 1.0))
                    if q_better and c_better and (q_strict or c_strict):
                        dominated = True
                        break
                if not dominated:
                    non_dominated.append(a)
                else:
                    front.append(a)

            # If nothing was identified as non-dominated (e.g., all equal), take all remaining
            if not non_dominated:
                non_dominated = remaining
                front = []

            # Sort non-dominated by cumulative_score then take what we need
            non_dominated.sort(key=lambda x: x.get("cumulative_score", 0), reverse=True)

            needed = beam_width - len(selected)
            selected.extend(non_dominated[:needed])
            remaining = non_dominated[needed:] + front

        return selected[:beam_width]

    async def _compare(self, requirement: str, ans_a: str, ans_b: str, 
                       agent_id: int, question: str, context: str) -> bool:
        agent = self.mas.agents[agent_id]
        role_desc = get_role_description(agent.type, self.task_type)
    
        template = ANSWER_EVALUATE_TEMPLATE.get(
            self.task_type, ANSWER_EVALUATE_TEMPLATE[TaskType.MATH]
        )
    
        prompt = template.format(
            agent_type=agent.type.value,
            role_description=role_desc,
            question=question.strip(),
            requirement=requirement,
            Answer_A=ans_a,
            Answer_B=ans_b,
        )
        raw_result = await async_call_llm(
            self.evaluator, prompt, temperature=0, max_tokens=1024,
            return_usage=True,
        )
        if self.cost_tracker is not None:
            self.cost_tracker.record_llm(
                "evaluator", self.cost_model_name, raw_result.get("usage", {})
            )
        return parse_comparison_result(raw_result["text"])

    async def _compare_intermediate(self, question: str, output_a: str, output_b: str) -> bool:
        template = INTERMEDIATE_COMPARE_TEMPLATE.get(
            self.task_type, INTERMEDIATE_COMPARE_TEMPLATE[TaskType.MATH]
        )
    
        prompt = template.format(
            question=question.strip(),
            output_a=output_a.strip() or "(empty)",
            output_b=output_b.strip() or "(empty)"
        )
        resp_result = await async_call_llm(
            self.evaluator, prompt, temperature=0.0, return_usage=True
        )
        if self.cost_tracker is not None:
            self.cost_tracker.record_llm(
                "evaluator", self.cost_model_name, resp_result.get("usage", {})
            )
        return parse_comparison_result(resp_result["text"])

    async def _compare_final_answer(self, question: str, requirement: str, 
                                    ans_a: str, ans_b: str) -> bool:
        template = FINAL_ANSWER_COMPARE_TEMPLATE.get(
            self.task_type, FINAL_ANSWER_COMPARE_TEMPLATE[TaskType.MATH]
        )
    
        prompt = template.format(
            question=question.strip(),
            requirement=requirement,
            Answer_A=ans_a.strip(),
            Answer_B=ans_b.strip()
        )
        resp_result = await async_call_llm(
            self.evaluator, prompt, temperature=0.0, return_usage=True
        )
        if self.cost_tracker is not None:
            self.cost_tracker.record_llm(
                "evaluator", self.cost_model_name, resp_result.get("usage", {})
            )
        return parse_comparison_result(resp_result["text"])
    
    @staticmethod
    def _compute_kendall_top2_overlap(ranking_before: List[str], ranking_after: List[str]) -> float:
        if len(ranking_before) < 1 or len(ranking_after) < 1:
            return 1.0
        
        return 1.0 if ranking_before[0] == ranking_after[0] else 0.0



    def _sanitize_prompt(self, prompt: str, old_p: str) -> str:

        def normalize_braces(text: str) -> str:
            text = re.sub(r'\{+', '{', text) 
            text = re.sub(r'\}+', '}', text)
            return text

        prompt = normalize_braces(prompt)
        old_p = normalize_braces(old_p)

        all_placeholders = re.findall(r'\{([^}]*)\}', prompt)
        new_patterns = set()
        numeric_placeholders = []

        for ph in all_placeholders:
            if ph == "question" or ph == "context":
                new_patterns.add("{" + ph + "}")
            elif ph.isdigit():
                numeric_placeholders.append(ph)
            elif ph == "": 
                pass
            else:
                return old_p


        escaped_prompt = prompt.replace("{", "{{").replace("}", "}}")
        final_prompt = escaped_prompt.replace("{{question}}", "{question}")
        final_prompt = final_prompt.replace("{{context}}", "{context}")

        question_count = final_prompt.count("{question}")
        if question_count == 0:
            final_prompt += "\nQuestion: {question}"
        elif question_count > 1:
            return old_p

        context_count = final_prompt.count("{context}")
        old_has_context = "{context}" in old_p

        if old_has_context:
            if context_count == 0:
                final_prompt += "\nContext: {context}"
            elif context_count > 1:
                return old_p
        elif context_count > 1:
            return old_p

        return final_prompt

    def _sanitize_prompt_memory(self, prompt: str, old_p: str) -> str:
        """Memory-branch sanitizer that preserves JSON literal braces.

        The legacy sanitizer rejects any unknown ``{...}`` fragment. That is
        safe for ordinary templates but rejects BFCL JSON examples. In the
        optional memory branch, only runtime placeholders are unescaped; all
        other braces are treated as literal text for ``str.format``.
        """
        prompt = re.sub(r'\{+', '{', prompt)
        prompt = re.sub(r'\}+', '}', prompt)
        sentinels = {
            "{question}": "__UNIMASO_QUESTION_PLACEHOLDER__",
            "{context}": "__UNIMASO_CONTEXT_PLACEHOLDER__",
            "{context_block}": "__UNIMASO_CONTEXT_BLOCK_PLACEHOLDER__",
        }
        for placeholder, sentinel in sentinels.items():
            prompt = prompt.replace(placeholder, sentinel)

        escaped = prompt.replace("{", "{{").replace("}", "}}")
        for placeholder, sentinel in sentinels.items():
            escaped = escaped.replace(sentinel, placeholder)

        question_count = escaped.count("{question}")
        if question_count == 0:
            escaped += "\nQuestion: {question}"
        elif question_count > 1:
            return old_p

        old_has_context = "{context}" in old_p or "{context_block}" in old_p
        context_count = escaped.count("{context}") + escaped.count("{context_block}")
        if old_has_context and context_count == 0:
            escaped += "\nContext: {context}"
        elif context_count > 1:
            return old_p
        return escaped

    def _build_model_pool_section(self, current_model: str) -> str:
        """Build the model pool info table for the prompt template."""
        if not self.heterogeneous:
            return ""
        header = "| Model        | Capability | Rel. Cost | Best For                          |\n"
        header += "|--------------|------------|-----------|----------------------------------|\n"
        rows = []
        for m in self.model_pool:
            c = get_model_cost(m)
            ratio = (c["input"] + c["output"]) / 10.0
            profile = get_model_profile(m)
            cap = profile["capability"]
            use = profile["best_for"]
            marker = " ← CURRENT" if m == current_model else ""
            rows.append(f"| {m:12s} | {cap:10s} | {ratio:.1f}x      | {use}{marker}")
        if self.lambda_cost > 0:
            lambda_note = f"λ={self.lambda_cost}: multiplicative discount. adjusted_score = win_rate × (cost_ratio)^(-λ) - 0.5. Upgrade to 2x cost → quality × 0.87. Downgrade to 0.5x → quality × 1.15."
        else:
            lambda_note = "λ=0: ignore cost, maximize quality only."
        return header + "\n".join(rows) + f"\n\n{lambda_note}"

    def _build_trajectory_section(self, agent_id: int) -> str:
        """Build optimization history section for the prompt template."""
        if not self.heterogeneous:
            return ""
        if agent_id not in self.trajectory_log or not self.trajectory_log[agent_id]:
            return "(Round 1 — no history yet.)"
        entries = self.trajectory_log[agent_id][-3:]
        lines = ["| Round | Model      | Score  |",
                 "|-------|------------|--------|"]
        for e in entries:
            lines.append(f"| {e['round']:5d} | {e['model']:10s} | {e['score']:.3f} |")
        best = entries[-1]
        return "OPTIMIZATION TRAJECTORY:\n" + "\n".join(lines) + \
               f"\n→ Current best: {best['model']} (score={best['score']:.3f})"

    async def _propose_new_prompt(self, requirement: str, old_p: str,
                                  qa: Dict[str, Dict[str, str]], agent_id: int,
                                  question: str,
                                  current_model: str = None,
                                  successor_info: str = "",
                                  experience_context: str = "") -> Tuple[str, str]:
        agent = self.mas.agents[agent_id]
        role_desc = get_role_description(agent.type, self.task_type)
        cur_model = current_model or self.best_model.get(agent_id, DEFAULT_AGENT_MODEL)

        samples_block = "\n\n".join([
            f"Problem {i+1}:\n{q.strip()}\n\nContext:\n{data['context'].strip() or '(no context)'}\n\nAgent Output:\n{data['output'].strip()}"
            for i, (q, data) in enumerate(qa.items())
        ])

        full_requirement = "Ensure the agent's role, responsibilities, and input format remain consistent. " + requirement
        audience_instruction = ""
        if successor_info:
            audience_instruction = (
                f"\n\n[DOWNSTREAM CONSTRAINT & FEEDBACK]\n"
                f"The output of this agent serves as INPUT for a downstream agent.\n"
                f"{successor_info}\n"
                f"**Optimization Goal**: Crucially, modify the prompt so the output addresses the issues above and strictly adheres to constraints to help the downstream agent succeed."
            )
        augmented_requirements = full_requirement + audience_instruction

        # Memory-enabled heterogeneous runs use a separate optimizer contract.
        # The legacy template remains the exact path for non-memory runs.
        memory_enabled = (self.heterogeneous and self.trajectory_memory is not None
                         and self.enable_legacy_memory_injection)
        if memory_enabled:
            prompt = MEMORY_PROMPT_OPTIMIZE_TEMPLATE.format(
                task_type=self.task_type.value,
                agent_id=agent_id,
                agent_type=agent.type.value,
                role_description=role_desc,
                current_model=cur_model,
                model_pool_section=self._build_model_pool_section(cur_model),
                requirements=augmented_requirements,
                prompt=old_p,
                samples=samples_block,
                experience_context=experience_context,
            )
        elif self.heterogeneous:
            template = PROMPT_OPTIMIZE_TEMPLATE.get(
                self.task_type, PROMPT_OPTIMIZE_TEMPLATE[TaskType.MATH]
            )
            prompt = template.format(
                agent_type=agent.type.value,
                role_description=role_desc,
                current_model=cur_model,
                model_pool_section=self._build_model_pool_section(cur_model),
                trajectory_section=self._build_trajectory_section(agent_id),
                requirements=augmented_requirements,
                prompt=old_p,
                samples=samples_block,
                lambda_cost=self.lambda_cost,
            )
        else:
            from prompts import PROMPT_OPTIMIZE_TEMPLATE_NO_HETERO
            template = PROMPT_OPTIMIZE_TEMPLATE_NO_HETERO.get(
                self.task_type, PROMPT_OPTIMIZE_TEMPLATE_NO_HETERO[TaskType.MATH]
            )
            prompt = template.format(
                agent_type=agent.type.value,
                role_description=role_desc,
                requirements=augmented_requirements,
                prompt=old_p,
                samples=samples_block,
            )

        raw_result = await async_call_llm(
            self.evaluator, prompt, temperature=0.7,
            max_tokens=2048, return_usage=True,
        )
        if self.cost_tracker is not None:
            self.cost_tracker.record_llm(
                "optimizer", self.cost_model_name, raw_result.get("usage", {}),
                agent_id=agent_id,
            )
        raw = raw_result["text"]
        try:
            raw_prompt = raw.split("<prompt>")[1].split("</prompt>")[0].strip()
            log = get_logger()
            log.debug(f"Agent-{agent_id}: Proposed new prompt ({len(raw_prompt)} chars)")
        except IndexError:
            log = get_logger()
            log.warning(f"Agent-{agent_id}: Failed to extract <prompt>...</prompt>, keeping old prompt")
            return old_p, cur_model

        # Parse model recommendation from LLM output
        if self.heterogeneous:
            try:
                raw_model = raw.split("<model>")[1].split("</model>")[0].strip()
                if raw_model not in self.model_pool or raw_model not in MODEL_REGISTRY:
                    log = get_logger()
                    log.warning(f"Agent-{agent_id}: Invalid model '{raw_model}', keeping {cur_model}")
                    raw_model = cur_model
            except IndexError:
                raw_model = cur_model

            # Parse model_decision for logging
            try:
                decision = raw.split("<model_decision>")[1].split("</model_decision>")[0].strip()
                log = get_logger()
                log.debug(f"Agent-{agent_id}: Model decision: {decision}")
            except IndexError:
                pass
        else:
            raw_model = cur_model

        sanitized_prompt = self._sanitize_prompt(raw_prompt, old_p)
        if self.trajectory_memory is not None and sanitized_prompt == old_p:
            sanitized_prompt = self._sanitize_prompt_memory(raw_prompt, old_p)

        # Deterministic guard for the memory branch: retrieved experience may
        # guide an edit, but must not cause a prompt explosion. The optimizer
        # model is still free to propose a useful change within this budget.
        if memory_enabled:
            max_prompt_len = max(len(old_p) + 120, int(len(old_p) * 1.5))
            if len(sanitized_prompt) > max_prompt_len:
                get_logger().warning(
                    f"Agent-{agent_id}: memory candidate too long "
                    f"({len(sanitized_prompt)}>{max_prompt_len}); keeping current prompt"
                )
                sanitized_prompt = old_p
                raw_model = cur_model
        return sanitized_prompt, raw_model

    async def _build_baseline_caches(self, questions: List[str],
                                      prompt_map: Dict[int, str]) -> Dict[str, InferenceCache]:
        mas_temp = MAS(
            self.mas.gtype, self.mas.task_type,
            Ns=self.mas.Ns, Na=self.mas.Na, Nr=self.mas.Nr, Nd=self.mas.Nd,
            agent_model=self.mas.agent_model,
            agent_models=self._expand_model_map(self.best_model),
        )
        mas_temp.inject_prompt_map(self._expand_prompt_map(prompt_map))
    
        async def run_one(q: str) -> Tuple[str, InferenceCache]:
            _, cache = await mas_temp.arun_with_cache(q)
            if self.cost_tracker is not None:
                self.cost_tracker.record_mas_cache(
                    cache,
                    model_map=self._expand_model_map(self.best_model),
                    metadata={"kind": "baseline", "question": q},
                )
            return q, cache
    
        baseline_limit = max(1, int(os.getenv("MAS_BASELINE_CONCURRENCY", "3")))
        sem = asyncio.Semaphore(baseline_limit)
        async def run_one_limited(q):
            async with sem:
                return await run_one(q)
        results = await asyncio.gather(*[run_one_limited(q) for q in questions])
        return {q: cache for q, cache in results}

    async def _evaluate_candidate(self,
                                cand_prompt: str,
                                agent_id: int,
                                eval_samples: List[str],
                                baseline_caches: Dict[str, InferenceCache],
                                temp_prompt_map: Dict[int, str],
                                predecessors: List[int],
                                terminal_id: int,
                                is_terminal: bool,
                                requirement: str,
                                pure_local_mode: bool = False,
                                use_lookahead_score: bool = False,
                                lookahead_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
                                random_sample_set: Optional[Set[str]] = None,
                                cand_model: str = None,
                                temp_model_map: Dict[int, str] = None,
                                base_model_map: Dict[int, str] = None) -> Dict[str, Any]:
        n_eval = len(eval_samples)
        # Preserve an explicitly supplied candidate model.  The previous
        # conditional-expression precedence ignored cand_model whenever
        # temp_model_map was None and silently fell back to qwen35-4b.
        candidate_model = (
            cand_model
            if cand_model is not None
            else (
                temp_model_map.get(agent_id, DEFAULT_AGENT_MODEL)
                if temp_model_map else DEFAULT_AGENT_MODEL
            )
        )
        candidate_id = hashlib.sha1(
            f"{candidate_model}\n{cand_prompt}".encode("utf-8")
        ).hexdigest()[:12]
        _baseline_model_map = base_model_map or self.best_model
        successors = []
        if use_lookahead_score and not is_terminal:
            successors = self.mas.get_successors(agent_id)
        
        async def process_pipeline(q: str):
            base_cache = baseline_caches[q]
            # Build model map: candidate agent uses candidate_model, others use base or temp map
            eval_model_map = dict(_baseline_model_map)
            if temp_model_map:
                for aid, m in temp_model_map.items():
                    eval_model_map[aid] = m
            for member in self.shared_model_groups.get(self._model_leader(agent_id), (agent_id,)):
                eval_model_map[member] = candidate_model

            mas_eval = MAS(
                self.mas.gtype, self.mas.task_type,
                Ns=self.mas.Ns, Na=self.mas.Na,
                Nr=self.mas.Nr, Nd=self.mas.Nd,
                agent_model=self.mas.agent_model,
                agent_models=eval_model_map,
            )
            mas_eval.inject_prompt_map(self._expand_prompt_map(temp_prompt_map))
            result, cand_cache = await mas_eval.arun_from_node(agent_id, base_cache, cand_prompt)
            if self.cost_tracker is not None:
                self.cost_tracker.record_mas_cache(
                    cand_cache,
                    model_map=eval_model_map,
                    agent_ids=[agent_id, *self.mas.get_all_successors(agent_id)],
                    metadata={"kind": "candidate", "question": q, "agent_id": agent_id},
                )
            cand_raw_trace = result["raw_trace"]
            
            base_output = base_cache.node_outputs_raw.get(agent_id, "")
            cand_output = cand_raw_trace.get(agent_id, "")
            ctx = base_cache.get_context_for_node(agent_id, predecessors)
            
            sample_info = {
                "question": q,
                "context": ctx,
                "output": cand_output
            }
            trace_info = {
                "question": q,
                "context": ctx,
                "before_agent_output": base_output,
                "candidate_agent_output": cand_output,
                "before_final_output": base_cache.node_outputs_raw.get(terminal_id, ""),
                "candidate_final_output": cand_raw_trace.get(terminal_id, ""),
                "before_successor_outputs": {
                    str(s): base_cache.node_outputs_raw.get(s, "")
                    for s in self.mas.get_successors(agent_id)
                },
                "candidate_successor_outputs": {
                    str(s): cand_raw_trace.get(s, "")
                    for s in self.mas.get_successors(agent_id)
                },
            }
            
            local_tasks = []
            local_metas = []
            if is_terminal:
                local_tasks.append(
                    self._compare(requirement, cand_output, base_output, agent_id, q, ctx)
                )
                local_metas.append({"type": "terminal", "info": sample_info})
            else:
                if not pure_local_mode:
                    base_final = base_cache.node_outputs_raw.get(terminal_id, "")
                    cand_final = cand_raw_trace.get(terminal_id, "")
                    local_tasks.append(
                        self._compare_final_answer(q, requirement, cand_final, base_final)
                    )
                    local_metas.append({"type": "global", "info": sample_info})
            
                local_tasks.append(
                    self._compare_intermediate(q, cand_output, base_output)
                )
                local_metas.append({"type": "local", "info": sample_info})
                if use_lookahead_score and successors:
                    for succ_id in successors:
                        succ_cand_out = cand_raw_trace.get(succ_id, "")
                        succ_base_out = base_cache.node_outputs_raw.get(succ_id, "")
                        if succ_cand_out and succ_base_out:
                            local_tasks.append(self._compare_intermediate(q, succ_cand_out, succ_base_out))
                            local_metas.append({"type": "next_local", "info": sample_info})
            
            sample_scores = await asyncio.gather(*local_tasks)
            base_cost = compute_mas_cost(base_cache, model_map=_baseline_model_map)
            cand_cost = compute_mas_cost(cand_cache)
            # Agent-level cost: this agent alone is responsible for its model choice
            base_model_name = _baseline_model_map.get(agent_id, DEFAULT_AGENT_MODEL)
            base_c = get_model_cost(base_model_name)
            agent_base_cost = (base_c["input"] * base_cache.node_input_tokens.get(agent_id, 0) +
                               base_c["output"] * base_cache.node_output_tokens.get(agent_id, 0))
            cand_c = get_model_cost(candidate_model)
            agent_cand_cost = (cand_c["input"] * cand_cache.node_input_tokens.get(agent_id, 0) +
                               cand_c["output"] * cand_cache.node_output_tokens.get(agent_id, 0))
            return (sample_scores, local_metas, base_cost, cand_cost,
                    agent_base_cost, agent_cand_cost, trace_info)

        candidate_limit = max(1, int(os.getenv("MAS_CANDIDATE_CONCURRENCY", "10")))
        candidate_sem = asyncio.Semaphore(candidate_limit)

        async def process_pipeline_limited(q: str):
            async with candidate_sem:
                return await process_pipeline(q)

        all_pipelines = await asyncio.gather(*[
            process_pipeline_limited(q) for q in eval_samples
        ])

        results = []
        compare_metadata = []
        total_base_cost = 0.0
        total_cand_cost = 0.0
        total_agent_base_cost = 0.0
        total_agent_cand_cost = 0.0
        sample_traces = []
        for (scores, metas, base_cost, cand_cost, agent_base_cost,
             agent_cand_cost, trace_info) in all_pipelines:
            results.extend(scores)
            compare_metadata.extend(metas)
            total_base_cost += base_cost
            total_cand_cost += cand_cost
            total_agent_base_cost += agent_base_cost
            total_agent_cand_cost += agent_cand_cost
            sample_traces.append(trace_info)
        
        wins_global = 0
        wins_local = 0
        wins_next_local = 0
        count_next_local = 0
        bad_sample_candidates = {} 
        case_analysis_map = {}
        
        for res, meta in zip(results, compare_metadata):
            is_win = res
            comp_type = meta["type"]
            sample_info = meta["info"]
            q_key = sample_info["question"]
            
            if q_key not in case_analysis_map:
                case_analysis_map[q_key] = {"info": sample_info}
            
            if comp_type == "local":
                case_analysis_map[q_key]["local"] = is_win
            elif comp_type == "global":
                case_analysis_map[q_key]["global"] = is_win
            elif comp_type == "terminal":
                case_analysis_map[q_key]["global"] = is_win
            elif comp_type == "next_local":
                wins_next_local += 1 if is_win else 0
                count_next_local += 1
                case_analysis_map[q_key]["next_local"] = is_win
            
            if not is_win and (comp_type == "local" or comp_type == "terminal"):
                bad_sample_candidates[q_key] = sample_info
        
        collected_bad_cases = list(bad_sample_candidates.values())[:3]


        misalignment_count = 0
        total_evaluable = 0
        
        collected_misleading = []
        candidates_with_priority = []
        
        for q, data in case_analysis_map.items():
            if "local" not in data:
                continue
            
            is_random_sample = (random_sample_set is None) or (q in random_sample_set)
            
            is_local_win = data.get("local", False)
            
            if not is_local_win:
                if is_random_sample:
                    total_evaluable += 1
                continue
            
            is_global_win = data.get("global", True)
            is_next_win = data.get("next_local", True)
            
            if is_random_sample:
                total_evaluable += 1
                if (not is_next_win) or (not is_global_win):
                    misalignment_count += 1
            
            priority = -1
            if (not is_next_win) and (not is_global_win):
                priority = 0
            elif not is_next_win:
                priority = 1
            elif not is_global_win:
                priority = 2
            
            if priority != -1:
                candidates_with_priority.append({
                    "priority": priority,
                    "info": data["info"]
                })
        
        candidates_with_priority.sort(key=lambda x: x["priority"])
        collected_misleading = [x["info"] for x in candidates_with_priority[:3]]

        misalignment_rate = misalignment_count / total_evaluable if total_evaluable > 0 else 0.0
        if collected_misleading:
            log = get_logger()
            log.debug(f"Agent-{agent_id}: found {len(collected_misleading)} misleading cases (misalign_rate={misalignment_rate:.3f}, count={misalignment_count}/{total_evaluable})")
        
        if is_terminal:
            wins_global = sum(results)
            win_rate = wins_global / len(eval_samples) if eval_samples else 0
        else:
            wins_global = sum(r for r, m in zip(results, compare_metadata) if m["type"] == "global")
            wins_local = sum(r for r, m in zip(results, compare_metadata) if m["type"] == "local")
            
            if pure_local_mode:
                win_rate = wins_local / len(eval_samples) if eval_samples else 0
            else:
                rate_local = wins_local / n_eval if n_eval > 0 else 0
                rate_global = wins_global / n_eval if n_eval > 0 else 0
                
                if use_lookahead_score and count_next_local > 0:
                    rate_next = wins_next_local / count_next_local
                    w_local, w_next, w_global = lookahead_weights
                    win_rate = (rate_local * w_local) + (rate_next * w_next) + (rate_global * w_global)
                    log = get_logger()
                    log.debug(f"Agent-{agent_id} candidate score: local={rate_local:.3f}(w={w_local:.2f}), next={rate_next:.3f}(w={w_next:.2f}), global={rate_global:.3f}(w={w_global:.2f}) → composite={win_rate:.4f}")
                else:
                    win_rate = (rate_global * 0.3 + rate_local * 0.7)
                    log = get_logger()
                    log.debug(f"Agent-{agent_id} candidate score: local={rate_local:.3f}, global={rate_global:.3f} → composite={win_rate:.4f}")

        # Keep both actual token-cost ratios and model-price ratios. The
        # actual Agent-level ratio is used for the cost-aware score below.
        norm_cost = total_cand_cost / max(total_base_cost, 1.0)
        agent_norm_cost = total_agent_cand_cost / max(total_agent_base_cost, 1.0)

        # Retain the model-price ratio for diagnostics. It does not include
        # the candidate's actual input/output token counts.
        base_model_name = _baseline_model_map.get(agent_id, DEFAULT_AGENT_MODEL)
        model_norm = (
            get_model_price_ratio(candidate_model, base_model_name)
            if self.heterogeneous else 1.0
        )
        # Use the current Agent's actual token-weighted cost ratio for scoring.
        # The lower bound prevents zero-cost traces from causing a negative
        # exponent error or an infinite discount.
        score_cost_norm = (
            max(agent_norm_cost, 1e-6)
            if self.heterogeneous else 1.0
        )
        discount = (
            score_cost_norm ** (-self.lambda_cost)
            if self.lambda_cost > 0 else 1.0
        )
        cost_penalty = 0.0  # kept for logging compatibility
        adjusted_score = win_rate * discount - 0.5

        # Detect "Quality Win, Efficiency Lose" misalignment:
        # candidate wins on quality but with disproportionate cost increase
        quality_cost_misalign = 0
        quality_cost_total = 0
        if total_evaluable > 0 and agent_norm_cost > 1.1:
            for q, data in case_analysis_map.items():
                if "local" not in data:
                    continue
                if data.get("local", False) and not data.get("global", True):
                    quality_cost_misalign += 1
                quality_cost_total += 1
        quality_cost_misalign_rate = (
            quality_cost_misalign / quality_cost_total if quality_cost_total > 0 else 0.0
        )

        log = get_logger()
        log.debug(
            f"Agent-{agent_id}: candidate_id={candidate_id} "
            f"prompt_len={len(cand_prompt)} "
            f"base_model={base_model_name} "
            f"cand_model={candidate_model} "
            f"base_cost={total_base_cost:.2f} "
            f"cand_cost={total_cand_cost:.2f} "
            f"model_price_norm={model_norm:.3f} "
            f"agent_cost_norm={agent_norm_cost:.3f} "
            f"score_cost_norm={score_cost_norm:.3f} "
            f"total_cost_norm={norm_cost:.3f} "
            f"discount={discount:.4f} "
            f"raw_score={win_rate:.4f} "
            f"adjusted={adjusted_score:.4f}"
        )

        local_rate = wins_local / n_eval if n_eval else 0.0
        global_rate = wins_global / n_eval if n_eval else 0.0
        if self.trajectory_memory is not None:
            current_prompt = temp_prompt_map.get(agent_id, self.best_prompt.get(agent_id, ""))
            current_model = _baseline_model_map.get(agent_id, DEFAULT_AGENT_MODEL)
            retrieval_view = self._build_memory_state(
                aid=agent_id,
                node_prompt=current_prompt,
                node_model=current_model,
                preds=predecessors,
                samples=eval_samples,
                baseline_caches=baseline_caches,
            )
            score_record = {
                "score": adjusted_score,
                "quality": win_rate,
                "local_rate": local_rate,
                "global_rate": global_rate,
                "local_delta": local_rate - 0.5,
                "global_delta": global_rate - 0.5,
                "next_delta": (wins_next_local / count_next_local - 0.5)
                    if count_next_local else 0.0,
                "cost_ratio": score_cost_norm,
                "model_price_ratio": model_norm,
                "agent_cost_ratio": agent_norm_cost,
                "norm_cost": norm_cost,
                "misalignment_rate": misalignment_rate,
                "wins_local": wins_local,
                "wins_global": wins_global,
                "wins_next_local": wins_next_local,
            }
            record = {
                "run_id": getattr(self.trajectory_memory, "run_dir", "" ).__str__(),
                "step": len(self.trajectory_memory.records),
                "record_type": "candidate_evaluation",
                "state": retrieval_view,
                "retrieval_view": retrieval_view,
                "action": {
                    "candidate_prompt": cand_prompt,
                    "candidate_model": candidate_model,
                },
                "score": score_record,
                "status": {
                    "accepted": adjusted_score > 0,
                    "accepted_by_optimizer": adjusted_score > 0,
                    "selection_policy": "adjusted_score > 0",
                },
                "payload": {
                    "before_prompt": current_prompt,
                    "before_model": current_model,
                    "candidate_prompt": cand_prompt,
                    "candidate_model": candidate_model,
                    "requirement": requirement,
                    "samples": sample_traces,
                },
            }
            await self.trajectory_memory.add(record)

        return {
            "candidate_id": candidate_id,
            "candidate_model": candidate_model,
            "candidate_prompt_len": len(cand_prompt),
            "score": adjusted_score,
            "quality": win_rate,
            "cost": norm_cost,
            "score_cost_norm": score_cost_norm,
            "model_price_norm": model_norm,
            "agent_norm_cost": agent_norm_cost,
            "wins_global": wins_global,
            "wins_local": wins_local,
            "wins_next_local": wins_next_local,
            "bad_cases": collected_bad_cases,
            "misleading_cases": collected_misleading,
            "misalignment_rate": misalignment_rate,
            "misalignment_count": misalignment_count,
            "total_evaluable": total_evaluable,
            "total_base_cost": total_base_cost,
            "total_cand_cost": total_cand_cost,
            "norm_cost": norm_cost,
            "quality_cost_misalign_rate": quality_cost_misalign_rate,
            "local_rate": local_rate,
            "global_rate": global_rate,
        }


    async def _optimize_agent(self, agent_id: int, requirement: str) -> str:
        agent = self.mas.agents[agent_id]
        best_p = self.best_prompt[agent_id]
        terminal_id = self.mas.get_terminal_id()
        is_terminal = (agent_id == terminal_id)
        predecessors = self.mas.get_predecessors(agent_id)
        log = get_logger()

        log.info(f"Agent-{agent_id} ({agent.type.value}){' [TERM]' if is_terminal else ''}: starting sequential optimization (max_depth=10, beam_width=2)")

        current_level = [{
            "prompt": best_p,
            "model": self.best_model.get(agent_id, DEFAULT_AGENT_MODEL),
            "cumulative_score": 0.0,
            "quality": 0.5,
            "cost": 1.0,
            "path": [best_p],
        }]
        max_depth = 10
        beam_width = 2
        best_overall = {"prompt": best_p, "model": self.best_model.get(agent_id, DEFAULT_AGENT_MODEL), "cumulative_score": 0.0}

        desc_prefix = f"Agent-{agent_id}({agent.type.value}){'[TERM]' if is_terminal else ''}"
        current_base_map = self.best_prompt.copy()
        with tqdm(total=max_depth, desc=desc_prefix, leave=False) as tree_bar:
            for depth in range(max_depth):
                eval_samples = self.get_eval_samples(depth)
                eval_samples_part1 = eval_samples[:5]
                eval_samples_part2 = eval_samples[5:]
                n_eval = len(eval_samples)

                node_results = await asyncio.gather(*[
                    self.process_single_node(
                        node=node,
                        aid=agent_id,
                        preds=predecessors,
                        term_id=terminal_id,
                        is_term=is_terminal,
                        req=requirement,
                        samples=eval_samples,
                        samples_p1=eval_samples_part1,
                        samples_p2=eval_samples_part2,
                        base_prompt_map=current_base_map,
                        agent_states=None,
                        use_stochastic_sampling=False,
                        use_feedback=False
                    )
                    for node in current_level
                ])

                all_next_nodes = []
                for res in node_results:
                    all_next_nodes.extend(res["nodes"])
                    if res["best_cumulative"] > best_overall["cumulative_score"]:
                        old_score = best_overall["cumulative_score"]
                        best_overall["prompt"] = res["best_prompt"]
                        best_overall["model"] = res.get("best_model", DEFAULT_AGENT_MODEL)
                        best_overall["cumulative_score"] = res["best_cumulative"]
                        self._set_best_model(agent_id, res.get("best_model", DEFAULT_AGENT_MODEL))
                        log.debug(f"Agent-{agent_id} depth={depth+1}: score improved {old_score:.4f} → {res['best_cumulative']:.4f}, model={res.get('best_model', '?')}, new prompt len={len(res['best_prompt'])}")

                all_next_nodes.sort(key=lambda x: x["cumulative_score"], reverse=True)
                current_level = self._beam_select(all_next_nodes, beam_width)

                beam_info = ", ".join([f"{n['cumulative_score']:.4f}" for n in current_level])
                log.debug(f"Agent-{agent_id} depth={depth+1}: beam=[{beam_info}]")
                tree_bar.set_postfix_str(f"depth {depth+1}, best={best_overall['cumulative_score']:.3f}")
                tree_bar.update(1)

                if not current_level:
                    log.debug(f"Agent-{agent_id}: beam empty at depth {depth+1}, stopping")
                    break

        log.info(f"Agent-{agent_id}: sequential optimization done, best_score={best_overall['cumulative_score']:.4f}, model={best_overall.get('model', '?')}, final_prompt_len={len(best_overall['prompt'])}")
        self._set_best_model(agent_id, best_overall.get("model", self.best_model.get(agent_id, DEFAULT_AGENT_MODEL)))
        return best_overall["prompt"]

    async def optimize_all(self, requirement: str = " ") -> Tuple[Dict[int, str], Dict[int, str]]:
        log = get_logger()
        order = self._optimizable_ids(self.mas._topo_order())
        log.info(f"Starting sequential optimization, topo_order={order}")
        round_initial_prompts = self.best_prompt.copy()
        round_initial_models = self.best_model.copy()

        with tqdm(total=len(order), desc="[OPT]") as topo_bar:
            for aid in order:
                if self.mas.agents[aid].type == AgentType.AGGREGATOR:
                    log.debug(f"Agent-{aid}: skipping (AGGREGATOR)")
                    topo_bar.update(1)
                    continue

                new_p = await self._optimize_agent(aid, requirement)
                self._set_best_prompt(aid, new_p)

                topo_bar.set_postfix_str(f"Agent-{aid} done")
                topo_bar.update(1)

        if self.co_evolver is not None:
            await self._apply_co_evolver(
                round_initial_prompts, round_initial_models, round_number=1
            )
        log.info("Sequential optimization complete")
        return self.best_prompt, self.best_model


    async def _optimize_agent_stateful(self,
                                    state: AgentOptState,
                                    requirement: str,
                                    max_depth: int = 10,
                                    beam_width: int = 2,
                                    patience: int = 3) -> Tuple[int, bool]:
        agent_id = state.agent_id
        if self.mas.agents[agent_id].type == AgentType.AGGREGATOR:
            return 0, False
        
        if state.total_layers_explored >= max_depth:
            return 0, False

        terminal_id = self.mas.get_terminal_id()
        is_terminal = (agent_id == terminal_id)
        predecessors = self.mas.get_predecessors(agent_id)
        initial_prompt_map = self.best_prompt.copy()
        current_best_prompt = initial_prompt_map[agent_id]

        if state.total_layers_explored == 0:
            initial_node = {
                "prompt": current_best_prompt,
                "model": self.best_model.get(agent_id, DEFAULT_AGENT_MODEL),
                "cumulative_score": 0.0,
                "quality": 0.5,
                "cost": 1.0,
                "path": [current_best_prompt]
            }
            state.current_beam = [initial_node]
            state.best_overall_node = initial_node.copy()

        depth = state.total_layers_explored
        eval_samples = self.get_eval_samples(depth)
        eval_samples_part1 = eval_samples[:5]
        eval_samples_part2 = eval_samples[5:]

        async def process_single_node(node: Dict, aid: int, preds: List[int], term_id: int,
                                    is_term: bool, req: str, samples: List[str],
                                    samples_p1: List[str], samples_p2: List[str]) -> Dict:
            node_prompt = node["prompt"]
            temp_prompt_map = self.best_prompt.copy()
            temp_prompt_map[aid] = node_prompt

            local_baseline_caches = await self._build_baseline_caches(samples, temp_prompt_map)
            experience_context = ""
            if self.enable_legacy_memory_injection and self.trajectory_memory is not None and samples:
                query_state = self._build_memory_state(
                    aid=aid,
                    node_prompt=node_prompt,
                    node_model=node.get("model", self.best_model.get(aid, DEFAULT_AGENT_MODEL)),
                    preds=preds,
                    samples=samples,
                    baseline_caches=local_baseline_caches,
                )
                await self._record_baseline_snapshot(
                    query_state, aid, node_prompt,
                    node.get("model", self.best_model.get(aid, DEFAULT_AGENT_MODEL)),
                    preds, samples, local_baseline_caches, req,
                )
                retrieved = await self.trajectory_memory.retrieve(
                    query_state,
                    max_items=3 * self.memory_per_category,
                    per_category=self.memory_per_category,
                )
                experience_context = self.trajectory_memory.format_for_optimizer(retrieved)
                if retrieved:
                    get_logger().debug(
                        f"Agent-{aid}: retrieved {len(retrieved)} trajectory experiences"
                    )
            qa_for_proposal1 = {}
            for q in samples_p1:
                cache = local_baseline_caches[q]
                ctx = cache.get_context_for_node(aid, preds)
                qa_for_proposal1[q] = {"context": ctx, "output": cache.node_outputs_raw.get(aid, "")}

            qa_for_proposal2 = {}
            for q in samples_p2:
                cache = local_baseline_caches[q]
                ctx = cache.get_context_for_node(aid, preds)
                qa_for_proposal2[q] = {"context": ctx, "output": cache.node_outputs_raw.get(aid, "")}

            sample1 = samples_p1[0] if samples_p1 else samples[0]
            sample2 = samples_p2[0] if samples_p2 else samples[-1]

            raw_cand_results = await asyncio.gather(
                self._propose_new_prompt(req, node_prompt, qa_for_proposal1, aid, sample1,
                                        current_model=node.get("model"),
                                        experience_context=experience_context),
                self._propose_new_prompt(req, node_prompt, qa_for_proposal2, aid, sample2,
                                        current_model=node.get("model"),
                                        experience_context=experience_context)
                , return_exceptions=True
            )
            cand_results = []
            for result in raw_cand_results:
                if isinstance(result, Exception):
                    get_logger().warning(
                        f"Agent-{aid}: prompt proposal failed; skipping candidate: "
                        f"{type(result).__name__}: {result}"
                    )
                    continue
                cand_results.append(result)
            node_model = node.get("model", self.best_model.get(aid, DEFAULT_AGENT_MODEL))
            # Build (prompt, model) candidates from LLM recommendations
            pm_candidates = [(node_prompt, node_model)]
            seen = {(node_prompt, node_model)}
            for cand_p, cand_m in cand_results:
                key = (cand_p, cand_m)
                if key not in seen:
                    seen.add(key)
                    pm_candidates.append(key)
            candidate_scores = {}

            for cand_p, cand_m in pm_candidates:
                if cand_p == node_prompt and cand_m == node_model:
                    candidate_scores[(cand_p, cand_m)] = {"score": 0.0, "wins_global": len(samples)//2, "wins_local": len(samples)//2, "quality": 0.5, "cost": 1.0, "norm_cost": 1.0}
                    continue
                try:
                    score_info = await self._evaluate_candidate(
                        cand_prompt=cand_p, agent_id=aid, eval_samples=samples,
                        baseline_caches=local_baseline_caches, temp_prompt_map=temp_prompt_map,
                        predecessors=preds, terminal_id=term_id, is_terminal=is_term, requirement=req,
                        cand_model=cand_m
                    )
                except Exception as exc:
                    get_logger().warning(
                        f"Agent-{aid}: candidate evaluation failed; keeping baseline: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    score_info = {"score": 0.0, "quality": 0.5, "cost": 1.0,
                                  "norm_cost": 1.0}
                expected_id = hashlib.sha1(
                    f"{cand_m}\n{cand_p}".encode("utf-8")
                ).hexdigest()[:12]
                scored_model = score_info.get("candidate_model", cand_m)
                scored_id = score_info.get("candidate_id", expected_id)
                if scored_model != cand_m or scored_id != expected_id:
                    raise RuntimeError(
                        f"candidate trace mismatch: pair=({cand_m},{expected_id}) "
                        f"scored=({scored_model},{scored_id})"
                    )
                score_info["candidate_model"] = scored_model
                score_info["candidate_id"] = scored_id
                score_info.setdefault("candidate_prompt_len", len(cand_p))
                candidate_scores[(cand_p, cand_m)] = score_info

            new_nodes = []
            local_best = {
                "prompt": node_prompt,
                "model": node_model,
                "score": 0.0,
                "candidate_id": hashlib.sha1(
                    f"{node_model}\n{node_prompt}".encode("utf-8")
                ).hexdigest()[:12],
                "norm_cost": 1.0,
            }
            for cand_p, cand_m in pm_candidates:
                info = candidate_scores[(cand_p, cand_m)]
                if info["score"] > 0:
                    new_node = {
                        "prompt": cand_p,
                        "model": cand_m,
                        "cumulative_score": node["cumulative_score"] + info["score"],
                        "quality": info.get("quality", 0.0),
                        "cost": info.get("norm_cost", 1.0),
                        "path": node["path"] + [cand_p],
                    }
                    new_nodes.append(new_node)
                    if info["score"] > local_best["score"]:
                        local_best = {
                            "prompt": cand_p,
                            "model": cand_m,
                            "score": info["score"],
                            "candidate_id": info.get("candidate_id"),
                            "norm_cost": info.get("norm_cost", 1.0),
                        }
                else:
                    new_nodes.append({
                        "prompt": node_prompt,
                        "model": node_model,
                        "cumulative_score": node["cumulative_score"],
                        "quality": node.get("quality", 0.5),
                        "cost": node.get("cost", 1.0),
                        "path": node["path"],
                    })

            return {
                "nodes": new_nodes,
                "best_prompt": local_best["prompt"],
                "best_model": local_best.get("model", node_model),
                "best_candidate_id": local_best.get("candidate_id"),
                "best_norm_cost": local_best.get("norm_cost", 1.0),
                "best_cumulative": node["cumulative_score"] + max(0, local_best["score"]),
            }

        if not state.current_beam:
            state.current_beam = [{
                "prompt": state.best_overall_node["prompt"],
                "cumulative_score": state.best_overall_node["cumulative_score"],
                "quality": state.best_overall_node.get("quality", 0.5),
                "cost": state.best_overall_node.get("cost", 1.0),
                "path": state.best_overall_node["path"]
            }]

        node_results = await asyncio.gather(*[
            process_single_node(
                node=node, aid=agent_id, preds=predecessors, term_id=terminal_id,
                is_term=is_terminal, req=requirement, samples=eval_samples,
                samples_p1=eval_samples_part1, samples_p2=eval_samples_part2
            )
            for node in state.current_beam
        ])

        all_next_nodes = []
        current_best_node = state.best_overall_node.copy()
        for res in node_results:
            all_next_nodes.extend(res["nodes"])
            if res["best_cumulative"] > current_best_node["cumulative_score"]:
                current_best_node = {
                    "prompt": res["best_prompt"],
                    "model": res.get("best_model", DEFAULT_AGENT_MODEL),
                    "candidate_id": res.get("best_candidate_id"),
                    "norm_cost": res.get("best_norm_cost", 1.0),
                    "cumulative_score": res["best_cumulative"],
                    "path": res.get("path", [])
                }

        all_next_nodes.sort(key=lambda x: x["cumulative_score"], reverse=True)
        state.current_beam = self._beam_select(all_next_nodes, beam_width)

        improved = current_best_node["cumulative_score"] > state.best_overall_node["cumulative_score"] + 1e-6
        trigger_patience = False
        base_score = state.best_overall_node["cumulative_score"]
        if improved:
            state.current_patience_count = 0
            state.best_overall_node = current_best_node
            self._set_best_model(agent_id, current_best_node.get("model", self.best_model.get(agent_id, DEFAULT_AGENT_MODEL)))
        else:
            state.current_patience_count += 1
            if state.current_patience_count >= patience:
                trigger_patience = True

        score_delta = state.best_overall_node["cumulative_score"] - base_score
        if score_delta > 0:
            old_prompt = self.best_prompt[agent_id]
            new_prompt = state.best_overall_node["prompt"]
            self._set_best_prompt(agent_id, new_prompt)
            self._set_best_model(agent_id, state.best_overall_node.get("model", self.best_model.get(agent_id, DEFAULT_AGENT_MODEL)))
            log = get_logger()
            log.info(f"Agent-{agent_id}: prompt improved (delta={score_delta:+.4f}), new_len={len(new_prompt)}")

        state.total_layers_explored += 1
        if trigger_patience:
            log = get_logger()
            log.debug(f"Agent-{agent_id}: early stopping (patience) at depth={state.total_layers_explored}")
        return 1, trigger_patience


    async def optimize_all_round_robin(self,
                                    requirement: str = " ",
                                    max_total_depth: int = 10,
                                    beam_width: int = 2,
                                    patience: int = 3) -> Dict[int, str]:
        log = get_logger()
        topo_order = self.mas._topo_order()
        optimizable_agents = [
            aid for aid in self._optimizable_ids(topo_order)
            if self.mas.agents[aid].type != AgentType.AGGREGATOR
        ]

        if not optimizable_agents:
            return self.best_prompt, self.best_model

        agent_states = {
            aid: AgentOptState(aid, self.best_prompt[aid],
                              initial_model=self.best_model.get(aid))
            for aid in optimizable_agents
        }

        log.info(f"Starting round-robin optimization: agents={optimizable_agents}, max_depth={max_total_depth}, patience={patience}")
        log.info(f"Initial model assignments: {{{', '.join(f'{aid}:{self.best_model.get(aid)}' for aid in optimizable_agents)}}}")

        round_num = 0
        while True:
            round_num += 1
            all_completed = all(
                state.total_layers_explored >= max_total_depth
                for state in agent_states.values()
            )
            if all_completed:
                break

            log.debug(f"Round-robin round {round_num}")
            round_initial_prompts = self.best_prompt.copy()
            round_initial_models = self.best_model.copy()
            for aid in optimizable_agents:
                state = agent_states[aid]
                if state.total_layers_explored >= max_total_depth:
                    continue

                agent = self.mas.agents[aid]
                agent_type_str = f"{agent.type.value}{' [TERM]' if aid == self.mas.get_terminal_id() else ''}"
                state.current_patience_count = 0

                log.debug(f"Round-robin: Agent-{aid} ({agent_type_str}) depth={state.total_layers_explored}/{max_total_depth}")
                while True:
                    if state.total_layers_explored >= max_total_depth:
                        break

                    layers_explored, trigger_patience = await self._optimize_agent_stateful(
                        state=state,
                        requirement=requirement,
                        max_depth=max_total_depth,
                        beam_width=beam_width,
                        patience=patience
                    )

                    if trigger_patience:
                        log.debug(f"Round-robin: Agent-{aid} paused (patience) at depth={state.total_layers_explored}")
                        break

            if self.co_evolver is not None:
                await self._apply_co_evolver(
                    round_initial_prompts, round_initial_models, round_num
                )

        log.info("Round-robin optimization complete")
        return self.best_prompt, self.best_model


    async def _optimize_agent_fixed_rounds(self,
                                        state: AgentOptState,
                                        requirement: str,
                                        agent_states: Dict[int, AgentOptState],
                                        use_dynamic_switching: bool = False,
                                        use_stochastic_sampling: bool = False,
                                        use_beam_refresh: bool = False,
                                        rounds_per_turn: int = 2,
                                        max_total_depth: int = 10,
                                        use_feedback: bool = False,
                                        use_misleading_sampling: bool = False,
                                        use_lookahead_score: bool = False,
                                        lookahead_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> int:
        agent_id = state.agent_id
        if self.mas.agents[agent_id].type == AgentType.AGGREGATOR:
            return 0

        log = get_logger()
        terminal_id = self.mas.get_terminal_id()
        is_terminal = (agent_id == terminal_id)
        predecessors = self.mas.get_predecessors(agent_id)

        if use_beam_refresh and state.total_layers_explored > 0 and state.current_beam:
            log.debug(f"Agent-{agent_id}: triggering Beam Refresh at depth={state.total_layers_explored}")
            await self._refresh_beam_scores(
                state, requirement, predecessors, terminal_id, is_terminal
            )

        completed_rounds = 0
        if state.total_layers_explored == 0:
            initial_node = {
                "prompt": state.best_overall_node["prompt"],
                "model": state.best_overall_node.get("model", DEFAULT_AGENT_MODEL),
                "cumulative_score": 0.0,
                "quality": 0.5,
                "cost": 1.0,
                "path": [state.best_overall_node["prompt"]]
            }
            state.current_beam = [initial_node]

        for round_i in range(rounds_per_turn):
            if state.total_layers_explored >= max_total_depth:
                break

            round_start = time.time()
            current_base_map = None
            if use_dynamic_switching and agent_states:
                current_base_map = self._get_dynamic_context(
                    agent_states,
                    current_agent_id=agent_id,
                    current_depth=state.total_layers_explored
                )

            eval_samples = self.get_eval_samples(state.total_layers_explored)
            eval_samples_part1 = eval_samples[:5]
            eval_samples_part2 = eval_samples[5:]

            if not state.current_beam:
                state.current_beam = [{
                    "prompt": state.best_overall_node["prompt"],
                    "cumulative_score": state.best_overall_node["cumulative_score"],
                    "quality": state.best_overall_node.get("quality", 0.5),
                    "cost": state.best_overall_node.get("cost", 1.0),
                    "path": state.best_overall_node["path"]
                }]

            node_results = await asyncio.gather(*[
                self.process_single_node(
                    node=node,
                    aid=agent_id,
                    preds=predecessors,
                    term_id=terminal_id,
                    is_term=is_terminal,
                    req=requirement,
                    samples=eval_samples,
                    samples_p1=eval_samples_part1,
                    samples_p2=eval_samples_part2,
                    base_prompt_map=current_base_map,
                    agent_states=agent_states,
                    use_stochastic_sampling=use_stochastic_sampling,
                    use_feedback=use_feedback,
                    use_misleading_sampling=use_misleading_sampling,
                    use_lookahead_score=use_lookahead_score,
                    lookahead_weights=lookahead_weights,
                )
                for node in state.current_beam
            ])
            depth_misalignment_rates = []
            for res in node_results:
                if "avg_misalignment_rate" in res:
                    depth_misalignment_rates.append(res["avg_misalignment_rate"])

            if depth_misalignment_rates:
                avg_depth_misalignment = sum(depth_misalignment_rates) / len(depth_misalignment_rates)
            else:
                avg_depth_misalignment = 0.0
            state.misalignment_rates_per_depth.append(avg_depth_misalignment)

            all_next_nodes = []
            current_best_node = state.best_overall_node.copy()
            current_best_bad_cases = []
            current_best_misleading = []

            for res in node_results:
                all_next_nodes.extend(res["nodes"])
                if res["best_cumulative"] > current_best_node["cumulative_score"]:
                    current_best_node = {
                        "prompt": res["best_prompt"],
                        "model": res.get("best_model", DEFAULT_AGENT_MODEL),
                        "candidate_id": res.get("best_candidate_id"),
                        "norm_cost": res.get("best_norm_cost", 1.0),
                        "cumulative_score": res["best_cumulative"],
                        "path": res.get("path", [])
                    }
                    current_best_bad_cases = res.get("best_bad_cases", [])
                    current_best_misleading = res.get("best_misleading_cases", [])
            all_next_nodes.sort(key=lambda x: x["cumulative_score"], reverse=True)
            state.current_beam = self._beam_select(all_next_nodes, self.beam_width)

            old_score = state.best_overall_node["cumulative_score"]
            if current_best_node["cumulative_score"] > state.best_overall_node["cumulative_score"] + 1e-6:
                state.best_overall_node = current_best_node
                self._set_best_prompt(agent_id, current_best_node["prompt"])
                self._set_best_model(agent_id, current_best_node.get("model", self.best_model.get(agent_id, DEFAULT_AGENT_MODEL)))
                log.info(f"Agent-{agent_id} depth={state.total_layers_explored+1}: score {old_score:.4f} → {current_best_node['cumulative_score']:.4f} (delta={current_best_node['cumulative_score']-old_score:+.4f}), model={current_best_node.get('model', '?')}, prompt_len={len(current_best_node['prompt'])}")
                log.debug(
                    f"Agent-{agent_id} selection_trace: "
                    f"candidate_id={current_best_node.get('candidate_id', '?')} "
                    f"model={current_best_node.get('model', '?')} "
                    f"cost_norm={current_best_node.get('norm_cost', 1.0):.3f} "
                    f"prompt_len={len(current_best_node['prompt'])} "
                    f"score={current_best_node['cumulative_score']:.4f}"
                )
                # Log trajectory for LLM model-selection context
                if agent_id not in self.trajectory_log:
                    self.trajectory_log[agent_id] = []
                self.trajectory_log[agent_id].append({
                    "round": state.total_layers_explored + 1,
                    "model": current_best_node.get("model", "?"),
                    "score": current_best_node["cumulative_score"],
                    "prompt": current_best_node["prompt"],
                })
                if current_best_bad_cases:
                    state.recent_bad_cases = current_best_bad_cases
                    log.debug(f"Agent-{agent_id}: {len(current_best_bad_cases)} bad cases collected")
                if current_best_misleading:
                    state.misleading_cases = current_best_misleading
                    log.debug(f"Agent-{agent_id}: {len(current_best_misleading)} misleading cases collected (misalign_rate={avg_depth_misalignment:.3f})")
            else:
                log.debug(f"Agent-{agent_id} depth={state.total_layers_explored+1}: no improvement (best={old_score:.4f}, curr_best={current_best_node['cumulative_score']:.4f})")

            beam_scores = ", ".join([f"{n['cumulative_score']:.4f}" for n in state.current_beam])
            log.debug(f"Agent-{agent_id} beam after depth {state.total_layers_explored+1}: [{beam_scores}], misalign={avg_depth_misalignment:.3f}, round_time={time.time()-round_start:.1f}s")

            state.total_layers_explored += 1
            completed_rounds += 1
        return completed_rounds

    async def optimize_all_fixed_rounds(self,
                                    requirement: str = " ",
                                    max_total_depth: int = 10,
                                    rounds_per_turn: int = 2,
                                    flexible_round_depths: Optional[List[int]] = None,
                                    beam_width: int = 2,
                                    use_dynamic_switching: bool = False,
                                    use_stochastic_sampling: bool = False,
                                    use_beam_refresh: bool = False,
                                    use_feedback: bool = False,
                                    use_misleading_sampling: bool = False,
                                    use_lookahead_score: bool = False,
                                    lookahead_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> Dict[int, str]:
        log = get_logger()
        self.beam_width = beam_width
        topo_order = self.mas._topo_order()
        optimizable_agents = [
            aid for aid in self._optimizable_ids(topo_order)
            if self.mas.agents[aid].type != AgentType.AGGREGATOR
        ]
        if not optimizable_agents:
            log.warning("No optimizable agents found")
            return self.best_prompt, self.best_model, {}

        agent_states = {
            aid: AgentOptState(aid, self.best_prompt[aid],
                              initial_model=self.best_model.get(aid))
            for aid in optimizable_agents
        }

        flags = []
        if use_dynamic_switching: flags.append("DAS")
        if use_stochastic_sampling: flags.append("SBCS")
        if use_beam_refresh: flags.append("BR")
        if use_feedback: flags.append("FB")
        if use_misleading_sampling: flags.append("MS")
        if use_lookahead_score: flags.append(f"LS{lookahead_weights}")
        if flexible_round_depths is not None:
            if not flexible_round_depths or any(depth <= 0 for depth in flexible_round_depths):
                raise ValueError("flexible_round_depths must contain positive integers")
            round_schedule = list(flexible_round_depths)
            max_total_depth = sum(round_schedule)
        else:
            if rounds_per_turn <= 0:
                raise ValueError("rounds_per_turn must be positive")
            round_schedule = None

        log.info(f"Starting fixed-rounds optimization: agents={optimizable_agents}, max_depth={max_total_depth}, rounds_per_turn={rounds_per_turn}, round_schedule={round_schedule}, beam_width={beam_width}, flags={flags}")
        if self.opt_seed is not None:
            log.info(f"Random seed: {self.opt_seed}")
        log.info(f"Initial model assignments: {{{', '.join(f'{aid}:{self.best_model.get(aid)}' for aid in optimizable_agents)}}}")

        round_num = 0
        if round_schedule is None:
            round_schedule = []
            remaining_depth = max_total_depth
            while remaining_depth > 0:
                steps = min(rounds_per_turn, remaining_depth)
                round_schedule.append(steps)
                remaining_depth -= steps

        for scheduled_steps in round_schedule:
            round_num += 1

            log.debug(f"Fixed-rounds: round {round_num}")
            round_initial_prompts = self.best_prompt.copy()
            round_initial_models = self.best_model.copy()
            for aid in optimizable_agents:
                state = agent_states[aid]
                if state.total_layers_explored >= max_total_depth:
                    continue

                agent = self.mas.agents[aid]
                agent_type_str = f"{agent.type.value}{' [TERM]' if aid == self.mas.get_terminal_id() else ''}"
                remaining = max_total_depth - state.total_layers_explored
                process_rounds = min(scheduled_steps, remaining)

                log.debug(f"Fixed-rounds: Agent-{aid} ({agent_type_str}), {process_rounds} rounds, depth={state.total_layers_explored}/{max_total_depth}")
                await self._optimize_agent_fixed_rounds(
                    state=state,
                    requirement=requirement,
                    agent_states=agent_states,
                    use_dynamic_switching=use_dynamic_switching,
                    use_stochastic_sampling=use_stochastic_sampling,
                    use_beam_refresh=use_beam_refresh,
                    rounds_per_turn=process_rounds,
                    max_total_depth=max_total_depth,
                    use_feedback=use_feedback,
                    use_misleading_sampling=use_misleading_sampling,
                    use_lookahead_score=use_lookahead_score,
                    lookahead_weights=lookahead_weights,
                )

            if self.co_evolver is not None:
                # Depth determines the model state for the next round; Co-Evolver
                # determines the prompt state for the next round.
                depth_best_models = self.best_model.copy()
                await self._apply_co_evolver(
                    round_initial_prompts,
                    round_initial_models,
                    round_num,
                    depth_best_models=depth_best_models,
                    agent_states=agent_states,
                )

        statistics = self._aggregate_statistics(agent_states, max_total_depth)
        self._log_statistics_summary(statistics)

        log.info("Fixed-rounds optimization complete")
        # Summary of the configuration that will actually be evaluated.
        summary_prompts = self.final_coevolver_prompt or self.best_prompt
        summary_models = self.final_coevolver_model or self.best_model
        for aid in optimizable_agents:
            agent = self.mas.agents[aid]
            final_len = len(summary_prompts.get(aid, ""))
            final_score = agent_states[aid].best_overall_node["cumulative_score"]
            final_model = summary_models.get(aid, DEFAULT_AGENT_MODEL)
            log.info(f"  Agent-{aid} ({agent.type.value}): final_score={final_score:.4f}, model={final_model}, prompt_len={final_len}")

        if self.final_coevolver_prompt:
            final_prompts = self._expand_prompt_map(self.final_coevolver_prompt)
            final_models = self._expand_model_map(self.final_coevolver_model)
        else:
            final_prompts = self._expand_prompt_map(self.best_prompt)
            final_models = self._expand_model_map(self.best_model)
        return final_prompts, final_models, statistics

    async def _apply_co_evolver(
        self,
        round_initial_prompts: Dict[int, str],
        round_initial_models: Dict[int, str],
        round_number: int,
        depth_best_models: Optional[Dict[int, str]] = None,
        agent_states: Optional[Dict[int, AgentOptState]] = None,
    ) -> Dict[str, Any]:
        """Run one independent Co-Evolver call per logical Agent at a round boundary."""
        if self.co_evolver is None:
            return {"enabled": False, "round": round_number, "proposals": []}

        profile = load_topology_profile(self.dataset_name, self.mas.gtype.value)
        if self.mas.gtype == GraphType.REFLECT:
            profile = expand_reflect_profile(
                profile=profile,
                runtime_agents=self.mas.agents,
                nr=self.mas.Nr,
            )
        agent_ids = [
            aid for aid in self._optimizable_ids(self.mas._topo_order())
            if self.mas.agents[aid].type != AgentType.AGGREGATOR
        ]
        global_prompts = self._expand_prompt_map(self.best_prompt)
        global_models = self._expand_model_map(self.best_model)
        for runtime_agent in self.mas.agents:
            global_prompts.setdefault(runtime_agent.id, runtime_agent.template)
            global_models.setdefault(
                runtime_agent.id,
                self.best_model.get(runtime_agent.id, runtime_agent.model_name),
            )
        proposals = []
        log = get_logger()

        for aid in agent_ids:
            memories = []
            if self.trajectory_memory is not None:
                query = {"current_prompt": global_prompts.get(aid, "")}
                memories = await self.trajectory_memory.retrieve(
                    query,
                    max_items=3 * self.memory_per_category,
                    per_category=self.memory_per_category,
                )
            try:
                result = await self.co_evolver.evolve(
                    agent_id=aid,
                    initial_prompt=round_initial_prompts.get(aid, global_prompts.get(aid, "")),
                    best_prompt=global_prompts.get(aid, ""),
                    initial_model=round_initial_models.get(aid, global_models.get(aid, DEFAULT_AGENT_MODEL)),
                    best_model=global_models.get(aid, DEFAULT_AGENT_MODEL),
                    global_prompts=global_prompts,
                    global_models=global_models,
                    profile=profile,
                    memories=memories,
                    round_number=round_number,
                )
                proposals.append(result)
                final_prompt = result.get("final_prompt", "")
                if final_prompt:
                    self._set_best_prompt(aid, final_prompt)
                log.info(
                    f"Co-Evolver round={round_number} agent={aid}: "
                    f"memory={len(memories)}, prompt_len={len(final_prompt)}, "
                    f"summary={result.get('decision_summary', '')}"
                )
            except Exception as exc:
                log.warning(
                    f"Co-Evolver round={round_number} agent={aid} failed; "
                    f"keeping current best: {type(exc).__name__}: {exc}"
                )

        merged = CoEvolver.merge_assignments(
            proposals,
            global_models,
            profile=profile,
            model_pool=self.model_pool,
            voter=self.voter,
        )

        # Keep only logical-agent assignments as the Co-Evolver result. Runtime
        # Council nodes are expanded only when the result is returned/evaluated.
        coevolver_models = {}
        for aid in agent_ids:
            model = merged.get(aid, global_models.get(aid))
            if model in self.model_pool:
                coevolver_models[aid] = model

        self.final_coevolver_prompt = {
            aid: self.best_prompt.get(aid, global_prompts.get(aid, ""))
            for aid in agent_ids
        }
        self.final_coevolver_model = coevolver_models

        if agent_states is None:
            # Preserve the historical round-robin behavior: its Co-Evolver
            # model assignment remains the active assignment.
            if self.heterogeneous:
                for aid, model in coevolver_models.items():
                    self._set_best_model(aid, model)
        else:
            # Fixed-rounds: the next round starts from Co-Evolver prompts, but
            # from the Depth-best models. Co-Evolver models are final-only.
            depth_models = (depth_best_models or self.best_model).copy()
            self.best_model = depth_models
            self._normalize_shared_models()

        # Reset each state beam so the next Depth round really starts from the
        # newly produced prompt and the Depth-best model, rather than continuing
        # from the previous round's stale beam node.
        if agent_states is not None:
            for aid in agent_ids:
                state = agent_states.get(aid)
                if state is None:
                    continue
                prompt = self.best_prompt.get(aid, "")
                model = self.best_model.get(aid, DEFAULT_AGENT_MODEL)
                initial_node = {
                    "prompt": prompt,
                    "model": model,
                    "cumulative_score": 0.0,
                    "quality": 0.0,
                    "cost": 0.0,
                    "path": [prompt],
                }
                state.current_beam = [initial_node]
                state.best_overall_node = initial_node
                state.current_patience_count = 0

        return {
            "enabled": True,
            "round": round_number,
            "proposals": proposals,
            "model_assignment": self._expand_model_map(coevolver_models),
        }

    def _aggregate_statistics(self, agent_states: Dict[int, AgentOptState], 
                            max_depth: int) -> Dict[str, Any]:

        terminal_id = self.mas.get_terminal_id()
        
        misalignment_per_agent = {}
        for aid, state in agent_states.items():
            if aid == terminal_id:
                continue
            misalignment_per_agent[aid] = state.misalignment_rates_per_depth.copy()
        
        averaged_by_depth = []
        for depth_idx in range(max_depth):
            rates_at_depth = []
            for aid, rates in misalignment_per_agent.items():
                if depth_idx < len(rates):
                    rates_at_depth.append(rates[depth_idx])
            if rates_at_depth:
                averaged_by_depth.append(sum(rates_at_depth) / len(rates_at_depth))
            else:
                averaged_by_depth.append(0.0)
        
        kendall_per_agent = {}
        all_kendall_scores = []
        for aid, state in agent_states.items():
            scores = [record["kendall_top2_overlap"] for record in state.beam_refresh_kendall_scores]
            kendall_per_agent[aid] = scores
            all_kendall_scores.extend(scores)
        
        avg_kendall = sum(all_kendall_scores) / len(all_kendall_scores) if all_kendall_scores else 1.0
        
        return {
            "misalignment_rates": {
                "per_agent": misalignment_per_agent,
                "averaged_by_depth": averaged_by_depth,
            },
            "kendall_scores": {
                "per_agent": kendall_per_agent,
                "averaged": avg_kendall,
                "all_scores": all_kendall_scores,
                "detailed_records": {
                    aid: state.beam_refresh_kendall_scores 
                    for aid, state in agent_states.items()
                }
            }
        }
    def _log_statistics_summary(self, statistics: Dict[str, Any]):
        log = get_logger()
        avg_by_depth = statistics["misalignment_rates"]["averaged_by_depth"]
        per_agent = statistics["misalignment_rates"]["per_agent"]

        log.info("=" * 50)
        log.info("Optimization Statistics Summary")
        log.info("=" * 50)

        # Misalignment rates per agent
        log.info("Misalignment Rates by Agent:")
        for aid, rates in per_agent.items():
            agent = self.mas.agents[aid]
            avg_rate = sum(rates) / len(rates) if rates else 0.0
            log.info(f"  Agent-{aid} ({agent.type.value}): avg_misalign={avg_rate:.4f}, rates={[f'{r:.3f}' for r in rates]}")

        # Averaged by depth
        if avg_by_depth:
            log.info(f"Misalignment Rates by Depth: {[f'{r:.3f}' for r in avg_by_depth]}")
            log.info(f"  Avg across all depths: {sum(avg_by_depth)/len(avg_by_depth):.4f}")

        # Kendall scores
        kendall_info = statistics["kendall_scores"]
        log.info(f"Beam Refresh Kendall Scores: avg={kendall_info['averaged']:.4f}, total_refreshes={len(kendall_info['all_scores'])}")
        for aid, scores in kendall_info["per_agent"].items():
            if scores:
                agent = self.mas.agents[aid]
                avg_k = sum(scores) / len(scores)
                log.info(f"  Agent-{aid} ({agent.type.value}): avg_kendall={avg_k:.4f}, refreshes={len(scores)}")


    def _build_memory_state(self, aid: int, node_prompt: str, node_model: str,
                            preds: List[int], samples: List[str],
                            baseline_caches: Dict[str, InferenceCache],
                            known_failure_signals: str = "") -> Dict[str, Any]:
        """Build the causal pre-candidate state used for memory retrieval."""
        agent = self.mas.agents[aid]
        predecessor_roles = [self.mas.agents[p].type.value for p in preds]
        successor_roles = [self.mas.agents[s].type.value
                           for s in self.mas.get_successors(aid)]
        sample_blocks = []
        for q in samples[:3]:
            cache = baseline_caches.get(q)
            if cache is None:
                continue
            predecessor_outputs = [
                cache.node_outputs_short.get(p, "") for p in preds
            ]
            sample_blocks.append({
                "question": q,
                "predecessor_outputs": predecessor_outputs,
                "current_agent_output": cache.node_outputs_raw.get(aid, ""),
            })
        return {
            "dataset": self.dataset_name,
            "task_type": self.task_type.value,
            "graph_type": self.mas.gtype.value,
            "agent_id": aid,
            "agent_type": agent.type.value,
            "depth": len(self.mas.get_all_predecessors(aid)),
            "predecessor_roles": predecessor_roles,
            "successor_roles": successor_roles,
            "current_model": node_model,
            "current_prompt": node_prompt,
            "baseline_samples": json.dumps(sample_blocks, ensure_ascii=False),
            "known_failure_signals": known_failure_signals,
        }

    async def _record_baseline_snapshot(self, retrieval_view: Dict[str, Any],
                                        aid: int, prompt: str, model: str,
                                        preds: List[int], samples: List[str],
                                        baseline_caches: Dict[str, InferenceCache],
                                        requirement: str) -> None:
        """Persist the pre-candidate state for audit, but exclude it from retrieval."""
        if self.trajectory_memory is None:
            return
        sample_traces = []
        for q in samples:
            cache = baseline_caches.get(q)
            if cache is None:
                continue
            sample_traces.append({
                "question": q,
                "predecessor_outputs": {
                    str(p): cache.node_outputs_raw.get(p, "") for p in preds
                },
                "current_agent_output": cache.node_outputs_raw.get(aid, ""),
                "successor_outputs": {
                    str(s): cache.node_outputs_raw.get(s, "")
                    for s in self.mas.get_successors(aid)
                },
            })
        await self.trajectory_memory.add({
            "run_id": str(getattr(self.trajectory_memory, "run_dir", "")),
            "step": len(self.trajectory_memory.records),
            "record_type": "baseline_snapshot",
            "state": retrieval_view,
            "retrieval_view": retrieval_view,
            "action": {"type": "baseline_snapshot"},
            "score": {"score": 0.0, "quality": 0.5, "local_rate": 0.5,
                      "global_rate": 0.5, "cost_ratio": 1.0},
            "status": {"accepted": True, "retrievable": False},
            "payload": {
                "before_prompt": prompt,
                "before_model": model,
                "candidate_prompt": None,
                "candidate_model": None,
                "requirement": requirement,
                "samples": sample_traces,
            },
        })

    async def process_single_node(self, node: Dict, aid: int, preds: List[int], term_id: int,
                                is_term: bool, req: str, samples: List[str],
                                samples_p1: List[str], samples_p2: List[str],
                                base_prompt_map: Optional[Dict[int, str]] = None,
                                agent_states: Optional[Dict[int, AgentOptState]] = None, 
                                use_stochastic_sampling: bool = False,
                                use_feedback: bool = False,
                                use_misleading_sampling: bool = False,
                                use_lookahead_score: bool = False,
                                lookahead_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> Dict:
        final_samples = []
        injected_questions = set()
        
        if use_misleading_sampling and agent_states:
            for pred_id in preds:
                if pred_id in agent_states:
                    traps = agent_states[pred_id].misleading_cases
                    for t in traps:
                        injected_questions.add(t["question"])
        
        injected_list = list(injected_questions)
        if len(injected_list) > 5:
            injected_list = random.sample(injected_list, 5)
        
        if injected_list:
            final_samples.extend(injected_list)
            log = get_logger()
            log.debug(f"Agent-{aid}: injected {len(injected_list)} misleading cases from predecessors {preds}")

        target_total = 10
        needed = max(0, target_total - len(final_samples))
        
        pool = [q for q in self.train_q if q not in injected_questions]
        
        random_samples = [] 
        if needed > 0:
            if len(pool) >= needed:
                random_samples = random.sample(pool, needed)
            else:
                random_samples = pool.copy()
            final_samples.extend(random_samples)
        
        random_sample_set = set(random_samples)
        
        current_samples = final_samples
        mid = len(current_samples) // 2
        current_samples_p1 = current_samples[:mid]
        current_samples_p2 = current_samples[mid:]
        node_prompt = node["prompt"]
        node_model = node.get("model", self.best_model.get(aid, DEFAULT_AGENT_MODEL))

        diverse_prompt_maps = []
        base_map_template = (base_prompt_map or self.best_prompt).copy()
        base_map_template[aid] = node_prompt 
        predecessors_set = set(preds)
        
        for i, sample in enumerate(current_samples):
            current_map = base_map_template.copy()
            is_robustness_sample = (i >= len(current_samples) * 0.7)
            if use_stochastic_sampling and agent_states and is_robustness_sample:
                for pred_id in predecessors_set:
                    if pred_id in agent_states and agent_states[pred_id].current_beam:
                        beam_nodes = agent_states[pred_id].current_beam
                        if len(beam_nodes) > 1:
                            alternatives = beam_nodes[1:]
                            chosen = random.choice(alternatives)
                            current_map[pred_id] = chosen["prompt"]
            diverse_prompt_maps.append(current_map)
        
        base_map_for_eval = diverse_prompt_maps[0] 
        local_baseline_caches = await self._build_diverse_baseline_caches(current_samples, diverse_prompt_maps)

        experience_context = ""
        if self.enable_legacy_memory_injection and self.trajectory_memory is not None and current_samples:
            signals = []
            if use_misleading_sampling:
                signals.append("misleading_sampling_enabled")
            if use_feedback:
                signals.append("downstream_feedback_enabled")
            query_state = self._build_memory_state(
                aid=aid,
                node_prompt=node_prompt,
                node_model=node_model,
                preds=preds,
                samples=current_samples,
                baseline_caches=local_baseline_caches,
                known_failure_signals=", ".join(signals),
            )
            await self._record_baseline_snapshot(
                query_state, aid, node_prompt, node_model,
                preds, current_samples, local_baseline_caches, req,
            )
            retrieved = await self.trajectory_memory.retrieve(
                query_state,
                max_items=3 * self.memory_per_category,
                per_category=self.memory_per_category,
            )
            experience_context = self.trajectory_memory.format_for_optimizer(retrieved)
            if retrieved:
                get_logger().debug(
                    f"Agent-{aid}: retrieved {len(retrieved)} trajectory experiences"
                )
        qa_for_proposal1 = {}
        for q in current_samples_p1:
            cache = local_baseline_caches[q]
            ctx = cache.get_context_for_node(aid, preds)
            qa_for_proposal1[q] = {"context": ctx, "output": cache.node_outputs_raw.get(aid, "")}
        
        qa_for_proposal2 = {}
        for q in current_samples_p2:
            cache = local_baseline_caches[q]
            ctx = cache.get_context_for_node(aid, preds)
            qa_for_proposal2[q] = {"context": ctx, "output": cache.node_outputs_raw.get(aid, "")}
        
        sample1 = current_samples_p1[0] if current_samples_p1 else samples[0]
        sample2 = current_samples_p2[0] if current_samples_p2 else samples[-1]
        
        successor_info_text = ""
        if use_feedback and agent_states:
            successors = self.mas.get_successors(aid)
            feedback_messages = []
            
            for succ_id in successors:
                if succ_id in agent_states:
                    bad_cases = agent_states[succ_id].recent_bad_cases
                    if bad_cases:
                        succ_agent_type = self.mas.agents[succ_id].type.value
                        
                        cases_str = "\n".join([
                            f"  - Case {i+1}:\n"
                            f"    [Problem]: \"{case['question']}\"\n"
                            f"    [Context Snippet]: \"{case['context'][:200]}\"\n" 
                            f"    The downstream agent ({succ_agent_type}) failed to produce a correct/better answer."
                            for i, case in enumerate(bad_cases[:2])
                        ])
                        
                        msg = (f"Feedback from downstream Agent-{succ_id} ({succ_agent_type}):\n"
                            f"Your previous outputs led to failures in the downstream task in the following cases:\n{cases_str}")
                        feedback_messages.append(msg)
            
            if feedback_messages:
                successor_info_text = "\n\n".join(feedback_messages)
                log = get_logger()
                log.debug(f"Agent-{aid}: received feedback from {len(feedback_messages)} downstream agents")
        # LLM generates both prompt and model recommendations
        raw_cand_results = await asyncio.gather(
            self._propose_new_prompt(req, node_prompt, qa_for_proposal1, aid, sample1,
                                    current_model=node_model, successor_info=successor_info_text,
                                    experience_context=experience_context),
            self._propose_new_prompt(req, node_prompt, qa_for_proposal2, aid, sample2,
                                    current_model=node_model, successor_info=successor_info_text,
                                    experience_context=experience_context)
            , return_exceptions=True
        )
        cand_results = []
        for result in raw_cand_results:
            if isinstance(result, Exception):
                get_logger().warning(
                    f"Agent-{aid}: prompt proposal failed; skipping candidate: "
                    f"{type(result).__name__}: {result}"
                )
                continue
            cand_results.append(result)

        async def evaluate_wrapper(cand_p: str, cand_m: str) -> Tuple[str, str, Dict]:
            """Evaluate a (prompt, model) candidate."""
            if cand_p == node_prompt and cand_m == node_model:
                return cand_p, cand_m, {
                    "score": 0.0,
                    "quality": 0.5,
                    "cost": 1.0,
                    "norm_cost": 1.0,
                    "wins_global": len(current_samples)//2,
                    "wins_local": len(current_samples)//2,
                    "bad_cases": [],
                    "misleading_cases": [],
                    "misalignment_rate": 0.0,
                    "misalignment_count": 0,
                    "total_evaluable": 0,
                }

            try:
                score_info = await self._evaluate_candidate(
                    cand_prompt=cand_p,
                    agent_id=aid,
                    eval_samples=current_samples,
                    baseline_caches=local_baseline_caches,
                    temp_prompt_map=base_map_for_eval,
                    predecessors=preds,
                    terminal_id=term_id,
                    is_terminal=is_term,
                    requirement=req,
                    use_lookahead_score=use_lookahead_score,
                    lookahead_weights=lookahead_weights,
                    random_sample_set=random_sample_set,
                    cand_model=cand_m,
                )
            except Exception as exc:
                get_logger().warning(
                    f"Agent-{aid}: candidate evaluation failed; keeping baseline: "
                    f"{type(exc).__name__}: {exc}"
                )
                score_info = {"score": 0.0, "quality": 0.5, "cost": 1.0,
                              "norm_cost": 1.0, "bad_cases": [],
                              "misleading_cases": [], "misalignment_rate": 0.0}
            expected_id = hashlib.sha1(
                f"{cand_m}\n{cand_p}".encode("utf-8")
            ).hexdigest()[:12]
            scored_model = score_info.get("candidate_model", cand_m)
            scored_id = score_info.get("candidate_id", expected_id)
            if scored_model != cand_m or scored_id != expected_id:
                raise RuntimeError(
                    f"candidate trace mismatch: pair=({cand_m},{expected_id}) "
                    f"scored=({scored_model},{scored_id})"
                )
            score_info["candidate_model"] = scored_model
            score_info["candidate_id"] = scored_id
            score_info.setdefault("candidate_prompt_len", len(cand_p))
            return cand_p, cand_m, score_info

        # Build (prompt, model) candidates from LLM recommendations
        pm_candidates = [(node_prompt, node_model)]  # baseline
        seen = {(node_prompt, node_model)}
        for cand_p, cand_m in cand_results:
            key = (cand_p, cand_m)
            if key not in seen:
                seen.add(key)
                pm_candidates.append(key)

        eval_results = await asyncio.gather(*[evaluate_wrapper(p, m) for p, m in pm_candidates])

        candidate_scores = {(p, m): info for p, m, info in eval_results}
        new_nodes = []
        local_best = {
            "prompt": node_prompt,
            "model": node_model,
            "score": 0.0,
            "candidate_id": hashlib.sha1(
                f"{node_model}\n{node_prompt}".encode("utf-8")
            ).hexdigest()[:12],
            "norm_cost": 1.0,
            "bad_cases": [],
            "misleading_cases": [],
            "misalignment_rate": 0.0,
        }
        all_misalignment_rates = []

        for cand_p, cand_m in pm_candidates:
            info = candidate_scores[(cand_p, cand_m)]
            all_misalignment_rates.append(info.get("misalignment_rate", 0.0))
            if info["score"] > local_best["score"]:
                local_best = {
                    "prompt": cand_p,
                    "model": cand_m,
                    "score": info["score"],
                    "candidate_id": info.get("candidate_id"),
                    "norm_cost": info.get("norm_cost", 1.0),
                    "bad_cases": info["bad_cases"],
                    "misleading_cases": info["misleading_cases"],
                    "misalignment_rate": info.get("misalignment_rate", 0.0),
                }
            if info["score"] > 0:
                new_node = {
                    "prompt": cand_p,
                    "model": cand_m,
                    "candidate_id": info.get("candidate_id"),
                    "cumulative_score": node["cumulative_score"] + info["score"],
                    "quality": info.get("quality", 0.0),
                    "cost": info.get("norm_cost", 1.0),
                    "path": node["path"] + [cand_p],
                    "_temp_bad_cases": info["bad_cases"],
                    "_temp_misleading": info["misleading_cases"],
                    "misalignment_rate": info.get("misalignment_rate", 0.0),
                }
                new_nodes.append(new_node)
            else:
                new_nodes.append({
                    "prompt": node_prompt,
                    "model": node_model,
                    "cumulative_score": node["cumulative_score"],
                    "quality": node.get("quality", 0.5),
                    "cost": node.get("cost", 1.0),
                    "path": node["path"],
                    "_temp_bad_cases": [],
                    "_temp_misleading": [],
                    "_temp_misalignment_rate": 0.0,
                })
        avg_misalignment_rate = sum(all_misalignment_rates) / len(all_misalignment_rates) if all_misalignment_rates else 0.0
        return {
            "nodes": new_nodes,
            "best_prompt": local_best["prompt"],
            "best_model": local_best.get("model", node_model),
            "best_candidate_id": local_best.get("candidate_id"),
            "best_norm_cost": local_best.get("norm_cost", 1.0),
            "best_cumulative": node["cumulative_score"] + max(0, local_best["score"]),
            "best_bad_cases": local_best["bad_cases"],
            "best_misleading_cases": local_best["misleading_cases"],
            "avg_misalignment_rate": avg_misalignment_rate,
        }


    def _get_dynamic_context(self, agent_states: Dict[int, AgentOptState], 
                           current_agent_id: int, 
                           current_depth: int) -> Dict[int, str]:

        context_map = self.best_prompt.copy()

        try:
            direct_predecessors = set(self.mas.graph.predecessors(current_agent_id))
        except AttributeError:
            direct_predecessors = set(self.mas.get_predecessors(current_agent_id))
        should_switch = (current_depth > 0 and current_depth % 2 != 0)

        if should_switch:
            log = get_logger()
            log.debug(f"Agent-{current_agent_id}: Dynamic Anchor Switching triggered at depth={current_depth}")

        for aid, state in agent_states.items():
            if aid == current_agent_id:
                continue

            if not state.current_beam:
                continue
            selected_prompt = state.best_overall_node["prompt"]
            if should_switch and aid in direct_predecessors:
                sorted_nodes = sorted(state.current_beam, key=lambda x: x["cumulative_score"], reverse=True)
                if len(sorted_nodes) >= 2:
                    selected_prompt = sorted_nodes[1]["prompt"]
                    log = get_logger()
                    log.debug(f"  Switched predecessor Agent-{aid} anchor: best({sorted_nodes[0]['cumulative_score']:.4f}) → alt({sorted_nodes[1]['cumulative_score']:.4f})")

            context_map[aid] = selected_prompt

        return context_map

    async def _build_diverse_baseline_caches(self, questions: List[str],
                                           prompt_maps: List[Dict[int, str]]) -> Dict[str, InferenceCache]:

        assert len(questions) == len(prompt_maps), "Questions and prompt_maps must align"
        async def run_one(q: str, p_map: Dict[int, str]) -> Tuple[str, InferenceCache]:
            mas_temp = MAS(
                self.mas.gtype, self.mas.task_type,
                Ns=self.mas.Ns, Na=self.mas.Na, Nr=self.mas.Nr, Nd=self.mas.Nd,
                agent_model=self.mas.agent_model,
                agent_models=self._expand_model_map(self.best_model),
            )
            mas_temp.inject_prompt_map(self._expand_prompt_map(p_map))
            _, cache = await mas_temp.arun_with_cache(q)
            if self.cost_tracker is not None:
                self.cost_tracker.record_mas_cache(
                    cache,
                    model_map=self._expand_model_map(self.best_model),
                    metadata={"kind": "diverse_baseline", "question": q},
                )
            return q, cache
    
        results = await asyncio.gather(*[
            run_one(q, p_map) for q, p_map in zip(questions, prompt_maps)
        ])
        return {q: cache for q, cache in results}


    async def _refresh_beam_scores(self, 
                                 state: AgentOptState, 
                                 requirement: str,
                                 predecessors: List[int],
                                 terminal_id: int,
                                 is_terminal: bool) -> None:

        if not state.current_beam:
            return

        log = get_logger()
        log.info(f"    Beam Refresh: re-evaluating {len(state.current_beam)} nodes at depth={state.total_layers_explored}")
        ranking_before = [node["prompt"] for node in sorted(
            state.current_beam, 
            key=lambda x: x["cumulative_score"], 
            reverse=True
        )]
        scores_before = {node["prompt"]: node["cumulative_score"] for node in state.current_beam}
        eval_samples = self.get_eval_samples(state.total_layers_explored)
        
        current_global_map = self.best_prompt.copy()
        prompt_maps = [current_global_map] * len(eval_samples)
        
        local_baseline_caches = await self._build_diverse_baseline_caches(eval_samples, prompt_maps)
        
        async def re_evaluate_node(node):
            prompt = node["prompt"]
            node_model = node.get("model", self.best_model.get(state.agent_id, DEFAULT_AGENT_MODEL))

            if prompt == self.best_prompt[state.agent_id] and node_model == self.best_model.get(state.agent_id, DEFAULT_AGENT_MODEL):
                node["cumulative_score"] = 0.0
                return node

            temp_map = current_global_map.copy()
            temp_map[state.agent_id] = prompt

            score_info = await self._evaluate_candidate(
                cand_prompt=prompt,
                agent_id=state.agent_id,
                eval_samples=eval_samples,
                baseline_caches=local_baseline_caches,
                temp_prompt_map=temp_map,
                predecessors=predecessors,
                terminal_id=terminal_id,
                is_terminal=is_terminal,
                requirement=requirement,
                cand_model=node_model,
            )

            # Use cost-adjusted score directly
            node["cumulative_score"] = score_info["score"]
            node["quality"] = score_info.get("quality", 0.0)
            node["cost"] = score_info.get("norm_cost", 1.0)
            return node

        new_beam = await asyncio.gather(*[re_evaluate_node(n) for n in state.current_beam])
        
        state.current_beam = sorted(new_beam, key=lambda x: x["cumulative_score"], reverse=True)

        ranking_after = [node["prompt"] for node in state.current_beam]
        scores_after = {node["prompt"]: node["cumulative_score"] for node in state.current_beam}
        
        kendall_score = self._compute_kendall_top2_overlap(ranking_before, ranking_after)
        
        state.beam_refresh_kendall_scores.append({
            "depth": state.total_layers_explored,
            "kendall_top2_overlap": kendall_score,
            "ranking_before": ranking_before[:3],
            "ranking_after": ranking_after[:3],
            "scores_before": {p[:50]: s for p, s in list(scores_before.items())[:3]},
            "scores_after": {p[:50]: s for p, s in list(scores_after.items())[:3]},
        })
        
        top_node = state.current_beam[0]
        state.best_overall_node = top_node.copy()

        log = get_logger()
        if top_node["prompt"] != self.best_prompt[state.agent_id]:
            log.info(f"    Beam Refresh: Anchor Shifted! New best score: {top_node['cumulative_score']:.3f}, kendall={kendall_score:.3f}")
            self._set_best_prompt(state.agent_id, top_node["prompt"])
            self._set_best_model(state.agent_id, top_node.get("model", self.best_model.get(state.agent_id, DEFAULT_AGENT_MODEL)))
        else:
            log.debug(f"    Beam Refresh: Anchor retained (score={top_node['cumulative_score']:.3f}, kendall={kendall_score:.3f})")

    async def _optimize_agent_simple_serial(self, agent_id: int, requirement: str,
                                            rounds: int = 9, sample_size: int = 3) -> str:
        agent = self.mas.agents[agent_id]
        log = get_logger()

        terminal_id = self.mas.get_terminal_id()
        is_terminal = (agent_id == terminal_id)
        predecessors = self.mas.get_predecessors(agent_id)

        current_prompt = self.best_prompt[agent_id]
        desc_prefix = f"SPO-Agent-{agent_id}({agent.type.value})"

        log.info(f"Agent-{agent_id} ({agent.type.value}): starting simple serial optimization (rounds={rounds}, sample_size={sample_size})")

        with tqdm(total=rounds, desc=desc_prefix, leave=False) as pbar:
            for r in range(rounds):

                eval_samples = random.sample(self.train_q, min(sample_size, len(self.train_q)))
                current_map = self.best_prompt.copy()
                baseline_caches = await self._build_baseline_caches(eval_samples, current_map)

                qa_for_proposal = {}
                for q in eval_samples:
                    cache = baseline_caches[q]
                    ctx = cache.get_context_for_node(agent_id, predecessors)
                    output = cache.node_outputs_raw.get(agent_id, "")
                    qa_for_proposal[q] = {"context": ctx, "output": output}

                sample_q = eval_samples[0]
                new_prompt, new_model = await self._propose_new_prompt(
                    requirement, current_prompt, qa_for_proposal, agent_id, sample_q,
                    current_model=self.best_model.get(agent_id)
                )

                if new_prompt == current_prompt:
                    log.debug(f"Agent-{agent_id} round={r+1}: prompt unchanged, skipping")
                    pbar.set_postfix_str(f"Skip (Same)")
                    pbar.update(1)
                    continue

                cand_map = self.best_prompt.copy()
                cand_map[agent_id] = new_prompt

                cand_model = new_model
                score_info = await self._evaluate_candidate(
                    cand_prompt=new_prompt,
                    agent_id=agent_id,
                    eval_samples=eval_samples,
                    baseline_caches=baseline_caches,
                    temp_prompt_map=cand_map,
                    predecessors=predecessors,
                    terminal_id=terminal_id,
                    is_terminal=is_terminal,
                    requirement=requirement,
                    pure_local_mode=True,
                    cand_model=cand_model,
                )

                if score_info["score"] > 0:
                    self._set_best_prompt(agent_id, new_prompt)
                    current_prompt = new_prompt
                    log.info(f"Agent-{agent_id} round={r+1}: improved, score={score_info['score']+0.5:.3f}")
                    pbar.set_postfix_str(f"Update! Score: {score_info['score'] + 0.5:.2f}")
                else:
                    pbar.set_postfix_str(f"Keep. Score: {score_info['score'] + 0.5:.2f}")
                
                pbar.update(1)

        return current_prompt

    async def optimize_all_simple_sequential(self, requirement: str = " ",
                                             rounds: int = 9) -> Dict[int, str]:
        log = get_logger()
        order = self._optimizable_ids(self.mas._topo_order())
        log.info(f"Starting simple sequential optimization: order={order}, rounds={rounds}")

        with tqdm(total=len(order), desc="[SPO]") as topo_bar:
            for aid in order:
                if self.mas.agents[aid].type == AgentType.AGGREGATOR:
                    log.debug(f"Agent-{aid}: skipping (AGGREGATOR)")
                    topo_bar.update(1)
                    continue
                await self._optimize_agent_simple_serial(
                    agent_id=aid,
                    requirement=requirement,
                    rounds=rounds,
                    sample_size=3
                )

                topo_bar.set_postfix_str(f"Agent-{aid} Done")
                topo_bar.update(1)

        log.info("Simple sequential optimization complete")
        return self.best_prompt, self.best_model
