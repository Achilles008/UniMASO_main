
import asyncio
import os
import re
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Tuple, Set

from config import GraphType, AgentType, TaskType, get_model, DEFAULT_AGENT_MODEL
from prompts import (
    get_agent_template,
    get_reflect_agi_template,
    get_gpqa_reflect_template,
    get_agi_reflect_template,
    COMPRESS_PROMPTS,
)
from utils import (
    async_call_llm, extract_answer, extract_code, extract_output,
    majority_vote, code_vote, normalize_qa_answer
)
from judges import LLMJudgeAgent, CodeJudgeAgent, HotpotQAJudge, BFCLJudge, LCBJudge


class InferenceCache:


    def __init__(self):
        self.question: str = ""
        self.node_inputs: Dict[int, str] = {}      # agent_id -> context input
        self.node_prompts: Dict[int, str] = {}     # agent_id -> full prompt sent to the model
        self.node_outputs_raw: Dict[int, str] = {} # agent_id -> raw output
        self.node_outputs_short: Dict[int, str] = {} # agent_id -> compressed/extracted output
        self.node_input_tokens: Dict[int, int] = {}   # agent_id -> prompt_tokens consumed
        self.node_output_tokens: Dict[int, int] = {}  # agent_id -> completion_tokens consumed
        self.node_model: Dict[int, str] = {}          # agent_id -> model_name used
        self.dynamic_steps: List[Dict[str, Any]] = []

    def set_question(self, question: str):
        self.question = question

    def set_node_data(self, agent_id: int, context: str, raw: str, short: str,
                      input_tokens: int = 0, output_tokens: int = 0, model: str = ""):
        self.node_inputs[agent_id] = context
        self.node_outputs_raw[agent_id] = raw
        self.node_outputs_short[agent_id] = short
        self.node_input_tokens[agent_id] = input_tokens
        self.node_output_tokens[agent_id] = output_tokens
        self.node_model[agent_id] = model
    
    def get_context_for_node(self, agent_id: int, predecessors: List[int], 
                              use_short: bool = True) -> str:
        if not predecessors:
            return ""
        
        outputs = self.node_outputs_short if use_short else self.node_outputs_raw
        ctx_list = [outputs.get(p, "") for p in predecessors if p in outputs]
        return "\n---\n".join(ctx_list)
    
    def has_all_predecessors(self, predecessors: List[int]) -> bool:
        return all(p in self.node_outputs_raw for p in predecessors)
    
    def clone(self) -> 'InferenceCache':
        new_cache = InferenceCache()
        new_cache.question = self.question
        new_cache.node_inputs = self.node_inputs.copy()
        new_cache.node_prompts = self.node_prompts.copy()
        new_cache.node_outputs_raw = self.node_outputs_raw.copy()
        new_cache.node_outputs_short = self.node_outputs_short.copy()
        new_cache.node_input_tokens = self.node_input_tokens.copy()
        new_cache.node_output_tokens = self.node_output_tokens.copy()
        new_cache.node_model = self.node_model.copy()
        new_cache.dynamic_steps = [dict(step) for step in self.dynamic_steps]
        return new_cache

    def clone_up_to(self, keep_nodes: Set[int]) -> 'InferenceCache':

        new_cache = InferenceCache()
        new_cache.question = self.question
        new_cache.node_inputs = {k: v for k, v in self.node_inputs.items() if k in keep_nodes}
        new_cache.node_prompts = {k: v for k, v in self.node_prompts.items() if k in keep_nodes}
        new_cache.node_outputs_raw = {k: v for k, v in self.node_outputs_raw.items() if k in keep_nodes}
        new_cache.node_outputs_short = {k: v for k, v in self.node_outputs_short.items() if k in keep_nodes}
        new_cache.node_input_tokens = {k: v for k, v in self.node_input_tokens.items() if k in keep_nodes}
        new_cache.node_output_tokens = {k: v for k, v in self.node_output_tokens.items() if k in keep_nodes}
        new_cache.node_model = {k: v for k, v in self.node_model.items() if k in keep_nodes}
        new_cache.dynamic_steps = [
            dict(step) for step in self.dynamic_steps
            if step.get("role_id") in keep_nodes
        ]
        return new_cache
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "node_inputs": self.node_inputs,
            "node_prompts": self.node_prompts,
            "node_outputs_raw": self.node_outputs_raw,
            "node_outputs_short": self.node_outputs_short,
            "node_input_tokens": self.node_input_tokens,
            "node_output_tokens": self.node_output_tokens,
            "node_model": self.node_model,
            "dynamic_steps": self.dynamic_steps,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceCache':
        cache = cls()
        cache.question = data.get("question", "")
        cache.node_inputs = {int(k): v for k, v in data.get("node_inputs", {}).items()}
        cache.node_prompts = {int(k): v for k, v in data.get("node_prompts", {}).items()}
        cache.node_outputs_raw = {int(k): v for k, v in data.get("node_outputs_raw", {}).items()}
        cache.node_outputs_short = {int(k): v for k, v in data.get("node_outputs_short", {}).items()}
        cache.node_input_tokens = {int(k): v for k, v in data.get("node_input_tokens", {}).items()}
        cache.node_output_tokens = {int(k): v for k, v in data.get("node_output_tokens", {}).items()}
        cache.node_model = {int(k): v for k, v in data.get("node_model", {}).items()}
        cache.dynamic_steps = list(data.get("dynamic_steps", []))
        return cache


def _format_functions_as_text(functions: list) -> str:
    """Convert OpenAI tool format to readable text for prompt embedding."""
    if not functions:
        return ""
    lines = ["Available Functions:"]
    for f in functions:
        func = f.get("function", f)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])
        lines.append(f"\n- {name}: {desc}")
        if params:
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                req = "(required)" if pname in required else "(optional)"
                if "enum" in pinfo:
                    req += f" [allowed: {', '.join(map(str, pinfo['enum']))}]"
                lines.append(f"    {pname} ({ptype}) {req}: {pdesc}")
    return "\n".join(lines)



class Agent:
    
    def __init__(self, agent_id: int, agent_type: AgentType, task_type: TaskType,
                 model_name: str = None, max_tokens: int = None):
        self.id = agent_id
        self.type = agent_type
        self.task_type = task_type
        self.template = get_agent_template(task_type, agent_type)
        self.last_raw = ""
        self.last_prompt = ""
        self.last_context_only_prompt = ""
        self.model_name = model_name or DEFAULT_AGENT_MODEL
        self._model_client = get_model(self.model_name)
        self._max_tokens = max_tokens
        self._skip_compression = False
        self._timeout_s = None
        self.last_error = None

    def _generation_timeout_s(self) -> int:
        if self._timeout_s is not None:
            return self._timeout_s
        if self.task_type != TaskType.CODE_GENERATION:
            return 0
        try:
            return int(os.getenv("LCB_AGENT_TIMEOUT", "0"))
        except ValueError:
            return 0

    async def arun(self, question: str, context: Optional[Any] = None, *, extract: bool = True) -> str:
        if self.type is AgentType.AGGREGATOR:
            return self._aggregate(context or [])

        try:
            context_text = context if context is not None else ""
            context_block = (
                f"其他 Agent 的参考结果：\n{context_text}"
                if str(context_text).strip() else ""
            )
            prompt = self.template.format(
                question=question, 
                context=context_text,
                context_block=context_block,
            )
        except (KeyError, IndexError, ValueError) as e:
            print(f"⚠️ Format error: {e}")
            prompt = self.template.replace("{question}", str(question))
            prompt = prompt.replace("{context}", str(context) if context else "")
            prompt = prompt.replace("{context_block}", context_block if 'context_block' in locals() else "")
        
        _mt = self._max_tokens if self._max_tokens is not None else (1024 if self.task_type == TaskType.FUNCTION_CALLING else 2048)
        timeout_s = self._generation_timeout_s()
        raw = await async_call_llm(
            self._model_client,
            prompt,
            temperature=0.0,
            max_tokens=_mt,
            timeout_s=timeout_s,
            timeout_label=f"agent={self.id},type={self.type.value},task={self.task_type.value},model={self.model_name}",
        )
        self.last_raw = raw
        if self.task_type == TaskType.CODE_GENERATION and not raw:
            return ""
        
        if self.type is AgentType.PREDICTOR and not extract:
            compress_template = COMPRESS_PROMPTS.get(self.task_type, COMPRESS_PROMPTS[TaskType.MATH])
            compress_prompt = compress_template.format(raw=raw)
            short = await async_call_llm(self._model_client, compress_prompt, 0.0)
            return short
        
        return extract_output(raw, self.task_type) if extract else raw

    async def arun_full(self, question: str, context: Optional[str] = None, functions: list = None) -> Tuple[str, str, int, int]:

        if self.type is AgentType.AGGREGATOR:
            if functions is not None:
                pass  # Aggregator doesn't use functions
            self.last_prompt = context or ""
            self.last_context_only_prompt = context or ""
            result = self._aggregate(context.split("\n---\n") if context else [])
            return result, result, 0, 0

        try:
            context_text = context if context is not None else ""
            context_block = (
                f"其他 Agent 的参考结果：\n{context_text}"
                if str(context_text).strip() else ""
            )
            # CS-only intermediate view: preserve the Agent template and actual
            # upstream context, but omit the task/question text.
            self.last_context_only_prompt = (
                self.template
                .replace("{question}", "")
                .replace("{context}", context_text)
                .replace("{context_block}", context_block)
            )
            prompt = self.template.format(
                question=question,
                context=context_text,
                context_block=context_block,
            )
        except (KeyError, IndexError, ValueError):
            prompt = self.template.replace("{question}", str(question))
            prompt = prompt.replace("{context}", str(context) if context else "")
            prompt = prompt.replace("{context_block}", context_block if 'context_block' in locals() else "")

        self.last_prompt = prompt
        _mt = self._max_tokens if self._max_tokens is not None else (1024 if self.task_type == TaskType.FUNCTION_CALLING else 2048)
        timeout_s = self._generation_timeout_s()
        raw_result = await async_call_llm(
            self._model_client,
            prompt,
            temperature=0.0,
            tools=functions,
            max_tokens=_mt,
            timeout_s=timeout_s,
            timeout_label=f"agent={self.id},type={self.type.value},task={self.task_type.value},model={self.model_name}",
            return_usage=True,
        )
        raw = raw_result["text"]
        self.last_raw = raw
        self.last_error = raw_result.get("error")
        raw_usage = raw_result["usage"]
        if self.task_type == TaskType.CODE_GENERATION and not raw:
            return "", "", raw_usage.get("prompt_tokens", 0), raw_usage.get("completion_tokens", 0)

        # Hierarchical/Council function-calling agents receive tools as text and
        # intentionally run with functions=None. Their upstream nodes produce
        # analysis, not a call to be rewritten. Do not send that analysis through
        # the generic JSON-preserving compressor, which can invent an incompatible
        # function/parameters shape before the final synthesizer sees it.
        if self.task_type == TaskType.FUNCTION_CALLING and functions is None:
            return (
                raw,
                raw,
                raw_usage.get("prompt_tokens", 0),
                raw_usage.get("completion_tokens", 0),
            )

        if self._skip_compression:
            short = extract_output(raw, self.task_type)
            return (raw, short, raw_usage.get("prompt_tokens", 0), raw_usage.get("completion_tokens", 0))

        compress_template = COMPRESS_PROMPTS.get(self.task_type, COMPRESS_PROMPTS[TaskType.MATH])
        compress_prompt = compress_template.format(raw=raw)
        short_result = await async_call_llm(
            self._model_client,
            compress_prompt,
            0.0,
            return_usage=True,
        )
        short = short_result["text"]
        short_usage = short_result["usage"]

        total_input = raw_usage.get("prompt_tokens", 0) + short_usage.get("prompt_tokens", 0)
        total_output = raw_usage.get("completion_tokens", 0) + short_usage.get("completion_tokens", 0)

        return raw, short, total_input, total_output

    def _aggregate(self, answers: List[str]) -> str:
        if not answers:
            return ""
        
        valid_answers = [a.strip() for a in answers if a and a.strip()]
        if not valid_answers:
            return ""
        
        extracted_answers = []
        for ans in valid_answers:
            extracted = extract_output(ans, self.task_type)
            if extracted:
                extracted_answers.append(extracted)
        
        if not extracted_answers:
            return extract_output(valid_answers[0], self.task_type)
        
        # Tool-call outputs are JSON-like structured data, not mathematical
        # answers. Never pass them through normalize_answer(), which rewrites
        # JSON square brackets into parentheses.
        if self.task_type == TaskType.FUNCTION_CALLING:
            from collections import Counter
            return Counter(extracted_answers).most_common(1)[0][0]

        if self.task_type in (TaskType.CODE, TaskType.CODE_GENERATION):
            return code_vote(extracted_answers)

        if self.task_type == TaskType.QA:
            from collections import Counter
            # Keep word boundaries for multi-word answers. The HotpotQA
            # judge normalizes whitespace itself, so do not use math's
            # whitespace-removing normalize_answer here.
            normalized_answers = [normalize_qa_answer(a) for a in extracted_answers]
            return Counter(normalized_answers).most_common(1)[0][0]

        return majority_vote(extracted_answers)



class MAS:
    
    def __init__(self, graph_type: GraphType, task_type: TaskType,
                 Ns: int = 1, Na: int = 3, Nr: int = 1, Nd: int = 1,
                 use_judge: bool = False, judge_client=None,
                 agent_model: str = None,
                 agent_models: Optional[Dict[int, str]] = None,
                 selector_model: str = "qwen35-9b", dataset_name: str = "bfcl_v3"):
        self.gtype = graph_type
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.Ns, self.Na, self.Nr, self.Nd = Ns, Na, Nr, Nd
        self.agent_model = agent_model or DEFAULT_AGENT_MODEL
        # Per-agent model assignment: {agent_id: model_name}. Falls back to agent_model.
        self.agent_models = agent_models or {}
        self.selector_model = selector_model
        self.selector_client = None
        self.dynamic_role_specs = []
        self.output_role_id = None
        self.max_agent_steps = None
        self.agents: List[Agent] = []
        self.edges: Dict[int, List[int]] = {}
        self._build()

        self.use_judge = use_judge
        if use_judge:
            if task_type == TaskType.CODE:
                self.judge = CodeJudgeAgent()
            elif task_type == TaskType.CODE_GENERATION:
                self.judge = LCBJudge()
            elif task_type == TaskType.FUNCTION_CALLING:
                self.judge = BFCLJudge()
            elif task_type == TaskType.QA:
                self.judge = HotpotQAJudge()
            elif task_type in [TaskType.MATH_CHOICE, TaskType.REASONING_CHOICE]:
                self.judge = LLMJudgeAgent(judge_client, "multi-choice problem")
            else:
                self.judge = LLMJudgeAgent(judge_client, "math problem")
        else:
            self.judge = None

    def _get_model_for(self, agent_id: int) -> str:
        """Return the model name to use for a given agent.

        Looks up agent_models dict first. Falls back to agent_model default.
        """
        return self.agent_models.get(agent_id, self.agent_model)

    def _build(self):
        a = self.agents
        tt = self.task_type

        if self.gtype is GraphType.SUMMARIZE:
            a.extend([Agent(0, AgentType.SUMMARIZER, tt, model_name=self._get_model_for(0)),
                      Agent(1, AgentType.PREDICTOR, tt, model_name=self._get_model_for(1))])
            self.edges = {0: [1], 1: []}

        elif self.gtype is GraphType.AGGREGATE:
            for i in range(self.Na):
                a.append(Agent(i, AgentType.PREDICTOR, tt, model_name=self._get_model_for(i)))
            agg_id = self.Na
            a.append(Agent(agg_id, AgentType.AGGREGATOR, tt, model_name=self._get_model_for(agg_id)))
            self.edges = {i: [agg_id] for i in range(self.Na)}
            self.edges[agg_id] = []

        elif self.gtype is GraphType.REFLECT:
            for i in range(self.Nr):
                pred_id = len(a)
                predictor = Agent(pred_id, AgentType.PREDICTOR, tt, model_name=self._get_model_for(pred_id), max_tokens=2048)
                refl_id = pred_id + 1
                reflector = Agent(refl_id, AgentType.REFLECTOR, tt, model_name=self._get_model_for(refl_id), max_tokens=2048)
                if tt is TaskType.REASONING_CHOICE:
                    predictor.template = get_agent_template(tt, AgentType.PREDICTOR)
                    reflector.template = get_agent_template(tt, AgentType.REFLECTOR)
                    predictor._max_tokens = 2048
                    reflector._max_tokens = 2048
                    predictor._skip_compression = False
                    reflector._skip_compression = False
                elif tt is TaskType.MATH:
                    predictor.template = get_agi_reflect_template(
                        "predictor_initial" if i == 0 else "predictor_revision"
                    )
                    reflector.template = get_agi_reflect_template("reflector")
                elif tt is TaskType.MATH_CHOICE:
                    predictor.template = get_reflect_agi_template(tt, AgentType.PREDICTOR)
                    reflector.template = get_reflect_agi_template(tt, AgentType.REFLECTOR)
                    predictor._max_tokens = 1024
                    reflector._max_tokens = 1024
                    predictor._skip_compression = True
                    reflector._skip_compression = True
                a.extend([predictor, reflector])

                self.edges[pred_id] = [refl_id]
                self.edges[refl_id] = []

                if i > 0:
                    prev_refl_id = pred_id - 1
                    self.edges[prev_refl_id] = [pred_id]

        elif self.gtype is GraphType.DEBATE:
            a.extend([
                Agent(0, AgentType.PREDICTOR, tt, model_name=self._get_model_for(0)),
                Agent(1, AgentType.PREDICTOR, tt, model_name=self._get_model_for(1)),
                Agent(2, AgentType.DEBATOR, tt, model_name=self._get_model_for(2))
            ])
            self.edges = {0: [2], 1: [2], 2: []}

        elif self.gtype is GraphType.LLM_AGG:
            for i in range(self.Na):
                a.append(Agent(i, AgentType.PREDICTOR, tt, model_name=self._get_model_for(i)))
            llm_agg_id = self.Na
            a.append(Agent(llm_agg_id, AgentType.LLM_AGG, tt, model_name=self._get_model_for(llm_agg_id)))
            self.edges = {i: [llm_agg_id] for i in range(self.Na)}
            self.edges[llm_agg_id] = []

        elif self.gtype is GraphType.DEBATE_LLM_AGG:
            for rnd in range(self.Nd):
                p0 = len(self.agents)
                self.agents.append(Agent(p0, AgentType.PREDICTOR, tt, model_name=self._get_model_for(p0)))
                p1 = len(self.agents)
                self.agents.append(Agent(p1, AgentType.PREDICTOR, tt, model_name=self._get_model_for(p1)))
                d = len(self.agents)
                self.agents.append(Agent(d, AgentType.DEBATOR, tt, model_name=self._get_model_for(d)))
                self.edges[p0] = [d]
                self.edges[p1] = [d]
                self.edges[d] = []

            llm_agg = len(self.agents)
            self.agents.append(Agent(llm_agg, AgentType.LLM_AGG, tt, model_name=self._get_model_for(llm_agg)))
            for rnd in range(self.Nd):
                self.edges[rnd*3 + 2] = [llm_agg]
            self.edges[llm_agg] = []
        elif self.gtype in (
                GraphType.SINGLE_AGENT,
                GraphType.SINGLE_AGENT_COT,
                GraphType.SINGLE_AGENT_TPE,
                GraphType.SINGLE_AGENT_SPO,
            ):
            # A true single-agent baseline: one model call, no compression call.
            agent = Agent(
                0,
                AgentType.PREDICTOR,
                tt,
                model_name=self._get_model_for(0),
            )
            agent._skip_compression = True
            try:
                agent._timeout_s = int(os.getenv("SINGLE_AGENT_TIMEOUT", "120"))
            except ValueError:
                agent._timeout_s = 120
            a.append(agent)
            self.edges = {0: []}
        elif self.gtype is GraphType.HIERARCHICAL:
            from prompts import HIERARCHICAL_PROMPTS

            agent_types = [
                AgentType.MATH_AGENT,  # 0
                AgentType.SCIENCE_AGENT,   # 1
                AgentType.CODE_AGENT,  # 2
                AgentType.TASK_SUMMARIZER    # 3
            ]

            current_prompts = HIERARCHICAL_PROMPTS.get(tt, HIERARCHICAL_PROMPTS[TaskType.MATH])
            for i in range(4):
                agent = Agent(i, agent_types[i], tt, model_name=self._get_model_for(i))
                agent.template = current_prompts[str(i)]
                a.append(agent)

            self.edges = {
                0: [3],
                1: [3],
                2: [3],
                3: []
            }

        elif self.gtype is GraphType.COUNCIL:
            from prompts import COUNCIL_PROMPTS
            current_prompts = COUNCIL_PROMPTS.get(tt, COUNCIL_PROMPTS[TaskType.MATH])
            # Round 1: 3 predictors with different personas
            for i in range(3):
                agent = Agent(i, AgentType.PREDICTOR, tt, model_name=self._get_model_for(i))
                agent.template = current_prompts[str(i)]
                a.append(agent)
            # Round 2: 3 predictors (receive round 1 outputs as context)
            for i in range(3, 6):
                agent = Agent(i, AgentType.PREDICTOR, tt, model_name=self._get_model_for(i))
                agent.template = current_prompts[str(i)]
                a.append(agent)
            # Mechanical voter (no LLM)
            a.append(Agent(6, AgentType.AGGREGATOR, tt, model_name=self._get_model_for(6)))
            self.edges = {
                0: [3, 4, 5], 1: [3, 4, 5], 2: [3, 4, 5],
                3: [6], 4: [6], 5: [6],
                6: []
            }

        elif self.gtype is GraphType.SELECTOR_DYNAMIC:
            from dynamic_prompts import (
                get_dynamic_role_specs,
            )
            self.dynamic_role_specs = get_dynamic_role_specs(self.dataset_name)
            self.output_role_id = 4
            self.max_agent_steps = 5
            for role_id, spec in enumerate(self.dynamic_role_specs):
                agent = Agent(
                    role_id,
                    spec["agent_type"] if "agent_type" in spec else AgentType.PREDICTOR,
                    tt,
                    model_name=self._get_model_for(role_id),
                )
                agent.role_name = spec["name"]
                agent.role_description = spec["description"]
                agent.template = spec["prompt"]
                agent._skip_compression = True
                a.append(agent)
            # Dynamic execution does not use a static edge DAG. Keep a
            # deterministic role order for the optimizer's variable order.
            self.edges = {i: [] for i in range(len(self.agents))}

    
    def _build_council_round2_context(self, cache) -> str:
        """Build truncated, labeled context for Council Round 2."""
        labels = [
            ("直接推理专家 (Direct Reasoner)", 0),
            ("多角度分析专家 (Multi-angle Thinker)", 1),
            ("反事实验证专家 (Counterfactual Verifier)", 2),
        ]
        parts = []
        for label, aid in labels:
            raw = cache.node_outputs_raw.get(aid, "")
            if len(raw) > 500:
                raw = raw[:500] + "...[truncated]"
            parts.append(f"- {label}:\n{raw}")
        return "以下其他专家的第一轮答案，请综合参考后给出你的最终答案：\n\n" + "\n\n".join(parts)

    def _topo_order(self) -> List[int]:
        if self.gtype is GraphType.SELECTOR_DYNAMIC:
            return [a.id for a in self.agents]
        in_degree = defaultdict(int)
        for src, dsts in self.edges.items():
            for d in dsts:
                in_degree[d] += 1
        
        queue = deque([a.id for a in self.agents if in_degree[a.id] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.edges.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order

    def _get_levels(self) -> List[List[int]]:
        if self.gtype is GraphType.SELECTOR_DYNAMIC:
            return [[a.id] for a in self.agents]
        in_degree = defaultdict(int)
        for src, dsts in self.edges.items():
            for d in dsts:
                in_degree[d] += 1

        queue = deque([a.id for a in self.agents if in_degree[a.id] == 0])
        levels = []
        while queue:
            level = list(queue)
            queue.clear()
            for node in level:
                for child in self.edges.get(node, []):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
            levels.append(level)
        return levels

    def get_predecessors(self, agent_id: int) -> List[int]:
        return [k for k, v in self.edges.items() if agent_id in v]

    def get_successors(self, agent_id: int) -> List[int]:
        return self.edges.get(agent_id, [])

    def get_all_predecessors(self, agent_id: int) -> Set[int]:
        predecessors = set()
        queue = deque(self.get_predecessors(agent_id))
        while queue:
            node = queue.popleft()
            if node not in predecessors:
                predecessors.add(node)
                queue.extend(self.get_predecessors(node))
        return predecessors

    def get_all_successors(self, agent_id: int) -> Set[int]:
        successors = set()
        queue = deque(self.get_successors(agent_id))
        while queue:
            node = queue.popleft()
            if node not in successors:
                successors.add(node)
                queue.extend(self.get_successors(node))
        return successors

    def get_terminal_id(self) -> int:
        if self.gtype is GraphType.SELECTOR_DYNAMIC:
            return self.output_role_id
        terminal_nodes = [i for i, v in self.edges.items() if not v]
        return terminal_nodes[0] if terminal_nodes else -1

    def inject_prompt_map(self, prompt_map: Dict[int, str]):
        for aid, p in prompt_map.items():
            if aid < len(self.agents) and self.agents[aid].type != AgentType.AGGREGATOR:
                self.agents[aid].template = p


    async def arun(self, question: str, context: Optional[str] = None, functions: list = None) -> Dict[str, Any]:
        result, _ = await self.arun_with_cache(question, context, functions=functions)
        return result

    async def arun_with_cache(self, question: str,
                               context: Optional[str] = None,
                               functions: list = None) -> Tuple[Dict[str, Any], InferenceCache]:
        if self.gtype is GraphType.SELECTOR_DYNAMIC:
            return await self._arun_selector_dynamic_with_cache(
                question, context=context, functions=functions
            )
        cache = InferenceCache()
        cache.set_question(question)

        levels = self._get_levels()
        terminal_id = self.get_terminal_id()

        # Embed functions as text for Council/Hierarchical FUNCTION_CALLING;
        # these branches use prompt-constrained JSON instead of native tools.
        _text_question = question
        if (self.task_type == TaskType.FUNCTION_CALLING and
            self.gtype in (GraphType.HIERARCHICAL, GraphType.COUNCIL) and functions):
            _text_question = question + "\n\n" + _format_functions_as_text(functions)

        for level in levels:
            tasks = []
            for agent_id in level:
                agent = self.agents[agent_id]
                predecessors = self.get_predecessors(agent_id)

                if predecessors:
                    if agent.type == AgentType.AGGREGATOR:
                        ctx = cache.get_context_for_node(agent_id, predecessors, use_short=False)
                    elif self.gtype == GraphType.COUNCIL and agent_id in (3, 4, 5):
                        ctx = self._build_council_round2_context(cache)
                    else:
                        ctx = cache.get_context_for_node(agent_id, predecessors, use_short=True)
                else:
                    ctx = context

                # For Council/Hierarchical FC: all agents get function text, no native FC
                if (self.task_type == TaskType.FUNCTION_CALLING and functions and
                    (self.gtype in (GraphType.HIERARCHICAL, GraphType.COUNCIL) or
                     ("deepseek" in str(agent.model_name or "").lower() and
                      self.gtype in (GraphType.AGGREGATE, GraphType.REFLECT,
                                     GraphType.SINGLE_AGENT,
                                     GraphType.SINGLE_AGENT_COT,
                                     GraphType.SINGLE_AGENT_TPE,
                                     GraphType.SINGLE_AGENT_SPO)))):
                    _q = question + "\n\n" + _format_functions_as_text(functions)
                    _f = None
                else:
                    _q = question
                    _f = functions

                tasks.append(self._run_single_agent(agent, _q, ctx, agent_id == terminal_id, functions=_f))

            results = await asyncio.gather(*tasks)

            for agent_id, (raw, short, input_tokens, output_tokens) in zip(level, results):
                predecessors = self.get_predecessors(agent_id)
                ctx = cache.get_context_for_node(agent_id, predecessors) if predecessors else (context or "")
                cache.set_node_data(agent_id, ctx, raw, short,
                                    input_tokens, output_tokens,
                                    self.agents[agent_id].model_name)
                cache.node_prompts[agent_id] = self.agents[agent_id].last_prompt

        # The terminal node is the MAS-level output for every task type.
        final_raw = cache.node_outputs_raw.get(terminal_id, "")
        final_answer = extract_output(final_raw, self.task_type)

        return {
            "final": final_answer,
            "raw_trace": cache.node_outputs_raw.copy(),
            "short_trace": cache.node_outputs_short.copy(),
        }, cache

    async def _select_dynamic_role(self, question: str, steps: List[Dict[str, Any]],
                                   candidates: List[int], remaining_steps: int) -> int:
        from dynamic_prompts import get_dynamic_selector_prompt

        if self.selector_client is None:
            self.selector_client = get_model(self.selector_model)
        roles = "\n".join(
            f"{idx}: {spec['name']} - {spec['description']}"
            for idx, spec in enumerate(self.dynamic_role_specs)
        )
        history = "\n\n".join(
            f"step {step['step_id']} role {step['role_id']}: {step['output']}"
            for step in steps
        ) or "No agent has responded yet."
        prompt = get_dynamic_selector_prompt(self.dataset_name).format(
            question=question,
            roles=roles,
            candidates=candidates,
            remaining_steps=remaining_steps,
            history=history,
        )
        raw = await async_call_llm(
            self.selector_client,
            prompt,
            temperature=0.0,
            max_tokens=16,
        )
        matches = re.findall(r"(?<!\d)([0-4])(?!\d)", str(raw))
        for match in matches:
            role_id = int(match)
            if role_id in candidates:
                return role_id
        return candidates[0]

    async def _arun_selector_dynamic_with_cache(
        self, question: str, context: Optional[str] = None,
        functions: list = None,
    ) -> Tuple[Dict[str, Any], InferenceCache]:
        cache = InferenceCache()
        cache.set_question(question)
        steps: List[Dict[str, Any]] = []

        for step_id in range(self.max_agent_steps):
            remaining_steps = self.max_agent_steps - len(steps)
            if remaining_steps == 1:
                role_id = self.output_role_id
                selection_source = "forced_output"
            else:
                role_id = await self._select_dynamic_role(
                    question, steps, list(range(len(self.agents))), remaining_steps
                )
                selection_source = "selector"

            agent = self.agents[role_id]
            previous_output = steps[-1]["output"] if steps else (
                "NONE. You are the first agent in the execution."
            )
            role_context = f"Previous agent output:\n{previous_output}"
            # DeepSeek BFCL endpoints do not accept the native tools payload in
            # this project. Match the existing Aggregate/Reflect compatibility
            # path by embedding the schema in the prompt instead.
            role_question = question
            role_functions = functions
            if (self.task_type == TaskType.FUNCTION_CALLING and functions and
                    "deepseek" in str(agent.model_name or "").lower()):
                role_question = question + "\n\n" + _format_functions_as_text(functions)
                role_functions = None
            raw, short, input_tokens, output_tokens = await agent.arun_full(
                role_question, role_context, functions=role_functions
            )
            step = {
                "step_id": step_id,
                "role_id": role_id,
                "role_name": getattr(agent, "role_name", agent.type.value),
                "input": agent.last_prompt,
                "output": raw,
                "short_output": short,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": agent.model_name,
                "selection_source": selection_source,
            }
            steps.append(step)

            # Keep role-level aggregates for compatibility with cost helpers;
            # preserve every invocation separately in dynamic_steps.
            cache.node_inputs[role_id] = role_context
            cache.node_prompts[role_id] = agent.last_prompt
            cache.node_outputs_raw[role_id] = raw
            cache.node_outputs_short[role_id] = short
            cache.node_input_tokens[role_id] = (
                cache.node_input_tokens.get(role_id, 0) + input_tokens
            )
            cache.node_output_tokens[role_id] = (
                cache.node_output_tokens.get(role_id, 0) + output_tokens
            )
            cache.node_model[role_id] = agent.model_name
            cache.dynamic_steps.append(step)

            if role_id == self.output_role_id:
                break

        final_raw = steps[-1]["output"] if steps else ""
        return {
            "final": extract_output(final_raw, self.task_type),
            "final_raw": final_raw,
            "raw_trace": cache.node_outputs_raw.copy(),
            "short_trace": cache.node_outputs_short.copy(),
            "trajectory": steps,
            "selected_path": [step["role_id"] for step in steps],
            "stop_reason": (
                "forced_output_at_limit"
                if steps and steps[-1]["selection_source"] == "forced_output"
                else "output_role_selected"
            ),
        }, cache

    async def _run_single_agent(self, agent: Agent, question: str,
                                 context: Optional[str], is_terminal: bool,
                                 functions: list = None) -> Tuple[str, str, int, int]:
        raw, short, input_tokens, output_tokens = await agent.arun_full(question, context, functions=functions)


        if agent.type == AgentType.AGGREGATOR:
            answers = context.split("\n---\n") if context else []
            result = agent._aggregate(answers)
            return result, result, 0, 0
        if is_terminal:
            short = extract_output(raw, self.task_type)

        return raw, short, input_tokens, output_tokens

    async def arun_from_node(self, start_node: int, 
                              base_cache: InferenceCache,
                              new_prompt: Optional[str] = None) -> Tuple[Dict[str, Any], InferenceCache]:
        question = base_cache.question
        terminal_id = self.get_terminal_id()
        
        
        nodes_to_run = {start_node} | self.get_all_successors(start_node)

        all_nodes = {a.id for a in self.agents}
        nodes_to_keep = all_nodes - nodes_to_run
        
        new_cache = base_cache.clone_up_to(nodes_to_keep)
        
        original_template = None
        if new_prompt is not None:
            original_template = self.agents[start_node].template
            self.agents[start_node].template = new_prompt
        
        try:
            topo_order = self._topo_order()
            nodes_to_run_sorted = [n for n in topo_order if n in nodes_to_run]
            
            for agent_id in nodes_to_run_sorted:
                agent = self.agents[agent_id]
                predecessors = self.get_predecessors(agent_id)

                ctx = new_cache.get_context_for_node(agent_id, predecessors, use_short=True)

                raw, short, input_tokens, output_tokens = await agent.arun_full(question, ctx if ctx else None)

                if agent_id == terminal_id:
                    short = extract_output(raw, self.task_type)

                new_cache.set_node_data(agent_id, ctx, raw, short,
                                        input_tokens, output_tokens,
                                        agent.model_name)
                new_cache.node_prompts[agent_id] = agent.last_prompt
            
            final_raw = new_cache.node_outputs_raw.get(terminal_id, "")
            final_answer = extract_output(final_raw, self.task_type)
            
            return {
                "final": final_answer,
                "raw_trace": new_cache.node_outputs_raw.copy(),
                "short_trace": new_cache.node_outputs_short.copy(),
            }, new_cache
            
        finally:
            if original_template is not None:
                self.agents[start_node].template = original_template

    async def arun_single_node_only(self, agent_id: int, question: str, 
                                     context: str, new_prompt: Optional[str] = None) -> Tuple[str, str]:
        agent = self.agents[agent_id]
        
        original_template = None
        if new_prompt is not None:
            original_template = agent.template
            agent.template = new_prompt
        
        try:
            raw, short, _, _ = await agent.arun_full(question, context)
            return raw, short
        finally:
            if original_template is not None:
                agent.template = original_template
