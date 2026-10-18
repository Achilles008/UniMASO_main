from __future__ import annotations

import asyncio
import itertools
import json
import math
import re
import uuid
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import (
    AVAILABLE_MODELS, DEFAULT_OPTIMIZER_MODEL, get_model, get_model_cost,
    get_model_profile, GraphType,
)
from utils import async_call_llm

PROFILE_PATH = Path(__file__).resolve().parent / "config" / "topology_agent_profiles.json"
PRIOR_PATH = Path(__file__).resolve().parent / "config" / "topology_allocation_priors.json"
MODEL_PRIOR_ALPHA = 0.5

def center_preference_weight(rank: int, model_count: int, min_weight: float = 0.2) -> float:
    """Linear prior that favors the middle capability tier in the active pool."""
    if model_count <= 1:
        return 1.0
    center = (model_count - 1) / 2
    distance = abs(rank - center) / center
    return min_weight + (1.0 - min_weight) * (1.0 - distance)

def model_prior_weights(model_pool: Iterable[str]) -> Dict[str, float]:
    pool = list(model_pool)
    return {
        model: center_preference_weight(rank, len(pool))
        for rank, model in enumerate(pool)
    }

def choose_weighted_model(votes: List[str], model_pool: Iterable[str]) -> Optional[str]:
    """Match the test scorer: vote count first, center prior only for close ties."""
    pool = list(model_pool)
    counts = Counter(votes)
    priors = model_prior_weights(pool)
    ranked = sorted(
        counts,
        key=lambda model: (
            -counts[model],
            -MODEL_PRIOR_ALPHA * priors.get(model, 0.0),
            pool.index(model),
        ),
    )
    return ranked[0] if ranked else None


def _trace_model_order(model_pool: Iterable[str]) -> List[str]:
    """Order active models by their configured capability rank."""
    pool = list(model_pool)
    ranked = []
    for index, model in enumerate(pool):
        profile = get_model_profile(model) or {}
        rank = profile.get("capability_rank", index)
        try:
            rank = float(rank)
        except (TypeError, ValueError):
            rank = float(index)
        ranked.append((rank, index, model))
    return [model for _, _, model in sorted(ranked)]


def _trace_boundary_risk(model: str, ordered_models: List[str]) -> float:
    """Return a soft risk for the two capability boundaries."""
    if len(ordered_models) <= 2:
        return 0.0
    position = ordered_models.index(model)
    z = position / float(len(ordered_models) - 1)
    return 1.0 - 4.0 * z * (1.0 - z)


def _trace_entropy_consensus(
    proposals: List[Dict[str, Any]],
    current: Dict[int, str],
    model_pool: Iterable[str],
    agent_ids: Iterable[int],
) -> Dict[int, str]:
    """Select a trace-supported assignment using entropy-aware consensus.

    Every selected model must have appeared at the same Agent position in at
    least one real Co-Evolver proposal. The method may recombine such
    position-level choices, but it never invents a model choice.
    """
    ordered_models = _trace_model_order(model_pool)
    if not ordered_models:
        return dict(current)

    normalized_proposals = []
    for proposal in proposals:
        assignment = proposal.get("model_assignment", {})
        if not isinstance(assignment, dict):
            continue
        normalized_proposals.append({
            int(key): str(value)
            for key, value in assignment.items()
            if str(value) in ordered_models
        })

    selected_ids = [int(aid) for aid in agent_ids]
    posteriors = {}
    for aid in selected_ids:
        counts = Counter(
            assignment[aid]
            for assignment in normalized_proposals
            if aid in assignment
        )
        if not counts:
            posteriors[aid] = {"support": {}, "candidates": []}
            continue
        total = sum(counts.values())
        probabilities = {
            model: counts.get(model, 0) / float(total)
            for model in ordered_models
        }
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities.values()
            if probability > 0
        )
        max_entropy = math.log(len(ordered_models)) if len(ordered_models) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy else 0.0
        confidence = 1.0 - normalized_entropy
        support = {
            model: probabilities[model] * confidence
            for model in counts
        }
        posteriors[aid] = {
            "support": support,
            "candidates": sorted(counts),
        }

    regular_models = ordered_models[1:-1] if len(ordered_models) > 2 else ordered_models
    candidate_sets = [
        posterior["candidates"] or [current.get(aid)]
        for aid, posterior in posteriors.items()
    ]
    if any(not candidates or candidates == [None] for candidates in candidate_sets):
        return dict(current)

    candidates = []
    for values in itertools.product(*candidate_sets):
        assignment = dict(zip(selected_ids, values))
        used = set(assignment.values())
        regular_used = used & set(regular_models)
        regular_positions = [regular_models.index(model) for model in regular_used]
        strong_regular_present = bool(regular_positions) and (
            max(regular_positions) == len(regular_models) - 1
        )
        boundary_risk = sum(
            _trace_boundary_risk(model, ordered_models)
            for model in assignment.values()
        ) / max(len(assignment), 1)
        support = sum(
            posteriors[aid]["support"].get(model, 0.0)
            for aid, model in assignment.items()
        )
        score = (
            1.0 * support
            + 2.0 * len(regular_used)
            + 3.0 * int(strong_regular_present)
            - 3.0 * boundary_risk
            - 1.0 * int(len(used) == 1)
        )
        candidates.append((score, tuple(sorted(assignment.items())), assignment))

    _, _, selected = sorted(
        candidates,
        key=lambda item: (-item[0], item[1]),
    )[0]
    return selected

def load_topology_profile(dataset: str, graph_type: str) -> Dict[str, Any]:
    data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    aliases = {"bfcl_v3": "bfcl", "livecodebench_v6": "lcb"}
    dataset = aliases.get(dataset, dataset)
    profile = data["datasets"][dataset]["graphs"][graph_type]
    result = dict(profile)
    result["dataset"] = dataset
    result["graph_type"] = graph_type
    result["task_type"] = data["datasets"][dataset].get("task_type", "")
    result["task_focus"] = data["datasets"][dataset].get("task_focus", "")
    return result


def expand_reflect_profile(
    profile: Dict[str, Any],
    runtime_agents: List[Any],
    nr: int,
) -> Dict[str, Any]:
    """Expand the two-role Reflect profile to the actual MAS runtime chain.

    The static profile describes one predictor/reflector pair.  MAS creates
    ``2 * Nr`` runtime agents, so the Co-Evolver must receive the expanded
    chain rather than the static two-agent profile.
    """
    if nr < 1:
        raise ValueError("Reflect nr must be >= 1")

    expected_count = 2 * nr
    if len(runtime_agents) < expected_count:
        raise ValueError(
            f"Expected {expected_count} Reflect runtime agents, "
            f"got {len(runtime_agents)}"
        )

    base_agents = {
        int(agent["agent_id"]): dict(agent)
        for agent in profile.get("agents", [])
    }
    if 0 not in base_agents or 1 not in base_agents:
        raise ValueError(
            "Reflect profile must contain predictor Agent 0 "
            "and reflector Agent 1"
        )

    expanded_agents = []
    for runtime_id in range(expected_count):
        logical_role_id = runtime_id % 2
        replica_id = runtime_id // 2
        agent = dict(base_agents[logical_role_id])
        runtime_agent = runtime_agents[runtime_id]

        agent["agent_id"] = runtime_id
        agent["runtime_role_id"] = logical_role_id
        agent["runtime_replica"] = replica_id
        agent["initial_prompt"] = runtime_agent.template
        agent["upstream_agents"] = [runtime_id - 1] if runtime_id else []
        agent["downstream_agents"] = (
            [runtime_id + 1]
            if runtime_id < expected_count - 1 else []
        )
        agent["is_terminal"] = runtime_id == expected_count - 1
        expanded_agents.append(agent)

    expanded = dict(profile)
    expanded.update({
        "agent_count": expected_count,
        "nr": nr,
        "runtime_agent_ids": list(range(expected_count)),
        "topological_order": list(range(expected_count)),
        "edges": [[i, i + 1] for i in range(expected_count - 1)],
        "agents": expanded_agents,
        "topology_purpose": (
            f"sequential solving and reflection with Nr={nr}; "
            "runtime predictor/reflector pairs"
        ),
    })
    return expanded

def load_topology_prior(graph_type: str) -> str:
    data = json.loads(PRIOR_PATH.read_text(encoding="utf-8"))
    return data.get("priors", {}).get(graph_type, {}).get("allocation_prior", "")

def _clip(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + "\n...[clipped]"

def _json_from_response(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    candidates = re.findall(r"\`\`\`(?:json)?\s*(.*?)\`\`\`", text, re.S | re.I) + [text]
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    result = {}
    m = re.search(r"<(?:final_prompt|prompt)>(.*?)</(?:final_prompt|prompt)>", text, re.S | re.I)
    if m:
        result["final_prompt"] = m.group(1).strip()
    m = re.search(r"<(?:model_assignment|models)>(.*?)</(?:model_assignment|models)>", text, re.S | re.I)
    if m:
        try:
            result["model_assignment"] = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    return result

class CoEvolver:
    """Round-boundary prompt/model co-evolutioner."""

    def __init__(self, optimizer_model: str = DEFAULT_OPTIMIZER_MODEL,
                 model_pool: Optional[Iterable[str]] = None,
                 max_prompt_chars: int = 0,
                 trace_dir: Optional[str] = None):
        self.optimizer_model = optimizer_model
        self.client = get_model(optimizer_model)
        self.model_pool = list(model_pool or AVAILABLE_MODELS)
        self.max_prompt_chars = max_prompt_chars
        self.trace_dir = Path(trace_dir) if trace_dir else None
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.trace_path = self.trace_dir / "co_evolver_traces.jsonl"
        else:
            self.trace_path = None
        self._trace_lock = asyncio.Lock()

    @staticmethod
    def _profile_text(profile: Dict[str, Any]) -> str:
        is_runtime_reflect = (
            profile.get("graph_type") == GraphType.REFLECT.value
            and profile.get("nr", 1) > 1
        )
        agent_label = (
            "Runtime Agents eligible for Prompt/model optimization:"
            if is_runtime_reflect
            else "Logical Agents eligible for Prompt/model optimization:"
        )
        lines = [
            f"Topology: {profile.get('dataset')}/{profile.get('task_type')}",
            f"Purpose: {profile.get('topology_purpose', '')}",
            f"Task focus: {profile.get('task_focus', '')}",
            f"Agent count: {profile.get('agent_count', len(profile.get('agents', [])))}",
            f"Logical edges: {profile.get('edges', [])}",
            f"Logical topological order: {profile.get('topological_order', [])}",
            agent_label,
        ]
        for agent in profile.get("agents", []):
            lines.append(
                f"- id={agent.get('agent_id')}, type={agent.get('agent_type')}, role={agent.get('role')}, "
                f"terminal={agent.get('is_terminal')}, optimizable={agent.get('is_optimizable')}"
            )
            lines.append(f"  initial_prompt={_clip(agent.get('initial_prompt', ''), 1600)}")
        stages = profile.get("execution_stages")
        if stages:
            lines.append("Runtime execution stages (informational; not separate optimization variables):")
            for stage in stages:
                lines.append(f"- stage {stage.get('stage')}: {stage.get('description')}")
        terminal = profile.get("terminal_operation")
        if terminal:
            lines.append(
                "Terminal operation (not an Agent and never receives a model assignment): "
                f"type={terminal.get('type')}; description={terminal.get('description')}; "
                f"inputs={terminal.get('input_agents', terminal.get('input_logical_agents', []))}"
            )
        groups = profile.get("shared_runtime_groups")
        if groups:
            lines.append(
                "Shared runtime groups (implementation detail; one model variable per group): "
                f"{groups}"
            )
        if is_runtime_reflect:
            lines.append(
                "For Reflect, every listed runtime predictor/reflector is an "
                "optimization variable. Only terminal mechanical operations are excluded."
            )
        else:
            lines.append(
                "Only the listed logical Agents are optimization variables. "
                "Runtime instances and terminal operations are excluded."
            )
        return "\n".join(lines)

    def _model_pool_text(self) -> str:
        lines = []
        for model in self.model_pool:
            profile = get_model_profile(model)
            cost = get_model_cost(model)
            input_cost = cost["input"]
            output_cost = cost["output"]
            lines.append(
                f"- {model}: capability={profile['capability']}; "
                f"capability_rank={profile['capability_rank']}; "
                f"input_cost_coefficient={input_cost}; "
                f"output_cost_coefficient={output_cost}; "
                f"suitable_for={profile['best_for']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _memory_text(memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return "No historical trajectory was retrieved."
        category_meanings = {
            "successful_low_cost": (
                "Successful and low-cost: this candidate achieved a good global outcome "
                "without substantially increasing inference cost."
            ),
            "local_win_global_lose": (
                "Local improvement but global failure: this candidate improved the current "
                "Agent locally but harmed the end-to-end multi-agent result."
            ),
            "successful_high_cost": (
                "Successful but high-cost: this candidate improved the global outcome "
                "but required substantially higher model or inference cost."
            ),
            "other": "Other historical trajectory.",
        }
        blocks = [
            "HISTORICAL TRAJECTORY REFERENCE",
            "The following historical optimization trajectories are provided as auxiliary "
            "references for the current Prompt evolution.",
        ]
        for i, item in enumerate(memories[:3], 1):
            payload = item.get("payload", {})
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {"raw": payload}
            score = item.get("score", {})
            category = item.get("category", "other")
            meaning = category_meanings.get(category, category_meanings["other"])
            blocks.append(
                f"\\nTrajectory {i}: {meaning}\\n"
                f"Local performance score: {score.get('local_rate')}\\n"
                f"Global performance score: {score.get('global_rate')}\\n"
                f"Cost ratio: {score.get('cost_ratio')}\\n"
                f"Accepted by optimizer: {item.get('status', {}).get('accepted')}\\n"
                f"Candidate Prompt proposed by that optimization:\\n"
                f"{_clip(payload.get('candidate_prompt', ''), 3000)}\\n"
                f"MAS output corresponding to that Candidate Prompt:\\n"
                f"{_clip(payload.get('samples'), 3500)}"
            )
        return "\\n".join(blocks)

    def build_prompt(self, *, agent_id: int, initial_prompt: str, best_prompt: str,
                     initial_model: str, best_model: str,
                     global_prompts: Dict[int, str], global_models: Dict[int, str],
                     profile: Dict[str, Any], memories: List[Dict[str, Any]]) -> str:
        allocation_prior = load_topology_prior(profile.get("graph_type", ""))
        eligible_ids = [
            int(agent["agent_id"])
            for agent in profile.get("agents", [])
            if agent.get("agent_type") != "aggregator"
        ]
        assignment_schema = json.dumps(
            {str(aid): "model_name" for aid in eligible_ids},
            ensure_ascii=False,
        )
        if profile.get("graph_type") == GraphType.REFLECT.value and profile.get("nr", 1) > 1:
            assignment_scope = (
                "all listed Reflect runtime Agent IDs, including every predictor "
                "and reflector created by Nr"
            )
        else:
            assignment_scope = "all listed optimizable logical LLM Agent IDs"
        return f"""You are the Co-Evolver at the end of one optimization round in a heterogeneous multi-agent system.
Use the available topology, role, prompt, trajectory, and model-pool evidence to construct a topology-aware co-evolution result.
This is a multi-angle view: Agent {agent_id} is the local perspective, but model_assignment must cover {assignment_scope} and must exclude mechanical aggregation or voting operations.

GLOBAL TOPOLOGY AND ROLE INFORMATION
{self._profile_text(profile)}

TOPOLOGY-SPECIFIC ALLOCATION PRIOR
{allocation_prior}

AVAILABLE MODEL POOL
{self._model_pool_text()}

CURRENT AGENT PROMPT EVOLUTION
Agent id: {agent_id}
Initial prompt at the start of this round:
{_clip(initial_prompt, 5000)}
Best prompt after depth optimization:
{_clip(best_prompt, 5000)}
Initial model: {initial_model}
Best model after depth optimization: {best_model}

RETRIEVED PROMPT-SIMILAR TRAJECTORIES
{self._memory_text(memories)}

TASK
1. Produce a final prompt for Agent {agent_id}. Use the round-start prompt as the stable foundation and incorporate only valuable improvements from the depth-optimized best prompt. Preserve role, input/output placeholders, and required answer tags.
   Prompt length rules:
   - The round-start prompt is the length baseline.
   - The final prompt must not exceed twice the character length of the round-start prompt.
   - Keep the original structure, role, placeholders, and output format whenever possible.
   - Do not expand the prompt merely to appear more optimized.
   - If the candidate is too long, compress it from the round-start prompt and retain only a few necessary improvements.
   - The system will check the output length after generation. If it still exceeds twice the round-start prompt length, the final prompt will be reverted to the round-start prompt.
   - The round-start prompt is already usable; do not rewrite or expand it without a clear benefit.
 2. Produce a complete model_assignment for {assignment_scope}. Do not include mechanical aggregator/voter nodes. Use only the available model pool.
   The model assignment should primarily consider:
   - the global topology;
   - each Agent's role;
   - upstream and downstream dependencies;
   - task difficulty;
   - critical-path responsibility;
   - model capability;
   - input and output cost coefficients.
   For every candidate model, explicitly consider both input_cost_coefficient and output_cost_coefficient shown in the model pool. A high-capability model is justified only when the Agent's role, task difficulty, or position on the critical path warrants its cost.
   Rules:
   - For complex code generation, complex reasoning, or other high-difficulty tasks, prefer a stronger model within the regular capability range for Agents on the critical path or responsible for core reasoning, result verification, or final integration.
   - For relatively simple tasks such as tool calling, format conversion, or information extraction, prefer a weaker model within the regular capability range for routine Agents that are not on the critical path, in order to reduce cost. Assign critical Agents according to their role and risk of propagating errors.
   - Model assignment must maintain balanced overall capability. When multiple
     capability levels are available, use a layered allocation that matches model
     strength to Agent importance and task difficulty. If a simple or non-critical
     Agent is assigned a weaker model within the non-extreme capability range,
     at least one important or critical-path Agent must be assigned a stronger
     model within the non-extreme capability range to maintain capability balance.
   - Do not assign the same model to every logical LLM Agent unless the topology and task difficulty clearly justify it.
   - Do not collapse into all-strong, all-weak, or all-middle allocation without a topology-based justification.
   - Follow all parameter-sharing and mechanical-node constraints specified by the selected topology prior.
   - Use historical trajectories only as auxiliary evidence for final Prompt evolution.
   - Do not use historical candidate Prompts, MAS outputs, or trajectory scores to determine model_assignment.
   - The lowest-capability and highest-capability models in the pool must be chosen very cautiously and must not be used as regular allocation choices.
   - Select either extreme capability tier only in a confirmed extreme case; when regular capability models are sufficient, prefer a regular capability model.
   - If an extreme tier is selected, briefly state the concrete necessity in decision_summary.
   - Determine model_assignment primarily from the current topology, Agent roles, task difficulty, critical-path responsibility, and model-pool capabilities and costs. Do not use the current initial model or depth-optimized best model as the basis for model assignment.
3. Return only JSON.

Return exactly:
{{
  "final_prompt": "one prompt string",
  "model_assignment": {assignment_schema},
  "decision_summary": "short factual summary"
}}
"""

    def _sanitize_assignment(
        self,
        assignment: Any,
        current: Dict[int, str],
        profile: Dict[str, Any],
    ) -> Dict[int, str]:
        """Validate assignments under MAS parameter sharing constraints."""
        frozen = {
            int(a["agent_id"])
            for a in profile.get("agents", [])
            if a.get("agent_type") == "aggregator"
        }
        logical_ids = {
            int(a["agent_id"])
            for a in profile.get("agents", [])
            if a.get("agent_type") != "aggregator"
        }
        # Co-Evolver results must expose only logical optimization variables.
        # Runtime Council instances are expanded later by the optimizer.
        current_norm = {}
        for key, value in (current or {}).items():
            try:
                aid = int(key)
            except (TypeError, ValueError):
                continue
            if aid in logical_ids and str(value) in self.model_pool:
                current_norm[aid] = str(value)
        result = dict(current_norm)
        if not isinstance(assignment, dict):
            return result
        allowed_ids = logical_ids
        proposed = {}
        for key, value in assignment.items():
            try:
                aid = int(key)
            except (TypeError, ValueError):
                continue
            if aid not in allowed_ids or aid in frozen or str(value) not in self.model_pool:
                continue
            proposed[aid] = str(value)
            result[aid] = str(value)

        # Council runtime nodes are intentionally not added here. The
        # optimizer expands logical assignments only for runtime execution.
        for aid in frozen:
            if aid in current_norm:
                result[aid] = current_norm[aid]
        return result


class DynamicCoEvolver(CoEvolver):
    """Copied Co-Evolver with only the static topology block replaced."""

    def build_prompt(self, *, agent_id: int, initial_prompt: str,
                     best_prompt: str, initial_model: str, best_model: str,
                     global_prompts: Dict[int, str], global_models: Dict[int, str],
                     profile: Dict[str, Any], memories: List[Dict[str, Any]]) -> str:
        role_lines = []
        for agent in profile.get("agents", []):
            aid = int(agent.get("agent_id", -1))
            role_lines.append(
                f"- Role {aid}: role={agent.get('role', '')}, "
                f"description={agent.get('description', '')}, "
                f"current_prompt={_clip(global_prompts.get(aid, ''), 1600)}, "
                f"current_model={global_models.get(aid, '')}, "
                f"terminal={agent.get('is_terminal')}, optimizable={agent.get('is_optimizable')}"
            )
        assignment_schema = json.dumps(
            {str(a.get("agent_id")): "model_name" for a in profile.get("agents", [])},
            ensure_ascii=False,
        )
        # This is the complete static Co-Evolver task body. Only the topology
        # evidence is replaced by the dynamic role-pool evidence above.
        return f"""You are the Co-Evolver at the end of one optimization round in a heterogeneous multi-agent system.
Use the available role-pool, prompt, trajectory, and model-pool evidence to construct a role-aware co-evolution result.
This is a multi-angle view: Agent {agent_id} is the local perspective, but model_assignment must cover all five optimizable logical LLM roles and must exclude the Selector.

DYNAMIC ROLE POOL INFORMATION
dataset={profile.get('dataset')}
task_type={profile.get('task_type')}
Selector model: {profile.get('selector_model', 'qwen35-9b')} (fixed; do not modify)
Selector prompt: fixed and dataset-specific (do not modify)
The Selector dynamically chooses roles for each question.
{chr(10).join(role_lines)}

DYNAMIC ROLE-POOL ALLOCATION PRIOR
Assign models according to role responsibility, task difficulty, effective responsibility observed from dynamic role usage, reliability requirements, and model capability/cost.

AVAILABLE MODEL POOL
{self._model_pool_text()}

CURRENT AGENT PROMPT EVOLUTION
Agent id: {agent_id}
Initial prompt at the start of this round:
{_clip(initial_prompt, 5000)}
Best prompt after depth optimization:
{_clip(best_prompt, 5000)}
Initial model: {initial_model}
Best model after depth optimization: {best_model}

RETRIEVED PROMPT-SIMILAR TRAJECTORIES
{self._memory_text(memories)}

TASK
1. Produce a final prompt for Agent {agent_id}. Use the round-start prompt as the stable foundation and incorporate only valuable improvements from the depth-optimized best prompt. Preserve role, input/output placeholders, and required answer tags.
   Prompt length rules:
   - The round-start prompt is the length baseline.
   - The final prompt must not exceed twice the character length of the round-start prompt.
   - Keep the original structure, role, placeholders, and output format whenever possible.
   - Do not expand the prompt merely to appear more optimized.
   - If the candidate is too long, compress it from the round-start prompt and retain only a few necessary improvements.
   - The system will check the output length after generation. If it still exceeds twice the round-start prompt length, the final prompt will be reverted to the round-start prompt.
   - The round-start prompt is already usable; do not rewrite or expand it without a clear benefit.
   - Preserve all delimiters, JSON brackets, tags, function-call syntax, code fences, and dataset-required output structures.
2. Produce a complete model_assignment for all five optimizable logical LLM roles using only the available model pool. Do not include the Selector.
   The model assignment should primarily consider:
   - each role's responsibility;
   - dynamic route frequency and effective critical-path responsibility;
   - task difficulty;
   - critical-path responsibility;
   - model capability;
   - input and output cost coefficients.
   For every candidate model, explicitly consider both input_cost_coefficient and output_cost_coefficient shown in the model pool. A high-capability model is justified only when the role, task difficulty, or observed dynamic responsibility warrants its cost.
   Rules:
   - For complex code generation, complex reasoning, or other high-difficulty tasks, prefer a stronger model within the regular capability range for roles on the effective critical path or responsible for core reasoning, result verification, or final integration.
   - For relatively simple tasks such as tool calling, format conversion, or information extraction, prefer a weaker model within the regular capability range for routine roles that are not on the effective critical path, in order to reduce cost. Assign critical roles according to their responsibility and risk of propagating errors.
   - Model assignment must maintain balanced overall capability. When multiple capability levels are available, use a layered allocation that matches model strength to role importance and task difficulty. If a simple or non-critical role is assigned a weaker model within the non-extreme capability range, at least one important role must be assigned a stronger model within the non-extreme capability range to maintain capability balance.
   - Do not assign the same model to every logical LLM role unless the role responsibilities and task difficulty clearly justify it.
   - Do not collapse into all-strong, all-weak, or all-middle allocation without a role-based justification.
   - Follow all parameter-sharing and mechanical-role constraints specified by the dynamic role pool.
   - Use historical trajectories only as auxiliary evidence for final Prompt evolution.
   - Do not use historical candidate Prompts, MAS outputs, or trajectory scores to determine model_assignment.
   - The lowest-capability and highest-capability models in the pool must be chosen very cautiously and must not be used as regular allocation choices.
   - Select either extreme capability tier only in a confirmed extreme case; when regular capability models are sufficient, prefer a regular capability model.
   - If an extreme tier is selected, briefly state the concrete necessity in decision_summary.
   - Determine model_assignment primarily from the current role responsibilities, task difficulty, effective critical-path responsibility, and model-pool capabilities and costs. Do not use the current initial model or depth-optimized best model as the basis for model assignment.
3. Return only JSON.

Return exactly:
{{
  "final_prompt": "one prompt string",
  "model_assignment": {assignment_schema},
  "decision_summary": "short factual summary"
}}
"""

    def _sanitize_prompt(self, prompt: Any, initial: str, best: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            return best or initial
        # Length-based rollback is intentionally disabled. Keep placeholder
        # validation so a malformed Prompt cannot break runtime formatting.
        value = prompt.strip()
        baseline = initial or ""
        required = set(re.findall(r"\{[^}]+\}", baseline))
        found = set(re.findall(r"\{[^}]+\}", value))
        return value if required.issubset(found) else (baseline or best)

    async def _write_trace(self, trace: Dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        async with self._trace_lock:
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    async def evolve(self, *, agent_id: int, initial_prompt: str, best_prompt: str,
                     initial_model: str, best_model: str, global_prompts: Dict[int, str],
                     global_models: Dict[int, str], profile: Dict[str, Any],
                     memories: List[Dict[str, Any]], round_number: Optional[int] = None) -> Dict[str, Any]:
        prompt = self.build_prompt(
            agent_id=agent_id, initial_prompt=initial_prompt, best_prompt=best_prompt,
            initial_model=initial_model, best_model=best_model, global_prompts=global_prompts,
            global_models=global_models, profile=profile, memories=memories)
        trace_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:10]}"
        started_at = datetime.now(timezone.utc).isoformat()
        trace_base = {
            "trace_id": trace_id,
            "started_at": started_at,
            "round": round_number,
            "agent_id": agent_id,
            "dataset": profile.get("dataset"),
            "graph_type": profile.get("graph_type"),
            "task_type": profile.get("task_type"),
            "optimizer_model": self.optimizer_model,
            "model_pool": self.model_pool,
            "input": {
                "initial_prompt": initial_prompt,
                "best_prompt": best_prompt,
                "initial_model": initial_model,
                "best_model": best_model,
                "topology_profile": profile,
                "retrieved_memories": memories,
                "generated_prompt": prompt,
            },
        }
        try:
            raw = await async_call_llm(
                self.client, prompt, temperature=0.0, max_tokens=4096,
                timeout_s=120, timeout_label=f"co_evolver_agent={agent_id}"
            )
            parsed = _json_from_response(raw)
            final_prompt = self._sanitize_prompt(
                parsed.get("final_prompt"), initial_prompt, best_prompt
            )
            assignment = self._sanitize_assignment(
                parsed.get("model_assignment"), global_models, profile
            )
            result = {
                "agent_id": agent_id,
                "final_prompt": final_prompt,
                "model_assignment": assignment,
                "decision_summary": _clip(parsed.get("decision_summary", ""), 600),
                "raw_response": _clip(raw, 5000),
            }
            trace = dict(trace_base)
            trace.update({
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "success",
                "output": {
                    "raw_response": raw,
                    "parsed_response": parsed,
                    "sanitized_result": result,
                },
            })
            await self._write_trace(trace)
            return result
        except Exception as exc:
            trace = dict(trace_base)
            trace.update({
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            })
            await self._write_trace(trace)
            raise

    @staticmethod
    def merge_assignments(
        proposals: List[Dict[str, Any]],
        current: Dict[int, str],
        profile: Optional[Dict[str, Any]] = None,
        model_pool: Optional[Iterable[str]] = None,
        voter: str = "majority",
    ) -> Dict[int, str]:
        current_norm = {}
        for key, value in (current or {}).items():
            try:
                current_norm[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        result = dict(current_norm)
        active_pool = list(model_pool or AVAILABLE_MODELS)
        if voter == "trace_entropy":
            if profile:
                eligible_ids = [
                    int(agent["agent_id"])
                    for agent in profile.get("agents", [])
                    if agent.get("agent_type") != "aggregator"
                ]
            else:
                eligible_ids = sorted(current_norm)

            # Council has three logical variables whose values are shared
            # by runtime pairs. Expand them only after voting.
            if profile and profile.get("graph_type") == GraphType.COUNCIL.value:
                eligible_ids = [0, 1, 2]

            selected = _trace_entropy_consensus(
                proposals=proposals,
                current=current_norm,
                model_pool=active_pool,
                agent_ids=eligible_ids,
            )
            result.update(selected)
            if profile and profile.get("graph_type") == GraphType.COUNCIL.value:
                for leader, member in ((0, 3), (1, 4), (2, 5)):
                    if leader in result:
                        result[member] = result[leader]
            return result

        # Original majority-vote path retained unchanged for the default
        # voter="majority" behavior.
        frozen = set()
        if profile:
            frozen = {
                int(a["agent_id"])
                for a in profile.get("agents", [])
                if a.get("agent_type") == "aggregator"
            }
        for aid in sorted(current_norm):
            if aid in frozen:
                continue
            votes = []
            for proposal in proposals:
                assignment = proposal.get("model_assignment", {})
                if not isinstance(assignment, dict):
                    continue
                value = assignment.get(aid, assignment.get(str(aid)))
                if value:
                    votes.append(value)
            if votes:
                result[aid] = choose_weighted_model(votes, active_pool)
        if profile and profile.get("graph_type") == GraphType.COUNCIL.value:
            for leader, member in ((0, 3), (1, 4), (2, 5)):
                votes = []
                for proposal in proposals:
                    assignment = proposal.get("model_assignment", {})
                    choice = assignment.get(leader, assignment.get(str(leader), assignment.get(member, assignment.get(str(member)))))
                    if choice:
                        votes.append(choice)
                if votes:
                    choice = choose_weighted_model(votes, active_pool)
                    if leader in result:
                        result[leader] = choice
                    if member in result:
                        result[member] = choice
        for aid in frozen:
            if aid in current_norm:
                result[aid] = current_norm[aid]
        return result
