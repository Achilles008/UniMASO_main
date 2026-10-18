"""Low-coupling LLM-as-a-judge evaluator for MARBLE-style CS.

This module is opt-in.  It only consumes a completed MAS trace and does not
change agent execution or the existing task/cost judges.
"""

import json
import re
from typing import Any, Dict, Mapping, Optional

from config import get_model, get_model_cost
from utils import async_call_llm


COMMUNICATION_PROMPT = r"""
[Context]

**Task / Problem:**
{task}

**Agent Interaction Records:**
{communications}

**Final MAS Output:**
{final_output}

[System]
Evaluate the communication quality between agents in the Graph structure. Focus on:

- **Information Exchange:** Was relevant information effectively transmitted?
- **Clarity:** Were intentions and messages clear?
- **Task Assistance:** Did communication help task completion?
- **Efficiency:** Was communication concise and purposeful?

Rate on a 5-point scale:
1. **1 point**: Poor communication with major failures.
2. **2 points**: Significant issues in clarity or relevance.
3. **3 points**: Adequate but required clarification.
4. **4 points**: Effective with minor improvements needed.
5. **5 points**: Clear, effective communication that maximized efficiency.

Apply a strict, conservative, and evidence-based scoring standard. Judge only the Agent's actual inputs, outputs, and interaction records. Do not infer communication quality from role descriptions, task instructions, or the correctness of the final answer alone.

5 points: Exceptional. All relevant information is transmitted accurately, completely, and in a timely manner; messages are clear, unambiguous, and consistent; communication directly supports task completion; and there is no unnecessary repetition, irrelevant content, or avoidable verbosity. Award 5 only when every criterion is fully and consistently satisfied with no observable weakness.

4 points: Good but not perfect. Information exchange is accurate and sufficiently complete, messages are clear, and communication supports task completion. Only very minor, non-impactful imperfections are allowed, such as slight verbosity or a small lack of concision. If there is any information omission, noticeable redundancy, need for clarification, unclear contribution, or substantive weakness in any criterion, do not award 4; the score must be 3 or lower.

3 points: Adequate but with clear shortcomings. Communication provides some useful information and enables task progress, but includes incomplete information, unclear expressions, required clarification, repeated messages, limited assistance, or insufficient evidence.

2 points: Poor. Significant omissions, confusion, ineffective repetition, irrelevant content, or delayed communication substantially reduce coordination efficiency, although a small amount of useful communication is present.

1 point: Very poor. There is almost no useful information exchange, or the communication is seriously irrelevant, confused, contradictory, or unable to support task progress.

If the evidence is insufficient to justify a higher score, choose the lower score. A correct final answer alone does not demonstrate excellent communication quality.
[Question]
You MUST respond with ONLY a JSON object in this EXACT format:
{{"rating": X}}
where X is a number between 1 and 5.

Example valid responses:
{{"rating": 4}}
{{"rating": 2}}

DO NOT include any other text, explanation, or formatting.
""".strip()


PLANNING_PROMPT = r"""
[Context]

**Task / Problem:**
{summary}

**Agent Interaction Records:**
{results}

**Final MAS Output:**
{final_output}

[System]
Evaluate the effectiveness of agent self-coordination in the Graph structure. Focus on:

- **Role Clarity:** Did agents understand their roles and responsibilities?
- **Task Alignment:** Were tasks aligned with goals?
- **Autonomy:** Did agents work independently without central oversight?

Rate on a 5-point scale:
1. **1 point**: Very poor self-coordination, with major inefficiencies.
2. **2 points**: Frequent role confusion, causing inefficiencies.
3. **3 points**: Moderate overlap or confusion in roles.
4. **4 points**: Effective with minor role clarification needed.
5. **5 points**: Clear roles, effective self-coordination.

Apply a strict, conservative, and evidence-based scoring standard. Judge only the Agent's actual task organization, role assignment, task alignment, handoffs, autonomy, and strategy adaptation. Do not infer planning quality from role descriptions, predefined task instructions, or the correctness of the final answer alone.

5 points: Exceptional. Agents clearly understand and maintain non-overlapping responsibilities; tasks are allocated rationally and fully aligned with the overall objective; agents autonomously make consistent progress, integrate intermediate results, and adapt or verify their strategies when necessary. Award 5 only when every criterion is fully and consistently satisfied with no observable weakness.

4 points: Good but not perfect. Roles and task allocation are clear and effective, agents remain aligned with the overall objective, and coordination supports task completion. Only very minor, non-impactful imperfections are allowed, such as slightly unclear boundaries or limited evidence of adaptation. If there is noticeable role overlap, task mismatch, missing planning evidence, excessive dependence on central control, lack of adaptation, or any substantive weakness in any criterion, do not award 4; the score must be 3 or lower.

3 points: Adequate but with clear shortcomings. Agents show some task organization and role coordination, but there is role overlap, incomplete task handoff, insufficient planning evidence, limited autonomy, or a need for additional coordination.

2 points: Poor. Significant role confusion, unreasonable task allocation, misalignment with the overall objective, duplicated work, or strong dependence on central control substantially reduces coordination efficiency, although some useful planning behavior is present.

1 point: Very poor. There is almost no identifiable task planning or role coordination; agent behaviors conflict, responsibilities are seriously mismatched, or the task cannot be effectively advanced.

If the evidence is insufficient to justify a higher score, choose the lower score. A correct final answer alone does not demonstrate excellent planning quality.
[Question]
You MUST respond with ONLY a JSON object in this EXACT format:
{{"rating": X}}
where X is a number between 1 and 5.

Example valid responses:
{{"rating": 4}}
{{"rating": 2}}

DO NOT include any other text, explanation, or formatting.
""".strip()


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def build_cs_prompt_inputs(
    problem: str,
    per_agent_tokens: Mapping[Any, Mapping[str, Any]],
    final_output: str,
    agent_roles: Optional[Mapping[Any, Mapping[str, str]]] = None,
    coordination_mode: str = "Centralized",
) -> Dict[str, str]:
    """Build the four input blocks used by the two CS prompts."""
    agent_roles = agent_roles or {}
    records = []

    def sort_key(item):
        try:
            return int(item[0])
        except (TypeError, ValueError):
            return str(item[0])

    for raw_id, data in sorted(per_agent_tokens.items(), key=sort_key):
        agent_id = str(raw_id)
        # `input` is the complete prompt sent to the current Agent. It already
        # contains the Agent's role/instructions, so do not add a separate
        # synthetic Role field or model name.
        # CS uses the context-only intermediate input when available.
        # The fallback preserves compatibility with old result records.
        agent_input = _text(data.get("context_only_input", data.get("input", "")))
        agent_output = _text(data.get("output", ""))
        records.append(
            f"Agent {agent_id}:\nInput:\n{agent_input}\nOutput:\n{agent_output}"
        )

    return {
        "task": _text(problem),
        "summary": _text(problem),
        "coordination_mode": _text(coordination_mode),
        "communications": "\n\n".join(records),
        "results": "\n\n".join(records),
        "final_output": _text(final_output),
    }


def _parse_rating(raw: str) -> int:
    text = _text(raw).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
        rating = int(payload["rating"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CS judge JSON: {raw!r}") from exc
    if rating < 1 or rating > 5:
        raise ValueError(f"CS rating must be between 1 and 5, got {rating}")
    return rating


class CSEvaluator:
    """Evaluate one completed MAS sample with two DeepSeek Flash calls."""

    def __init__(self, model_name: str = "deepseek-v4-flash"):
        self.model_name = model_name
        self.client = get_model(model_name)

    async def _judge(self, prompt: str) -> Dict[str, Any]:
        response = await async_call_llm(
            self.client,
            prompt,
            temperature=0.0,
            max_tokens=128,
            return_usage=True,
            timeout_label=f"cs_judge:{self.model_name}",
        )
        raw = response.get("text", "") if isinstance(response, dict) else response
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        return {"rating": _parse_rating(raw), "raw": raw, "usage": usage}

    async def evaluate(
        self,
        problem: str,
        per_agent_tokens: Mapping[Any, Mapping[str, Any]],
        final_output: str,
        agent_roles: Optional[Mapping[Any, Mapping[str, str]]] = None,
        coordination_mode: str = "Centralized",
    ) -> Dict[str, Any]:
        values = build_cs_prompt_inputs(
            problem=problem,
            per_agent_tokens=per_agent_tokens,
            final_output=final_output,
            agent_roles=agent_roles,
            coordination_mode=coordination_mode,
        )
        communication_prompt = COMMUNICATION_PROMPT.format(**values)
        planning_prompt = PLANNING_PROMPT.format(**values)
        communication = await self._judge(communication_prompt)
        planning = await self._judge(planning_prompt)
        communication_score = communication["rating"]
        planning_score = planning["rating"]
        usage = [communication.get("usage", {}), planning.get("usage", {})]
        cost_cfg = get_model_cost(self.model_name)
        judge_cost = sum(
            int(u.get("prompt_tokens", 0)) * cost_cfg["input"]
            + int(u.get("completion_tokens", 0)) * cost_cfg["output"]
            for u in usage
        )
        return {
            "communication_score": communication_score,
            "planning_score": planning_score,
            "coordination_score": (communication_score + planning_score) / 2.0,
            "judge_model": self.model_name,
            "judge_temperature": 0.0,
            "judge_cost": judge_cost,
            "judge_usage": usage,
            "judge_raw": {
                "communication": communication["raw"],
                "planning": planning["raw"],
            },
            "prompts": {
                "communication": communication_prompt,
                "planning": planning_prompt,
            },
        }
