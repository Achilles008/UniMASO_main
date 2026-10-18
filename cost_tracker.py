"""Cost accounting for UniMASO optimization and evaluation runs.

Embedding calls are intentionally excluded from all cost accounting.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from config import get_model_cost


class CostTracker:
    """Accumulate weighted token costs by optimization stage.

    Stages used by this project:
      optimizer, evaluator, co_evolver, sample_pool, evaluation_inference
    """

    def __init__(self) -> None:
        self._stage = defaultdict(lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
        })
        self._calls = []
        self._seen_cache_ids = set()

    def record_llm(
        self,
        stage: str,
        model_name: str,
        usage: Optional[Dict[str, Any]],
        *,
        round_number: Optional[int] = None,
        agent_id: Optional[int] = None,
        substage: Optional[str] = None,
    ) -> float:
        usage = usage or {}
        input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        coeff = get_model_cost(model_name)
        cost = input_tokens * coeff["input"] + output_tokens * coeff["output"]

        entry = self._stage[stage]
        entry["calls"] += 1
        entry["input_tokens"] += input_tokens
        entry["output_tokens"] += output_tokens
        entry["cost"] += cost
        self._calls.append({
            "stage": stage,
            "substage": substage,
            "round": round_number,
            "agent_id": agent_id,
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_coefficient": coeff["input"],
            "output_coefficient": coeff["output"],
            "weighted_cost": cost,
        })
        return cost

    def record_mas_cache(
        self,
        cache: Any,
        *,
        model_map: Optional[Dict[int, str]] = None,
        stage: str = "sample_pool",
        metadata: Optional[Dict[str, Any]] = None,
        agent_ids: Optional[Any] = None,
    ) -> float:
        """Record actual MAS generation represented by one cache.

        A cache object is counted once. This prevents reused baseline caches
        from being charged repeatedly across candidate comparisons.
        """
        cache_key = id(cache)
        if cache_key in self._seen_cache_ids:
            return 0.0
        self._seen_cache_ids.add(cache_key)

        model_map = model_map or {}
        input_tokens = getattr(cache, "node_input_tokens", {}) or {}
        output_tokens = getattr(cache, "node_output_tokens", {}) or {}
        node_models = getattr(cache, "node_model", {}) or {}
        if agent_ids is None:
            selected_ids = set(input_tokens) | set(output_tokens)
        else:
            selected_ids = set(agent_ids)
        total = 0.0
        for aid in selected_ids:
            model_name = model_map.get(aid) or node_models.get(aid) or "qwen35-4b"
            total += self.record_llm(
                stage,
                model_name,
                {
                    "prompt_tokens": input_tokens.get(aid, 0),
                    "completion_tokens": output_tokens.get(aid, 0),
                },
                agent_id=int(aid),
                substage="mas_agent",
            )
        if metadata:
            for call in reversed(self._calls):
                if call["stage"] == stage and call.get("substage") == "mas_agent":
                    call["cache_metadata"] = dict(metadata)
                    break
        return total

    def record_external_cost(
        self,
        stage: str,
        cost: float,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        substage: str = "external",
    ) -> None:
        entry = self._stage[stage]
        entry["calls"] += 1
        entry["input_tokens"] += int(input_tokens)
        entry["output_tokens"] += int(output_tokens)
        entry["cost"] += float(cost)
        self._calls.append({
            "stage": stage,
            "substage": substage,
            "model": None,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "weighted_cost": float(cost),
        })

    def stage_summary(self, stage: str) -> Dict[str, Any]:
        return dict(self._stage.get(stage, {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
        }))

    def summary(self) -> Dict[str, Any]:
        optimization_stages = ("optimizer", "evaluator", "co_evolver", "sample_pool")
        optimization = {
            stage: self.stage_summary(stage) for stage in optimization_stages
        }
        optimization_total = sum(item["cost"] for item in optimization.values())
        evaluation = self.stage_summary("evaluation_inference")
        return {
            "optimization_cost": {
                "total": optimization_total,
                **optimization,
            },
            "evaluation_inference_cost": {
                "total": evaluation["cost"],
                **evaluation,
            },
            "summary": {
                "total_optimization_cost": optimization_total,
                "total_evaluation_inference_cost": evaluation["cost"],
                "total_experiment_cost": optimization_total + evaluation["cost"],
            },
        }

    def calls(self):
        return list(self._calls)

    def save(self, summary_path: str | Path, calls_path: str | Path | None = None) -> None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if calls_path is not None:
            calls_path = Path(calls_path)
            calls_path.parent.mkdir(parents=True, exist_ok=True)
            with calls_path.open("w", encoding="utf-8") as f:
                for item in self._calls:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
