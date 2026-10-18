"""Dynamic-only optimizer for selector topology.

Unlike static topologies, every candidate is evaluated by replaying the full
selector trajectory. The selector model and dataset-specific selector prompt
remain fixed; only the five role prompt/model pairs are optimized.
"""

import asyncio
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config import DEFAULT_AGENT_MODEL, GraphType, compute_mas_cost
from agent import MAS, InferenceCache
from optimizers import MAPromptOptimizer
from dynamic_co_evolver import DynamicCoEvolver
from logger import get_logger


class DynamicMAPromptOptimizer(MAPromptOptimizer):
    """Dynamic optimizer with route-aware local/next/global scoring."""

    # Tuple order follows the base optimizer API: local, next, global.
    DYNAMIC_WEIGHTS = (0.2, 0.2, 0.6)

    def _is_dynamic(self) -> bool:
        return self.mas.gtype is GraphType.SELECTOR_DYNAMIC

    def _dynamic_optimization_order(self) -> List[int]:
        return list(range(len(self.mas.agents)))

    def _dynamic_profile(self) -> Dict[str, Any]:
        path = Path(__file__).resolve().parent / "config" / "dynamic_topology_roles.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = data.get(self.dataset_name, {})
        roles = entry.get("roles", [])
        agents = []
        for role in roles:
            aid = int(role["agent_id"])
            agents.append({
                "agent_id": aid,
                "agent_type": "predictor",
                "role": role.get("name", ""),
                "description": role.get("description", ""),
                "is_terminal": aid == 4,
                "is_optimizable": True,
                "initial_prompt": self.best_prompt.get(aid, ""),
            })
        profile = {
            "dataset": self.dataset_name,
            "task_type": self.mas.task_type.value,
            "graph_type": GraphType.SELECTOR_DYNAMIC.value,
            "agent_count": len(agents),
            "agents": agents,
            "selector_model": self.mas.selector_model,
        }
        return profile

    async def _build_baseline_caches(
        self, questions: List[str], prompt_map: Dict[int, str]
    ) -> Dict[str, InferenceCache]:
        """Build baselines with the correct dataset-specific selector prompt."""
        mas_temp = MAS(
            GraphType.SELECTOR_DYNAMIC,
            self.mas.task_type,
            Ns=self.mas.Ns, Na=self.mas.Na, Nr=self.mas.Nr, Nd=self.mas.Nd,
            agent_model=self.mas.agent_model,
            agent_models=self.best_model.copy(),
            selector_model=self.mas.selector_model,
            dataset_name=self.mas.dataset_name,
        )
        mas_temp.inject_prompt_map(prompt_map)
        limit = max(1, int(os.getenv("MAS_BASELINE_CONCURRENCY", "3")))
        sem = asyncio.Semaphore(limit)

        async def run_one(question: str):
            async with sem:
                _, cache = await mas_temp.arun_with_cache(question)
                if self.cost_tracker is not None:
                    self.cost_tracker.record_mas_cache(
                        cache,
                        model_map=self.best_model,
                        metadata={"kind": "dynamic_baseline", "question": question},
                    )
                return question, cache

        return dict(await asyncio.gather(*(run_one(q) for q in questions)))

    @staticmethod
    def _role_steps(cache: InferenceCache, role_id: int) -> List[Dict[str, Any]]:
        return [s for s in cache.dynamic_steps
                if int(s.get("role_id", -1)) == role_id]

    @staticmethod
    def _final_raw(cache: InferenceCache, terminal_id: int) -> str:
        if cache.dynamic_steps:
            return str(cache.dynamic_steps[-1].get("output", ""))
        return cache.node_outputs_raw.get(terminal_id, "")

    async def _evaluate_candidate(
        self, cand_prompt: str, agent_id: int, eval_samples: List[str],
        baseline_caches: Dict[str, InferenceCache],
        temp_prompt_map: Dict[int, str], predecessors: List[int],
        terminal_id: int, is_terminal: bool, requirement: str,
        pure_local_mode: bool = False, use_lookahead_score: bool = False,
        lookahead_weights: Tuple[float, float, float] = DYNAMIC_WEIGHTS,
        random_sample_set: Optional[Set[str]] = None,
        cand_model: str = None, temp_model_map: Dict[int, str] = None,
        base_model_map: Dict[int, str] = None,
    ) -> Dict[str, Any]:
        # These parameters are part of the parent contract; dynamic scoring
        # intentionally uses its own route-aware semantics.
        del predecessors, is_terminal, pure_local_mode, use_lookahead_score
        del lookahead_weights, random_sample_set
        candidate_prompt_map = self.best_prompt.copy()
        candidate_prompt_map.update(temp_prompt_map or {})
        candidate_prompt_map[agent_id] = cand_prompt
        candidate_model_map = (base_model_map or self.best_model).copy()
        candidate_model_map.update(temp_model_map or {})
        candidate_model = cand_model or candidate_model_map.get(agent_id, DEFAULT_AGENT_MODEL)
        candidate_model_map[agent_id] = candidate_model

        async def run_candidate(question: str):
            mas_eval = MAS(
                GraphType.SELECTOR_DYNAMIC,
                self.mas.task_type,
                Ns=self.mas.Ns, Na=self.mas.Na, Nr=self.mas.Nr, Nd=self.mas.Nd,
                agent_model=self.mas.agent_model,
                agent_models=candidate_model_map,
                selector_model=self.mas.selector_model,
                dataset_name=self.mas.dataset_name,
            )
            mas_eval.inject_prompt_map(candidate_prompt_map)
            result, cache = await mas_eval.arun_with_cache(question)
            if self.cost_tracker is not None:
                self.cost_tracker.record_mas_cache(
                    cache,
                    model_map=candidate_model_map,
                    metadata={"kind": "dynamic_candidate", "question": question, "agent_id": agent_id},
                )
            return result, cache

        async def compare_question(question: str):
            base_cache = baseline_caches[question]
            candidate_result, candidate_cache = await run_candidate(question)
            base_steps = base_cache.dynamic_steps
            cand_steps = candidate_cache.dynamic_steps
            base_role_steps = self._role_steps(base_cache, agent_id)
            cand_role_steps = self._role_steps(candidate_cache, agent_id)
            comparisons = []

            # End-to-end quality is always evaluated.
            base_final = self._final_raw(base_cache, terminal_id)
            cand_final = candidate_result.get(
                "final_raw", self._final_raw(candidate_cache, terminal_id)
            )
            comparisons.append(("global", await self._compare_final_answer(
                question, requirement, cand_final, base_final
            )))

            # Compare matching occurrences. Missing role occurrences are not
            # automatic losses because the selector may legitimately route
            # around a role for this sample.
            for base_step, cand_step in zip(base_role_steps, cand_role_steps):
                local_win = await self._compare_intermediate(
                    question, str(cand_step.get("output", "")),
                    str(base_step.get("output", ""))
                )
                comparisons.append(("local", local_win))
                base_index = base_steps.index(base_step)
                cand_index = cand_steps.index(cand_step)
                if base_index + 1 < len(base_steps) and cand_index + 1 < len(cand_steps):
                    next_win = await self._compare_intermediate(
                        question,
                        str(cand_steps[cand_index + 1].get("output", "")),
                        str(base_steps[base_index + 1].get("output", ""))
                    )
                    comparisons.append(("next", next_win))

            return question, comparisons, base_cache, candidate_cache, {
                "question": question,
                "role_id": agent_id,
                "baseline_path": [s.get("role_id") for s in base_steps],
                "candidate_path": [s.get("role_id") for s in cand_steps],
            }

        limit = max(1, int(os.getenv("MAS_CANDIDATE_CONCURRENCY", "10")))
        sem = asyncio.Semaphore(limit)

        async def limited(question: str):
            async with sem:
                return await compare_question(question)

        rows = await asyncio.gather(*(limited(q) for q in eval_samples))
        wins = {"local": 0, "next": 0, "global": 0}
        counts = {"local": 0, "next": 0, "global": 0}
        traces = []
        total_base_cost = 0.0
        total_candidate_cost = 0.0
        for _, comparisons, base_cache, candidate_cache, trace in rows:
            traces.append(trace)
            total_base_cost += compute_mas_cost(
                base_cache, model_map=base_model_map or self.best_model
            )
            total_candidate_cost += compute_mas_cost(
                candidate_cache, model_map=candidate_model_map
            )
            for kind, result in comparisons:
                counts[kind] += 1
                wins[kind] += int(bool(result))

        rates = {kind: wins[kind] / counts[kind] if counts[kind] else 0.0
                 for kind in wins}
        w_local, w_next, w_global = self.DYNAMIC_WEIGHTS
        terms = [(w_local, rates["local"], counts["local"]),
                 (w_next, rates["next"], counts["next"]),
                 (w_global, rates["global"], counts["global"])]
        weight_sum = sum(w for w, _, count in terms if count > 0)
        quality = sum(w * rate for w, rate, count in terms if count > 0) / max(weight_sum, 1e-9)
        norm_cost = total_candidate_cost / max(total_base_cost, 1.0)
        discount = norm_cost ** (-self.lambda_cost) if self.lambda_cost else 1.0
        adjusted_score = quality * discount - 0.5
        # Keep the exact candidate-id contract used by the parent loop.
        candidate_id = hashlib.sha1(
            f"{candidate_model}\n{cand_prompt}".encode("utf-8")
        ).hexdigest()[:12]
        score_info = {
            "candidate_id": candidate_id,
            "candidate_model": candidate_model,
            "candidate_prompt_len": len(cand_prompt),
            "score": adjusted_score, "quality": quality,
            "cost": norm_cost, "norm_cost": norm_cost,
            "score_cost_norm": norm_cost, "agent_norm_cost": norm_cost,
            "model_price_norm": 1.0,
            "wins_local": wins["local"], "wins_next_local": wins["next"],
            "wins_global": wins["global"],
            "count_local": counts["local"], "count_next_local": counts["next"],
            "count_global": counts["global"],
            "local_rate": rates["local"], "next_rate": rates["next"],
            "global_rate": rates["global"], "total_evaluable": len(eval_samples),
            "bad_cases": [], "misleading_cases": [],
            "misalignment_rate": 0.0, "misalignment_count": 0,
            "total_base_cost": total_base_cost,
            "total_cand_cost": total_candidate_cost,
            "quality_cost_misalign_rate": 0.0,
            "dynamic_traces": traces,
            "dynamic_score_weights": {"local": w_local, "next": w_next, "global": w_global},
        }
        await self._record_dynamic_memory(
            agent_id, cand_prompt, candidate_model, eval_samples,
            baseline_caches, score_info, requirement,
        )
        return score_info

    async def _record_dynamic_memory(
        self, agent_id: int, cand_prompt: str, candidate_model: str,
        eval_samples: List[str], baseline_caches: Dict[str, InferenceCache],
        score_info: Dict[str, Any], requirement: str,
    ) -> None:
        if self.trajectory_memory is None:
            return
        retrieval_view = self._build_memory_state(
            aid=agent_id,
            node_prompt=cand_prompt,
            node_model=candidate_model,
            preds=[],
            samples=eval_samples,
            baseline_caches=baseline_caches,
        )
        await self.trajectory_memory.add({
            "run_id": str(self.trajectory_memory.run_dir),
            "step": len(self.trajectory_memory.records),
            "record_type": "dynamic_candidate_evaluation",
            "state": retrieval_view,
            "retrieval_view": retrieval_view,
            "action": {
                "agent_id": agent_id,
                "candidate_prompt": cand_prompt,
                "candidate_model": candidate_model,
            },
            "score": {
                "score": score_info["score"],
                "quality": score_info["quality"],
                "local_rate": score_info["local_rate"],
                "global_rate": score_info["global_rate"],
                "next_rate": score_info["next_rate"],
                "local_delta": score_info["local_rate"] - 0.5,
                "global_delta": score_info["global_rate"] - 0.5,
                "next_delta": score_info["next_rate"] - 0.5,
                "cost_ratio": score_info["norm_cost"],
                "dynamic_weights": score_info["dynamic_score_weights"],
            },
            "status": {"accepted": score_info["score"] > 0},
            "payload": {
                "before_prompt": self.best_prompt.get(agent_id, ""),
                "before_model": self.best_model.get(agent_id, DEFAULT_AGENT_MODEL),
                "candidate_prompt": cand_prompt,
                "candidate_model": candidate_model,
                "requirement": requirement,
                "samples": score_info.get("dynamic_traces", []),
            },
        })

    async def _apply_co_evolver(
        self, round_initial_prompts: Dict[int, str],
        round_initial_models: Dict[int, str], round_number: int,
        depth_best_models: Optional[Dict[int, str]] = None,
        agent_states: Optional[Dict[int, Any]] = None,
    ) -> Dict[str, Any]:
        """Run Co-Evolver with role-pool context, never static topology text."""
        if self.co_evolver is None:
            return {"enabled": False, "round": round_number, "proposals": []}
        profile = self._dynamic_profile()
        agent_ids = list(range(len(self.mas.agents)))
        global_prompts = self.best_prompt.copy()
        global_models = self.best_model.copy()
        proposals = []
        log = get_logger()
        for aid in agent_ids:
            memories = []
            if self.trajectory_memory is not None:
                memories = await self.trajectory_memory.retrieve(
                    {"current_prompt": global_prompts.get(aid, "")},
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
                if result.get("final_prompt"):
                    self._set_best_prompt(aid, result["final_prompt"])
                log.info(
                    "Dynamic Co-Evolver round=%s agent=%s: memory=%s prompt_len=%s",
                    round_number, aid, len(memories), len(result.get("final_prompt", "")),
                )
            except Exception as exc:
                log.warning("Dynamic Co-Evolver agent=%s failed: %s", aid, exc)

        merged = DynamicCoEvolver.merge_assignments(
            proposals, global_models, profile=profile,
            model_pool=self.model_pool, voter=self.voter,
        )
        self.final_coevolver_prompt = {
            aid: self.best_prompt.get(aid, global_prompts.get(aid, ""))
            for aid in agent_ids
        }
        self.final_coevolver_model = {
            aid: merged.get(aid, global_models.get(aid, DEFAULT_AGENT_MODEL))
            for aid in agent_ids
            if merged.get(aid, global_models.get(aid)) in self.model_pool
        }
        if agent_states is None and self.heterogeneous:
            for aid, model in self.final_coevolver_model.items():
                self._set_best_model(aid, model)
        elif agent_states is not None:
            self.best_model = (depth_best_models or self.best_model).copy()
            self._normalize_shared_models()
            for aid, state in agent_states.items():
                prompt = self.best_prompt.get(aid, "")
                model = self.best_model.get(aid, DEFAULT_AGENT_MODEL)
                state.current_beam = [{"prompt": prompt, "model": model,
                                       "cumulative_score": 0.0, "quality": 0.0,
                                       "cost": 0.0, "path": [prompt]}]
                state.best_overall_node = state.current_beam[0]
        return {"enabled": True, "round": round_number, "proposals": proposals}

    async def _refresh_beam_scores(
        self, state, requirement: str, predecessors: List[int],
        terminal_id: int, is_terminal: bool,
    ) -> None:
        """Re-rank beam nodes by complete dynamic rollouts."""
        if not state.current_beam:
            return
        ranking_before = [
            node.get("prompt", "") for node in sorted(
                state.current_beam,
                key=lambda x: x.get("cumulative_score", 0.0),
                reverse=True,
            )
        ]
        eval_samples = self.get_eval_samples(state.total_layers_explored)
        baseline = await self._build_baseline_caches(eval_samples, self.best_prompt)
        refreshed = []
        for node in state.current_beam:
            prompt = node.get("prompt", self.best_prompt[state.agent_id])
            model = node.get("model", self.best_model.get(state.agent_id, DEFAULT_AGENT_MODEL))
            info = await self._evaluate_candidate(
                prompt, state.agent_id, eval_samples, baseline,
                self.best_prompt.copy(), predecessors, terminal_id, is_terminal,
                requirement, use_lookahead_score=True, cand_model=model,
                base_model_map=self.best_model.copy(),
            )
            item = dict(node)
            item.update({"cumulative_score": info["score"],
                        "quality": info["quality"], "cost": info["norm_cost"],
                        "last_score_info": info})
            refreshed.append(item)
        state.current_beam = sorted(
            refreshed, key=lambda x: x["cumulative_score"], reverse=True
        )
        ranking_after = [node.get("prompt", "") for node in state.current_beam]
        kendall = self._compute_kendall_top2_overlap(ranking_before, ranking_after)
        state.beam_refresh_kendall_scores.append({
            "depth": state.total_layers_explored,
            "kendall_top2_overlap": kendall,
            "ranking_before": ranking_before[:3],
            "ranking_after": ranking_after[:3],
            "dynamic": True,
        })
        if state.current_beam:
            state.best_overall_node = state.current_beam[0].copy()
            self._set_best_prompt(state.agent_id, state.best_overall_node["prompt"])
            self._set_best_model(
                state.agent_id,
                state.best_overall_node.get("model", self.best_model.get(state.agent_id))
            )

    async def optimize_all_fixed_rounds(self, *args, **kwargs):
        """Use dynamic-safe scoring while retaining the base round loop."""
        kwargs["use_beam_refresh"] = True
        kwargs["use_lookahead_score"] = True
        kwargs["lookahead_weights"] = self.DYNAMIC_WEIGHTS
        kwargs["use_dynamic_switching"] = False
        kwargs["use_stochastic_sampling"] = False
        kwargs["use_feedback"] = False
        kwargs["use_misleading_sampling"] = False
        return await super().optimize_all_fixed_rounds(*args, **kwargs)

    async def optimize_all_round_robin(self, *args, **kwargs):
        kwargs["use_beam_refresh"] = True
        kwargs["use_lookahead_score"] = True
        kwargs["lookahead_weights"] = self.DYNAMIC_WEIGHTS
        return await super().optimize_all_round_robin(*args, **kwargs)
