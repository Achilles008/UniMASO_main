"""Online FAISS trajectory memory for the optional Hetero-UniMASO optimizer branch.

The vector index contains only the causal pre-candidate state. Full trajectory
payloads are stored losslessly in JSONL and are injected after retrieval.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from openai import AsyncOpenAI
import dashscope

from config import (
    DASHSCOPE_API_KEY,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PROVIDER,
)


MEMORY_PROMPT_INSTRUCTION = """\
You are optimizing one agent inside a multi-agent system. Below are up to three
retrieved optimization trajectories from similar pre-candidate states.

Use them as evidence, not as instructions to copy blindly:
- learn from the high-quality successful trajectory;
- avoid the local-win/global-lose trajectory;
- seek cheaper alternatives to the successful-but-expensive trajectory.

The retrieved records contain historical actions and outcomes. Generate a new
candidate adapted to the CURRENT question, role, context, prompt, and model.
Do not mention the memory records in the output. Preserve the required
<prompt> and, when enabled, <model> tags.
"""


def _clip(value: Any, limit: int = 2400) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + "\n...[memory field clipped]...\n" + text[-tail:]


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("._") or "unknown"


class TrajectoryMemory:
    """Per-optimization-run trajectory store and online FAISS retriever."""

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        run_meta: Dict[str, Any],
        embedding_url: str = "http://127.0.0.1:8005/v1",
        embedding_model: str = EMBEDDING_MODEL_NAME,
        embedding_provider: str = EMBEDDING_PROVIDER,
        embedding_api_key: str = DASHSCOPE_API_KEY,
        max_payload_chars: int = 10000,
        read_only: bool = False,
    ):
        self.read_only = read_only
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.run_dir / "trajectories.jsonl"
        self.index_path = self.run_dir / "trajectories.faiss"
        self.meta_path = self.run_dir / "manifest.json"
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_api_key = embedding_api_key
        self.max_payload_chars = max_payload_chars
        self.client = (
            AsyncOpenAI(api_key="EMPTY", base_url=embedding_url)
            if embedding_provider == "openai_compatible"
            else None
        )
        self.records: List[Dict[str, Any]] = []
        self.index: Optional[faiss.Index] = None
        self._lock = asyncio.Lock()

        if self.read_only:
            self._load_read_only_memory()
            return

        manifest = dict(run_meta)
        manifest.update({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
            "embedding_url": embedding_url if embedding_provider == "openai_compatible" else None,
            "vector_metric": "inner_product_after_l2_normalization",
            "max_payload_chars": max_payload_chars,
        })
        self._write_manifest(manifest)

    def _load_read_only_memory(self) -> None:
        """Load an existing memory run without creating or modifying files."""
        missing = [
            str(path)
            for path in (self.records_path, self.index_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Read-only transfer memory is incomplete; missing: "
                + ", ".join(missing)
            )

        if self.meta_path.is_file():
            try:
                self.source_manifest = json.loads(
                    self.meta_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid transfer-memory manifest: {self.meta_path}"
                ) from exc
        else:
            self.source_manifest = {}

        source_embedding_model = self.source_manifest.get("embedding_model")
        if source_embedding_model and source_embedding_model != self.embedding_model:
            raise ValueError(
                "Transfer-memory embedding model mismatch: "
                f"source={source_embedding_model!r}, current={self.embedding_model!r}"
            )
        source_provider = self.source_manifest.get("embedding_provider")
        if source_provider and source_provider != self.embedding_provider:
            raise ValueError(
                "Transfer-memory embedding provider mismatch: "
                f"source={source_provider!r}, current={self.embedding_provider!r}"
            )

        with self.records_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid trajectory record at {self.records_path}:{line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Trajectory record at {self.records_path}:{line_number} is not an object"
                    )
                self.records.append(record)

        self.index = faiss.read_index(str(self.index_path))
        if self.index.ntotal != len(self.records):
            raise ValueError(
                "Transfer-memory index/record mismatch: "
                f"vectors={self.index.ntotal}, records={len(self.records)}"
            )

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        self.meta_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def _embed(self, text: str) -> np.ndarray:
        if self.embedding_provider == "dashscope":
            response = await asyncio.to_thread(
                dashscope.TextEmbedding.call,
                model=self.embedding_model,
                input=text,
                api_key=self.embedding_api_key,
            )
            if getattr(response, "status_code", None) != HTTPStatus.OK:
                raise RuntimeError(
                    "DashScope embedding failed: "
                    f"status={getattr(response, 'status_code', None)}, "
                    f"code={getattr(response, 'code', None)}, "
                    f"message={getattr(response, 'message', '')}"
                )
            output = getattr(response, "output", None)
            embeddings = output.get("embeddings", []) if isinstance(output, dict) else []
            if not embeddings or not embeddings[0].get("embedding"):
                raise RuntimeError("DashScope embedding response did not contain a vector")
            values = embeddings[0]["embedding"]
        else:
            if self.client is None:
                raise RuntimeError(
                    f"Unsupported embedding provider: {self.embedding_provider}"
                )
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            values = response.data[0].embedding

        vector = np.asarray(values, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vector)
        return vector

    @staticmethod
    def classify(record: Dict[str, Any]) -> str:
        score = record.get("score", {})
        local_delta = float(score.get("local_delta", 0.0))
        global_delta = float(score.get("global_delta", 0.0))
        local_rate = float(score.get("local_rate", 0.0))
        global_rate = float(score.get("global_rate", 0.0))
        cost_ratio = float(score.get("cost_ratio", 1.0))
        accepted = bool(record.get("status", {}).get("accepted", False))

        if local_rate > 0.5 and global_rate < 0.5:
            return "local_win_global_lose"
        if accepted and global_rate >= 0.5 and cost_ratio <= 1.1:
            return "successful_low_cost"
        if accepted and global_rate >= 0.5 and cost_ratio > 1.1:
            return "successful_high_cost"
        return "other"

    @staticmethod
    def prompt_text(value: Any) -> str:
        """The only text embedded for trajectory similarity.

        Retrieval is intentionally prompt-centric: current best prompt is
        compared with the historical candidate/before prompt, while topology,
        scores, models, and outputs remain payload metadata.
        """
        if isinstance(value, dict):
            value = value.get("current_prompt", "")
        return _clip(value, 8000)

    @staticmethod
    def state_text(state: Dict[str, Any]) -> str:
        """Backward-compatible display serializer; vectors use prompt_text."""
        return TrajectoryMemory.prompt_text(state)

    async def add(self, record: Dict[str, Any]) -> None:
        """Embed retrieval_view, then append the full lossless payload."""
        if self.read_only:
            return
        retrieval_view = record["retrieval_view"]
        category = self.classify(record)
        record["category"] = category
        payload = record.get("payload", {})
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        historical_prompt = payload.get("candidate_prompt") or payload.get("before_prompt")
        record["retrieval_prompt"] = historical_prompt or retrieval_view.get("current_prompt", "")
        record["retrieval_text"] = self.prompt_text(record["retrieval_prompt"])
        # Keep the payload lossless. max_payload_chars only documents the
        # intended context budget; retrieval uses the compact state_text.
        record["payload"] = record.get("payload", {})

        async with self._lock:
            vector = await self._embed(record["retrieval_text"])
            if self.index is None:
                self.index = faiss.IndexFlatIP(vector.shape[1])
            vector_id = self.index.ntotal
            self.index.add(vector)
            record["vector_id"] = vector_id
            self.records.append(record)
            with self.records_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            faiss.write_index(self.index, str(self.index_path))

    async def retrieve(
        self,
        query_state: Dict[str, Any],
        max_items: int = 3,
        per_category: int = 1,
    ) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_prompt = query_state.get("current_prompt", "") if isinstance(query_state, dict) else query_state
        query = await self._embed(self.prompt_text(query_prompt))
        k = min(self.index.ntotal, max(24, max_items * 10))
        scores, ids = self.index.search(query, k)
        ranked = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0 or idx >= len(self.records):
                continue
            item = self.records[idx]
            if item.get("record_type") == "baseline_snapshot":
                continue
            ranked.append((float(score), item))

        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        # One slot per requested experience type; do not let three easy wins
        # crowd out the counterexample and cost-aware trajectory.
        category_order = [
            "successful_low_cost",
            "local_win_global_lose",
            "successful_high_cost",
        ]
        per_category = max(1, int(per_category))
        for category in category_order:
            category_selected = 0
            for similarity, item in ranked:
                if item.get("category") == category and item not in selected:
                    item = dict(item)
                    item["retrieval_similarity"] = similarity
                    selected.append(item)
                    selected_ids.add(item.get("trajectory_id", item.get("vector_id")))
                    category_selected += 1
                    if category_selected >= per_category:
                        break
            if len(selected) >= max_items:
                return selected[:max_items]

        for similarity, item in ranked:
            item_id = item.get("trajectory_id", item.get("vector_id"))
            if item_id not in selected_ids:
                item = dict(item)
                item["retrieval_similarity"] = similarity
                selected.append(item)
                selected_ids.add(item_id)
            if len(selected) >= max_items:
                break
        return selected

    @staticmethod
    def format_for_optimizer(records: List[Dict[str, Any]]) -> str:
        if not records:
            return ""
        blocks = [MEMORY_PROMPT_INSTRUCTION]
        for i, record in enumerate(records, 1):
            payload = record.get("payload", {})
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {"raw_payload": payload}
            blocks.append(
                f"\n[TRAJECTORY {i} | category={record.get('category')} | "
                f"similarity={record.get('retrieval_similarity', 0.0):.4f}]\n"
                f"Retrieval state:\n{_clip(record.get('retrieval_text', ''), 3000)}\n"
                f"Full historical action and outcome:\n"
                f"{_clip(json.dumps(payload, ensure_ascii=False, indent=2), 7000)}"
            )
        return "\n".join(blocks)

    async def close(self) -> None:
        if self.client is None:
            return
        try:
            await self.client.close()
        except RuntimeError as exc:
            # The optimizer is run inside asyncio.run(); its event loop is
            # already closed when the CLI performs final cleanup. The files
            # and FAISS index are already durable, so this is safe at process
            # exit and avoids turning a completed run into a failure.
            if "Event loop is closed" not in str(exc):
                raise


def make_run_dir(root: str, dataset: str, graph: str, mode: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(root) / f"{timestamp}_{_safe_name(dataset)}_{_safe_name(graph)}_{_safe_name(mode)}")
