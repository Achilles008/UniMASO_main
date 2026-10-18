
import asyncio
from typing import List

import os
import sys
from utils import async_call_llm, execute_code_with_tests

class LLMJudgeAgent:
    
    _TEMPLATE = (
        "Below is a {problem_type}, its standard answer, and a candidate solution. "
        "Please determine whether the candidate solution produced the correct answer as its final output. "
        "Only output 'True' or 'False', nothing else.\n\n"
        "Problem: {question}\n"
        "Standard answer: {gold}\n"
        "Candidate solution: {raw_output}\n\n"
        "Correct:"
    )

    def __init__(self, client, problem_type: str = "math problem"):
        self.client = client
        self.problem_type = problem_type

    async def ajudge(self, question: str, gold: str, raw_output: str) -> bool:
        prompt = self._TEMPLATE.format(
            problem_type=self.problem_type,
            question=question, 
            gold=gold, 
            raw_output=raw_output
        )
        resp = await async_call_llm(self.client, prompt, temperature=0.0)
        return resp.strip().lower() == "true"


class CodeJudgeAgent:
    
    def __init__(self):
        pass
    
    async def ajudge(self, code: str, test_list: List[str]) -> bool:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, execute_code_with_tests, code, test_list)
        return result["success"]

import json
from typing import Union
# ================================================================
# Official HotpotQA Judge (ported from MAS-PromptBench langgraph_hotpotqa.py)
# ================================================================
import re as _re
import string as _string
from collections import Counter as _Counter

class HotpotQAJudge:
    """Official HotpotQA evaluation with EM and F1 scoring.

    Ported verbatim from MAS-PromptBench topologies/centralized/langgraph/
    hotpotqa/langgraph_hotpotqa.py, which follows hotpot_evaluate_v1.py.
    """

    @staticmethod
    def normalize_answer(s: str) -> str:
        """Lowercase, strip punctuation, strip articles a/an/the."""
        s = s.lower()
        s = "".join(ch for ch in s if ch not in set(_string.punctuation))
        s = _re.sub(r"\b(a|an|the)\b", " ", s)
        s = " ".join(s.split())
        return s

    @staticmethod
    def exact_match_score(pred: str, gold: str) -> float:
        return float(
            HotpotQAJudge.normalize_answer(pred) == HotpotQAJudge.normalize_answer(gold)
        )

    @staticmethod
    def f1_score(pred: str, gold: str) -> tuple:
        """Returns (f1, precision, recall)."""
        npred = HotpotQAJudge.normalize_answer(pred)
        ngold = HotpotQAJudge.normalize_answer(gold)
        zero = (0.0, 0.0, 0.0)
        if npred in {"yes", "no", "noanswer"} and npred != ngold:
            return zero
        if ngold in {"yes", "no", "noanswer"} and npred != ngold:
            return zero
        pred_tokens = npred.split()
        gold_tokens = ngold.split()
        common = _Counter(pred_tokens) & _Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return zero
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall

    async def ajudge(self, raw_output: str, gold: str, **kwargs) -> dict:
        """UniMASO-compatible async judge.

        Returns: {"correct": bool, "em": float, "f1": float, "precision": float, "recall": float}
        """
        pred = (raw_output or "").strip()
        gold = (gold or "").strip()
        if not pred:
            return {"correct": False, "em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        em = self.exact_match_score(pred, gold)
        f1, prec, rec = self.f1_score(pred, gold)
        return {
            "correct": em == 1.0,
            "em": em,
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        }


# ================================================================
# Official BFCL Judge (ported from MAS-PromptBench langgraph_bfcl.py)
# ================================================================
import json as _json

class BFCLJudge:
    """Official BFCL v3 evaluation via bfcl_eval's AST checker.

    Ported from MAS-PromptBench topologies/centralized/langgraph/bfcl/
    langgraph_bfcl.py.
    """

    _NAME_ARGS_PAIRS = (
        ("fn_name", "args"),
        ("name", "arguments"),
        ("function_name", "arguments"),
        ("function", "arguments"),
    )

    _FENCED_RE = _re.compile(r"```(?:\w*)\s*([\s\S]*?)\s*```")

    def __init__(self, model_id: str = "Qwen/Qwen3-8B"):
        self.model_id = model_id

    @staticmethod
    def _normalize_call(d: dict) -> dict:
        """Normalize wrong shapes like {fn_name,args} back to {name:args}."""
        for name_key, args_key in BFCLJudge._NAME_ARGS_PAIRS:
            if (name_key in d and args_key in d
                    and isinstance(d[name_key], str)
                    and isinstance(d[args_key], dict)):
                return {d[name_key]: d[args_key]}
        return d

    @staticmethod
    def _parse_json_candidate(candidate: str):
        candidate = (candidate or "").strip()
        if not candidate:
            return None
        try:
            return _json.loads(candidate)
        except _json.JSONDecodeError:
            pass

        # Some models wrap the JSON payload in parentheses inside a fence.
        if candidate.startswith("(") and candidate.endswith(")"):
            try:
                return _json.loads(candidate[1:-1].strip())
            except _json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _candidate_json_spans(text: str) -> list:
        """Fallback JSON extraction for bare arrays/objects outside fences."""
        spans = []
        for open_ch, close_ch in (("[", "]"), ("{", "}")):
            start = text.find(open_ch)
            while start != -1:
                depth = 0
                in_str = False
                esc = False
                for idx in range(start, len(text)):
                    ch = text[idx]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    if ch == '"':
                        in_str = True
                    elif ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        depth -= 1
                        if depth == 0:
                            spans.append(text[start:idx + 1])
                            break
                start = text.find(open_ch, start + 1)
        return spans

    @staticmethod
    def _canonicalize(parsed) -> list | None:
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
            return [BFCLJudge._normalize_call(d) for d in parsed]
        return None

    @staticmethod
    def extract_canonical(text: str) -> list | None:
        """Extract BFCL canonical calls, preferring MAS-PromptBench fenced JSON."""
        import re as _re2
        text = _re2.sub(r"\bTERMINATE\b", "", text or "")
        candidates = [m.group(1) for m in BFCLJudge._FENCED_RE.finditer(text)]
        for cand in reversed(candidates):
            canonical = BFCLJudge._canonicalize(BFCLJudge._parse_json_candidate(cand))
            if canonical:
                return canonical

        # Compatibility fallback for legacy UniMASO prompts that emitted bare JSON.
        canonical = BFCLJudge._canonicalize(BFCLJudge._parse_json_candidate(text.strip()))
        if canonical:
            return canonical
        for cand in reversed(BFCLJudge._candidate_json_spans(text)):
            canonical = BFCLJudge._canonicalize(BFCLJudge._parse_json_candidate(cand))
            if canonical:
                return canonical
        return None

    async def ajudge(self, raw_output: str, function_schemas: list = None,
                     ground_truth: list = None, category: str = "simple", **kwargs) -> dict:
        """UniMASO-compatible async judge.

        Args:
            raw_output: The full MAS agent output text (may contain JSON fence).
            function_schemas: The BFCL function schema list.
            ground_truth: The expected canonical call list.
            category: One of "simple", "multiple", "parallel", "parallel_multiple".

        Returns:
            {"correct": bool, "valid": bool, "error_type": str|None, "details": dict}
        """
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

        if function_schemas is None or ground_truth is None:
            # Fallback for missing schemas/gt
            model_output = self.extract_canonical(raw_output)
            return {"correct": False, "valid": False, "error_type": "missing_schemas_or_gt"}

        model_output = self.extract_canonical(raw_output)
        if model_output is None:
            return {
                "correct": False,
                "valid": False,
                "error_type": "no_canonical_extracted",
                "model_output": None,
            }
        try:
            result = ast_checker(
                function_schemas,
                model_output,
                ground_truth,
                Language.PYTHON,
                category,
                self.model_id,
            )
        except Exception as e:
            return {
                "correct": False,
                "valid": False,
                "error_type": f"{type(e).__name__}: {e}",
                "model_output": model_output,
            }

        is_valid = bool(result.get("valid"))
        return {
            "correct": is_valid,
            "valid": is_valid,
            "error_type": result.get("error_type"),
            "details": result,
            "model_output": model_output,
        }


# ================================================================
# Official LCB Judge (ported from MAS-PromptBench langgraph_lcb.py)
# ================================================================
import base64 as _base64
import pickle as _pickle
import zlib as _zlib
from decimal import Decimal as _Decimal, InvalidOperation as _InvalidOperation
import tempfile as _tempfile

class LCBJudge:
    """Official LiveCodeBench evaluation with stdin/functional test execution.

    Ported from MAS-PromptBench topologies/centralized/langgraph/lcb/
    langgraph_lcb.py. Do NOT modify the scoring logic.
    """

    _FUNCTIONAL_WORKER = """
import os, sys, json, platform, resource, shutil, subprocess, builtins, faulthandler


def _reliability_guard(max_memory_bytes):
    if max_memory_bytes:
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (max_memory_bytes, max_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (max_memory_bytes, max_memory_bytes))
    faulthandler.disable()
    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for _name in (
        "kill","system","putenv","remove","removedirs","rmdir","fchdir","setuid",
        "fork","forkpty","killpg","rename","renames","truncate","replace","unlink",
        "fchmod","fchown","chmod","chown","chroot","lchflags","lchmod","lchown","chdir",
    ):
        if hasattr(os, _name):
            try:
                setattr(os, _name, None)
            except Exception:
                pass
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None
    for _mod in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[_mod] = None


_max_mem = int(sys.argv[1])
_fn = sys.argv[2]
_args = json.loads(sys.argv[3])
_outfile = sys.argv[4]
_code = sys.stdin.read()

_reliability_guard(_max_mem)

_out = None
try:
    _ns = {}
    exec(_code, _ns)
    if "Solution" in _ns:
        _target = getattr(_ns["Solution"](), _fn)
    elif _fn in _ns:
        _target = _ns[_fn]
    else:
        _out = {"ok": False, "error": "neither Solution." + _fn + " nor " + _fn + " defined"}
    if _out is None:
        _r = _target(*_args)
        if isinstance(_r, tuple):
            _r = list(_r)
        _out = {"ok": True, "result": _r}
except BaseException as _e:
    _out = {"ok": False, "error": type(_e).__name__ + ": " + str(_e)}

with open(_outfile, "w") as _f:
    _f.write(json.dumps(_out, default=list))
"""

    _FUNCTIONAL_MEMORY_BYTES = int(os.environ.get(
        "LCB_FUNCTIONAL_MEMORY_BYTES", str(4 * 1024 ** 3)
    ))

    def __init__(self, timeout_s: int = 5, test_limit: int = None):
        self.timeout_s = timeout_s
        if test_limit is None:
            test_limit = int(os.environ.get("LCB_TEST_LIMIT", "3"))
        self.test_limit = test_limit

    @staticmethod
    def _compare_stdout(actual: str, expected: str) -> bool:
        if actual.strip() == expected.strip():
            return True
        actual_lines = actual.splitlines()
        expected_lines = expected.splitlines()
        if len(actual_lines) != len(expected_lines):
            return False
        for a, e in zip(actual_lines, expected_lines):
            if a.strip() == e.strip():
                continue
            try:
                a_parts = [_Decimal(x) for x in a.split()]
                e_parts = [_Decimal(x) for x in e.split()]
            except (_InvalidOperation, ValueError):
                return False
            if a_parts != e_parts:
                return False
        return True

    @staticmethod
    def _run_stdin_test(code: str, tc: dict, timeout_s: int) -> dict:
        import subprocess as _sp
        try:
            result = _sp.run(
                [sys.executable, "-c", code],
                input=tc.get("input", ""),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except _sp.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

        expected = tc.get("output", "")
        actual = result.stdout
        ok = LCBJudge._compare_stdout(actual, expected) and result.returncode == 0
        return {
            "ok": ok,
            "expected": expected.strip(),
            "actual": actual.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode,
        }

    @staticmethod
    def _parse_maybe_json(value):
        if isinstance(value, str):
            try:
                return _json.loads(value)
            except _json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def _run_functional_test(code: str, tc: dict, timeout_s: int) -> dict:
        import subprocess as _sp
        fn_name = tc.get("fn_name") or tc.get("func_name")
        if not fn_name:
            return {"ok": False, "error": "functional test missing fn_name"}

        args = LCBJudge._parse_maybe_json(tc.get("input", "[]"))
        if not isinstance(args, (list, tuple)):
            args = [args]

        with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            outfile = f.name

        try:
            proc = _sp.run(
                [
                    sys.executable, "-c", LCBJudge._FUNCTIONAL_WORKER,
                    str(LCBJudge._FUNCTIONAL_MEMORY_BYTES),
                    fn_name,
                    _json.dumps(args),
                    outfile,
                ],
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except _sp.TimeoutExpired:
            os.unlink(outfile)
            return {"ok": False, "error": "timeout"}
        except Exception as e:
            os.unlink(outfile)
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

        try:
            with open(outfile) as f:
                payload = _json.load(f)
        except (_json.JSONDecodeError, FileNotFoundError, OSError):
            return {
                "ok": False,
                "error": f"worker produced no parseable result (exit={proc.returncode})",
                "stderr": (proc.stderr or "").strip(),
            }
        finally:
            try:
                os.unlink(outfile)
            except FileNotFoundError:
                pass

        if not payload.get("ok"):
            return {"ok": False, "error": payload.get("error", "worker error")}
        actual = payload["result"]
        expected = LCBJudge._parse_maybe_json(tc.get("output"))
        return {"ok": actual == expected, "expected": expected, "actual": actual}

    def run_tests(self, code: str, tests: list, timeout_s: int = None, test_limit: int = None) -> dict:
        """Run code against LCB-style tests (stdin or functional).

        Returns: {"pass", "total", "pass_rate", "details"}
        """
        if timeout_s is None:
            timeout_s = self.timeout_s
        if test_limit is None:
            test_limit = self.test_limit
        if not tests:
            return {"pass": 0, "total": 0, "pass_rate": 0.0, "details": [], "uncapped_total": 0, "test_limit": test_limit}

        uncapped_total = len(tests)
        if test_limit and test_limit > 0:
            tests = tests[:test_limit]
        passed = 0
        details = []
        for i, tc in enumerate(tests):
            testtype = tc.get("testtype")
            fn_name = tc.get("fn_name") or tc.get("func_name")
            is_functional = testtype == "functional" or (testtype is None and fn_name is not None)

            if is_functional:
                r = self._run_functional_test(code, tc, timeout_s)
                r["mode"] = "functional"
            else:
                r = self._run_stdin_test(code, tc, timeout_s)
                r["mode"] = "stdin"

            passed += int(r.get("ok", False))
            details.append({"test": i, **r})

        return {
            "pass": passed,
            "total": len(tests),
            "pass_rate": passed / len(tests),
            "details": details,
            "uncapped_total": uncapped_total,
            "test_limit": test_limit,
        }

    @staticmethod
    def _decode_private_tests(blob: str) -> list:
        """LCB private tests: base64(zlib(pickle(json_str)))."""
        if not blob:
            return []
        try:
            decompressed = _zlib.decompress(_base64.b64decode(blob))
        except Exception:
            return []
        try:
            payload = _pickle.loads(decompressed)
            if isinstance(payload, str):
                return _json.loads(payload)
            if isinstance(payload, list):
                return payload
        except Exception:
            pass
        try:
            return _json.loads(decompressed)
        except Exception:
            return []

    async def ajudge(self, raw_output: str, tests: list = None, **kwargs) -> dict:
        """Run extracted code against LCB test cases.

        Returns: {"correct": bool, "pass": int, "total": int, "pass_rate": float}
        """
        code = raw_output  # already extracted via extract_output
        if not code or not code.strip():
            total = len(tests) if tests else 0
            return {"correct": False, "pass": 0, "total": total, "pass_rate": 0.0}

        test_limit = kwargs.get("test_limit", self.test_limit)
        result = self.run_tests(code, tests, timeout_s=self.timeout_s, test_limit=test_limit)
        return {
            "correct": result["pass_rate"] == 1.0,
            "pass": result["pass"],
            "total": result["total"],
            "pass_rate": result["pass_rate"],
            "uncapped_total": result.get("uncapped_total", result["total"]),
            "test_limit": result.get("test_limit", test_limit),
            "details": result.get("details", []),
        }
