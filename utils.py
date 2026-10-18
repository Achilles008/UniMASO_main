import re
import html
import asyncio
import subprocess
import tempfile
import os
import random
from typing import List, Dict, Any, Optional
from collections import Counter

from config import TaskType

DS_API_CONCURRENCY_LIMIT = 60 
ds_semaphore = asyncio.Semaphore(DS_API_CONCURRENCY_LIMIT)
def async_retry(tries: int = 8, delay: float = 2.0, max_delay: float = 30.0,
                jitter: float = 0.25):
    def deco(func):
        async def wrapper(*args, **kwargs):
            d = delay
            for i in range(1, tries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if i == tries:
                        raise
                    sleep_s = min(max_delay, d)
                    sleep_s *= 1.0 + random.uniform(-jitter, jitter)
                    print(
                        f"⚠️ LLM request failed ({type(exc).__name__}); "
                        f"retry {i}/{tries - 1} in {sleep_s:.1f}s",
                        flush=True,
                    )
                    await asyncio.sleep(max(0.1, sleep_s))
                    d *= 2
            return None
        return wrapper
    return deco

@async_retry(tries=8, delay=2.0, max_delay=30.0, jitter=0.25)
async def async_call_llm(client, prompt: str, temperature: float = 0.0,
                         max_tokens: int = 4096, tools: list = None,
                         timeout_s: int = None, timeout_label: str = "",
                         return_usage: bool = False):
    """Call LLM via a ModelClient (round-robin across endpoints).

    Args:
        client: ModelClient instance (carries model_path + endpoint list).
        tools: Optional list of OpenAI tool definitions for function calling.
    """
    if getattr(client, "api_style", "chat") == "responses":
        request_kwargs = {
            "model": client.model_path,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "extra_body": dict(getattr(client, "extra_body", {}) or {}),
        }

        async def _create_response():
            if client._semaphore is not None:
                async with client._semaphore:
                    return await client.responses.create(**request_kwargs)
            return await client.responses.create(**request_kwargs)

        try:
            if timeout_s and timeout_s > 0:
                resp = await asyncio.wait_for(_create_response(), timeout=timeout_s)
            else:
                resp = await _create_response()
        except asyncio.TimeoutError:
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            label = f" ({timeout_label})" if timeout_label else ""
            print(f"⚠️ LLM call timeout after {timeout_s}s{label}; returning empty output.", flush=True)
            result = {"text": "", "usage": usage, "error": f"timeout after {timeout_s}s"}
            return result if return_usage else ""

        usage_obj = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": getattr(usage_obj, "input_tokens", 0) if usage_obj else 0,
            "completion_tokens": getattr(usage_obj, "output_tokens", 0) if usage_obj else 0,
        }
        text_parts = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    text_parts.append(value)
        text = "\n".join(text_parts)
        if not text:
            text = getattr(resp, "output_text", "") or ""
        return {"text": text, "usage": usage} if return_usage else text

    request_kwargs = {
        "model": client.model_path,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": dict(getattr(client, "extra_body", {}) or {}),
    }
    if tools:
        request_kwargs["tools"] = tools
        request_kwargs["tool_choice"] = "auto"

    async def _create_completion():
        if client._semaphore is not None:
            async with client._semaphore:
                return await client.chat.completions.create(**request_kwargs)
        return await client.chat.completions.create(**request_kwargs)

    try:
        if timeout_s and timeout_s > 0:
            resp = await asyncio.wait_for(_create_completion(), timeout=timeout_s)
        else:
            resp = await _create_completion()
    except asyncio.TimeoutError:
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        label = f" ({timeout_label})" if timeout_label else ""
        print(f"⚠️ LLM call timeout after {timeout_s}s{label}; returning empty output.", flush=True)
        result = {"text": "", "usage": usage, "error": f"timeout after {timeout_s}s"}
        return result if return_usage else ""

    # Capture token usage for cost computation (vLLM / OpenAI-compatible)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if hasattr(resp, 'usage') and resp.usage:
        usage = {
            "prompt_tokens": getattr(resp.usage, 'prompt_tokens', 0),
            "completion_tokens": getattr(resp.usage, 'completion_tokens', 0),
        }

    msg = resp.choices[0].message
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        import json
        tcs = []
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            tcs.append({"name": tc.function.name, "arguments": args})
        text = json.dumps(tcs, ensure_ascii=False)
    else:
        text = msg.content if msg.content else ""

    return {"text": text, "usage": usage} if return_usage else text

def extract_answer(raw: str) -> str:
    raw = raw.strip()
    for _ in range(3):
        new_raw = html.unescape(raw)
        if new_raw == raw:
            break
        raw = new_raw

    matches = re.findall(r"<answer>(.*?)</answer>", raw, re.S)
    if matches:
        return matches[-1].strip()

    box_pat = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.S)
    m = box_pat.search(raw)
    if m:
        return m.group(1).strip()

    sentences = re.split(r'[。\n;]+', raw)
    last = sentences[-1].strip()
    return last[-30:] if len(last) > 30 else last

def extract_function_call(raw: str) -> str:
    """Extract all function calls from model output (Native FC JSON or text fallback)."""
    import json
    raw = raw.strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass
    json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\}'
    matches = re.findall(json_pattern, raw, re.DOTALL)
    if matches:
        all_calls = []
        for match in matches:
            try:
                parsed = json.loads(match)
                all_calls.append(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
        if all_calls:
            return json.dumps(all_calls, ensure_ascii=False)
    return raw

def extract_code(raw: str) -> str:
    raw = raw.strip()

    candidates = []
    seen = set()

    python_pattern = r'```\s*python\s*(.*?)\s*```'
    for m in re.findall(python_pattern, raw, flags=re.DOTALL | re.IGNORECASE):
        code = m.strip()
        if code and code not in seen:
            seen.add(code)
            candidates.append(code)

    general_pattern = r'```\s*(.*?)\s*```'
    for m in re.findall(general_pattern, raw, flags=re.DOTALL):
        code = m.strip()
        if code and code not in seen:
            code = re.sub(r'^python\s*', '', code, flags=re.IGNORECASE).strip()
            seen.add(code)
            candidates.append(code)

    code_tag_pattern = r'<code>(.*?)</code>'
    for m in re.findall(code_tag_pattern, raw, flags=re.DOTALL | re.IGNORECASE):
        code = m.strip()
        if code and code not in seen:
            seen.add(code)
            candidates.append(code)

    for code in reversed(candidates):
        if 'def ' in code:
            return code

    if 'def ' in raw:
        def_pattern = r'def\s+[\w_]+\(.*?\):\s*\n?(.|\n)*?(?=\n\s*\n|```|\Z)'
        def_match = re.search(def_pattern, raw, flags=re.IGNORECASE)
        if def_match:
            return def_match.group(0).strip()

    return ''

def extract_code_from_generation(raw: str) -> str:
    """Extract code from a code generation model output."""
    match = re.search(r'```(?:python)?\s*\n(.*?)```', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*\n(.*?)```', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()




# ---- AST-validated Python code extraction (ported from MAS-PromptBench code_extract.py) ----
def extract_python_code(text: str) -> str | None:
    """Return the last non-empty fenced block that is plausibly Python code.

    Uses ast.parse() to validate syntax. Ignores blocks that start with
    JSON-like braces. Returns None if no valid Python code is found.
    """
    import ast as _ast
    raw = text or ""
    py_fence = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    bare_fence = re.compile(r"```\s*\n(.*?)```", re.DOTALL)

    for pattern in (py_fence, bare_fence):
        for block in reversed(pattern.findall(raw)):
            candidate = block.strip()
            if _is_python_code(candidate):
                return candidate
    return None


def _is_python_code(candidate: str) -> bool:
    """Check if a string is syntactically valid Python code (not JSON)."""
    import ast as _ast
    if not candidate:
        return False
    stripped = candidate.lstrip()
    if stripped.startswith(("{", "[")):
        return False
    try:
        _ast.parse(candidate)
    except SyntaxError:
        return False
    return True

def extract_output(raw: str, task_type: TaskType) -> str:
    if task_type == TaskType.CODE:
        return extract_code(raw)
    elif task_type == TaskType.FUNCTION_CALLING:
        return raw  # BFCL judge does its own canonical extraction
    elif task_type == TaskType.CODE_GENERATION:
        code = extract_python_code(raw)
        return code if code else extract_code_from_generation(raw)
    else:
        return extract_answer(raw)

def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r'\$(.*?)\$', r'\1', answer)
    answer = re.sub(r'\\\[(.*?)\\\]', r'\1', answer, flags=re.S)
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\boxed\s*{((?:[^{}]|{[^}]*})*?)}', r'\1', answer)
    answer = re.sub(r'\\\((.*?)\\\)', r'\1', answer)
    answer = re.sub(r'\\?°', '', answer)
    answer = re.sub(r'\^?\\?circ', '', answer)
    answer = re.sub(r'\s+', '', answer)
    answer = re.sub(r'\\sqrt\s*{([^}]*)}', r'sqrt(\1)', answer)
    answer = re.sub(r'√(\d+)', r'sqrt(\1)', answer)
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', answer)
    answer = re.sub(r'\\dfrac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', answer)
    answer = re.sub(r'\\pi', 'π', answer)
    answer = re.sub(r'\\left|\\right', '', answer)
    answer = answer.replace('[', '(').replace(']', ')')
    if re.fullmatch(r'[,\s\-0-9]+', answer):
        nums = [int(x) for x in re.findall(r'-?\d+', answer)]
        return ','.join(map(str, sorted(nums)))
    return answer.lower()

def normalize_qa_answer(answer: str) -> str:
    """Normalize QA answers: lowercase, strip punctuation/articles, normalize whitespace."""
    import re
    answer = answer.lower().strip()
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = re.sub(r'\b(a|an|the)\s+', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer.strip()

def execute_code_with_tests(code: str, test_list: List[str], timeout: int = 5) -> Dict[str, Any]:
    
    required_imports = set()
    for test in test_list:
        if 'math.' in test:
            required_imports.add('import math')
    
    existing_imports = set()
    for imp in required_imports:
        if imp in code:
            existing_imports.add(imp)
    
    missing_imports = required_imports - existing_imports
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        temp_file = f.name
        
        for imp in sorted(missing_imports):
            f.write(imp + '\n')
        
        if missing_imports:
            f.write('\n')
        
        f.write(code + '\n\n')
        
        for test in test_list:
            f.write(test + '\n')
    
    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        os.unlink(temp_file)
        
        if result.returncode == 0:
            return {"success": True, "passed": len(test_list), "total": len(test_list), "errors": []}
        else:
            return {"success": False, "passed": 0, "total": len(test_list), "errors": [result.stderr]}
    except subprocess.TimeoutExpired:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return {"success": False, "passed": 0, "total": len(test_list), "errors": ["Execution timeout"]}
    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return {"success": False, "passed": 0, "total": len(test_list), "errors": [str(e)]}

def extract_function_name_from_tests(test_list: List[str]) -> Optional[str]:
    BUILTIN_SKIP = {
        'assert', 'math', 'set', 'list', 'dict', 'tuple', 'str', 'int', 'float',
        'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'sum',
        'max', 'min', 'abs', 'round', 'all', 'any', 'isinstance', 'type',
        'print', 'input', 'open', 'isclose', 'sqrt', 'pow', 'ceil', 'floor'
    }
    
    function_candidates = []
    
    for test in test_list:
        matches = re.findall(r'\b([a-zA-Z_]\w*)\s*\(', test)
        for match in matches:
            if match not in BUILTIN_SKIP:
                function_candidates.append(match)
    
    if not function_candidates:
        return None
    
    most_common = Counter(function_candidates).most_common(1)
    return most_common[0][0] if most_common else None

def majority_vote(answers: List[str]) -> str:
    from collections import defaultdict
    
    if not answers:
        return ""
    
    counts = defaultdict(int)
    for a in answers:
        if a:
            normalized = normalize_answer(a)
            counts[normalized] += 1
    
    if not counts:
        return answers[0] if answers else ""
    
    return max(counts.items(), key=lambda x: x[1])[0]

def code_vote(codes: List[str]) -> str:
    def normalize_code(code: str) -> str:
        lines = code.split('\n')
        normalized = []
        for line in lines:
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            if line:
                normalized.append(line)
        return '\n'.join(normalized)
    
    if not codes:
        return ""
    
    normalized_codes = [normalize_code(c) for c in codes if c]
    
    if not normalized_codes:
        return codes[0] if codes else ""
    
    code_hashes = [hash(nc) for nc in normalized_codes]
    most_common_hash = Counter(code_hashes).most_common(1)[0][0]
    
    for i, h in enumerate(code_hashes):
        if h == most_common_hash:
            return codes[i]
    
    return codes[0]

def parse_comparison_result(raw: str) -> bool:
    if "<choose>" in raw and "</choose>" in raw:
        try:
            choice = raw.split("<choose>")[1].split("</choose>")[0]
            return "A" in choice
        except:
            pass
    for c in reversed(raw.strip().upper()):
        if c.isalpha():
            return c == 'A'
    return True
