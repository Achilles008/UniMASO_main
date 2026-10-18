
import json
import random
from typing import List, Dict, Any

from config import DATASET_CONFIG, TaskType


def _wrap_openai_tools(functions: list) -> list:
    """Wrap raw BFCL function definitions in OpenAI tools format."""
    wrapped = []
    for func in functions:
        params = func.get("parameters", {})
        # Fix "dict" -> "object" for OpenAI API
        if params.get("type") == "dict":
            params = {**params, "type": "object"}
        wrapped.append({
            "type": "function",
            "function": {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": params,
            }
        })
    return wrapped




# ---- HotpotQA prompt builder (ported from MAS-PromptBench hotpotqa_pretreive.py) ----
def build_hotpotqa_prompt(question: str, context: list = None) -> str:
    """Build HotpotQA prompt with pre-retrieved Wikipedia context.

    Args:
        question: The question string.
        context: List of [title, [sentences...]] from cached context.

    Returns:
        Formatted prompt string.
    """
    if not context:
        return f"Question: {question}"

    context_parts = []
    for doc in context:
        title = doc[0]
        sentences = doc[1]
        context_parts.append(
            "Title: " + title + "\n" + "\n".join("- " + s for s in sentences)
        )
    context_text = "\n\n".join(context_parts)
    return "Documents:\n\n" + context_text + "\n\nQuestion: " + question

def load_test_data(dataset: str, max_items: int = None) -> List[Dict[str, Any]]:
    config = DATASET_CONFIG.get(dataset)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    file_path = config["path"]
    task_type = config["task_type"]
    problem_key = config["problem_key"]
    answer_key = config["answer_key"]
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            
            entry = {
                'problem': item[problem_key],
                'unique_id': item.get('unique_id', item.get('task_id', str(len(data)))),
                'task_type': task_type,
            }
            
            if task_type == TaskType.CODE:
                entry['answer'] = ''
                entry['test_list'] = item.get('test_list', [])
            elif task_type == TaskType.FUNCTION_CALLING:
                raw_funcs = item.get('functions', [])
                entry['answer'] = item.get(answer_key, '')
                entry['functions'] = _wrap_openai_tools(raw_funcs)  # OpenAI-wrapped for MAS native FC
                entry['function_schemas'] = raw_funcs  # Raw schemas for ast_checker
                # Parse ground_truth for the judge
                raw_gt = item.get('ground_truth_raw', [])
                entry['ground_truth_raw'] = raw_gt
                entry['ground_truth'] = raw_gt  # Already a list of canonical dicts
                entry['category'] = item.get('subset', 'simple')
            elif task_type == TaskType.CODE_GENERATION:
                entry['answer'] = item.get(answer_key, '')
                entry['starter_code'] = item.get('starter_code', '')
                # Parse test cases from answer (JSON string with public/private test cases)
                tests = []
                raw_answer = item.get(answer_key, '')
                if raw_answer:
                    try:
                        import json as _json
                        ans_data = _json.loads(raw_answer) if isinstance(raw_answer, str) else raw_answer
                        tests = ans_data.get('public_test_cases', []) + ans_data.get('private_test_cases', [])
                    except Exception:
                        pass
                entry['tests'] = tests
            elif task_type == TaskType.QA:
                entry['answer'] = item.get(answer_key, '')
                entry['type'] = item.get('type', '')
                entry['level'] = item.get('level', '')
                # Build context from HotpotQA documents: [[title, [sentences]], ...]
                context_parts = []
                for doc in item.get('context', []):
                    title = doc[0]
                    sentences = doc[1]
                    context_parts.append('Title: ' + title + '\n' + '\n'.join('- ' + s for s in sentences))
                context_text = '\n\n'.join(context_parts)
                entry['context_text'] = context_text
                # Build problem as: documents + question
                entry['problem'] = 'Documents:\n\n' + context_text + '\n\nQuestion: ' + item.get('question', item.get(problem_key, ''))
            else:
                entry['answer'] = item.get(answer_key, '')
            
            data.append(entry)
            if max_items is not None and len(data) >= max_items:
                break
    
    return data

def load_train_for_opt(dataset: str, k: int = 50, rng=None) -> List[str]:
    data = load_test_data(dataset)
    if k > len(data):
        k = len(data)
    sampler = rng or random
    return [item["problem"] for item in sampler.sample(data, k)]

def get_task_type(dataset: str) -> TaskType:
    config = DATASET_CONFIG.get(dataset)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset}")
    return config["task_type"]

def get_default_use_judge(dataset: str) -> bool:
    config = DATASET_CONFIG.get(dataset)
    if not config:
        return False
    return config.get("default_use_judge", False)
