import asyncio
import itertools
import os
from enum import Enum
from openai import AsyncOpenAI


class ModelClient:
    """Round-robin load-balanced client for one model with multiple endpoints.

    Usage:
        mc = ModelClient("Qwen3.5-9B",
                         ["http://127.0.0.1:8002/v1", "http://127.0.0.1:8002/v1"])
        resp = await mc.chat.completions.create(model=mc.model_path, ...)
    """

    def __init__(self, model_path: str, base_urls: list, concurrency: int = 0,
                 api_key: str = "EMPTY", provider: str = "vllm",
                 extra_body: dict = None, api_style: str = "chat"):
        self.model_path = model_path
        self.provider = provider
        self.api_style = api_style
        self.extra_body = extra_body if extra_body is not None else (
            {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
            if provider == "vllm" else {}
        )
        self._clients = [
            AsyncOpenAI(api_key=api_key or "EMPTY", base_url=url) for url in base_urls
        ]
        self._counter = itertools.cycle(range(len(self._clients)))
        self._semaphore = asyncio.Semaphore(concurrency) if concurrency > 0 else None

    def get(self):
        """Return next client in round-robin order."""
        return self._clients[next(self._counter)]

    async def acquire(self):
        if self._semaphore:
            await self._semaphore.acquire()

    def release(self):
        if self._semaphore:
            self._semaphore.release()

    @property
    def chat(self):
        return self.get().chat

    @property
    def responses(self):
        return self.get().responses


# ============================================================
# Model Registry — add new models here
# ============================================================
# Set API credentials in the process environment; never commit them.
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
# DASHSCOPE_API_KEY is used by the DashScope chat and embedding clients.
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

MODEL_REGISTRY = {
    "qwen35-9b": ModelClient(
        model_path="Qwen3.5-9B",
        base_urls=[
            "http://127.0.0.1:8004/v1",
        ],
    ),
    "qwen35-27b": ModelClient(
        model_path=os.getenv("QWEN35_27B_MODEL_ID", "Qwen3.5-27B"),
        base_urls=[os.getenv("QWEN35_27B_BASE_URL", "http://localhost:8002/v1")],
    ),
    "qwen35-27b-api": ModelClient(
        model_path="qwen3.5-27b",
        base_urls=["https://dashscope.aliyuncs.com/compatible-mode/v1"],
        api_key=DASHSCOPE_API_KEY,
        provider="dashscope",
        extra_body={
            "enable_thinking": False,
            "top_k": 20,
        },
        api_style="chat",
    ),
    "qwen35-4b": ModelClient(
        model_path="Qwen3.5-4B",
        base_urls=[
            "http://127.0.0.1:8003/v1",
        ],
    ),
    "glm-5.2": ModelClient(
        model_path="glm-5.2",
        base_urls=["https://open.bigmodel.cn/api/paas/v4/"],
        api_key=GLM_API_KEY,
        provider="zhipu",
        extra_body={"thinking": {"type": "disabled"}},
    ),
    "deepseek-v4-flash": ModelClient(
        model_path="deepseek-v4-flash",
        base_urls=["https://api.deepseek.com"],
        api_key=DEEPSEEK_API_KEY,
        provider="deepseek",
        extra_body={"thinking": {"type": "disabled"}},
    ),
    "deepseek-v4-pro": ModelClient(
        model_path="deepseek-v4-pro",
        base_urls=["https://api.deepseek.com"],
        api_key=DEEPSEEK_API_KEY,
        provider="deepseek",
        extra_body={"thinking": {"type": "disabled"}},
    ),
}



EMBEDDING_MODEL_NAME = "qwen3.7-text-embedding"
EMBEDDING_PROVIDER = "dashscope"
EMBEDDING_API_KEY = DASHSCOPE_API_KEY
EMBEDDING_BASE_URL = "http://localhost:8005/v1"  # retained for OpenAI-compatible fallback

# Embedding models use a different API surface from chat models and therefore
# live in a separate registry instead of MODEL_REGISTRY.
EMBEDDING_MODEL_REGISTRY = {
    "qwen3.7-text-embedding": {
        "provider": "dashscope",
        "model": "qwen3.7-text-embedding",
        "api_key_source": "DASHSCOPE_API_KEY",
        "dimension": 1024,
    },
}

# Default model assignments (can be overridden via CLI)
# Prompt-facing metadata is kept in configuration so the Co-Evolver can
# describe whatever model pool is active at runtime.
MODEL_PROFILES = {
    "qwen35-4b": {"capability": "very_weak", "best_for": "simple extraction, formatting, routine classification, and low-difficulty subtasks", "capability_rank": 1},
    "qwen35-9b": {"capability": "medium", "best_for": "basic reasoning, routine analysis, standard verification, and relatively simple independent solving", "capability_rank": 2},
    "qwen35-27b": {"capability": "strong", "best_for": "standard multi-step reasoning, moderate analysis, intermediate synthesis, and ordinary task-solving roles", "capability_rank": 3},
    "qwen35-27b-api": {"capability": "strong", "best_for": "standard multi-step reasoning, moderate analysis, intermediate synthesis, and ordinary task-solving roles", "capability_rank": 3},
    "deepseek-v4-flash": {"capability": "very strong", "best_for": "difficult reasoning, critical-path agents, complex synthesis, and tasks requiring substantially stronger reliability than standard models", "capability_rank": 4},
    "deepseek-v4-pro": {"capability": "extremely_strong", "best_for": "exceptionally difficult reasoning, highly complex coding, or globally decisive tasks where strong models are unlikely to be reliable", "capability_rank": 5},
    "glm-5.2": {"capability": "strong", "best_for": "complex reasoning and coding", "capability_rank": 3},
}

def get_model_profile(model_name: str) -> dict:
    return MODEL_PROFILES.get(model_name, {
        "capability": "unknown",
        "best_for": "use according to the model's documented capabilities",
        "capability_rank": 0,
    })

DEFAULT_AGENT_MODEL = "qwen35-4b"
DEFAULT_OPTIMIZER_MODEL = "deepseek-v4-flash"
DEFAULT_EVALUATOR_MODEL = "deepseek-v4-flash"


def get_model(model_name: str) -> ModelClient:
    """Look up a model by name. Raises KeyError if not found."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


# ============================================================
# Cost configuration for Hetero-UniMASO
# ============================================================
MODEL_COST = {
    "qwen35-4b":  {"input": 1, "output": 4},
    "qwen35-9b":  {"input": 2, "output": 8},
    "qwen35-27b": {"input": 8, "output": 32},
    "qwen35-27b-api": {"input": 8, "output": 32},
    "deepseek-v4-flash":   {"input": 24, "output": 72},
    "deepseek-v4-pro":     {"input": 72, "output": 216},
    "glm-5.2":             {"input": 107, "output": 373},
}

# Models available for optimization selection (weakest → strongest)
AVAILABLE_MODELS = [
    "qwen35-4b", "deepseek-v4-flash", "qwen35-9b",
    "glm-5.2", "deepseek-v4-pro", "qwen35-27b", "qwen35-27b-api",
]

DEFAULT_LAMBDA_COST = 0.20
DEFAULT_SEED = 42


def get_model_cost(model_name: str) -> dict:
    """Return the {input, output} cost coefficients for a model.
    Falls back to qwen35-9b pricing if model not found.
    """
    return MODEL_COST.get(model_name, {"input": 2, "output": 8})


def get_model_price_ratio(cand_model: str, base_model: str) -> float:
    """Pure model cost ratio, independent of agent token usage.

    Ratio = cand_price / base_price, where price = sum of (input+output) coefficients.
    4b->9b = 2.0, 27b->9b = 0.25, 9b->27b = 4.0, etc.
    """
    base = MODEL_COST.get(base_model, MODEL_COST["qwen35-9b"])
    cand = MODEL_COST.get(cand_model, MODEL_COST["qwen35-9b"])
    base_price = base["input"] + base["output"]
    cand_price = cand["input"] + cand["output"]
    return cand_price / base_price


def compute_mas_cost(
    cache,  # InferenceCache — duck-typed so we don't import from agent.py
    model_map=None,  # Optional: Dict[int, str] agent_id → model_name
) -> float:
    """Compute total cost of a full MAS inference trace.

    Iterates over agent IDs in the cache, looks up each agent's model,
    and sums input/output token costs.

    Args:
        cache: InferenceCache with node_input_tokens, node_output_tokens, node_model.
        model_map: Optional override {agent_id: model_name}. If None, uses
                   cache.node_model (stored at execution time).

    Returns:
        Total cost (weighted token sum).
    """
    total = 0.0
    # Prefer explicit model_map (for "what if" cost estimation), fall back to
    # the model that was actually used during execution.
    for agent_id in cache.node_input_tokens:
        if model_map and agent_id in model_map:
            model = model_map[agent_id]
        elif hasattr(cache, 'node_model') and agent_id in cache.node_model:
            model = cache.node_model[agent_id]
        else:
            model = DEFAULT_AGENT_MODEL
        c = get_model_cost(model)
        total += c["input"] * cache.node_input_tokens.get(agent_id, 0)
        total += c["output"] * cache.node_output_tokens.get(agent_id, 0)
    return total



# BFCL model registration (required by bfcl_eval.ast_checker)
try:
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    _BFCL_MODEL_ID = "Qwen/Qwen3-8B"
    if _BFCL_MODEL_ID not in MODEL_CONFIG_MAPPING:
        _template = MODEL_CONFIG_MAPPING.get("qwen3-8b")
        if _template:
            MODEL_CONFIG_MAPPING[_BFCL_MODEL_ID] = ModelConfig(
                model_name=_BFCL_MODEL_ID,
                display_name=_BFCL_MODEL_ID,
                url=_template.url,
                org=_template.org,
                license=_template.license,
                model_handler=_template.model_handler,
                is_fc_model=True,
                underscore_to_dot=False,
            )
except ImportError:
    pass  # bfcl_eval not installed

class GraphType(Enum):
    SUMMARIZE = "summarize"
    AGGREGATE = "Independent"
    REFLECT = "Sequential"
    DEBATE = "debate"
    LLM_AGG = "llm_agg"
    DEBATE_LLM_AGG = "debate_llm_agg"

    SELF_REFINE = "self_refine"
    HIERARCHICAL = "Centralized"
    COUNCIL = "Decentralized"
    SELECTOR_DYNAMIC = "Dynamic"
    SINGLE_AGENT = "Single_Agent"
    SINGLE_AGENT_COT = "Single_Agent_CoT"
    SINGLE_AGENT_TPE = "Single_Agent_TPE"
    SINGLE_AGENT_SPO = "Single_Agent_SPO"

class AgentType(Enum):
    PREDICTOR = "predictor"
    SUMMARIZER = "summarizer"
    AGGREGATOR = "aggregator"
    REFLECTOR = "reflector"
    DEBATOR = "debator"
    LLM_AGG = "llm_agg"
    MATH_EXPERT = "math_expert"
    MATH_ANALYST = "math_analyst"
    PROGRAMMING_EXPERT = "programming_expert"
    PROJECT_MANAGER = "project_manager"
    PROGRAMMER = "programmer"
    TEST_ANALYST = "test_analyst"
    BUG_FIXER = "bug_fixer"
    KNOWLEDGEABLE_EXPERT = "knowledgeable_expert"
    CRITIC = "critic"
    PSYCHOLOGIST = "psychologist"
    HISTORIAN = "historian"
    REFINER = "refiner"  
    MATH_AGENT = "math_agent"
    SCIENCE_AGENT = "science_agent"
    CODE_AGENT = "code_agent"
    TASK_SUMMARIZER = "task_summarizer"

class TaskType(Enum):
    MATH = "math"               
    MATH_CHOICE = "math_choice"    
    REASONING_CHOICE = "reasoning_choice"
    CODE = "code"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    QA = "qa"              

DATASET_CONFIG = {
    "math": {
        "path": "dataset/math-500/math_500.jsonl",
        "task_type": TaskType.MATH,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False, 
    },
    "aime": {
        "path": "dataset/aime-2025/aime2025.jsonl",
        "task_type": TaskType.MATH,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "agi": {
        "path": "dataset/AGIEval_MATH/test.jsonl",
        "task_type": TaskType.MATH,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "aqua": {
        "path": "dataset/aqua/aqua.jsonl",
        "task_type": TaskType.MATH_CHOICE,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "gpqa": {
        "path": "dataset/gpqa/gpqa_diamond.jsonl",
        "task_type": TaskType.REASONING_CHOICE,
        "problem_key": "question",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "mbpp": {
        "path": "dataset/mbpp/sanitized-mbpp.jsonl",
        "task_type": TaskType.CODE,
        "problem_key": "prompt",
        "answer_key": None,
        "default_use_judge": True,
    },
    "humaneval": {
        "path": "dataset/humaneval-et/HumanEval_ET.jsonl",
        "task_type": TaskType.CODE,
        "problem_key": "prompt",
        "answer_key": None,
        "default_use_judge": True,
    },
    "hotpotqa": {
        "path": "dataset/hotpotqa/hotpotqa_eval_500.jsonl",
        "task_type": TaskType.QA,
        "problem_key": "question",
        "answer_key": "answer",
        "default_use_judge": False,
    },
    "bfcl_v3": {
        "path": "dataset/bfcl_v3/bfcl_eval_500.jsonl",
        "task_type": TaskType.FUNCTION_CALLING,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": True,
    },
    "livecodebench_v6": {
        "path": "dataset/livecodebench_v6/lcb_eval_131.jsonl",
        "task_type": TaskType.CODE_GENERATION,
        "problem_key": "problem",
        "answer_key": "answer",
        "default_use_judge": True,
    }
}

def create_main_client() -> ModelClient:
    return get_model(DEFAULT_AGENT_MODEL)


def create_judge_client() -> ModelClient:
    return get_model(DEFAULT_EVALUATOR_MODEL)


def create_evaluator_client() -> ModelClient:
    return get_model(DEFAULT_EVALUATOR_MODEL)


aclient = create_main_client()
bclient = create_judge_client()


def get_default_use_judge(dataset: str) -> bool:
    config = DATASET_CONFIG.get(dataset)
    if not config:
        return False
    return config.get("default_use_judge", False)
