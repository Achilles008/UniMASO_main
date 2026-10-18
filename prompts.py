from config import AgentType, GraphType, TaskType

ROLE_DESCRIPTIONS = {
    AgentType.PREDICTOR: {
        TaskType.MATH: "Primary solver that provides step-by-step mathematical reasoning and final answer.",
        TaskType.MATH_CHOICE: "Primary solver that provides step-by-step mathematical reasoning and final answer.",
        TaskType.REASONING_CHOICE: "Primary solver that provides step-by-step reasoning and final answer.",
        TaskType.CODE: "Primary solver that provides step-by-step reasoning and final answer.",


    TaskType.CODE_GENERATION: "Competitive programming agent that reads a problem statement, designs an efficient algorithm, and writes clean Python code that reads from stdin and prints to stdout. Must handle all edge cases and pass hidden test cases.",
        TaskType.FUNCTION_CALLING: "Tool-calling agent that selects the most appropriate function and accurately extracts parameter values from the user query.",
        TaskType.QA: "Multi-hop QA agent that reads multiple documents, identifies bridging connections across them, and extracts the correct short answer.",
    },
    AgentType.REFLECTOR: {
        TaskType.MATH: "Critical reviewer that identifies errors in previous solutions and provides corrected reasoning.",
        TaskType.MATH_CHOICE: "Critical reviewer that identifies errors in previous solutions and provides corrected reasoning.",
        TaskType.REASONING_CHOICE: "Critical reviewer that identifies errors in previous solutions and provides corrected reasoning.",
        TaskType.CODE: "Critical reviewer that identifies errors in previous solutions and provides corrected reasoning.",
        TaskType.FUNCTION_CALLING: "Critical reviewer that checks function selection, parameter completeness, value accuracy, and type correctness, then provides the corrected function call.",
        TaskType.QA: "Critical reviewer that verifies cross-document evidence chains, identifies missing or misread information, and provides the corrected answer.",
        TaskType.CODE_GENERATION: "Code reviewer that spots bugs, logic errors, and edge-case gaps in the previous solution, then outputs only the corrected Python code.",
    },
    AgentType.SUMMARIZER: "Extractor of relevant information from context to assist problem solving.",
    
    AgentType.DEBATOR: "Evaluator of multiple solutions that synthesizes the best approach from competing answers.",
    AgentType.LLM_AGG: "Integrator that compares multiple candidate solutions and produces a single authoritative answer.",
    AgentType.AGGREGATOR: "Voting mechanism that selects the most frequent answer from multiple predictors.",
    AgentType.MATH_EXPERT: "You are a math expert.",
    AgentType.MATH_ANALYST: "You are a mathematical analyst.",
    AgentType.PROGRAMMING_EXPERT: "You are a programming expert.",
    AgentType.PROJECT_MANAGER: "You are a project manager.",
    AgentType.PROGRAMMER: "You are a programming expert.",
    AgentType.TEST_ANALYST: "You are a test analyst.",
    AgentType.BUG_FIXER: "You are a bug fixer.",

    AgentType.KNOWLEDGEABLE_EXPERT: "You are a knowledgeable expert in question answering.",
    AgentType.CRITIC: "You are an excellent critic.",
    AgentType.PSYCHOLOGIST: "You are a psychologist.",
    AgentType.HISTORIAN: "You research and analyze cultural, economic, political, and social events in the past.",
}

def get_role_description(agent_type: AgentType, task_type: TaskType = None) -> str:
    desc = ROLE_DESCRIPTIONS.get(agent_type, "General reasoning agent")
    if isinstance(desc, dict) and task_type:
        return desc.get(task_type, list(desc.values())[0])
    return desc if isinstance(desc, str) else list(desc.values())[0]

# Reflect-AGIEval short-answer templates.
REFLECT_AGI_TEMPLATES = {
    TaskType.MATH_CHOICE: {
        AgentType.PREDICTOR: (
            "Solve the following problem. Output only the final answer; do not include analysis, derivation, explanation, or Markdown.\n"
            "If the problem provides answer choices, output only the correct option; otherwise, output only the answer requested by the problem.\n"
            "The final answer must appear exactly once between the <answer> and </answer> tags.\n\n"
            "Question: {question}\n\n<answer>"
        ),
        AgentType.REFLECTOR: (
            "Check whether the previous agent's answer is correct. Output only the corrected final answer; do not include your critique, reasoning, explanation, or Markdown.\n"
            "If the previous answer is correct, preserve it exactly; if it is incorrect, output only the correct answer.\n"
            "If the problem provides answer choices, output only the correct option; otherwise, output only the answer requested by the problem.\n"
            "The final answer must appear exactly once between the <answer> and </answer> tags.\n\n"
            "Question: {question}\nPrevious agent's answer: {context}\n\n<answer>"
        ),
    },
    TaskType.REASONING_CHOICE: {
        AgentType.PREDICTOR: (
            "Solve the following problem. Output only the final answer; do not include analysis, derivation, explanation, or Markdown.\n"
            "Output only the correct option or the shortest answer requested by the problem.\n"
            "The final answer must appear exactly once between the <answer> and </answer> tags.\n\n"
            "Question: {question}\n\n<answer>"
        ),
        AgentType.REFLECTOR: (
            "Check whether the previous agent's answer is correct. Output only the corrected final answer; do not include your critique, reasoning, explanation, or Markdown.\n"
            "If the previous answer is correct, preserve it exactly; if it is incorrect, output only the correct answer.\n"
            "Output only the correct option or the shortest answer requested by the problem.\n"
            "The final answer must appear exactly once between the <answer> and </answer> tags.\n\n"
            "Question: {question}\nPrevious agent's answer: {context}\n\n<answer>"
        ),
    },
}

GPQA_REFLECT_TEMPLATES = {
    "predictor_initial": (
        "Solve the following multiple-choice question.\n\n"
        "Reason through the problem in 3?5 concise steps.\n"
        "Do not omit the key reasoning needed to determine the answer.\n\n"
        "Finally, output only the selected option letter in this format:\n"
        "<answer>A/B/C/D</answer>\n\n"
        "Question:\n{question}"
    ),
    "predictor_revision": (
        "Solve the following multiple-choice question.\n\n"
        "A previous attempt is provided below. Carefully check it for possible errors "
        "and independently solve the problem again. Do not blindly accept the previous answer.\n\n"
        "Reason through the problem in 3?5 concise steps.\n\n"
        "Finally, output only the corrected option letter in this format:\n"
        "<answer>A/B/C/D</answer>\n\n"
        "Question:\n{question}\n\n"
        "Previous attempt:\n{context}"
    ),
    "reflector": (
        "Check the Predictor's reasoning and answer for the following question.\n\n"
        "Determine whether the answer is correct. If it is incorrect, correct it.\n\n"
        "Output only the final option letter. Do not output explanations, "
        "reasoning, criticism, or Markdown.\n\n"
        "The answer must appear exactly once in this format:\n"
        "<answer>A/B/C/D</answer>\n\n"
        "Question:\n{question}\n\n"
        "Predictor's attempt:\n{context}"
    ),
}

def get_gpqa_reflect_template(template_name: str):
    return GPQA_REFLECT_TEMPLATES.get(template_name)

AGI_REFLECT_TEMPLATES = {
    "predictor_initial": (
        "Solve the following mathematics problem.\n\n"
        "Reason through it in 3-5 concise steps and check the result carefully.\n\n"
        "Output the final answer exactly once:\n"
        "<answer>YOUR_FINAL_ANSWER</answer>\n\n"
        "Question:\n{question}"
    ),
    "predictor_revision": (
        "Solve the following mathematics problem again.\n\n"
        "A previous attempt is provided below. Check it carefully and correct it "
        "if necessary. Do not blindly accept it.\n\n"
        "Reason through it in 3-5 concise steps.\n\n"
        "Output the final answer exactly once:\n"
        "<answer>YOUR_FINAL_ANSWER</answer>\n\n"
        "Question:\n{question}\n\n"
        "Previous attempt:\n{context}"
    ),
    "reflector": (
        "Review the previous mathematical solution and independently verify the answer.\n\n"
        "Correct it if necessary. Use 3-5 concise reasoning steps.\n\n"
        "Output the verified final answer exactly once:\n"
        "<answer>YOUR_FINAL_ANSWER</answer>\n\n"
        "Question:\n{question}\n\n"
        "Previous solution:\n{context}"
    ),
}

def get_agi_reflect_template(template_name: str):
    return AGI_REFLECT_TEMPLATES.get(template_name)

def get_reflect_agi_template(task_type: TaskType, agent_type: AgentType):
    return REFLECT_AGI_TEMPLATES.get(task_type, {}).get(agent_type)

AGENT_TEMPLATES = {
    TaskType.MATH: {
        AgentType.PREDICTOR: (
            "Analyze the problem in 2-3 key steps. Show your final answer between <answer> and </answer> tags.\n"
            "{context}\n"
            "Question: {question}\nReasoning:"
        ),
        AgentType.SUMMARIZER: (
            "Based on the math problem, retrieve relevant information from the provided context that is ONLY "
            "helpful in solving the problem. Do not repeat irrelevant context. Start with 'Summary: '.\n"
            "Question: {question}\nContext: {context}\nSummary:"
        ),
        AgentType.REFLECTOR: (
            "Please review the solution above and criticize where it might be wrong. "
            "Show your final answer bracketed between <answer> and </answer>.\n"
            "Question: {question}\nSolution: {context}\nFeedback: indicating the reflection of the solution given the question.\nCorrect answer:"
        ),
        AgentType.DEBATOR: (
            "These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale, finish by giving an updated answer. "
            "Show your final answer bracketed between <answer> and </answer>.\n"
            "Question: {question}\nSolutions: {context}\nAnswer:"
        ),
        AgentType.LLM_AGG: (
            "You are given several candidate solutions to the following math problem. "
            "Carefully compare their reasoning and answers, then produce a single final solution. "
            "Show your final answer bracketed between <answer> and </answer> tags.\n"
            "Question: {question}\n\n"
            "Candidate solutions:\n{context}\n"
            "Answer:"
        ),
    },
    
    TaskType.MATH_CHOICE: {
        AgentType.PREDICTOR: (
            "Analyze the problem in 2-3 key steps. Show your final option between <answer> and </answer> tags.\n"
            "{context}\n"
            "Question: {question}\nReasoning:"
        ),
        AgentType.SUMMARIZER: (
            "Based on the math problem, retrieve relevant information from the provided context that is ONLY "
            "helpful in solving the problem. Do not repeat irrelevant context. Start with 'Summary: '.\n"
            "Question: {question}\nContext: {context}\nSummary:"
        ),
        AgentType.REFLECTOR: (
            "Please review the solution above and criticize where it might be wrong. "
            "Show your final option with one letter bracketed between <answer> and </answer>.\n"
            "Question: {question}\nSolution: {context}\nFeedback: indicating the reflection of the solution given the question.\nCorrect answer:"
        ),
        AgentType.DEBATOR: (
            "These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale, finish by giving an updated answer. "
            "Show your final option with one letter bracketed between <answer> and </answer>.\n"
            "Question: {question}\nSolutions: {context}\nAnswer:"
        ),
        AgentType.LLM_AGG: (
            "You are given several candidate solutions to the following math problem. "
            "Carefully compare their reasoning and answers, then produce a single final solution. "
            "Show your final option with one letter bracketed between <answer> and </answer> tags.\n"
            "Question: {question}\n\n"
            "Candidate solutions:\n{context}\n"
            "Answer:"
        ),
    },
    
    TaskType.REASONING_CHOICE: {
        AgentType.PREDICTOR: (
            "Analyze the problem in 2-3 key steps. Show your final option between <answer> and </answer> tags.\n"
            "{context}\n"
            "Question: {question}\nReasoning:"
        ),
        AgentType.SUMMARIZER: (
            "Based on the multi-choice problem, retrieve relevant information from the provided context that is ONLY "
            "helpful in solving the problem. Do not repeat irrelevant context. Start with 'Summary: '.\n"
            "Question: {question}\nContext: {context}\nSummary:"
        ),
        AgentType.REFLECTOR: (
            "Please review the solution above and criticize where it might be wrong. "
            "Show your final option with one letter bracketed between <answer> and </answer>.\n"
            "Question: {question}\nSolution: {context}\nFeedback: indicating the reflection of the solution given the question.\nCorrect answer:"
        ),
        AgentType.DEBATOR: (
            "These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale, finish by giving an updated answer. "
            "Show your final option with one letter bracketed between <answer> and </answer>.\n"
            "Question: {question}\nSolutions: {context}\nAnswer:"
        ),
        AgentType.LLM_AGG: (
            "You are given several candidate solutions to the following multi-choice problem. "
            "Carefully compare their reasoning and answers, then produce a single final solution. "
            "Show your final option with one letter bracketed between <answer> and </answer> tags.\n"
            "Question: {question}\n\n"
            "Candidate solutions:\n{context}\n"
            "Answer:"
        ),
    },
    
    TaskType.CODE: {
        AgentType.PREDICTOR: (
            "You are an expert Python programmer. Write clean, efficient, and well-documented code.\n"
            "Problem: {question}\n"
            "{context}\n"
            "Provide a complete and correct code implementation in python.\n"
            "Code:"
        ),
        AgentType.SUMMARIZER: (
            "Based on the programming problem, extract key requirements and constraints.\n"
            "Problem: {question}\n"
            "Context: {context}\n"
            "Summary of key requirements:"
        ),
        AgentType.REFLECTOR: (
            "Review the following Python code and identify potential bugs or improvements.\n"
            "Problem: {question}\n"
            "Code: {context}\n"
            "Provide an improved version in python at the end of your response.\n"
            "Improved Code:"
        ),
        AgentType.DEBATOR: (
            "Compare these Python solutions and synthesize the best approach.\n"
            "Problem: {question}\n"
            "Solutions:\n{context}\n"
            "Provide the best solution in python at the end of your response.\n"
            "Final Code:"
        ),
        AgentType.LLM_AGG: (
            "Analyze several Python solutions and produce the most robust one.\n"
            "Problem: {question}\n\n"
            "Candidate solutions:\n{context}\n"
            "Provide your final solution in python at the end of your response.\n"
            "Final Code:"
        ),
    },

    TaskType.QA: {
        AgentType.PREDICTOR: (
            "Read all documents carefully. Answer the question by finding and connecting information across multiple documents. "
            "Think step by step: (1) identify relevant documents, (2) extract key facts, (3) bridge information across documents, "
            "(4) derive the final answer. Put your step-by-step reasoning BEFORE the answer tags. The <answer>...</answer> tags must contain ONLY the final short answer (1-5 words), with NO reasoning inside.\n"
            "{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        AgentType.SUMMARIZER: (
            "Read the provided documents and extract only the facts relevant to answering the question. "
            "Do not repeat irrelevant context. Start with 'Summary: '.\n"
            "Question: {question}\nContext: {context}\nSummary:"
        ),
        AgentType.REFLECTOR: (
            "Review the previous answer against all provided documents and the question below. "
            "Check for: (1) missing cross-document connections, (2) incorrect information extraction, "
            "(3) incomplete reasoning chains. Put your review reasoning BEFORE the answer tags. The <answer>...</answer> tags must contain ONLY the corrected short answer (1-5 words), with NO reasoning inside.\n"
            "{question}\n\n"
            "Previous answer: {context}\n\n"
            "Review and corrected answer:"
        ),
        AgentType.DEBATOR: (
            "Compare these answers to the multi-hop QA question. Examine each answer's evidence grounding "
            "and reasoning. Synthesize the best short answer (1-5 words) within <answer>...</answer> tags. Do NOT include reasoning inside the tags.\n"
            "Question: {question}\n"
            "Candidate answers:\n{context}\n"
            "Best answer:"
        ),
        AgentType.LLM_AGG: (
            "You are given several candidate answers to a multi-hop QA question. "
            "Carefully compare their evidence grounding and reasoning, then produce a single final answer. "
            "Output the final answer within <answer>...</answer> tags.\n"
            "Question: {question}\n\n"
            "Candidate answers:\n{context}\n"
            "Final answer:"
        ),
    },


TaskType.CODE_GENERATION: {
    AgentType.PREDICTOR: (
        "You are a competitive programming expert. Solve the problem below by writing clean, correct Python 3 code.\n"
        "CRITICAL RULES:\n"
        "1. Read input from sys.stdin (standard input)\n"
        "2. Print output to sys.stdout (standard output)\n"
        "3. Output ONLY the Python code inside a ```python code block\n"
        "4. Do NOT include explanations, analysis, or markdown outside the code block\n"
        "5. The code must handle all edge cases within the given constraints\n\n"
        "{context}\n\nProblem:\n{question}\n\n"
        "Your solution (```python block only):"
    ),
    AgentType.SUMMARIZER: (
        "Extract the core algorithmic approach from this code solution. "
        "What is the key insight, data structure, or technique used? "
        "Keep it to one sentence, max 30 words.\n"
        "Code: {context}\nQuestion: {question}\nSummary:"
    ),
    AgentType.REFLECTOR: (
        "You are a strict code reviewer. The code below attempts to solve the given problem.\n"
        "Find any bugs, logic errors, edge cases, or performance issues.\n"
        "Then OUTPUT ONLY the fully corrected Python code inside a ```python code block.\n"
        "DO NOT write analysis, explanations, or lists of issues. ONLY output the corrected code.\n\n"
        "Problem:\n{question}\n\n"
        "Current code:\n{context}\n\n"
        "Corrected code (```python block only):"
    ),
    AgentType.DEBATOR: (
        "Compare the following candidate code solutions for the given problem.\n"
        "Evaluate each on: correctness, efficiency, edge case handling, and code clarity.\n"
        "Choose the best one and output ONLY that code in a ```python block.\n\n"
        "Problem: {question}\n"
        "Candidates:\n{context}\n\n"
        "Best solution (```python block only):"
    ),
    AgentType.LLM_AGG: (
        "You are given several candidate code solutions for the following problem.\n"
        "Compare them carefully, identify the most correct and efficient one,\n"
        "and output a single final Python code in a ```python block.\n\n"
        "Problem: {question}\n\n"
        "Candidates:\n{context}\n\n"
        "Final code (```python block only):"
    ),
},

    TaskType.FUNCTION_CALLING: {
        AgentType.PREDICTOR: (
            "Based on the user query, select the most appropriate function from the available tools and provide accurate parameter values. "
            "Pay close attention to: (1) exact function name, (2) required vs optional parameters, (3) correct parameter types (integer, float, string, boolean, list, dict), "
            "and (4) extracting values precisely as they appear in the query.\n"
            "End with one fenced ```json``` block containing a non-empty list of canonical function-call dicts.\n"
            "{context}\n"
            "User Query: {question}\n"
            "Function Call:"
        ),
        AgentType.SUMMARIZER: (
            "Extract the core intent from the user query: what operation is requested, what values are mentioned, and which parameters they correspond to. "
            "Ignore irrelevant details. End with one fenced ```json``` block containing a non-empty list of canonical function-call dicts.\n"
            "Question: {question}\nContext: {context}\nFunction Call:"
        ),
        AgentType.REFLECTOR: (
            "Review the function call above for correctness. Check:\n"
            "1. Function name: is the correct function selected for this query?\n"
            "2. Parameters: are all required parameters present? Are any hallucinated parameters added?\n"
            "3. Values: are parameter values extracted accurately from the user query?\n"
            "4. Types: are parameter types correct (e.g., integer not string, float not integer)?\n"
            "If any errors are found, provide the corrected function call as a non-empty list of canonical function-call dicts inside one fenced ```json``` block. "
            "If the call is already correct, repeat it as-is.\n\n"
            "Question: {question}\n"
            "Previous call: {context}\n"
            "Corrected call:"
        ),
        AgentType.DEBATOR: (
            "Compare the following function call candidates for the given user query. "
            "Evaluate each on: function selection correctness, parameter completeness, value accuracy, and type correctness. "
            "Choose the best one and output it as a non-empty list of canonical function-call dicts inside one fenced ```json``` block.\n"
            "Question: {question}\nCalls:\n{context}\nBest call:"
        ),
        AgentType.LLM_AGG: (
            "You are given several candidate function calls for the following user query. "
            "Compare them carefully, identify the most accurate one, and output a single final function call as a non-empty list of canonical function-call dicts "
            "inside one fenced ```json``` block.\n"
            "Question: {question}\n\n"
            "Candidate calls:\n{context}\nFinal call:"
        ),
    },
}

def get_agent_template(task_type: TaskType, agent_type: AgentType) -> str:
    """获取Agent模板"""
    return AGENT_TEMPLATES.get(task_type, {}).get(agent_type)

COMPRESS_PROMPTS = {
    TaskType.MATH: (
        "Below is a solution to a math problem. "
        "Summarize the key reasoning steps and the final answer in at most 30 words.\n\n"
        "{raw}\n\nSummary:"
    ),
    TaskType.MATH_CHOICE: (
        "Below is a solution to a math problem. "
        "Summarize the key reasoning steps and the final answer in at most 30 words.\n\n"
        "{raw}\n\nSummary:"
    ),
    TaskType.REASONING_CHOICE: (
        "Below is a solution to a multi-choice problem. "
        "Summarize the key reasoning steps and the final answer in at most 30 words.\n\n"
        "{raw}\n\nSummary:"
    ),
    TaskType.CODE: (
        "Summarize the key logic of this Python code in at most 50 words:\n\n"
        "{raw}\n\nSummary:"
    ),
    TaskType.QA: (
        "Compress the following QA output into a concise summary (no more than 30 words), "
        "keeping only the final answer and key evidence facts.\n\n"
        "Raw QA output: {raw}\n\n"
        "Compressed summary:"
    ),


    TaskType.CODE_GENERATION: (
        "Summarize this code solution: what algorithm does it use? Max 20 words.\n\n"
        "{raw}\n\nSummary:"
    ),
    TaskType.FUNCTION_CALLING: (
        "Summarize the following function-calling analysis for a downstream synthesizer. "
        "Do not invent, repair, or reformat a function call. Preserve exact function names, "
        "parameter names, values, and types when they appear. If the text is only analysis, "
        "keep it as analysis and do not create JSON.\n\n"
        "{raw}\n\nOutput:"
    ),
}

# === Simplified templates for non-heterogeneous mode (original UniMASO prompt-only) ===
# These omit model pool, trajectory, model selection steps, and lambda_cost.
PROMPT_OPTIMIZE_TEMPLATE_NO_HETERO = {

    TaskType.FUNCTION_CALLING: """
You are optimizing a prompt for an agent in a multi-agent reasoning system.

Agent Type: {agent_type}
System Role: {role_description}

Sample Execution Traces (Question + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

DECISION FRAMEWORK:
Step 1 - Diagnose function call quality: What is wrong?
  - Wrong JSON format ("function"/"parameters" instead of "name"/"arguments")? Wrong function name? Missing required parameters? Incorrect parameter types? Hallucinated values?
Step 2 — Output the optimized prompt.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Analysis of function call quality.</analyse>
<prompt>The complete optimized prompt. Preserve {{question}} and {{context}} placeholders. CRITICAL: The JSON output format specification — e.g., "Output ONLY a JSON array: [{{"name": "...", "arguments": {{...}}}}]" — MUST be kept verbatim. Do NOT rephrase, rewrite, or remove the format instruction. The exact field names "name" and "arguments" are mandatory for correct system parsing.</prompt>
""",

    TaskType.CODE: """
You are optimizing a prompt for an agent in a multi-agent code generation system.

Agent Type: {agent_type}
System Role: {role_description}

Sample Execution Traces (Problem + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

DECISION FRAMEWORK:
Step 1 — Diagnose bugs: What specific coding errors appear in the sample outputs?
Step 2 — Output the optimized prompt.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Detailed analysis of bugs found.</analyse>
<prompt>The complete, optimized prompt (keep it under 300 words). Preserve placeholders.</prompt>
""",

    TaskType.MATH_CHOICE: """
You are optimizing a prompt for an agent in a multiple-choice reasoning system.

Agent Type: {agent_type}
System Role: {role_description}

Sample Execution Traces (Question + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

DECISION FRAMEWORK:
Step 1 - Diagnose function call quality: What is wrong?
  - Wrong JSON format ("function"/"parameters" instead of "name"/"arguments")? Wrong function name? Missing required parameters? Incorrect parameter types? Hallucinated values?
Step 2 — Output the optimized prompt.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Analysis of function call quality.</analyse>
<prompt>The complete optimized prompt. Preserve {{question}} and {{context}} placeholders.</prompt>
""",

    TaskType.REASONING_CHOICE: """
You are optimizing a prompt for an agent in a problem-solving system.

Agent Type: {agent_type}
System Role: {role_description}

Sample Execution Traces (Question + Context + Agent Output)
{samples}
Requirements:
{requirements}
Reference prompt:
{prompt}

### Analysis Focus:
1. Truncation Check: Are outputs cut off? If yes, the prompt may be too demanding — simplify first.
2. Reasoning Quality: Are answers correct? If not, is guidance missing?

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Analysis of function call quality.</analyse>
<prompt>The complete optimized prompt. Preserve {{question}} and {{context}}.</prompt>
""",

    TaskType.QA: """
You are optimizing a prompt for an agent in a multi-hop question answering system.

Agent Type: {agent_type}
System Role: {role_description}

Sample Execution Traces (Documents + Question + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

DECISION FRAMEWORK:
Step 1 — Diagnose output quality: What QA-specific problems appear?
  - Missing cross-document connections? Incomplete evidence chains? Answers too long or off-target?
Step 2 — Output the optimized prompt.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Analysis of QA output quality.</analyse>
<prompt>The complete optimized prompt. Preserve {{question}} and {{context}} placeholders.</prompt>
""",

    TaskType.CODE_GENERATION: """
You are optimizing a prompt for an agent in a competitive programming system.

Agent Type: {agent_type}
System Role: {role_description}

Sample Execution Traces (Problem + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

DECISION FRAMEWORK:
Step 1 - Diagnose code quality: What specific issues appear in the sample outputs?
  - Algorithm errors? Wrong I/O format? Missing edge cases? Inefficient solutions?
Step 2 - Output the optimized prompt.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Analysis of code quality issues.</analyse>
<prompt>The complete optimized prompt. Preserve {{question}} and {{context}} placeholders.</prompt>
""",

    TaskType.MATH: """
You are optimizing a prompt for an agent in a multi-agent reasoning system.

Agent Type: {agent_type}
System Role: {role_description}

Sample Execution Traces (Question + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

DECISION FRAMEWORK:
Step 1 - Diagnose function call quality: What is wrong?
  - Wrong JSON format ("function"/"parameters" instead of "name"/"arguments")? Wrong function name? Missing required parameters? Incorrect parameter types? Hallucinated values?
Step 2 — Output the optimized prompt.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Analysis of function call quality.</analyse>
<prompt>The complete optimized prompt. Preserve {{question}} and {{context}} placeholders.</prompt>
""",
}


PROMPT_OPTIMIZE_TEMPLATE = {
    TaskType.MATH: """
You are optimizing a prompt AND selecting a model for an agent in a multi-agent reasoning system.

Agent Type: {agent_type}
System Role: {role_description}
Current Model: {current_model}

============================== MODEL POOL ==============================
{model_pool_section}
=======================================================================

{trajectory_section}

Sample Execution Traces (Question + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

DECISION FRAMEWORK:
Step 1 - Diagnose function call quality: What is wrong?
  - Wrong JSON format ("function"/"parameters" instead of "name"/"arguments")? Wrong function name? Missing required parameters? Incorrect parameter types? Hallucinated values?
Step 2 — Assess model adequacy: Is {current_model} overkill, adequate, or insufficient?
  - Overkill (quality already good) → consider downgrading to save cost.
  - Insufficient (errors, shallow reasoning) → consider upgrading for better quality.
  - Adequate → keep current model, focus on prompt improvement.
Step 3 — Output the optimized prompt and selected model.

Rules (cost-benefit balance, λ={lambda_cost}):
- Optimize the prompt first. Then decide if model change is needed.
- Upgrade when quality is clearly insufficient. Downgrade when quality is clearly excessive.
- You may jump at most ONE tier (Weak↔Medium or Medium↔Strong), unless lambda=0.
- If outputs contain clear errors that prompt fixes cannot solve, do NOT downgrade.
- Both upgrade and downgrade are valid choices — let evidence decide.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Analysis of output quality and model adequacy.</analyse>
<model_decision>One of: Keep | Upgrade to <name> | Downgrade to <name>. One sentence why.</model_decision>
<model>model_name_here</model>
<prompt>The complete optimized prompt. Preserve {{question}} and {{context}} placeholders.</prompt>
""",

    TaskType.CODE: """
You are optimizing a prompt AND selecting a model for an agent in a multi-agent code generation system.

Agent Type: {agent_type}
System Role: {role_description}
Current Model: {current_model}

============================== MODEL POOL ==============================
{model_pool_section}
=======================================================================

{trajectory_section}

Sample Execution Traces (Problem + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

DECISION FRAMEWORK:
Step 1 — Diagnose bugs: What specific coding errors appear in the sample outputs?
Step 2 — Assess model adequacy: Is {current_model} overkill, adequate, or insufficient?
  - Overkill (clean code without errors) → consider downgrading to save cost.
  - Insufficient (bugs, syntax errors) → consider upgrading for better quality.
  - Adequate → keep current, focus on prompt fixes.
Step 3 — Output the optimized prompt and selected model.

Rules (cost-benefit balance, λ={lambda_cost}):
- Fix the prompt first. Then decide if model change is needed.
- Upgrade when the model clearly cannot handle the coding task. Downgrade when it's clearly overkill.
- You may jump at most ONE tier per round, unless lambda=0.
- If outputs contain unfixable bugs (logic errors beyond prompt fixes), do NOT downgrade.
- Both upgrade and downgrade are valid — let evidence and cost trade-off decide.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Detailed analysis of bugs found and model adequacy assessment.</analyse>
<model_decision>One of: Keep | Upgrade to <name> | Downgrade to <name>. One sentence why.</model_decision>
<model>model_name_here</model>
<prompt>The complete, optimized prompt (keep it under 300 words). Preserve placeholders.</prompt>
""",

    TaskType.MATH_CHOICE: """
You are optimizing a prompt AND selecting a model for an agent in a multiple-choice reasoning system.

Agent Type: {agent_type}
System Role: {role_description}
Current Model: {current_model}

============================== MODEL POOL ==============================
{model_pool_section}
=======================================================================

{trajectory_section}

Sample Execution Traces (Question + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

DECISION FRAMEWORK:
Step 1 - Diagnose function call quality: What is wrong?
  - Wrong JSON format ("function"/"parameters" instead of "name"/"arguments")? Wrong function name? Missing required parameters? Incorrect parameter types? Hallucinated values?
Step 2 — Assess model adequacy: Is {current_model} overkill, adequate, or insufficient?
  - Overkill → consider downgrading to save cost.
  - Insufficient (errors, shallow reasoning) → consider upgrading for better quality.
  - Adequate → keep current, focus on prompt improvement.
Step 3 — Output the optimized prompt and selected model.

Rules (cost-benefit balance, λ={lambda_cost}):
- Optimize the prompt first. Then decide if model change is needed.
- Jump at most ONE tier per round, unless lambda=0.
- Upgrade when quality is insufficient. Downgrade when quality is excessive.
- Do NOT downgrade if outputs have unfixable errors that better prompting cannot solve.
- Both upgrade and downgrade are equally valid — let evidence and cost trade-off decide.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Analysis of output quality and model adequacy.</analyse>
<model_decision>One of: Keep | Upgrade to <name> | Downgrade to <name>. One sentence why.</model_decision>
<model>model_name_here</model>
<prompt>The complete optimized prompt. Preserve {{question}} and {{context}} placeholders.</prompt>
""",

    TaskType.REASONING_CHOICE: """
You are optimizing a prompt AND selecting a model for an agent in a problem-solving system.

Agent Type: {agent_type}
System Role: {role_description}
Current Model: {current_model}

============================== MODEL POOL ==============================
{model_pool_section}
=======================================================================

{trajectory_section}

Sample Execution Traces (Question + Context + Agent Output)
{samples}
Requirements:
{requirements}
Reference prompt:
{prompt}

### Analysis Focus:
1. Truncation Check: Are outputs cut off? If yes, the prompt may be too demanding — simplify first.
2. Reasoning Quality: Are answers correct? If not, is guidance missing, or is {current_model} too weak?
3. Cost-Benefit: After optimizing the prompt, weigh quality gain/loss against cost change. A cheaper model that scores slightly lower may be the better choice; an expensive model that fixes major errors may be worth it.

### Model Selection — Cost-Benefit Framework:
- λ = {lambda_cost} balances cost vs quality. Higher λ = cost matters more.
- Upgrading increases cost (penalty in scoring). Do it ONLY when quality gain justifies the cost.
- Downgrading decreases cost (reward in scoring). Do it when quality stays acceptable.
- Prompt optimization comes FIRST: fix the prompt, THEN decide if model change is needed.
- The scoring formula already accounts for cost — trust the trade-off. No default direction.

### Model Rules:
- Jump at most ONE tier per round, unless lambda=0.
- Do NOT change model if the current model works well with the optimized prompt.
- Consider BOTH upgrade and downgrade based on evidence, not preference.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Analysis of output quality and model adequacy.</analyse>
<model_decision>One of: Keep | Upgrade to <name> | Downgrade to <name>. One sentence why.</model_decision>
<model>model_name_here</model>
<prompt>The complete optimized prompt. Preserve {{question}} and {{context}}.</prompt>
""",
    TaskType.QA: """
You are optimizing a prompt AND selecting a model for an agent in a multi-hop question answering system.

Agent Type: {agent_type}
System Role: {role_description}
Current Model: {current_model}

============================== MODEL POOL ==============================
{model_pool_section}
=======================================================================

{trajectory_section}

Sample Execution Traces (Documents + Question + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

DECISION FRAMEWORK:
Step 1 — Diagnose output quality: What QA-specific problems appear?
  - Missing cross-document connections? Incomplete evidence chains? Answers too long or off-target?
Step 2 — Assess model adequacy: Is {current_model} overkill, adequate, or insufficient?
  - Overkill (accurate, concise answers) -> consider downgrading to save cost.
  - Insufficient (misses connections, fabricates answers) -> consider upgrading.
  - Adequate -> keep current, focus on prompt improvement.
Step 3 — Output the optimized prompt and selected model.

Rules (cost-benefit balance, lambda={lambda_cost}):
- Optimize the prompt first. Then decide if model change is needed.
- Upgrade when the model cannot handle multi-hop reasoning. Downgrade when clearly overkill.
- Jump at most ONE tier per round, unless lambda=0.
- If outputs contain hallucinated facts, do NOT downgrade.
- Both upgrade and downgrade are valid — let evidence and cost trade-off decide.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Analysis of QA output quality and model adequacy.</analyse>
<model_decision>One of: Keep | Upgrade to <name> | Downgrade to <name>. One sentence why.</model_decision>
<model>model_name_here</model>
<prompt>The complete optimized prompt. Preserve {{question}} and {{context}} placeholders.</prompt>
""",

    TaskType.CODE_GENERATION: """
You are optimizing a prompt AND selecting a model for an agent in a competitive programming system.

Agent Type: {agent_type}
System Role: {role_description}
Current Model: {current_model}

============================== MODEL POOL ==============================
{model_pool_section}
=======================================================================

{trajectory_section}

Sample Execution Traces (Problem + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

DECISION FRAMEWORK:
Step 1 - Diagnose code quality: What specific issues appear in the sample outputs?
  - Algorithm errors? Wrong I/O format? Missing edge cases? Inefficient solutions?
Step 2 - Assess model adequacy: Is {current_model} overkill, adequate, or insufficient?
  - Overkill (correct, efficient solutions) -> consider downgrading to save cost.
  - Insufficient (bugs, incomplete, timeout-prone) -> consider upgrading.
  - Adequate -> keep current, focus on prompt improvement.
Step 3 - Output the optimized prompt and selected model.

Rules (cost-benefit balance, lambda={lambda_cost}):
- Fix the prompt first. Then decide if model change is needed.
- Upgrade when the model clearly cannot solve these problems. Downgrade when clearly overkill.
- Jump at most ONE tier per round, unless lambda=0.
- If outputs contain algorithmic errors beyond prompt fixes, do NOT downgrade.
- Both upgrade and downgrade are valid - let evidence and cost trade-off decide.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Analysis of code quality issues and model adequacy.</analyse>
<model_decision>One of: Keep | Upgrade to <name> | Downgrade to <name>. One sentence why.</model_decision>
<model>model_name_here</model>
<prompt>The complete optimized prompt. Preserve {{question}} and {{context}} placeholders.</prompt>
""",

    TaskType.FUNCTION_CALLING: """
You are optimizing a prompt AND selecting a model for an agent in a tool-calling / function selection system.

Agent Type: {agent_type}
System Role: {role_description}
Current Model: {current_model}

============================== MODEL POOL ==============================
{model_pool_section}
=======================================================================

{trajectory_section}

Sample Execution Traces (User Query + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```

DECISION FRAMEWORK:
Step 1 — Diagnose function call quality: What is wrong?
  - Wrong JSON format ("function"/"parameters" instead of "name"/"arguments")? Wrong function name? Missing required parameters? Incorrect parameter types? Hallucinated values?
Step 2 — Assess model adequacy: Is {current_model} overkill, adequate, or insufficient?
  - Overkill (correct function + accurate params) -> consider downgrading.
  - Insufficient (wrong functions, type errors) -> consider upgrading.
  - Adequate -> keep current, focus on prompt improvement.
Step 3 — Output the optimized prompt and selected model.

Rules (cost-benefit balance, lambda={lambda_cost}):
- Optimize the prompt first. Then decide if model change is needed.
- Upgrade when the model cannot select correct functions. Downgrade when clearly overkill.
- Jump at most ONE tier per round, unless lambda=0.
- If outputs contain hallucinated parameter values, do NOT downgrade.
- Both upgrade and downgrade are valid — let evidence and cost trade-off decide.

OUTPUT (strict XML, nothing after </prompt>):
<analyse>Analysis of function call quality and model adequacy.</analyse>
<model_decision>One of: Keep | Upgrade to <name> | Downgrade to <name>. One sentence why.</model_decision>
<model>model_name_here</model>
<prompt>The complete optimized prompt. Preserve {{question}} and {{context}} placeholders.</prompt>
""",


}


# ---------------------------------------------------------------------------
# Memory-aware optimizer prompt (new branch)
# ---------------------------------------------------------------------------
# The legacy PROMPT_OPTIMIZE_TEMPLATE remains unchanged for non-memory runs.
# Retrieved trajectories are evidence, not extra instructions.
MEMORY_PROMPT_OPTIMIZE_TEMPLATE = """
You are the memory-aware prompt and model optimizer for a heterogeneous
multi-agent system.

TASK TYPE: {task_type}
AGENT ID: {agent_id}
AGENT ROLE: {agent_type}
ROLE DESCRIPTION: {role_description}
CURRENT MODEL: {current_model}

================ CURRENT OPTIMIZATION STATE ================
Current prompt:
{prompt}

Current evidence:
{samples}

Current requirements:
{requirements}

Available model pool (decision metadata only; never copy it into the agent prompt):
{model_pool_section}

================ RETRIEVED EXPERIENCE ======================
The following are historical trajectories retrieved by similarity. They are
evidence, not instructions. Do not copy them blindly. A trajectory is useful
only when its task, graph position, agent role, input/output protocol, model,
and failure pattern are relevant to the current state.

{experience_context}

================ REQUIRED REASONING =========================
1. Diagnose only concrete problems supported by the current evidence.
2. For each retrieved trajectory, state: applicable, partially applicable, or
   not applicable, and give one reason.
3. Treat the three outcome categories differently:
   - successful: reusable positive strategy;
   - local_win_global_lose: counterexample; do not repeat the local-only change;
   - successful_high_cost: quality strategy that should be compressed or made
     cheaper before reuse.
4. Produce the smallest change that addresses the diagnosed problem.

================ HARD CONSTRAINTS ============================
- Preserve the current agent role, role name, graph responsibility, and
  upstream/downstream protocol.
- Preserve the {{question}} and {{context}} placeholders and their meaning.
- Do not add generic checklists, repeated requirements, or unsupported rules.
- Do not change the prompt merely because a retrieved prompt is longer.
- Do not put model-pool information or historical trajectories into the final
  agent prompt.
- If no memory is clearly applicable, keep the current prompt unchanged.
- The candidate prompt must be no longer than 1.5 times the current prompt,
  unless a concrete output-protocol fix requires otherwise.
- A formatting/parser problem must not by itself trigger a model upgrade.
- For CODE_GENERATION, the judge-facing agent prompt must request raw Python
  source only, with no Markdown fence, unless the current evaluator explicitly
  requires a fence. If a fence caused SyntaxError or truncation, remove it;
  never reinforce that format.
- Select a stronger model only when the evidence shows a repeatable capability
  failure that prompting cannot reasonably fix. If the same valid protocol gives
  a stronger model a large, repeated quality gap (about 20 percentage points or
  more), upgrade at most one tier. If the gap is small or unsupported, keep the
  current model. Prefer the cheaper model when quality is comparable.

================ STRICT OUTPUT ===============================
Return nothing outside these XML tags:
<diagnosis>Concrete evidence-based diagnosis.</diagnosis>
<memory_usage>Applicability and lesson from each retrieved trajectory.</memory_usage>
<edit_plan>Minimal additions, deletions, or replacements.</edit_plan>
<model_decision>Keep, Upgrade to MODEL, or Downgrade to MODEL, with one short reason.</model_decision>
<model>{current_model}</model>
<prompt>The complete candidate prompt, preserving required placeholders.</prompt>
"""

ANSWER_EVALUATE_TEMPLATE = {
    TaskType.MATH: """
You are evaluating two outputs (A and B) from an agent of type {agent_type}.
Based on the input, requirements and the agent's role, evaluate the two responses, A and B, and determine which one is better.

The agent's specific role: {role_description}

# Input to the Agent
Question: {question}

# Requirement
{requirement}

# A
{Answer_A}

# B
{Answer_B}

Guidelines:
- If one contains a clearly correct final answer and the other does not, choose the correct one — even if its explanation is shorter.
- If both answers appear correct, prefer the one with clearer, logically sound reasoning.
- Do NOT favor verbose, or confident-sounding outputs if they are wrong.
- Pay attention to the format of final answer (<answer>...</answer>).

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
<analyse>Some analysis</analyse>
<choose>A/B</choose>
""",
    
    TaskType.MATH_CHOICE: """
You are evaluating two outputs (A and B) from an agent of type {agent_type}.
Based on the input, requirements and the agent's role, evaluate the two responses, A and B, and determine which one is better.

The agent's specific role: {role_description}

# Input to the Agent
Question: {question}

# Requirement
{requirement}

# A
{Answer_A}

# B
{Answer_B}

Guidelines:
- If one contains a clearly correct final answer and the other does not, choose the correct one — even if its explanation is shorter.
- If both answers appear correct, prefer the one with clearer, logically sound reasoning.
- Do NOT favor verbose, or confident-sounding outputs if they are wrong.
- Pay attention to the format of final answer (only one letter between <answer> and </answer> representing the correct option).

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
<analyse>Some analysis</analyse>
<choose>A/B</choose>
""",
    
    TaskType.CODE: """
You are evaluating two code outputs (A and B) from an agent of type {agent_type}.
Based on the problem requirements and the agent's role, evaluate the two code solutions, A and B, and determine which one is better.

The agent's specific role: {role_description}

# Programming Problem
{question}

# Requirements
{requirement}

# Code Solution A
{Answer_A}

# Code Solution B
{Answer_B}

Guidelines:
- If one solution's code is syntactically correct and passes all possible test cases while the other doesn't, choose the correct one.
- If both solutions' code appear correct, prefer the one with:
  * Better code quality (readability, maintainability)
  * More efficient algorithm or implementation
  * Clearer variable names and comments
  * Clearer reasoning before final code
- Do NOT favor verbose or over-engineered solutions if a simpler one is correct.
- Pay attention to whether the code is properly enclosed in ```python ... ``` blocks.

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
<analyse>Some analysis focusing on correctness, code quality, and efficiency</analyse>
<choose>A/B</choose>
""",

    TaskType.REASONING_CHOICE: """
You are an expert judge evaluating two reasoning outputs (A and B) from an agent of type {agent_type}.
Based on the input, requirements and the agent's role, evaluate the two responses, A and B, and determine which one is better.

The agent's specific role: {role_description}

# Input to the Agent
Question: {question}

# Requirement
{requirement}

# A
{Answer_A}

# B
{Answer_B}

### Evaluation Criteria (Priority Order):
1. **Completeness:** Is the output truncated? Complete outputs are always preferred.
2. **Correctness:** Is the reasoning sound and the answer likely correct?
3. **Efficiency:** 
   - Concise, focused reasoning is BETTER than verbose, over-structured reasoning.
   - Unnecessary section headers and formatting reduce quality.
   - The goal is correct answers, not beautiful formatting.
### Decision:
- If one is truncated, choose the complete one.
- If both complete, choose the one with better reasoning (not better formatting).
- Penalize outputs that waste tokens on structure instead of thinking.

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
<analyse>
Compare Reasoning (if both complete).
</analyse>
<choose>A/B</choose>
""",
    TaskType.QA: """
You are evaluating two outputs (A and B) from a multi-hop QA agent of type {agent_type}.
Based on the documents, question, and the agent's role, evaluate the two responses and determine which one is better.

The agent's specific role: {role_description}

# Documents and Question
{question}

# Requirement
{requirement}

# A
{Answer_A}

# B
{Answer_B}

Guidelines for QA evaluation:
- If one answer is factually correct (grounded in the documents) and the other is not, choose the correct one.
- If both appear correct, prefer the one with clearer cross-document evidence chains.
- Penalize answers that fabricate facts not present in the documents (hallucination).
- Prefer concise, well-targeted answers over verbose ones with irrelevant details.
- Do NOT favor longer or more confident-sounding outputs if they are wrong.
- Pay attention to the format: answer must be in <answer>...</answer> tags and contain only the short answer.

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
<analyse>Compare evidence grounding, factual accuracy, and answer precision.</analyse>
<choose>A/B</choose>
""",


    TaskType.CODE_GENERATION: """
You are evaluating two code outputs (A and B) from an agent of type {agent_type}.
Based on the problem requirements and the agent's role, evaluate the two code solutions, A and B, and determine which one is better.

The agent's specific role: {role_description}

# Programming Problem
{question}

# Requirements
{requirement}

# Code Solution A
{Answer_A}

# Code Solution B
{Answer_B}

Guidelines for code evaluation:
- If one code is algorithmically correct and passes all test cases while the other is buggy, choose the correct one.
- If both appear correct, prefer the more efficient algorithm (time/space complexity).
- If both have bugs, choose the one that is closer to a correct solution.
- Check: correct stdin/stdout handling, no extra prints, edge case coverage.

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
<analyse>Analysis focusing on correctness, edge cases, and code quality</analyse>
<choose>A/B</choose>
""",
    TaskType.FUNCTION_CALLING: """
You are evaluating two outputs (A and B) from a tool-calling agent of type {agent_type}.
Based on the user query and the agent's role, evaluate the two function calls and determine which one is better.

The agent's specific role: {role_description}

# User Query and Available Tools
{question}

# Requirement
{requirement}

# A
{Answer_A}

# B
{Answer_B}

Guidelines for function call evaluation:
- OUTPUT FORMAT (CRITICAL): The only valid format is a JSON array [{{"name": "function_name", "arguments": {{...}}}}]. The field names "name" and "arguments" are MANDATORY. Outputs using "function"/"parameters", raw text, or any other format are INVALID. If one output uses the correct format and the other does not, strongly prefer the correct one.
- Function name: Must match the correct function for the user's intent.
- Required parameters: All mandatory parameters must be present with correct values.
- Parameter types: Integer vs float vs string vs boolean — type errors are critical.
- Optional parameters: Missing optional params are acceptable; wrong optional params are not.
- Parameter values: Must be extracted precisely from the user query (no hallucination).
- If one call is structurally correct and the other has type errors or missing params, choose the correct one.
- Do NOT favor longer or more verbose outputs if the function call is wrong.

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
<analyse>Compare function name accuracy, parameter completeness, and type correctness.</analyse>
<choose>A/B</choose>
""",


}

INTERMEDIATE_COMPARE_TEMPLATE = {
    TaskType.MATH: """
You are comparing two intermediate outputs from an agent in a multi-agent mathematical reasoning system.
Your task is to decide which output is more likely to help the system eventually produce the CORRECT FINAL ANSWER.

Prefer the output that:
- Contains mathematically accurate reasoning (no factual/logical errors)
- Clearly states intermediate results or assumptions
- Avoids misleading statements or ambiguous conclusions
- Provides enough detail for downstream agents to verify or build upon

Problem: {question}
Output A:
{output_a}
Output B:
{output_b}

Which output is more conducive to obtaining the correct final answer? Respond ONLY with "A" or "B".
""",
    
    TaskType.CODE: """
You are comparing two outputs from a 'Predictor' agent in a code generation pipeline.
The next agent in the pipeline is a 'Reflector' or 'Debugger' that will try to fix any errors.

Your goal is to select the output that provides the **best starting point** for the Reflector.

Criteria for selection (in order of importance):
1. **Correctness**: If one output appears to be functionally correct python code while the other has logic errors, choose the correct one.
2. **Adherence to Requirements**: The code must NOT contain exception handling, input validation, or type hints if the prompt forbids them. It must be a standalone function.
3. **Format**: The code must be easily extractable (e.g., inside ```python``` blocks).
4. **Logic Clarity**: If both have bugs, choose the one with the clearer algorithmic logic that is easier to debug.

Problem: {question}

Output A:
{output_a}

Output B:
{output_b}

Which output is better? Respond ONLY with "A" or "B".
""",

    TaskType.MATH_CHOICE: """
You are comparing two intermediate outputs from an agent in a multi-agent mathematical reasoning system.
Your task is to decide which output is more likely to help the system eventually produce the CORRECT FINAL ANSWER.

Prefer the output that:
- Contains mathematically accurate reasoning (no factual/logical errors)
- Clearly states intermediate results or assumptions
- Avoids misleading statements or ambiguous conclusions
- Provides enough detail for downstream agents to verify or build upon

Problem: {question}
Output A:
{output_a}
Output B:
{output_b}

Which output is more conducive to obtaining the correct final answer? Respond ONLY with "A" or "B".
""",

    TaskType.REASONING_CHOICE: """
You are comparing two intermediate outputs from an agent in a multi-agent problem-solving reasoning system.
Your task is to decide which output is more likely to help the system eventually produce the CORRECT FINAL ANSWER.

### Evaluation Priority:
**1. Completeness:**
- Is the output truncated (ends mid-sentence, incomplete thought)?
- A truncated output cannot effectively help downstream agents.
- If one is truncated and one is not, select the complete one.
**2. Reasoning Correctness (If both complete):**
- Are scientific facts and principles correctly stated?
- Is the logical reasoning valid?
- Are option eliminations properly justified?
**3. Usefulness for Next Agent:**
- Does it provide clear intermediate conclusions?
- Can the next agent build upon this reasoning?
### Decision Rules:
- Truncated output = automatic disqualification (unless both truncated)
- If both complete: prioritize scientific accuracy over verbosity
- If both truncated: choose the one with more valid reasoning content

Problem: {question}
Output A:
{output_a}
Output B:
{output_b}

Which output is more conducive to obtaining the correct final answer? Respond ONLY with "A" or "B".
""",
    TaskType.QA: """
You are comparing two intermediate outputs from an agent in a multi-hop QA system.
Your task is to decide which output is more likely to help the system eventually produce the CORRECT FINAL ANSWER.

Focus on:
- Evidence quality: Did the agent identify the right documents and extract key facts?
- Cross-document connections: Did it recognize bridging relationships?
- Reasoning soundness: Is the logic grounded in the documents?
- Conciseness: Did it avoid irrelevant information?

Problem: {question}
Output A:
{output_a}
Output B:
{output_b}

Which output is more conducive to obtaining the correct final answer? Respond ONLY with "A" or "B".
""",


    TaskType.CODE_GENERATION: """
    You are comparing two intermediate outputs from a code generation agent.
    Your task is to decide which output is closer to a correct solution.
    Consider: algorithm correctness, edge case handling, and input/output format compliance.
    
    Problem: {question}
    Output A:
    {output_a}
    Output B:
    {output_b}
    
    Which output is more likely to lead to a correct final solution? Respond ONLY with "A" or "B".
    """,
    TaskType.FUNCTION_CALLING: """
You are comparing two intermediate outputs from an agent in a tool-calling system.
Your task is to decide which output is more likely to help the system eventually produce the CORRECT FUNCTION CALL.

CRITICAL FORMAT RULE: The correct final output format is a JSON array [{{"name": "function_name", "arguments": {{...}}}}]. Outputs using "function"/"parameters" or other field names are INVALID. Prefer outputs that already use the correct "name" and "arguments" field names.

Focus on:
- Output format: Does it use "name"/"arguments" (correct) or "function"/"parameters" (wrong)?
- Function identification: Did it recognize the right tool for the user's intent?
- Parameter extraction: Did it correctly identify required vs optional parameters?
- Type awareness: Did it distinguish between integer, float, string, boolean, list, dict types?
- Value accuracy: Are parameter values extracted precisely from the user query?

Problem: {question}
Output A:
{output_a}
Output B:
{output_b}

Which output is more conducive to obtaining the correct function call? Respond ONLY with "A" or "B".
""",


}


FINAL_ANSWER_COMPARE_TEMPLATE = {
    TaskType.MATH: """
You are comparing two FINAL answers to a math problem.
Your SOLE task is to determine which answer is more likely to be **mathematically correct**.

Do NOT favor longer, more detailed, or better-formatted responses unless they are also correct.
If one answer is clearly correct and the other is wrong, choose the correct one—even if its reasoning is minimal.
If both seem plausible, prefer the one with clearer, error-free reasoning.

Problem: {question}
Requirement: {requirement}

Answer A:
{Answer_A}

Answer B:
{Answer_B}

Respond with only "A" or "B".
""",
    
    TaskType.CODE: """
You are comparing two FINAL Python code solutions.
Your SOLE criterion is **Functional Correctness**.

Problem: {question}
Requirement: {requirement}

Answer A:
{Answer_A}

Answer B:
{Answer_B}

Evaluation Steps:
1. Mentally trace both codes with edge case inputs.
2. If Answer A is correct and Answer B is buggy (infinite loop, wrong logic, syntax error), choose A.
3. If Answer B is correct and Answer A is buggy, choose B.
4. If both are correct, choose the one that is **more concise** and follows standard Pythonic practices.
5. If both are buggy, choose the one that is "less broken" (closer to the solution).

Respond with only "A" or "B".
""",

    TaskType.MATH_CHOICE: """
You are comparing two FINAL answers to a math problem.
Your SOLE task is to determine which answer is more likely to be **mathematically correct**.

Do NOT favor longer, more detailed, or better-formatted responses unless they are also correct.
If one answer is clearly correct and the other is wrong, choose the correct one—even if its reasoning is minimal.
If both seem plausible, prefer the one with clearer, error-free reasoning.

Problem: {question}
Requirement: {requirement}

Answer A:
{Answer_A}

Answer B:
{Answer_B}

Respond with only "A" or "B".
""",

    TaskType.REASONING_CHOICE: """
You are comparing two FINAL answers to a problem.
Your SOLE task is to determine which answer is more likely to be **correct**.

### Evaluation Process:
**Step 1: Check Completeness**
- Does each output have a clear final answer (preferably in <answer>X</answer> format)?
- If one is truncated/incomplete and the other is complete, choose the complete one.
**Step 2: Extract and Compare Answers (if both complete)**
- Identify the final letter choice from each output
- If answers are the same, choose the one with better reasoning
- If answers differ, proceed to Step 3
**Step 3: Evaluate Reasoning Quality (if answers differ)**
- Which output correctly applies relevant scientific principles?
- Which output properly eliminates incorrect options?
- Which reasoning chain is more logically sound?
**Step 4: Make Decision**
Priority: Complete > Truncated
Among complete outputs: Correct reasoning > Incorrect reasoning > Better format

Problem: {question}
Requirement: {requirement}

Answer A:
{Answer_A}

Answer B:
{Answer_B}

Respond with only "A" or "B".
""",
    TaskType.QA: """
You are comparing two FINAL answers to a multi-hop QA question.
Your SOLE task is to determine which answer is more likely to be **factually correct** based on the provided documents.

# Question and Documents
{question}

# Requirement
{requirement}

# Answer A
{Answer_A}

# Answer B
{Answer_B}

Evaluation criteria (for multi-hop QA):
Step 1: Factual Correctness — Which answer better matches the documents? Any hallucinations?
Step 2: Completeness — Does the answer fully address the question?
Step 3: Conciseness — Is the answer precise and to the point?
Priority: Factually correct > Complete > Concise

Respond with only "A" or "B".
""",


    TaskType.CODE_GENERATION: """
    Compare two FINAL code solutions for the same competitive programming problem.
    Evaluate: correctness (passes all test cases), efficiency (time/space), and code clarity.
    
    Requirement: {requirement}
    Problem: {question}
    
    Code A: {Answer_A}
    Code B: {Answer_B}
    
    Which code is more correct? Respond ONLY with "A" or "B".
    """,
    TaskType.FUNCTION_CALLING: """
You are comparing two FINAL function calls.
Your SOLE task is to determine which function call is more likely to be **correct**.

# User Query and Available Tools
{question}

# Requirement
{requirement}

# Function Call A
{Answer_A}

# Function Call B
{Answer_B}

Evaluation criteria (for function calling):
Step 0: OUTPUT FORMAT (CRITICAL) — The valid format is [{{"name": "...", "arguments": {{...}}}}]. If one output uses "name"/"arguments" and the other uses "function"/"parameters" or any other field names, STRONGLY prefer the one with "name"/"arguments". Wrong field names make the output invalid regardless of content correctness.
Step 1: Function Name — Is the function name correct for the user's intent?
Step 2: Required Parameters — Are all mandatory parameters present with correct values?
Step 3: Parameter Types — Are integer, float, string, boolean, list, dict types correct?
Step 4: Optional Parameters — Are optional parameters correctly valued if present?
Priority: Correct format > Correct function + required params > Correct types > Optional params

Respond with only "A" or "B".
""",


}


OPTIMIZATION_REQUIREMENTS = {
    TaskType.MATH: (
        "The agent must produce a clear, step-by-step reasoning process. "
        "Crucially, the **final answer must appear ONLY once**, enclosed strictly between <answer> and </answer> tags, "
        "with **no additional text, explanation, unit, or reasoning inside these tags**. "
        "The content within `<answer>...</answer>` must be the minimal, canonical mathematical answer."
    ),
    TaskType.MATH_CHOICE: (
        "The agent must produce a clear, step-by-step reasoning process. "
        "Crucially, the **final answer must appear ONLY once**, enclosed strictly between <answer> and </answer> tags, "
        "with **no additional text, explanation, unit, or reasoning inside these tags**. "
        "The content within `<answer>...</answer>` must be only one letter representing the correct option among the given option."
    ),
    TaskType.REASONING_CHOICE: (
        "Crucially, the **final answer must appear ONLY once**, enclosed strictly between <answer> and </answer> tags, "
        "with **no additional text, explanation, unit, or reasoning inside these tags**. "
        "The content within `<answer>...</answer>` must be only one letter representing the correct option among the given option."
    ),
    TaskType.CODE: (
        "The agent must produce a clear, step-by-step reasoning process followed by syntactically correct Python code that passes all possible test cases. "
        "The format of code must be like ```python\ndef function_name(input_arguments): \n#code here\nreturn output```without any exception detection, input validation, type annotations, or edge-case handling. "
    ),
    TaskType.QA: (
        "Optimize the prompt for multi-hop QA. The agent must: (1) read all documents thoroughly, "
        "(2) identify and connect relevant information across documents, (3) output the answer in <answer> tags. "
        "The prompt must not fabricate answers and must only use information from the provided documents."
    ),


    TaskType.CODE_GENERATION: (
        "The agent must generate correct Python code for competitive programming problems. "
        "The code reads from stdin and writes to stdout. "
        "Optimization should focus on: "
        "1. Algorithm correctness (passes all test cases) "
        "2. Input/output format exact match "
        "3. Edge case handling (min/max constraints, zero, negative) "
        "4. Time and space complexity within limits "
        "5. No extra output, prompts, or debugging prints "
    ),

    TaskType.FUNCTION_CALLING: (
        "The agent must: (1) select the correct function with the exact name as defined, "
        "(2) provide all required parameters with values exactly as they appear in the user query, "
        "(3) respect parameter types strictly (integer vs float vs string vs boolean), "
        "(4) never hallucinate parameters not defined in the function signature, "
        "and (5) omit optional parameters when not specified by the user.\n"
        "CRITICAL OUTPUT FORMAT: The output MUST be a valid JSON array in the exact format "
        "[{\"name\": \"function_name\", \"arguments\": {\"param1\": value1, ...}}]. "
        "For multiple calls, include multiple objects in the array. "
        "The field names \"name\" and \"arguments\" are fixed and MUST NOT be changed."
    ),
}

AGENT_DROPOUT_MATH_PROMPTS = {
    "0": (
        "You are a math expert. "
        "You will be given a multiple-choice question and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final choice with only a capital letter, for example: <answer>A</answer>\n"
        "Question: {question}\n"
        "{context}"
    ),
    "1": (
        "You are a mathematical analyst. "
        "You will be given a multiple-choice question, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results. "
        "The last line of your output contains only the final choice with only a capital letter, for example: <answer>A</answer>\n"
        "Question: {question}\n"
        "{context}"
    ),
    "2": (
        "You are a programming expert. "
        "You will be given a multiple-choice question, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve multiple-choice question. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the answer variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n    x = 10\n    y = 20\n    return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "3": (
        "You are a math expert. "
        "You will be given a multiple-choice question and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final choice with only a capital letter, for example: <answer>A</answer>\n"
        "Question: {question}\n"
        "{context}"
    ),
}


AGENT_DROPOUT_MATH_OPEN_PROMPTS = {
    "0": (
        "You are a math expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "Show your final answer bracketed between <answer> and </answer> tags.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "1": (
        "You are a mathematical analyst. "
        "You will be given a math problem, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results. "
        "Show your final answer bracketed between <answer> and </answer> tags.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "2": (
        "You are a programming expert. "
        "You will be given a math problem, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve the math problem. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the answer variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n    x = 10\n    y = 20\n    return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "3": (
        "You are a math expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "Show your final answer bracketed between <answer> and </answer> tags.\n"
        "Question: {question}\n"
        "{context}"
    ),
}

AGENT_DROPOUT_CODE_PROMPTS = {
    "0": (
        "You are a project manager. "
        "You will be given a function signature and its docstring by the user. "
        "You are responsible for overseeing the overall structure of the code, ensuring that the code is structured to complete the task. Implement code concisely and correctly without pursuing over-engineering. "
        "You need to suggest optimal design patterns to ensure that the code follows best practices for maintainability and flexibility. "
        "You can specify the overall design of the code, including the classes that need to be defined (maybe none) and the functions used (maybe only one function). "
        "I hope your reply will be more concise. Preferably within fifty words. Don't list too many points.\n"
        "Problem: {question}\n"
        "{context}"
    ),
    "1": (
        "You are a programming expert. "
        "You will be given a function signature and its docstring by the user. "
        "You may be able to get the output results of other agents. They may have passed internal tests, but they may not be completely correct. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```\n"
        "Do not include anything other than Python code blocks in your response. "
        "Do not change function names and input variable types in tasks. "
        "Analyze the problem in 2-3 key steps.\n"
        "Problem: {question}\n"
        "{context}"
    ),
    "2": (
        "You are a test analyst. "
        "You will be given a function signature and its docstring by the user. "
        "You need to provide problems in the current code or solution based on the test data and possible test feedback in the question. "
        "You need to provide additional special use cases, boundary conditions, etc. that should be paid attention to when writing code. "
        "You can point out any potential errors in the code. "
        "I hope your reply will be more concise. Preferably within fifty words. Don't list too many points.\n"
        "Problem: {question}\n"
        "{context}"
    ),
    "3": (
        "You are a bug fixer. "
        "You will be given a function signature and its docstring by the user. "
        "You need to provide modified and improved python code based on the current overall code design, algorithm framework, code implementation or test problems. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```\n"
        "Do not include anything other than Python code blocks in your response. "
        "Do not change function names and input variable types in tasks.\n"
        "Problem: {question}\n"
        "{context}"
    ),
}

AGENT_DROPOUT_REASONING_PROMPTS = {
    "0": (
        "You are a knowledgeable expert in question answering. "
        "Please give several key entities that need to be searched in wikipedia to solve the problem. "
        "Key entities that need to be searched are included between two '@' when output, for example: @catfish effect@, @broken window effect@, @Shakespeare@. "
        "If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "1": (
        "You are an excellent critic. "
        "Please point out potential issues in other agent's analysis point by point.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "2": (
        "You are a psychologist. "
        "You are good at psychology, sociology, and philosophy. "
        "You give people scientific suggestions that will make them feel better. "
        "Based on the question and hints from other agents, provide your analysis and reasoning. "
        "Show your final option with one letter bracketed between <answer> and </answer> tags.\n"
        "Question: {question}\n"
        "{context}"
    ),
    "3": (
        "You are a historian. "
        "You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history. "
        "Based on the question and hints from other agents, provide your final analysis and answer. "
        "Show your final option with one letter bracketed between <answer> and </answer> tags.\n"
        "Question: {question}\n"
        "{context}"
    ),
}


HIERARCHICAL_PROMPTS = {
    TaskType.MATH: {
        "0": ( # Planner
            "You are a math agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>.\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
        "1": ( # Critic
            "You are a science agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>.\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
        "2": ( # Refiner
            "You are a code agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>.\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
        "3": ( # Judger
            "You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>.\n\n"
            "Content from Previous Agent:\n{context}\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
    },
    
    TaskType.MATH_CHOICE: {
        "0": "You are a math agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nInput Question: {question}\n\nYour response:",
        "1": "You are a science agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nInput Question: {question}\n\nYour response:",
        "2": "You are a code agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nInput Question: {question}\n\nYour response:",
        "3": "You are a task summarizer. Given the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nContent from Previous Agent:\n{context}\n\nInput Question: {question}\n\nYour response:",
    },
    TaskType.REASONING_CHOICE: {
        "0": "You are a math agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nInput Question: {question}\n\nYour response:",
        "1": "You are a science agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nInput Question: {question}\n\nYour response:",
        "2": "You are a code agent. Given the input question, reason step-by-step and put the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nInput Question: {question}\n\nYour response:",
        "3": "You are a task summarizer. Given the final answer inside <answer>YOUR_FINAL_CHOICE</answer>. Your final answer must be selected from A,B,C,D. For example <answer>A</answer>. Do not add any other contents inside the box. \n\nContent from Previous Agent:\n{context}\n\nInput Question: {question}\n\nYour response:",
    },
    TaskType.CODE: {
        "0": ( # Planner
            "You are a math agent. Given the input question, reason step by step and provide an efficient and self-contained Python function that solves the following problem in a markdown code block. You must put all python code as self-contained Python function in markdown code blocks. "
            "For example ```python\nimport needed_library\ndef FUNC_NAME(a, b):\n    return a + b```. "
            "Do not add any other contents inside the markdown code block.\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
        "1": ( # Critic
            "You are a science agent. Given the input question, reason step by step and provide an efficient and self-contained Python function that solves the following problem in a markdown code block. You must put all python code as self-contained Python function in markdown code blocks. "
            "For example ```python\nimport needed_library\ndef FUNC_NAME(a, b):\n    return a + b```. "
            "Do not add any other contents inside the markdown code block.\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
        "2": ( # Refiner
            "You are a code agent. Given the input question, reason step by step and provide an efficient and self-contained Python function that solves the following problem in a markdown code block. You must put all python code as self-contained Python function in markdown code blocks. "
            "For example ```python\nimport needed_library\ndef FUNC_NAME(a, b):\n    return a + b```. "
            "Do not add any other contents inside the markdown code block.\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
        "3": ( # Judger
            "You are a task summarizer. Given the input question and responses from previous agents as reference, reason step by step and provide an efficient and self-contained Python function that solves the following problem in a markdown code block.\n\n"
            "For example ```python\nimport needed_library\ndef FUNC_NAME(a, b):\n    return a + b```. "
            "Do not add any other contents inside the markdown code block.\n\n"
            "Content from Previous Agent:\n{context}\n\n"
            "Input Question: {question}\n\n"
            "Your response:"
        ),
    },
    TaskType.QA: {
        "0": (
            "You are a Document Analyst. Read all provided documents and extract all facts relevant to the question. "
            "List key entities, relationships, and potential connections between documents. Be thorough and precise.\n\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Relevant facts and connections:"
        ),
        "1": (
            "You are an Evidence Validator. Review the extracted facts and verify each one against the source documents. "
            "Flag any unsupported claims, identify the strongest evidence chains, and note any gaps in reasoning.\n\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Validated evidence and gaps:"
        ),
        "2": (
            "You are a Reasoning Specialist. Using the validated evidence, construct a step-by-step reasoning chain "
            "that connects the documents to answer the question. Each step must be grounded in specific document facts.\n\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Step-by-step reasoning chain:"
        ),
        "3": (
            "You are a Task Summarizer. Review all the analyses above and produce the final answer to the question. "
            "Synthesize the document analysis, evidence validation, and reasoning chain. "
            "Output ONLY the short final answer (1-5 words) in <answer>...</answer> tags. Do NOT include reasoning inside the tags.\n\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Final answer:"
        ),
    },
    TaskType.FUNCTION_CALLING: {
        "0": (  # Function Selector - text analysis only
            "You are a function selection expert. Given the user query and available functions below, "
            "analyze which function should be called and explain why.\n"
            "Consider: (1) what is the user's intent, (2) which function best matches this intent, "
            "(3) why other available functions are not suitable.\n"
            "Do NOT output a function call JSON -- only provide a brief text analysis.\n\n"
            "Input Question: {question}\n\n"
            "Your analysis:"
        ),
        "1": (  # Parameter Extractor - text analysis only
            "You are a parameter extraction expert. Given the user query and available functions below, "
            "identify all parameter values that should be passed to the function.\n"
            "For each parameter: state its value, its type, and where in the user query this value comes from.\n"
            "Check that all required parameters are accounted for.\n"
            "Do NOT output a function call JSON -- only provide a brief text analysis.\n\n"
            "Input Question: {question}\n\n"
            "Your analysis:"
        ),
        "2": (  # Format Validator - text analysis only
            "You are a format validator. Given the user query and available functions below, "
            "verify parameter types, check required vs optional fields, and ensure values are valid.\n"
            "Specifically check: (1) integers are not passed as strings, (2) enum values match allowed choices, "
            "(3) all required parameters have values, (4) no hallucinated parameters.\n"
            "Do NOT output a function call JSON -- only provide a brief text analysis.\n\n"
            "Input Question: {question}\n\n"
            "Your analysis:"
        ),
        "3": (  # Final Call Synthesizer - outputs JSON directly
            "You are a task synthesizer. Below are analyses from your teammates, "
            "the user query, and the available functions.\n"
            "Based on these, produce the final function call.\n"
            "Output ONLY a JSON array in this exact format -- no other text:\n\n"
            "[{\"name\": \"<function_name>\", \"arguments\": {<param>: <value>, ...}}]\n"
            "\n"
            "Teammate Analyses:\n{context}\n\n"
            "User Query & Available Functions:\n{question}\n\n"
            "Function Call:"
        ),
    },

    # Previous CODE_GENERATION hierarchical prompts kept for reference.
    # TaskType.CODE_GENERATION: {
    #     "0": (  # Algorithm Analyst - text analysis only
    #         "You are an algorithm analyst. For the coding problem below, analyze the algorithm strategy.\n"
    #         "Describe: (1) what type of computational problem this is, (2) the algorithm that solves it, "
    #         "(3) what key data structures are needed.\n"
    #         "Do NOT write code -- only provide a brief text analysis.\n\n"
    #         "Problem:\n{question}\n\n"
    #         "Your analysis:"
    #     ),
    #     "1": (  # Edge Case Analyst - text analysis only
    #         "You are a test analyst. For the coding problem below, identify edge cases and pitfalls.\n"
    #         "List: (1) boundary conditions given the constraints, (2) tricky inputs that could break naive solutions, "
    #         "(3) input/output format requirements.\n"
    #         "Do NOT write code -- only provide a brief text analysis.\n\n"
    #         "Problem:\n{question}\n\n"
    #         "Your analysis:"
    #     ),
    #     "2": (  # Complexity Analyst - text analysis only
    #         "You are a complexity analyst. For the coding problem below, analyze performance requirements.\n"
    #         "Consider: (1) what time complexity is needed given the input constraints, "
    #         "(2) acceptable space complexity, (3) any optimization trade-offs to consider.\n"
    #         "Do NOT write code -- only provide a brief text analysis.\n\n"
    #         "Problem:\n{question}\n\n"
    #         "Your analysis:"
    #     ),
    #     "3": (  # Code Synthesizer - produces final code
    #         "You are a competitive programmer. Below are three analyses from your teammates, followed by the coding problem.\n"
    #         "Synthesize these analyses and write the correct, efficient Python solution.\n"
    #         "Output ONLY the code inside a ```python code block. No other text.\n\n"
    #         "Previous analyses:\n{context}\n\n"
    #         "Problem:\n{question}\n\n"
    #         "Your code (```python block ONLY):"
    #     ),
    # },
    TaskType.CODE_GENERATION: {
        "0": (
            "You are an algorithm analyst for a competitive programming problem.\n"
            "Output EXACTLY 3 lines and nothing else.\n"
            "Line 1: Algorithm: <at most 8 words>\n"
            "Line 2: Data structures: <at most 8 words>\n"
            "Line 3: Key idea: <at most 12 words>\n"
            "Do NOT write code. Do NOT explain. Do NOT use bullets or markdown.\n\n"
            "Problem:\n{question}\n"
        ),
        "1": (
            "You are an edge-case analyst for a competitive programming problem.\n"
            "Output EXACTLY 3 lines and nothing else.\n"
            "Line 1: Edge case 1: <at most 12 words>\n"
            "Line 2: Edge case 2: <at most 12 words>\n"
            "Line 3: IO note: <at most 12 words>\n"
            "Do NOT write code. Do NOT explain. Do NOT use bullets or markdown.\n\n"
            "Problem:\n{question}\n"
        ),
        "2": (
            "You are a complexity analyst for a competitive programming problem.\n"
            "Output EXACTLY 3 lines and nothing else.\n"
            "Line 1: Time: O(...)\n"
            "Line 2: Space: O(...)\n"
            "Line 3: Constraint driver: <at most 12 words>\n"
            "Do NOT write code. Do NOT explain. Do NOT use bullets or markdown.\n\n"
            "Problem:\n{question}\n"
        ),
        "3": (
            "You are a competitive programmer.\n"
            "Use the three short teammate notes below only as hints.\n"
            "Output RAW Python 3 code only. No markdown fence. No prose. No comments.\n"
            "The first line must start with import, from, or def.\n"
            "The code must read from stdin and write to stdout.\n"
            "Keep the implementation concise and avoid restating the analysis.\n\n"
            "Teammate notes:\n{context}\n\n"
            "Problem:\n{question}\n"
        ),
    },


}
ROLE_DESCRIPTIONS.update({
    AgentType.MATH_AGENT: "You are a math agent.",
    AgentType.SCIENCE_AGENT: "You are a science agent.",
    AgentType.CODE_AGENT: "You are a code agent.",
    AgentType.TASK_SUMMARIZER: "You are a task summarizer.",
})


# ============================================================
# Dataset x graph initial prompts for the 3x4 MAS experiments.
#
# These prompt maps are intentionally stored in the same in-memory style as
# optimized UniMASO prompt files: {agent_id: prompt}. The runner can inject them
# through MAS.inject_prompt_map without changing the rest of the execution path.
# ============================================================

_LCB_AGGREGATE_PROMPTS = {
    0: (
        "You are Algorithm Coder in an independent competitive-programming team. "
        "Prioritize the correct algorithm and proof of feasibility, then write a complete Python 3 solution.\n"
        "Rules: read from stdin, write to stdout, output only Python code in a ```python code block, "
        "and include no prose outside the code block.\n\n"
        "Problem:\n{question}\n\n"
        "Solution:"
    ),
    1: (
        "You are Edge-case Coder in an independent competitive-programming team. "
        "Focus on boundary cases, tricky input formats, off-by-one errors, and degenerate cases before writing code.\n"
        "Rules: read from stdin, write to stdout, output only Python code in a ```python code block, "
        "and include no prose outside the code block.\n\n"
        "Problem:\n{question}\n\n"
        "Solution:"
    ),
    2: (
        "You are Implementation Coder in an independent competitive-programming team. "
        "Write a concise, robust Python 3 implementation that follows the problem's exact input/output format.\n"
        "Rules: read from stdin, write to stdout, output only Python code in a ```python code block, "
        "and include no prose outside the code block.\n\n"
        "Problem:\n{question}\n\n"
        "Solution:"
    ),
}

_LCB_REFLECT_PROMPTS = {
    0: (
        "You are Initial Solver for a competitive-programming pipeline. Produce a complete Python 3 solution.\n"
        "Rules: read from stdin, write to stdout, output only Python code in a ```python code block.\n\n"
        "Problem:\n{question}\n\n"
        "Initial solution:"
    ),
    1: (
        "You are Code Reviewer. Review the previous solution for correctness, complexity, input/output handling, "
        "and hidden edge cases. Output only a corrected Python 3 solution in a ```python code block.\n\n"
        "Problem:\n{question}\n\n"
        "Previous solution:\n{context}\n\n"
        "Corrected solution:"
    ),
    2: (
        "You are Code Rewriter. Use the review context to rewrite the solution into the cleanest complete Python 3 program. "
        "Preserve the exact stdin/stdout contract. Output only a ```python code block.\n\n"
        "Problem:\n{question}\n\n"
        "Context from previous stage:\n{context}\n\n"
        "Rewritten solution:"
    ),
    3: (
        "You are Final Verifier. Perform a final mental check for edge cases, complexity, and I/O format, then emit the final code. "
        "Output only Python code in a ```python code block, with no explanation.\n\n"
        "Problem:\n{question}\n\n"
        "Candidate solution:\n{context}\n\n"
        "Final code:"
    ),
}

_LCB_HIERARCHICAL_PROMPTS = {
    0: (
        "You are Algorithm Analyst for a competitive-programming problem. "
        "Do not write code. Provide a compact plan with the algorithm, key data structures, and why it fits the constraints.\n\n"
        "Problem:\n{question}\n\n"
        "Algorithm notes:"
    ),
    1: (
        "You are Edge-case Analyst for a competitive-programming problem. "
        "Do not write code. List the main boundary cases, tricky inputs, and exact I/O concerns the final coder must handle.\n\n"
        "Problem:\n{question}\n\n"
        "Edge-case notes:"
    ),
    2: (
        "You are Complexity Analyst for a competitive-programming problem. "
        "Do not write code. State the required time and space complexity and the input constraints that drive them.\n\n"
        "Problem:\n{question}\n\n"
        "Complexity notes:"
    ),
    3: (
        "You are Code Synthesizer. Use the teammate notes only as guidance and write the final Python 3 solution. "
        "Output raw Python code only: no markdown fence, no prose, no comments. The code must read from stdin and write to stdout.\n\n"
        "Teammate notes:\n{context}\n\n"
        "Problem:\n{question}\n\n"
        "Final code:"
    ),
}

_LCB_DEBATE_PROMPTS = {
    0: (
        "You are Direct Solver in a two-solver code debate. Write the most straightforward correct Python 3 solution. "
        "Output only Python code in a ```python code block.\n\n"
        "Problem:\n{question}\n\n"
        "Direct solution:"
    ),
    1: (
        "You are Alternative Solver in a two-solver code debate. Seek a different valid angle or implementation strategy. "
        "Output only Python code in a ```python code block.\n\n"
        "Problem:\n{question}\n\n"
        "Alternative solution:"
    ),
    2: (
        "You are Code Arbiter. Compare the candidate solutions for correctness, complexity, and edge-case handling. "
        "If needed, repair them. Output only the final Python code in a ```python code block.\n\n"
        "Problem:\n{question}\n\n"
        "Candidate solutions:\n{context}\n\n"
        "Final code:"
    ),
}

_BFCL_AGGREGATE_PROMPTS = {
    0: (
        "You are Intent Caller in an independent function-calling team. Identify the user's intent and choose the best function. "
        "End with one fenced ```json``` block containing a non-empty list of canonical function-call dicts.\n\n"
        "{context}\n"
        "User query and available functions:\n{question}\n\n"
        "Function call:"
    ),
    1: (
        "You are Schema Caller in an independent function-calling team. Match the query to the provided schemas, paying special attention "
        "to required parameters and allowed enum values. End with one fenced ```json``` block containing a non-empty list of canonical function-call dicts.\n\n"
        "{context}\n"
        "User query and available functions:\n{question}\n\n"
        "Function call:"
    ),
    2: (
        "You are Type-safe Caller in an independent function-calling team. Extract parameter values exactly and preserve their correct JSON types. "
        "Do not hallucinate parameters. End with one fenced ```json``` block containing a non-empty list of canonical function-call dicts.\n\n"
        "{context}\n"
        "User query and available functions:\n{question}\n\n"
        "Function call:"
    ),
}

_BFCL_REFLECT_PROMPTS = {
    0: (
        "You are Initial Caller. Produce the best function call for the query and schemas. "
        "End with one fenced ```json``` block containing a non-empty list of canonical function-call dicts.\n\n"
        "{context}\n"
        "User query and available functions:\n{question}\n\n"
        "Initial call:"
    ),
    1: (
        "You are Schema Reviewer. Check the previous call against the available schemas: function name, required parameters, optional parameters, "
        "and enum constraints. End with one fenced ```json``` block containing the corrected non-empty list of canonical function-call dicts.\n\n"
        "User query and available functions:\n{question}\n\n"
        "Previous call:\n{context}\n\n"
        "Corrected call:"
    ),
    2: (
        "You are Argument Repairer. Repair missing, incorrect, hallucinated, or poorly typed arguments in the previous function call. "
        "End with one fenced ```json``` block containing a non-empty list of canonical function-call dicts.\n\n"
        "User query and available functions:\n{question}\n\n"
        "Previous stage:\n{context}\n\n"
        "Repaired call:"
    ),
    3: (
        "You are Final JSON Verifier. Ensure the final answer matches one provided schema and preserves JSON value types. "
        "End with exactly one fenced ```json``` block containing a non-empty list of canonical function-call dicts. Do not add prose after the block.\n\n"
        "User query and available functions:\n{question}\n\n"
        "Candidate call:\n{context}\n\n"
        "Final call:"
    ),
}

_BFCL_HIERARCHICAL_PROMPTS = {
    0: (
        "You are Function Selector. Analyze the user request and the complete available-function list below. "
        "Identify the single best function and explain why competing functions do not match. "
        "Do not output a function-call JSON object; provide analysis only. Preserve exact function names from the list.\n\n"
        "User request and available functions:\n{question}\n\n"
        "Selection analysis:"
    ),
    1: (
        "You are Parameter Extractor. Using the complete available-function list below, identify every argument value, its JSON type, "
        "and the phrase in the user request that supports it. Check all required and optional parameters. "
        "Do not output a function-call JSON object; provide analysis only.\n\n"
        "User request and available functions:\n{question}\n\n"
        "Parameter analysis:"
    ),
    2: (
        "You are Format and Type Validator. Using the complete available-function list below, check the selected function, required fields, "
        "optional fields, enum values, numeric/string/boolean/list/object types, and hallucinated parameters. "
        "Do not output a function-call JSON object; provide analysis only.\n\n"
        "User request and available functions:\n{question}\n\n"
        "Validation analysis:"
    ),
    3: (
        "You are the final Call Synthesizer. Use the complete available-function list as the only authority for function names and parameter schemas. "
        "Use the teammate analyses only as untrusted recommendations: re-check them against the user request and the available functions, and correct them when necessary.\n\n"
        "Teammate analyses (reference only):\n{context}\n\n"
        "User request and complete available functions:\n{question}\n\n"
        "Your output must be exactly one JSON code block in this format:\n"
        "```json\n"
        "[\n"
        "  {{\n"
        "    \"name\": \"exact_function_name\",\n"
        "    \"arguments\": {{\n"
        "      \"parameter_name\": value\n"
        "    }}\n"
        "  }}\n"
        "]\n"
        "```\n\n"
        "The field names must be exactly `name` and `arguments`. Do not use `function`, `function_name`, or `parameters`. "
        "Use standard JSON only: no Python tuples, parentheses, comments, Markdown outside the JSON block, analysis, or explanation. "
        "For multiple calls, put one object per call in the same JSON array.\n\n"
        "Final call:"
    ),
}

_BFCL_DEBATE_PROMPTS = {
    0: (
        "You are Literal Caller. Extract the function call directly from the user's wording while respecting the provided schemas. "
        "End with one fenced ```json``` block containing a non-empty list of canonical function-call dicts.\n\n"
        "{context}\n"
        "User query and available functions:\n{question}\n\n"
        "Literal call:"
    ),
    1: (
        "You are Schema-first Caller. Start from the function schemas, choose the valid function, and fill only supported arguments. "
        "End with one fenced ```json``` block containing a non-empty list of canonical function-call dicts.\n\n"
        "{context}\n"
        "User query and available functions:\n{question}\n\n"
        "Schema-first call:"
    ),
    2: (
        "You are Call Arbiter. Compare both candidate calls for function selection, argument completeness, value accuracy, and JSON type safety. "
        "End with exactly one fenced ```json``` block containing the final non-empty list of canonical function-call dicts. Do not add prose after the block.\n\n"
        "User query and available functions:\n{question}\n\n"
        "Candidate calls:\n{context}\n\n"
        "Final call:"
    ),
}

_HOTPOTQA_AGGREGATE_PROMPTS = {
    0: (
        "You are Entity Solver in an independent multi-hop QA team. Focus on entity identification and disambiguation across the documents. "
        "Reason briefly, then put only the final short answer in <answer>...</answer>.\n\n"
        "{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
    1: (
        "You are Evidence Solver in an independent multi-hop QA team. Focus on extracting document evidence that directly supports the answer. "
        "Reason briefly, then put only the final short answer in <answer>...</answer>.\n\n"
        "{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
    2: (
        "You are Bridge Solver in an independent multi-hop QA team. Focus on connecting facts across documents to bridge from the question to the answer. "
        "Reason briefly, then put only the final short answer in <answer>...</answer>.\n\n"
        "{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
}

_HOTPOTQA_REFLECT_PROMPTS = {
    0: (
        "You are Initial Answerer for multi-hop QA. Read all provided documents, identify the bridge entities, and give an initial answer. "
        "Put only the final short answer in <answer>...</answer>.\n\n"
        "{context}\n\n"
        "Question:\n{question}\n\n"
        "Initial answer:"
    ),
    1: (
        "You are Evidence Auditor. Check whether the previous answer is supported by the documents and identify missing or contradicted evidence. "
        "Then provide a corrected short answer in <answer>...</answer>.\n\n"
        "Question:\n{question}\n\n"
        "Previous answer:\n{context}\n\n"
        "Audit and corrected answer:"
    ),
    2: (
        "You are Bridge Repairer. Repair the cross-document reasoning chain using the evidence audit, then provide the best short answer. "
        "Put only the final answer in <answer>...</answer>.\n\n"
        "Question:\n{question}\n\n"
        "Previous stage:\n{context}\n\n"
        "Repaired answer:"
    ),
    3: (
        "You are Final Answer Normalizer. Verify the answer, remove extra wording, and output a 1-5 word answer. "
        "Output only <answer>short answer</answer>.\n\n"
        "Question:\n{question}\n\n"
        "Candidate answer and reasoning:\n{context}\n\n"
        "Final answer:"
    ),
}

_HOTPOTQA_HIERARCHICAL_PROMPTS = {
    0: (
        "You are Document Analyst. Extract relevant entities, dates, titles, relationships, and facts from the provided documents. "
        "Do not guess the final answer unless the evidence is explicit.\n\n"
        "{context}\n\n"
        "Question:\n{question}\n\n"
        "Relevant facts:"
    ),
    1: (
        "You are Evidence Validator. Verify which document facts support or refute possible answers, and identify the strongest evidence chain. "
        "Do not output a final answer tag.\n\n"
        "{context}\n\n"
        "Question:\n{question}\n\n"
        "Validated evidence:"
    ),
    2: (
        "You are Reasoning Specialist. Build a concise multi-hop reasoning path that connects the question to the answer using document facts. "
        "Do not output a final answer tag.\n\n"
        "{context}\n\n"
        "Question:\n{question}\n\n"
        "Reasoning path:"
    ),
    3: (
        "You are Answer Summarizer. Synthesize the teammate analyses and output only the final short answer. "
        "Use <answer>...</answer> tags and put no reasoning inside the tags.\n\n"
        "Teammate analyses:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Final answer:"
    ),
}

_HOTPOTQA_DEBATE_PROMPTS = {
    0: (
        "You are Evidence-first Solver. Solve the multi-hop question by grounding every step in the provided documents. "
        "Put only the final short answer in <answer>...</answer>.\n\n"
        "{context}\n\n"
        "Question:\n{question}\n\n"
        "Evidence-first answer:"
    ),
    1: (
        "You are Reasoning-first Solver. Solve the multi-hop question by explicitly identifying the bridge relation between documents. "
        "Put only the final short answer in <answer>...</answer>.\n\n"
        "{context}\n\n"
        "Question:\n{question}\n\n"
        "Reasoning-first answer:"
    ),
    2: (
        "You are Answer Arbiter. Compare both candidate answers for evidence grounding and bridge reasoning. "
        "Output only the best short final answer in <answer>...</answer>.\n\n"
        "Question:\n{question}\n\n"
        "Candidate answers:\n{context}\n\n"
        "Final answer:"
    ),
}

_SINGLE_AGENT_PROMPTS = {
    "agi": (
        "Output the final answer inside <answer>...</answer>.\n\n"
        "Problem:\n{question}\n\nAnswer:"
    ),
    "gpqa": (
        "Output one option letter A, B, C, or D inside <answer>...</answer>.\n\n"
        "Question:\n{question}\n\nAnswer:"
    ),
    "bfcl": (
        "Output exactly one fenced JSON block containing a non-empty JSON array. "
        "Each item must be {\"function_name\": {\"parameter\": value}}. "
        "Put all calls in the same array. Output no other text.\n\n"
        "{question}\n\nOutput:"
    ),
    "hotpotqa": (
        "Output only a concise answer inside <answer>...</answer>.\n\n"
        "{question}\n\nAnswer:"
    ),
    "lcb": (
        "Output only complete raw Python 3 code, without Markdown or explanation.\n\n"
        "Problem:\n{question}\n\nCode:"
    ),
}

_SINGLE_AGENT_COT_PROMPTS = {
    "agi": (
        "Think step by step and show the reasoning. Then output the final answer "
        "inside <answer>...</answer>.\n\nProblem:\n{question}\n\nAnswer:"
    ),
    "gpqa": (
        "Think step by step and show the reasoning. Then output one option letter "
        "A, B, C, or D inside <answer>...</answer>.\n\nQuestion:\n{question}\n\nAnswer:"
    ),
    "bfcl": (
        "Think step by step about the function call. Then output exactly one fenced "
        "JSON block containing a non-empty JSON array. Each item must be "
        "{\"function_name\": {\"parameter\": value}}. Put all calls in "
        "the same array. Keep reasoning outside the JSON block.\n\n"
        "{question}\n\nOutput:"
    ),
    "hotpotqa": (
        "Think step by step and show the reasoning. Then output only a concise "
        "answer inside <answer>...</answer>.\n\n{question}\n\nAnswer:"
    ),
    "lcb": (
        "Think step by step and show the reasoning. Then output one fenced Python 3 "
        "code block containing only the complete solution code.\n\n"
        "Problem:\n{question}\n\nCode:"
    ),
}


_SINGLE_AGENT_TPE_PROMPTS = {
    "agi": (
        "Solve the problem carefully. Check the final result before answering. "
        "Return only the final answer inside <answer>...</answer>. "
        "Do not output any text outside the tags.\n\n"
        "Problem:\n{question}\n\nAnswer:"
    ),
    "gpqa": (
        "Determine the best choice. Return exactly one letter: A, B, C, or D. "
        "Put the letter inside <answer>...</answer> and output nothing outside the tags.\n\n"
        "Question:\n{question}\n\nAnswer:"
    ),
    "bfcl": (
        "Use only the functions provided in the prompt. Return exactly one fenced "
        "JSON block containing a non-empty JSON array of function calls. "
        "Each call must use the form {\"function_name\": {\"parameter\": value}}. "
        "Use valid JSON, include all required arguments, and output no other text.\n\n"
        "{question}\n\nOutput:"
    ),
    "hotpotqa": (
        "Identify the answer requested by the question and provide the shortest "
        "correct answer supported by the given context. Put only that answer inside "
        "<answer>...</answer>, with no text outside the tags.\n\n"
        "{question}\n\nAnswer:"
    ),
    "lcb": (
        "Implement the complete solution. Check that the code is valid Python 3 and "
        "handles the required input and output. Return only executable raw Python 3 "
        "code, without Markdown or explanation.\n\n"
        "Problem:\n{question}\n\nCode:"
    ),
}

_SINGLE_AGENT_SPO_PROMPTS = {
    "agi": (
        "Solve the problem carefully and check the final result. "
        "Return the exact final answer inside <answer>...</answer>. "
        "Do not output any text outside the tags.\n\n"
        "Problem:\n{question}\n\nAnswer:"
    ),
    "gpqa": (
        "Determine the best choice from the four options. Return exactly one letter: "
        "A, B, C, or D. Put the letter inside <answer>...</answer> and output "
        "nothing outside the tags.\n\n"
        "Question:\n{question}\n\nAnswer:"
    ),
    "bfcl": (
        "Follow only the function definitions provided in the prompt. Output one "
        "fenced JSON block with a non-empty array of function calls. "
        "Every call must have exactly the form {\"function_name\": {\"parameter\": value}}. "
        "Use the exact function and argument names, valid JSON, and no other text.\n\n"
        "{question}\n\nOutput:"
    ),
    "hotpotqa": (
        "Identify the answer requested by the question and respond with the shortest "
        "correct answer supported by the given context. Put only that answer inside "
        "<answer>...</answer>.\n\n"
        "{question}\n\nAnswer:"
    ),
    "lcb": (
        "Write the complete Python 3 program required by the problem. "
        "Implement the algorithm and the required standard input/output behavior. "
        "Return only the raw program text, with no Markdown or explanation.\n\n"
        "Problem:\n{question}\n\nCode:"
    ),
}

DATASET_GRAPH_INITIAL_PROMPTS = {
    "agi": {
        GraphType.SINGLE_AGENT: {0: _SINGLE_AGENT_PROMPTS["agi"]},
        GraphType.SINGLE_AGENT_COT: {0: _SINGLE_AGENT_COT_PROMPTS["agi"]},
        GraphType.SINGLE_AGENT_TPE: {0: _SINGLE_AGENT_TPE_PROMPTS["agi"]},
        GraphType.SINGLE_AGENT_SPO: {0: _SINGLE_AGENT_SPO_PROMPTS["agi"]},
    },
    "gpqa": {
        GraphType.SINGLE_AGENT: {0: _SINGLE_AGENT_PROMPTS["gpqa"]},
        GraphType.SINGLE_AGENT_COT: {0: _SINGLE_AGENT_COT_PROMPTS["gpqa"]},
        GraphType.SINGLE_AGENT_TPE: {0: _SINGLE_AGENT_TPE_PROMPTS["gpqa"]},
        GraphType.SINGLE_AGENT_SPO: {0: _SINGLE_AGENT_SPO_PROMPTS["gpqa"]},
    },
    "lcb": {
        GraphType.AGGREGATE: _LCB_AGGREGATE_PROMPTS,
        GraphType.REFLECT: _LCB_REFLECT_PROMPTS,
        GraphType.HIERARCHICAL: _LCB_HIERARCHICAL_PROMPTS,
        GraphType.DEBATE: _LCB_DEBATE_PROMPTS,
        GraphType.SINGLE_AGENT: {0: _SINGLE_AGENT_PROMPTS["lcb"]},
        GraphType.SINGLE_AGENT_COT: {0: _SINGLE_AGENT_COT_PROMPTS["lcb"]},
        GraphType.SINGLE_AGENT_TPE: {0: _SINGLE_AGENT_TPE_PROMPTS["lcb"]},
        GraphType.SINGLE_AGENT_SPO: {0: _SINGLE_AGENT_SPO_PROMPTS["lcb"]},
    },
    "bfcl": {
        GraphType.AGGREGATE: _BFCL_AGGREGATE_PROMPTS,
        GraphType.REFLECT: _BFCL_REFLECT_PROMPTS,
        GraphType.HIERARCHICAL: _BFCL_HIERARCHICAL_PROMPTS,
        GraphType.DEBATE: _BFCL_DEBATE_PROMPTS,
        GraphType.SINGLE_AGENT: {0: _SINGLE_AGENT_PROMPTS["bfcl"]},
        GraphType.SINGLE_AGENT_COT: {0: _SINGLE_AGENT_COT_PROMPTS["bfcl"]},
        GraphType.SINGLE_AGENT_TPE: {0: _SINGLE_AGENT_TPE_PROMPTS["bfcl"]},
        GraphType.SINGLE_AGENT_SPO: {0: _SINGLE_AGENT_SPO_PROMPTS["bfcl"]},
    },
    "hotpotqa": {
        GraphType.AGGREGATE: _HOTPOTQA_AGGREGATE_PROMPTS,
        GraphType.REFLECT: _HOTPOTQA_REFLECT_PROMPTS,
        GraphType.HIERARCHICAL: _HOTPOTQA_HIERARCHICAL_PROMPTS,
        GraphType.DEBATE: _HOTPOTQA_DEBATE_PROMPTS,
        GraphType.SINGLE_AGENT: {0: _SINGLE_AGENT_PROMPTS["hotpotqa"]},
        GraphType.SINGLE_AGENT_COT: {0: _SINGLE_AGENT_COT_PROMPTS["hotpotqa"]},
        GraphType.SINGLE_AGENT_TPE: {0: _SINGLE_AGENT_TPE_PROMPTS["hotpotqa"]},
        GraphType.SINGLE_AGENT_SPO: {0: _SINGLE_AGENT_SPO_PROMPTS["hotpotqa"]},
    },
}

DATASET_PROMPT_ALIASES = {
    "agi": "agi",
    "math": "agi",
    "gpqa": "gpqa",
    "livecodebench": "lcb",
    "livecodebench_454": "lcb",
    "livecodebench_v6": "lcb",
    "lcb_eval_100": "lcb",
    "bfcl_simple": "bfcl",
    "bfcl_v3": "bfcl",
    "bfcl_eval_100": "bfcl",
    "hotpotqa": "hotpotqa",
    "hotpotqa_eval_100": "hotpotqa",
}


# HotpotQA-only final-answer contract. The original templates above remain
# unchanged; this suffix is applied only to the selected HotpotQA graphs.
_HOTPOTQA_FINAL_OUTPUT_SUFFIX = (
    "\n\nFinal output rule: Return exactly one short answer phrase inside one "
    "<answer>...</answer> tag. Put only the entity, title, place, person, "
    "organization, or date that directly answers the question inside the tag. "
    "Do not include explanations, evidence, reasoning, citations, or a complete sentence."
)


def get_dataset_graph_prompt_map(dataset: str, graph_type: GraphType):
    """Return a dataset- and graph-specific initial prompt map, if available."""
    canonical = DATASET_PROMPT_ALIASES.get(dataset)
    if not canonical:
        return None
    prompt_map = DATASET_GRAPH_INITIAL_PROMPTS.get(canonical, {}).get(graph_type)

    # Council has no dataset-specific map in the legacy table. Build a
    # HotpotQA-only copy from the generic QA Council templates so that the
    # generic QA prompts used by AGI/GPQA remain untouched.
    if canonical == "hotpotqa" and graph_type is GraphType.COUNCIL:
        council_map = COUNCIL_PROMPTS.get(TaskType.QA, {})
        prompt_map = {
            int(agent_id): template
            for agent_id, template in council_map.items()
            if int(agent_id) < 6
        }
        return {
            agent_id: template + _HOTPOTQA_FINAL_OUTPUT_SUFFIX
            for agent_id, template in prompt_map.items()
        }

    if not prompt_map:
        return None

    # Work on a copy so the original HotpotQA templates are preserved.
    result = dict(prompt_map)
    if canonical == "hotpotqa":
        if graph_type is GraphType.AGGREGATE:
            target_ids = list(result)
        elif graph_type is GraphType.HIERARCHICAL:
            target_ids = [3]
        else:
            # Reflect intentionally keeps its original prompt version.
            target_ids = []
        for agent_id in target_ids:
            if agent_id in result:
                result[agent_id] += _HOTPOTQA_FINAL_OUTPUT_SUFFIX
    return result
# ============================================================
# Council (Decentralized 2-Round Peer Review) Prompts
# ============================================================
# Round 1: Agents 0,1,2 answer independently with different personas
# Round 2: Agents 3,4,5 see Round 1 answers (truncated) and revise
# Agent 6: AGGREGATOR (mechanical majority vote, no LLM)

COUNCIL_PROMPTS = {
    TaskType.MATH: {
        "0": (
            "You are a Direct Reasoner. Given the math problem, reason step-by-step "
            "and put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>. "
            "Output ONLY the step-by-step reasoning and the final answer.\n\n"
            "Problem: {question}\n\n"
            "Your solution:"
        ),
        "1": (
            "You are a Multi-angle Thinker. Consider at least two different approaches "
            "to solve this math problem, then choose the best one. "
            "Put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>. "
            "Output ONLY your analysis and the final answer.\n\n"
            "Problem: {question}\n\n"
            "Your solution:"
        ),
        "2": (
            "You are a Counterfactual Verifier. Before solving, think about common "
            "mistakes and pitfalls that could lead to wrong answers. Then solve carefully. "
            "Put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>. "
            "Output ONLY your verification and the final answer.\n\n"
            "Problem: {question}\n\n"
            "Your solution:"
        ),
        "3": (
            "You are a Direct Reasoner. Other experts have provided their answers below. "
            "Review their solutions, identify agreements and disagreements, then provide "
            "your revised final answer. Put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>. "
            "Output ONLY your revised reasoning and the final answer.\n\n"
            "Other experts answers:\n{context}\n\n"
            "Problem: {question}\n\n"
            "Your revised solution:"
        ),
        "4": (
            "You are a Multi-angle Thinker. Other experts have provided their answers below. "
            "Compare their approaches, synthesize the best ideas, then provide your revised "
            "final answer. Put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>. "
            "Output ONLY your synthesis and the final answer.\n\n"
            "Other experts answers:\n{context}\n\n"
            "Problem: {question}\n\n"
            "Your revised solution:"
        ),
        "5": (
            "You are a Counterfactual Verifier. Other experts have provided their answers below. "
            "Check their work for errors, edge cases, and overlooked pitfalls, then provide "
            "your verified final answer. Put the final answer inside <answer>YOUR_FINAL_ANSWER</answer>. "
            "Output ONLY your verification and the final answer.\n\n"
            "Other experts answers:\n{context}\n\n"
            "Problem: {question}\n\n"
            "Your verified solution:"
        ),
    },

    TaskType.MATH_CHOICE: {
        "0": (
            "You are a Direct Reasoner. Solve the multiple-choice problem step-by-step. "
            "Your final answer must be a single letter A, B, C, or D inside <answer>X</answer>. "
            "Output ONLY your reasoning and the final choice.\n\n"
            "Problem: {question}\n\n"
            "Your solution:"
        ),
        "1": (
            "You are a Multi-angle Thinker. Consider at least two different ways to solve "
            "this choice problem, then select the best answer. Your final answer must be "
            "a single letter inside <answer>X</answer>. Output ONLY your analysis and the final choice.\n\n"
            "Problem: {question}\n\n"
            "Your solution:"
        ),
        "2": (
            "You are a Counterfactual Verifier. Before answering, consider why each wrong "
            "option might seem correct and explain the trap. Then give the correct choice "
            "inside <answer>X</answer>. Output ONLY your verification and the final choice.\n\n"
            "Problem: {question}\n\n"
            "Your solution:"
        ),
        "3": (
            "You are a Direct Reasoner. Other experts answers are below. Review them and "
            "provide your revised final choice inside <answer>X</answer>. Output ONLY your "
            "revised reasoning and the final choice.\n\n"
            "Other experts answers:\n{context}\n\n"
            "Problem: {question}\n\n"
            "Your revised solution:"
        ),
        "4": (
            "You are a Multi-angle Thinker. Other experts answers are below. Synthesize "
            "their insights and provide your revised final choice inside <answer>X</answer>. "
            "Output ONLY your synthesis and the final choice.\n\n"
            "Other experts answers:\n{context}\n\n"
            "Problem: {question}\n\n"
            "Your revised solution:"
        ),
        "5": (
            "You are a Counterfactual Verifier. Other experts answers are below. Verify "
            "their reasoning, catch any errors, and provide your final choice inside "
            "<answer>X</answer>. Output ONLY your verification and the final choice.\n\n"
            "Other experts answers:\n{context}\n\n"
            "Problem: {question}\n\n"
            "Your verified solution:"
        ),
    },

    TaskType.REASONING_CHOICE: {
        "0": (
            "You are a Direct Reasoner. Analyze the question and reason step-by-step to "
            "the correct choice. Your final answer must be a single letter inside "
            "<answer>X</answer>. Output ONLY your reasoning and the final choice.\n\n"
            "Question: {question}\n\n"
            "Your analysis:"
        ),
        "1": (
            "You are a Multi-angle Thinker. Approach this question from multiple "
            "perspectives, then select the best answer. Your final answer must be a "
            "single letter inside <answer>X</answer>. Output ONLY your analysis and the final choice.\n\n"
            "Question: {question}\n\n"
            "Your analysis:"
        ),
        "2": (
            "You are a Counterfactual Verifier. Identify why each distractor might be "
            "tempting, explain why it is wrong, then give the correct choice inside "
            "<answer>X</answer>. Output ONLY your verification and the final choice.\n\n"
            "Question: {question}\n\n"
            "Your analysis:"
        ),
        "3": (
            "You are a Direct Reasoner. Other experts answers are below. Review them and "
            "provide your revised final choice inside <answer>X</answer>. Output ONLY your "
            "revised reasoning and the final choice.\n\n"
            "Other experts answers:\n{context}\n\n"
            "Question: {question}\n\n"
            "Your revised analysis:"
        ),
        "4": (
            "You are a Multi-angle Thinker. Other experts answers are below. Synthesize "
            "their insights and provide your revised final choice inside <answer>X</answer>. "
            "Output ONLY your synthesis and the final choice.\n\n"
            "Other experts answers:\n{context}\n\n"
            "Question: {question}\n\n"
            "Your revised analysis:"
        ),
        "5": (
            "You are a Counterfactual Verifier. Other experts answers are below. Verify "
            "their reasoning and provide your final choice inside <answer>X</answer>. "
            "Output ONLY your verification and the final choice.\n\n"
            "Other experts answers:\n{context}\n\n"
            "Question: {question}\n\n"
            "Your verified analysis:"
        ),
    },

    TaskType.CODE: {
        "0": (
            "You are a Direct Implementer. Write a clean, self-contained Python function "
            "that solves the problem. Put all code in a markdown code block. "
            "Output ONLY the Python code block.\n\n"
            "Problem: {question}\n\n"
            "Your code:"
        ),
        "1": (
            "You are a Multi-angle Implementer. Consider at least two algorithmic approaches, "
            "compare their trade-offs, then implement the best one as a self-contained "
            "Python function. Output ONLY the Python code block.\n\n"
            "Problem: {question}\n\n"
            "Your code:"
        ),
        "2": (
            "You are a Defensive Implementer. Think about edge cases, invalid inputs, and "
            "boundary conditions first, then write a robust self-contained Python function. "
            "Output ONLY the Python code block.\n\n"
            "Problem: {question}\n\n"
            "Your code:"
        ),
        "3": (
            "You are a Direct Implementer. Other experts implementations are below. Review "
            "them and provide your revised self-contained Python function. Output ONLY "
            "the Python code block.\n\n"
            "Other experts code:\n{context}\n\n"
            "Problem: {question}\n\n"
            "Your revised code:"
        ),
        "4": (
            "You are a Multi-angle Implementer. Other experts implementations are below. "
            "Synthesize the best ideas from all solutions into a single self-contained "
            "Python function. Output ONLY the Python code block.\n\n"
            "Other experts code:\n{context}\n\n"
            "Problem: {question}\n\n"
            "Your revised code:"
        ),
        "5": (
            "You are a Defensive Implementer. Other experts implementations are below. "
            "Check them for edge-case bugs and correctness, then write a verified Python "
            "function. Output ONLY the Python code block.\n\n"
            "Other experts code:\n{context}\n\n"
            "Problem: {question}\n\n"
            "Your verified code:"
        ),
    },

    TaskType.CODE_GENERATION: {
        "0": (
            "You are an Algorithm-first Coder. Design the correct algorithm first, then "
            "write a complete Python 3 solution that reads from stdin and writes to stdout. "
            "Output ONLY Python code in a ```python code block.\n\n"
            "Problem:\n{question}\n\n"
            "Solution:"
        ),
        "1": (
            "You are an Efficiency-first Coder. Prioritize optimal time and space complexity, "
            "then write the most efficient complete Python 3 solution (stdin/stdout). "
            "Output ONLY Python code in a ```python code block.\n\n"
            "Problem:\n{question}\n\n"
            "Solution:"
        ),
        "2": (
            "You are a Robustness-first Coder. Identify all edge cases and boundary "
            "conditions first, then write a bulletproof Python 3 solution (stdin/stdout). "
            "Output ONLY Python code in a ```python code block.\n\n"
            "Problem:\n{question}\n\n"
            "Solution:"
        ),
        "3": (
            "You are an Algorithm-first Coder. Other experts solutions are below. Review "
            "them and provide your revised Python 3 solution. Output ONLY Python code in "
            "a ```python code block.\n\n"
            "Other experts solutions:\n{context}\n\n"
            "Problem:\n{question}\n\n"
            "Revised solution:"
        ),
        "4": (
            "You are an Efficiency-first Coder. Other experts solutions are below. "
            "Synthesize the best algorithmic ideas into an optimal Python 3 solution. "
            "Output ONLY Python code in a ```python code block.\n\n"
            "Other experts solutions:\n{context}\n\n"
            "Problem:\n{question}\n\n"
            "Revised solution:"
        ),
        "5": (
            "You are a Robustness-first Coder. Other experts solutions are below. Audit "
            "them for edge-case failures and correctness bugs, then write a verified "
            "Python 3 solution. Output ONLY Python code in a ```python code block.\n\n"
            "Other experts solutions:\n{context}\n\n"
            "Problem:\n{question}\n\n"
            "Verified solution:"
        ),
    },

    TaskType.FUNCTION_CALLING: {
        "0": 'Read the user query and extract the function call directly from what is stated. Do not infer, do not add. Output a JSON array: [{"name": "function_name", "arguments": {...}}].\n\nUser Query and Available Functions:\n{question}\n\nFunction Call:',
        "1": 'Consider multiple possible interpretations of the user query, then choose the most reasonable one. Output a JSON array: [{"name": "function_name", "arguments": {...}}].\n\nUser Query and Available Functions:\n{question}\n\nFunction Call:',
        "2": 'Before outputting, check for common errors: wrong function name, missing required arguments, wrong types. Output a JSON array: [{"name": "function_name", "arguments": {...}}].\n\nUser Query and Available Functions:\n{question}\n\nFunction Call:',
        "3": 'Below are the outputs from three independent analyses of the same query. Consider all perspectives, then output your best function call. Output a JSON array: [{"name": "function_name", "arguments": {...}}].\n\nOther Analyses:\n{context}\n\nUser Query and Available Functions:\n{question}\n\nFunction Call:',
        "4": 'Below are the outputs from three independent analyses of the same query. Consider all perspectives, then output your best function call. Output a JSON array: [{"name": "function_name", "arguments": {...}}].\n\nOther Analyses:\n{context}\n\nUser Query and Available Functions:\n{question}\n\nFunction Call:',
        "5": 'Below are the outputs from three independent analyses of the same query. Consider all perspectives, then output your best function call. Output a JSON array: [{"name": "function_name", "arguments": {...}}].\n\nOther Analyses:\n{context}\n\nUser Query and Available Functions:\n{question}\n\nFunction Call:',
    },

    TaskType.QA: {
        "0": (
            "You are a Document Analyst. Read all provided documents and extract all facts "
            "relevant to the question. Identify key entities and cross-document connections. "
            "Output ONLY the final short answer (1-5 words) in <answer>...</answer> tags.\n\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        "1": (
            "You are an Evidence Tracer. Trace the evidence chain across documents, verify "
            "each hop in the reasoning, and identify the strongest direct evidence. "
            "Output ONLY the final short answer (1-5 words) in <answer>...</answer> tags.\n\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        "2": (
            "You are a Contradiction Detector. Check for conflicting information across "
            "documents, resolve any contradictions by weighing source reliability, and "
            "extract the correct answer. Output ONLY the final short answer (1-5 words) "
            "in <answer>...</answer> tags.\n\n"
            "{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        "3": (
            "You are a Document Analyst. Other experts answers are below. Compare their "
            "findings with the documents, resolve disagreements, and output the final "
            "answer in <answer>...</answer> tags.\n\n"
            "Other experts answers:\n{context}\n\n"
            "Question: {question}\n\n"
            "Revised answer:"
        ),
        "4": (
            "You are an Evidence Tracer. Other experts answers are below. Re-check the "
            "evidence chain for each candidate answer and output the best-supported answer "
            "in <answer>...</answer> tags.\n\n"
            "Other experts answers:\n{context}\n\n"
            "Question: {question}\n\n"
            "Revised answer:"
        ),
        "5": (
            "You are a Contradiction Detector. Other experts answers are below. Identify "
            "and resolve any conflicts between them, then output the correct answer in "
            "<answer>...</answer> tags.\n\n"
            "Other experts answers:\n{context}\n\n"
            "Question: {question}\n\n"
            "Revised answer:"
        ),
    },
}


# Compact, task-type-specific Council prompts.
# Agents 0/3, 1/4, and 2/5 intentionally share the exact same template;
# the second-round context is injected by Agent at runtime.
def _make_council_prompts(output_contract: str) -> dict:
    roles = (
        "Direct Solver: use a direct and reliable method to solve the problem.",
        "Multi-angle Analyst: examine the problem from different angles and choose the most credible conclusion.",
        "Result Verifier: focus on finding errors, contradictions, omissions, and edge cases.",
    )
    prompts = {}
    for role_index, role in enumerate(roles):
        prompt = (
            f"You are a {role}\n\n"
            "Solve the problem independently. If reference results from other Agents are provided, check their correctness, correct them when necessary, and do not copy them blindly.\n\n"
            "Question:\n{question}\n\n"
            "{context_block}\n\n"
            f"{output_contract}"
        )
        prompts[str(role_index)] = prompt
        prompts[str(role_index + 3)] = prompt
    return prompts


_COUNCIL_TEXT_CONTRACT = (
    "Retain the necessary reasoning and write the final result exactly as follows:\n"
    "<answer>FINAL_ANSWER</answer>\n\n"
    "The tag must contain only the final answer, with no analysis or explanation."
)

_COUNCIL_CODE_CONTRACT = (
    "The final output must be complete, executable Python code.\n"
    "Output only one Python code block; do not output analysis, explanation, or any other text:\n\n"
    "```python\n# code\n```"
)

_COUNCIL_TOOL_CONTRACT = (
    "You are a tool-calling agent.\n\n"
    "Use the user request, tool definitions, and other agents' results to produce the correct function calls. "
    "Correct any errors you find.\n\n"
    "The required format is a JSON array of function-call objects. "
    "In each object, the key is the exact function name and the value is an object of parameter names and values. "
    "Do not write field names such as function_name or parameter.\n\n"
    "Example format (replace the placeholders; do not output the placeholders):\n\n"
    "[\n"
    "  {{\n"
    "    \"<exact_function_name>\": {{\n"
    "      \"<exact_parameter_name>\": <value>\n"
    "    }}\n"
    "  }}\n"
    "]\n"
    "Use scalar or structured values exactly as declared by the tool schema; do not wrap values in extra arrays. "
    "Put multiple calls in the same array. Output no analysis, explanation, or Markdown."
)


# Override the legacy verbose Council prompts above with the shared templates.
# The mapping keeps the existing TaskType interface unchanged.
COUNCIL_PROMPTS = {
    TaskType.MATH: _make_council_prompts(_COUNCIL_TEXT_CONTRACT),
    TaskType.MATH_CHOICE: _make_council_prompts(_COUNCIL_TEXT_CONTRACT),
    TaskType.REASONING_CHOICE: _make_council_prompts(_COUNCIL_TEXT_CONTRACT),
    TaskType.QA: _make_council_prompts(_COUNCIL_TEXT_CONTRACT),
    TaskType.CODE: _make_council_prompts(_COUNCIL_CODE_CONTRACT),
    TaskType.CODE_GENERATION: _make_council_prompts(_COUNCIL_CODE_CONTRACT),
    TaskType.FUNCTION_CALLING: _make_council_prompts(_COUNCIL_TOOL_CONTRACT),
}
