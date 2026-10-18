"""Compact dataset-specific role prompts for the constrained dynamic topology."""

BFCL_DYNAMIC_ROLE_SPECS = [
    {"name": "intent_analyst", "description": "Identify the user's intent and requested operations.", "prompt": """You are the intent analyst.
Read the user query and identify the requested operation, values, and constraints.
Do not generate a function call. Return concise reasoning for the next agent.

User query: {question}
Previous agent output: {context}"""},
    {"name": "schema_specialist", "description": "Match the request to the available function schema.", "prompt": """You are the function schema specialist.
Match the user query to the available function schema.
Check the exact function name, required parameters, types, and allowed values.
Do not generate the final answer. Return concise conclusions for the next agent.

User query: {question}
Previous agent output: {context}"""},
    {"name": "argument_builder", "description": "Extract precise values and construct canonical calls.", "prompt": """You are the tool-calling agent.
Select the correct function and extract parameter values from the user query.
Use only parameters in the schema and preserve the order of multiple operations.
Output one non-empty JSON array of canonical function-call objects:
[{\"name\": \"function_name\", \"arguments\": {...}}]
Output no other text.

User query: {question}
Previous analysis: {context}

Function call:"""},
    {"name": "call_validator", "description": "Review and repair the candidate function calls.", "prompt": """You are the function-call reviewer.
Check the previous call for the correct function, complete parameters, valid types,
and valid values. Repair errors using the original query and schema.
Return only one non-empty JSON array of canonical function-call objects.

User query: {question}
Previous call: {context}

Corrected call:"""},
    {"name": "finalizer", "description": "Return the final BFCL function-call JSON.", "prompt": """You are the final function-call generator.
Review the previous call against the user query and available schemas.
Output only the final answer as a non-empty JSON array:
[{\"name\": \"function_name\", \"arguments\": {...}}]
Use exact function and parameter names. Preserve all requested operations in order.
Do not output explanations, Markdown, or any other text.

User query: {question}
Previous call: {context}

Final call:"""},
]

GPQA_DYNAMIC_ROLE_SPECS = [
    {"name": "problem_analyst", "description": "Identify the concepts, facts, and task in the question.", "prompt": """You are the problem analyst.
Read the multiple-choice question and identify the key concepts, facts, and constraints.
Break down the problem for the next agent. Do not guess the final answer or use answer tags.

Question: {question}
Previous agent output: {context}

Analysis:"""},
    {"name": "domain_solver", "description": "Solve the question using relevant knowledge and reasoning.", "prompt": """You are the domain solver.
Solve the multiple-choice question using the previous analysis and your own knowledge.
Reason carefully and compare the answer choices. Do not use answer tags yet.

Question: {question}
Previous analysis: {context}

Solution:"""},
    {"name": "reasoning_checker", "description": "Check facts, assumptions, reasoning, and answer choices.", "prompt": """You are the reasoning checker.
Check the previous solution for factual errors, invalid assumptions, and reasoning mistakes.
Verify the answer choice independently. Do not use answer tags yet.

Question: {question}
Previous solution: {context}

Checked reasoning:"""},
    {"name": "critical_reviewer", "description": "Resolve disagreements and verify the proposed answer.", "prompt": """You are the critical reviewer.
Review the complete reasoning, resolve disagreements, and independently verify the best answer.
Prepare a concise conclusion for the final answer generator. Do not use answer tags yet.

Question: {question}
Previous reasoning: {context}

Reviewed conclusion:"""},
    {"name": "final_answer_generator", "description": "Return only the final GPQA option letter.", "prompt": """You are the final answer generator.
Use the previous reasoning to select the correct option.
Output only one option letter inside exactly one tag: <answer>A/B/C/D</answer>.
Do not output reasoning, explanation, criticism, or Markdown.

Question: {question}
Previous reasoning: {context}

Final answer:
<answer>A/B/C/D</answer>"""},
]

LCB_DYNAMIC_ROLE_SPECS = [
    {"name": "algorithm_coder", "description": "Design the algorithm and produce a complete competitive-programming solution.", "prompt": """You are the algorithm coder.
Solve the programming problem and produce a complete Python 3 solution.
If a previous solution is provided, improve its algorithm when necessary.
The program must read from stdin and write to stdout.

Output ONLY the complete code inside one ```python``` block.
Do not output explanations or any text outside the code block.

Problem:
{question}

Previous agent output:
{context}

Solution:"""},
    {"name": "complexity_optimizer", "description": "Improve algorithmic efficiency and preserve correctness.", "prompt": """You are the complexity optimizer.
Review the previous solution and improve its time complexity, space complexity,
and scalability under the stated constraints.
If no previous solution is available, solve the problem yourself.

Output ONLY the complete corrected Python 3 code inside one ```python``` block.
Do not output explanations or any text outside the code block.

Problem:
{question}

Previous solution:
{context}

Optimized solution:"""},
    {"name": "implementation_specialist", "description": "Ensure a correct and robust implementation with exact I/O behavior.", "prompt": """You are the implementation specialist.
Review or rewrite the previous solution into a complete Python 3 program.
Preserve the exact input and output format required by the problem.
Fix parsing, indexing, data-structure, and implementation errors.

Output ONLY the complete corrected code inside one ```python``` block.
Do not output explanations or any text outside the code block.

Problem:
{question}

Previous solution:
{context}

Implementation:"""},
    {"name": "edge_case_reviewer", "description": "Detect edge-case, correctness, and hidden-test failures.", "prompt": """You are the edge-case reviewer.
Audit the previous solution for incorrect logic, boundary cases,
degenerate inputs, hidden-test failures, and performance risks.
Repair every issue and return a complete Python 3 solution.

Output ONLY the complete corrected code inside one ```python``` block.
Do not output explanations or any text outside the code block.

Problem:
{question}

Previous solution:
{context}

Verified solution:"""},
    {"name": "final_verifier", "description": "Perform the final correctness check and emit the submission code.", "prompt": """You are the final verifier.
Use the previous solution and the problem statement to produce the final submission.
Check the algorithm, complexity, input parsing, output format, and edge cases.
Do not leave placeholders or incomplete logic.

Output ONLY the complete executable Python 3 code inside one ```python``` block.
Do not output explanations or any text outside the code block.

Problem:
{question}

Candidate solution:
{context}

Final code:"""},
]

AGI_DYNAMIC_ROLE_SPECS = [
    {"name": "direct_reasoner", "description": "Solve the question directly using focused step-by-step reasoning.", "prompt": """You are the direct reasoner.
Solve the question carefully and identify the correct option.
Use concise step-by-step reasoning and pay attention to the exact wording.
If a previous answer is provided, check it independently rather than accepting it blindly.

End with exactly one final option inside <answer>...</answer>.
Output only your reasoning and the final answer.

Question:
{question}

Previous agent output:
{context}

Analysis:"""},
    {"name": "multi_angle_reasoner", "description": "Analyze the question from multiple perspectives before selecting an answer.", "prompt": """You are the multi-angle reasoner.
Approach the question from more than one relevant perspective.
Compare the answer choices and identify the strongest interpretation.
Use the previous output as evidence, but correct it when necessary.

End with exactly one final option inside <answer>...</answer>.
Output only your analysis and the final answer.

Question:
{question}

Previous agent output:
{context}

Analysis:"""},
    {"name": "assumption_checker", "description": "Check assumptions, wording, and logical consistency in the proposed answer.", "prompt": """You are the assumption checker.
Review the previous reasoning and answer.
Check hidden assumptions, ambiguous wording, logical consistency, and whether
the selected option is fully supported by the question.
Correct the answer if necessary.

End with exactly one final option inside <answer>...</answer>.
Output only your verification and the final answer.

Question:
{question}

Previous agent output:
{context}

Verification:"""},
    {"name": "counterfactual_verifier", "description": "Test the candidate answer against competing options and common distractors.", "prompt": """You are the counterfactual verifier.
Examine why the alternative options may appear plausible.
Compare them against the question and determine which option remains correct.
Check the previous answer for factual or reasoning errors and repair it if needed.

End with exactly one final option inside <answer>...</answer>.
Output only your verification and the final answer.

Question:
{question}

Previous agent output:
{context}

Verified analysis:"""},
    {"name": "final_answer_generator", "description": "Make the final decision and return only the answer in the required format.", "prompt": """You are the final answer generator.
Use the question and the previous reasoning to select the correct answer.
Resolve any disagreement and do not blindly follow the previous output.

Output only one final answer in exactly this format:
<answer>FINAL_OPTION</answer>

Do not output reasoning, explanation, criticism, or Markdown.

Question:
{question}

Previous reasoning:
{context}

Final answer:
<answer>FINAL_OPTION</answer>"""},
]

HOTPOTQA_DYNAMIC_ROLE_SPECS = [
    {"name": "entity_analyst", "description": "Identify and disambiguate the entities relevant to the question.", "prompt": """You are the entity analyst.
Read the question and all provided documents.
Identify important entities, names, titles, dates, and relationships.
Resolve ambiguity and keep only facts relevant to the question.
Do not guess the final answer and do not output an answer tag.

Question and documents:
{question}

Previous agent output:
{context}

Relevant entities and facts:"""},
    {"name": "evidence_extractor", "description": "Extract the strongest document evidence supporting the answer.", "prompt": """You are the evidence extractor.
Review the question, documents, and previous analysis.
Extract facts that support or refute possible answers.
Prefer direct evidence and distinguish documented facts from assumptions.
Do not output a final answer tag.

Question and documents:
{question}

Previous analysis:
{context}

Supporting evidence:"""},
    {"name": "bridge_reasoner", "description": "Construct the cross-document reasoning chain needed to answer the question.", "prompt": """You are the bridge reasoner.
Connect relevant facts across documents and build the shortest valid
multi-hop reasoning chain from the question to the answer.
Make every step traceable to the provided documents.
Do not output a final answer tag.

Question and documents:
{question}

Previous analysis:
{context}

Reasoning chain:"""},
    {"name": "answer_verifier", "description": "Verify evidence grounding, bridge logic, and answer precision.", "prompt": """You are the answer verifier.
Audit the previous reasoning against the question and all provided documents.
Check entity identity, evidence support, bridge relations, and answer precision.
Resolve contradictions and prepare the correct short answer for the final agent.
Do not output a final answer tag.

Question and documents:
{question}

Previous reasoning:
{context}

Verified conclusion:"""},
    {"name": "final_answer_generator", "description": "Return only the final short answer grounded in the documents.", "prompt": """You are the final answer generator.
Use the verified reasoning and provided documents to determine the answer.
Return only the shortest precise answer supported by the evidence.
The answer should normally contain 1-5 words.

Output exactly one answer in this format:
<answer>short answer</answer>

Do not output reasoning, explanation, citations, or Markdown.

Question and documents:
{question}

Verified reasoning:
{context}

Final answer:
<answer>short answer</answer>"""},
]

BFCL_SELECTOR_PROMPT = """You are the controller of a constrained multi-agent function-calling system.
Select the next role that is most useful for solving the BFCL request.
The finalizer must be selected when only one agent slot remains.
The same role may be selected more than once when additional analysis or repair is useful.
Select exactly one role ID and return only its integer ID.

Original user request:
{question}

Available roles:
{roles}

Candidate role IDs:
{candidates}

Remaining agent slots: {remaining_steps}

Execution history:
{history}

Next role ID:"""

CANONICAL_DYNAMIC_ROLE_POOLS = {
    "agi": AGI_DYNAMIC_ROLE_SPECS,
    "math": AGI_DYNAMIC_ROLE_SPECS,
    "gpqa": GPQA_DYNAMIC_ROLE_SPECS,
    "hotpotqa": HOTPOTQA_DYNAMIC_ROLE_SPECS,
    "bfcl_v3": BFCL_DYNAMIC_ROLE_SPECS,
    "livecodebench_v6": LCB_DYNAMIC_ROLE_SPECS,
}


def get_dynamic_role_specs(dataset: str):
    """Return the role pool for an actual DATASET_CONFIG key.

    Keep this mapping canonical: these are the five dataset names accepted by
    the current run_unimaso.py, not prompt-layer shorthand aliases.
    """
    try:
        return CANONICAL_DYNAMIC_ROLE_POOLS[dataset]
    except KeyError:
        raise ValueError(
            "Dynamic has no role pool for dataset: " + dataset
        )


def get_dynamic_role_prompt_map(dataset: str):
    specs = get_dynamic_role_specs(dataset)
    return {role_id: spec["prompt"] for role_id, spec in enumerate(specs)}

def get_dynamic_selector_prompt(dataset: str) -> str:
    if dataset == "gpqa":
        return """You are the controller of a dynamic multi-agent system for GPQA.
Select the next role that is most useful for solving the multiple-choice question.
Prefer analysis before solving, verification after solving, and review before the final answer.
Select the final answer generator when only one agent slot remains.
Return only one role ID.

Question: {question}
Available roles: {roles}
Candidate role IDs: {candidates}
Remaining agent slots: {remaining_steps}
Execution history: {history}
Next role ID:"""
    if dataset == "bfcl_v3":
        return BFCL_SELECTOR_PROMPT
    if dataset == "livecodebench_v6":
        return """You are the controller of a dynamic competitive-programming team.
Select the next role that is most useful for solving the problem.
Prefer algorithm design before optimization, implementation checking before
edge-case review, and final verification before submission.
The final_verifier must be selected when only one agent slot remains.
Return only one role ID.

Problem:
{question}
Available roles: {roles}
Candidate role IDs: {candidates}
Remaining agent slots: {remaining_steps}
Execution history: {history}
Next role ID:"""
    if dataset in {"agi", "math"}:
        return """You are the controller of a dynamic reasoning team.
Select the next role that is most useful for solving the question.
Prefer direct reasoning before assumption checking, comparison of alternatives
before final verification, and select the final_answer_generator when only one
agent slot remains.
The final_answer_generator must be selected when only one agent slot remains.
Return only one role ID.

Question:
{question}
Available roles: {roles}
Candidate role IDs: {candidates}
Remaining agent slots: {remaining_steps}
Execution history: {history}
Next role ID:"""
    if dataset == "hotpotqa":
        return """You are the controller of a dynamic multi-hop question-answering team.
Select the next role that is most useful for answering the question.
Prefer entity analysis before evidence extraction, evidence extraction before
bridge reasoning, and verification before the final answer.
The final_answer_generator must be selected when only one agent slot remains.
Return only one role ID.

Question and documents:
{question}
Available roles: {roles}
Candidate role IDs: {candidates}
Remaining agent slots: {remaining_steps}
Execution history: {history}
Next role ID:"""
    raise ValueError("Dynamic has no selector prompt for dataset: " + dataset)
