def prompt_template(query: str):
    return f"""Break down the following multi-hop question into a sequence of logically connected single-hop questions, following a chronological order. Each single-hop question should progress towards answering the original question, and the final single-hop question should directly address the original query. The step of final single-hop query addressing original query is of utmost importance. The single-hop queries should not be more than 5 in number

Output: Structure your answer in the exact JSON format provided below, replacing placeholders with the actual question breakdown.
{{
    "single_hop_queries": [
        "First single-hop question that helps address part of the original query.",
        "Second single-hop question based on previous answer.",
        "...",
        "Final single-hop question that directly answers the original query."
    ]
}}
Query: {query}"""

def prompt_template_coder_agent(query):
    return f"""
You are a highly skilled AI coding assistant. Please generate Python code that meets the following requirements:

Task:
{query}

Make sure the code is valid and well-commented. Even if code is avaiable in above context, you are supposed to rewrite it below. Provide the code inside the following format:

```python
# Your generated code here
"""

def correction_prompt_coder_agent(response):
    return f"""
The following Python code was generated but contains errors or is not executable:

{response}

Please review and correct the code so that it works as expected. Ensure the fixed code is provided in the same format:

```python
# Corrected code here
"""

def pattern_correction_coder_agent(response):
    return f"""
The generated response \n{response}\n does not conform to the required format. Ensure the Python code is enclosed within the following format:

```python
# Correct code format here
"""
def pt_math(query):
    return f"""
Answer the below given query in twenty to thirty words.
Task:
{query}
You are strictly not allowed to exceed 20-30 words and do not generate any type of code.
Answer:

"""

def prompt_template_doc(python_code):
    return f"""
You are an expert documentation assistant. Given the following Python code, generate its documentation in the JSON format below:

JSON Schema:
function_info = {{
    "name": function_name,          # The name of the function.
    "description": description,     # A brief description of what the function does.
    "parameters": parameters_dict,  # A dictionary of parameter names and their descriptions.
    "returns": returns              # A brief description of the return value.
}}

Python Code:
{python_code}

Provide only the JSON documentation as the output. Do not include explanations or additional text.
"""
def pattern_correction_prompt_template_doc(previous_response):
    return f"""
The documentation generated does not match the required JSON schema. Ensure the documentation adheres to the format below:

JSON Schema:
function_info = {{
    "name": function_name,          # The name of the function.
    "description": description,     # A brief description of what the function does.
    "parameters": parameters_dict,  # A dictionary of parameter names and their descriptions.
    "returns": returns              # A brief description of the return value.
}}

The previous response was:
{previous_response}

Revise the response so it matches the exact JSON schema. Provide only the JSON documentation as the output.
"""

def prompt_temp(query, l1, l2, l3):
    """
    Combines the current query with long-term memory (l1), short-term memory (l2),
    and database query results (l3) to form a cohesive prompt.

    Parameters:
    query (str): The current query or input from the user.
    l1 (str): Long-term memory context or prior conversation context.
    l2 (str): Short-term memory from recent subquery context.
    l3 (str): Database query results or external retrieved context.

    Returns:
    str: A prompt that integrates all input components.
    """
    prompt = "### User Query:\n"
    prompt += f"{query}\n\n"

    if l1:
        prompt += "### Long-term Context:\n"
        prompt += f"{l1}\n\n"

    if l2:
        prompt += "### Short-term Context:\n"
        prompt += f"{l2}\n\n"

    if l3:
        prompt += "### Relevant Data from Database:\n"
        prompt += f"{l3}\n\n"

    prompt += "### Generate a cohesive response based on the above information."
    return prompt

