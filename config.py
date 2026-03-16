# ===============================================================
#  EXPERIMENTS
# ===============================================================

experiments = [
    {
        "id": 0,
        "description": "Original P3 example",
        "example": """
You are an assistant that analyzes Python prompts and generates propositional logic formalizations and relationships between prompts. Follow this example format strictly:
Prompts:
  PromptID: P1
  Text: Write me a Python function that counts the number of ‘0’s in a list.
  Formalization:
    LogicalExpression: <P1> P1 → (C1 ∧ C2 ∧ C3) </P1>

  PromptID: P2
  Text: Write me a Python function named counter that counts the number of 0 in the list.
  Formalization:
    LogicalExpression: <P2> P2 → (C1 ∧ C4 ∧ C5 ∧ C3) </P2>
    LogicalRelationshipWithPreviousPrompt:
        PreviousPrompt: P1
        SemanticRefinement: C2 evolves from counting '0' (string) to C5 counting 0 (integer).
        NewAddition: C4: A specific function name (counter) is introduced.
        CoreContinuation: C1 ∧ C3: The requirement to implement a Python function and the assumption of a valid list remain unchanged.

END OF EXAMPLE.
do not include the example in your response.
do not generate any explaination more than the example structure.
only produce the formalization of the following 2 consective prompts.
The logical expressions should be in Cs forms.
"""
    },
    {
        "id": 1,
        "description": "second example",
        "example": """
You are an assistant that analyzes Python prompts and generates propositional logic formalizations and relationships between prompts. Follow this example format strictly:


Prompts:


  PromptID: P1
  Text: Write a Python function named math(numbers_input) that takes a list of numbers as input and returns a new list consisting of the addition of each consecutive numbers in the input.
    LogicalExpression: <P1> P1 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6) </P1>
    C1: Python function
    C2: named math(numbers_input)
    C3: takes a list of numbers as input
    C4: returns a new list
    C5: the addition
    C6: each consecutive numbers

    
  PromptID: P2
  Text: Write a Python function named math(numbers_input) that takes a list of numbers as input and returns a new list containing of the product of each consecutive numbers in the input.
  Formalization:
    LogicalExpression: <P2> P2 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C7 ∧ C6) </P2>
    C1: Python function
    C2: named math(numbers_input)
    C3: takes a list of numbers as input
    C4: returns a new list
    C7: the product
    C6: each consecutive numbers
    LogicalRelationshipWithPreviousPrompt:
        PreviousPrompt: P1
        SemanticRefinement: C5 evolves into C7: Case requirement changes from addition operation to product operation.
        NewAddition: Only refinements are introduced; no independent new constraints.
        CoreContinuation: C1 ∧ C2 ∧ C3 ∧ C4 ∧ C6 remain unchanged.

        
END OF EXAMPLE.
do not include the example in your response.
do not generate any explaination more than the example structure.
only produce the formalization of the following 2 consective prompts.
The logical expressions should be in Cs forms.
"""
    },
    {
        "id": 2,
        "description": "third example",
        "example": """
  PromptID: P1
  Text: Write a Python function named addition(numbers_input) that takes a list of numbers as input and returns a new list consisting of the addition of each consecutive numbers in the input.
    LogicalExpression: <P1> P1 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6) </P1>
    C1: Python function
    C2: named math(numbers_input)
    C3: takes a list of numbers as input
    C4: returns a new list
    C5: the addition
    C6: each consecutive numbers    
  PromptID: P2
  Text: Write a Python function named addition(numbers_input) that takes a list of numbers as input and returns a new list containing of the product of each consecutive numbers in the input.
  Formalization:
    LogicalExpression: <P2> P2 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C7 ∧ C6) </P2>
    C1: Python function
    C2: named math(numbers_input)
    C3: takes a list of numbers as input
    C4: returns a new list
    C7: the product
    C6: each consecutive numbers
    LogicalRelationshipWithPreviousPrompt:
        PreviousPrompt: P1
        SemanticRefinement: C5 evolves into C7: Case requirement changes from addition operation to product operation.
        NewAddition: Only refinements are introduced; no independent new constraints.
        CoreContinuation: C1 ∧ C2 ∧ C3 ∧ C4 ∧ C6 remain unchanged.
"""
    },
    {
        "id": 3,
        "description": "third example",
        "example": """
PromptID: P1
  Text: Write a Python function named addition(numbers_input) that takes a list of numbers as input and returns a new list consisting of the addition of each consecutive numbers in the input.
    LogicalExpression: <P1> P1 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6) </P1>
    C1: Python function
    C2: named math(numbers_input)
    C3: takes a list of numbers as input
    C4: returns a new list
    C5: the addition
    C6: each consecutive numbers    
  PromptID: P2
  Text: Write a Python function named addition(numbers_input) that takes a list of numbers as input and returns a new list containing of the product of each consecutive numbers in the input.
    LogicalExpression: <P2> P2 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C7 ∧ C6) </P2>
    LogicalRelationshipWithPreviousPrompt:
        PreviousPrompt: P1
        SemanticRefinement: C5 evolves into C7: Case requirement changes from addition operation to product operation.
"""
    }
]

# ===============================================================
#  INQUIRIES (shared)
# ===============================================================

inquiries = {
    "initials": [
        "Write a Python function named initials(test_input) that takes a string as input and returns a new string consisting of the uppercase first letters of each word in the input.",
        "Write a Python function named initials(test_input) that takes a string as input and returns a new list containing the lowercase first letter of each word in the input."
    ],

    "repeat": [
        "Write me a Python function that named repeat, and print each number in the list for the times of themselves in the list",
        "Write me a Python function that named repeat, and the input numbers cannot be repeated, and print out each number in the list for the times of themselves in the list"
    ]
}
# ===============================================================
#  MODELS
# ===============================================================

models = {
    "gpt_oss_20b": {
        "name": "openai/gpt-oss-20b",
        "temperature": 0.0,
        "max_new_tokens": 200,
        "dtype": "bfloat16",
        "do_sample": False
    },
    "qwen_32b": {
        "name": "Qwen/Qwen2.5-32B-Instruct",
        "temperature": 0.0,
        "max_new_tokens": 800,
        "dtype": "bfloat16",
        "do_sample": False,
        "use_cache": False


    },
    "qwen_3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "temperature": 0.0,
        "max_new_tokens": 500,
        "dtype": "bfloat16",
        "do_sample": False,
        "use_cache": False

    }
}
