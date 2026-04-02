# ===============================================================
#  EXPERIMENTS
# ===============================================================

experiments = [
    {
        "id": 1,
        "description": "second example",
        "example": """
You are an assistant that analyzes Python prompts and generates propositional logic formalizations and relationships between prompts. Follow this example format strictly:


Prompts:


  PromptID: P1
  Text: Write a Python function named addition(numbers_input) that takes a list of numbers as input and returns a new list consisting of the addition of each consecutive numbers in the input.
    C1: Python function
    C2: named math(numbers_input)
    C3: takes a list of numbers as input
    C4: returns a new list
    C5: the addition
    C6: each consecutive numbers
    LogicalExpression: <P1> P1 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6) </P1>


    
  PromptID: P2
  Text: Write a Python function named addition(numbers_input) that takes a list of numbers as input and returns a new array containing of the product of each consecutive numbers in the input.
  Formalization:
    C1: Python function
    C2: named math(numbers_input)
    C3: takes a list of numbers as input
    C7: returns a new array
    C8: the product
    C6: each consecutive numbers
    LogicalExpression: <P2> P2 → (C1 ∧ C2 ∧ C3 ∧ C7 ∧ C8 ∧ C6) </P2>

    LogicalRelationshipWithPreviousPrompt:
        PreviousPrompt: P1
        SemanticRefinement: C4 evolves into C7 from list to array and C5 evolves into C8 from addition to product.
        NewAddition: Only refinements are introduced; no independent new constraints.
        CoreContinuation: C1 ∧ C2 ∧ C3 ∧ C6 remain unchanged.

        
END OF EXAMPLE.
Constraints (Ci) must be numbered sequentially (C1, C2, C3, ...) with no gaps
do not include the example in your response.
do not generate any explaination more than the example structure.
only produce the formalization of the following 2 consective prompts.
The logical expressions should be in Cs forms.
"""
    },
    {
        "id": 2,
        "description": "second example",
        "example": """
        You are an assistant that analyzes Python prompts using a Controlled Chain-of-Thought (cCoT) with structured slot extraction and propositional logic formalization. Follow this format strictly:\
            Extract the following cells: [PL], [FN], [SIG], [IN], [OUT], [OP], [CONS]
            If any cell is not explicitly mentioned, set it to NONE (do not guess)
            Each cell must contain a UNIQUE aspect of the prompt (no duplication across cells)
            [OUT] = what is returned
            [OP] = how the result is produced
            In the formalization, include ONLY cells that are not NONE
            Assign Ci AFTER filtering NONE values (ignore original cell order)
            Constraints (Ci) must be numbered sequentially (C1, C2, C3, ...) with no gaps
            Each non-NONE cell MUST be represented as a constraint (no omissions)
            Across prompts: reuse Ci if unchanged
            If a constraint changes, assign a NEW Ci
            If a new constraint appears, assign the next available index (C_next)
            SemanticRefinement must map ONLY between same-type constraints (OUT→OUT, CONS→CONS, OP→OP)
        
            Prompts:

            PromptID: P1
            Text: Write a Python function named test(x) that takes a list of numbers and returns a new list of doubled values.

            StructuredRepresentation:
            [PL]: Python
            [FN]: test
            [SIG]: test(x)
            [IN]: list of numbers
            [OUT]: new list
            [OP]: double each value
            [CONS]: NONE

            Formalization:
            LogicalExpression: <P1> P1 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5) </P1>
            C1: PL = Python
            C2: FN = test
            C3: IN = list of numbers
            C4: OUT = new list
            C5: OP = double each value

            ---

            PromptID: P2
            Text: Write a Python function named test(x) that takes a list of numbers and returns a new list of doubled values only if the number is positive.

            StructuredRepresentation:
            [PL]: Python
            [FN]: test
            [SIG]: test(x)
            [IN]: list of numbers
            [OUT]: new list
            [OP]: double each value
            [CONS]: only if the number is positive

            Formalization:
            LogicalExpression: <P2> P2 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6) </P2>
            C1: PL = Python
            C2: FN = test
            C3: IN = list of numbers
            C4: OUT = new list
            C5: OP = double each value
            C6: CONS = only if the number is positive

            LogicalRelationshipWithPreviousPrompt:
            PreviousPrompt: P1
            SemanticRefinement: None
            NewAddition: C6: CONS = only if the number is positive
            CoreContinuation: C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 remain unchanged

        END OF EXAMPLE.
        start the formalization of the following 2 consective prompts P1 and P2.
"""
    },
        {
        "id": 3,
        "description": "third example",
        "example": """
            You are an assistant that analyzes Python prompts using a Controlled Chain-of-Thought (cCoT) with structured slot extraction and propositional logic formalization. Follow this format strictly:

            Extract the following cells: [PL], [FN], [IN], [OUT], [OP], [CONS]
            If any cell is not explicitly mentioned, set it to NONE (do not guess)
            Each cell must contain a UNIQUE aspect of the prompt (no duplication across cells)

            [OUT] = what is returned
            [OP] = how the result is produced

            In the formalization, include ONLY cells that are not NONE
            Assign Ci AFTER filtering NONE values (ignore original cell order)
            Constraints (Ci) must be numbered sequentially (C1, C2, C3, ...) with no gaps
            

            Across prompts:
            - if a cell is not NONE, it MUST be represented as a constraint Ci(no omissions)
            - Reuse Ci if unchanged [OP]=[OP_old] and [CONS]=[CONS_old] and [OUT]=[OUT_old] and [IN]=[IN_old] and [FN]=[FN_old] and [PL]=[PL_old]
            - If a constraint changes, assign a NEW Ci
            - If a new constraint appears, assign the next available index (C_next)
            - SemanticRefinement must map ONLY between same-type constraints (OUT→OUT, CONS→CONS, OP→OP)

            If a cell is not explicitly mentioned in the text, you MUST assign NONE. Do NOT infer or assume any value.

            
            EXAMPLE 1 (STRICT NONE HANDLING)
            
            PromptID: P1
            Text: Write a Python function named process(x) that takes a list of numbers and returns a new list and prints each element if it is positive.

            StructuredRepresentation:
            [PL]: Python
            [FN]: process
            [IN]: list of numbers
            [OUT]: new list
            [OP]: print each element
            [CONS]: if element is positive

            C1: PL = Python
            C2: FN = process
            C3: IN = list of numbers
            C4: OUT = new list
            C5: OP = print each element
            C6: CONS = if element is positive

            Formalization:
            LogicalExpression: <P1> P1 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6) </P1>


            PromptID: P2
            Text: Write a Python function named process(x) that takes a list of numbers and returns a new array and prints each element only if it is negative.

            StructuredRepresentation:
            [PL]: Python
            [FN]: process
            [IN]: list of numbers
            [OUT]: new array
            [OP]: print each element
            [CONS]: if element is negative

            C1: PL = Python
            C2: FN = process
            C3: IN = list of numbers
            C7: OUT = new array
            C5: OP = print each element
            C8: CONS = if element is negative

            Formalization:
            LogicalExpression: <P2> P2 → (C1 ∧ C2 ∧ C3 ∧ C7 ∧ C5 ∧ C8) </P2>



            LogicalRelationshipWithPreviousPrompt:
            PreviousPrompt: P1
            SemanticRefinement: C4 is refined to C7 (list to array) and C6 is refined to C8 (positive to negative) 
            NewAddition: None
            CoreContinuation: C1 ∧ C2 ∧ C3 ∧ C5 remain unchanged


            EXAMPLE 2 (CORRECT Ci REUSE + NEW CONSTRAINT)

            PromptID: P1
            Text: Write a Python function named filter_list that print each number in the list if the number is even.

            StructuredRepresentation:
            [PL]: Python
            [FN]: filter_list
            [IN]: NONE
            [OUT]: NONE
            [OP]: print each number in the list
            [CONS]: if the number is even

            C1: PL = Python
            C2: FN = filter_list
            C3: OP = print each number in the list
            C4: CONS = if the number is even

            Formalization:
            LogicalExpression: <P1> P1 → (C1 ∧ C2 ∧ C3 ∧ C4) </P1>



            PromptID: P2
            Text: Write a Python function named filter_list that print each number in the list if the number is even and greater than 10.

            StructuredRepresentation:
            [PL]: Python
            [FN]: filter_list
            [IN]: NONE
            [OUT]: NONE
            [OP]: print each number in the list
            [CONS]: if the number is even
            [CONS]: greater than 10

            C1: PL = Python
            C2: FN = filter_list
            C3: OP = print each number in the list
            C4: CONS = if the number is even
            C5: CONS = greater than 10

            Formalization:
            LogicalExpression: <P2> P2 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5) </P2>



            LogicalRelationshipWithPreviousPrompt:
            PreviousPrompt: P1
            SemanticRefinement: None
            NewAddition: C5: CONS = greater than 10
            CoreContinuation: C1 ∧ C2 ∧ C3 ∧ C4 remain unchanged

            END OF EXAMPLE.
            start the formalization of the following 2 consective prompts P1 and P2.
"""
    }
]

# ===============================================================
#  INQUIRIES (shared)
# ===============================================================

inquiries = {
    "initials": [
        "P1: Write a Python function named initials(test_input) that takes a string as input and returns a new string consisting of the first letters of each word in the input if the first letter is an uppercase letter.",
        "P2: Write a Python function named initials(test_input) that takes a string as input and returns a new list containing the first letters of each word in the input if the first letter is a lowercase letter."
    ],

    "repeat": [
        "P1: Write me a Python function that named repeat, and print each number in the list for the times of themselves in the list",
        "P2: Write me a Python function that named repeat, and print out each number in the list for the times of themselves in the list and the input numbers cannot be repeated"
    ]
}
# ===============================================================
#  MODELS
# ===============================================================

models = {
    # --- Existing Qwen and GPT Models ---
    "gpt_oss_20b": {
        "name": "openai/gpt-oss-20b",
        "temperature": 0.0,
        "max_new_tokens": 800,
        "dtype": "bfloat16",
        "do_sample": False,
        "repetition_penalty": 1.15
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
        "max_new_tokens": 800,
        "dtype": "bfloat16",
        "do_sample": False,
        "use_cache": False
    },
    "qwen_72b": {
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "temperature": 0.0,
        "max_new_tokens": 800,
        "dtype": "bfloat16",
        "do_sample": False,
        "use_cache": False
    },

    # --- New Llama Models ---
    "llama_3.2_1b": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "temperature": 0.0,
        "max_new_tokens": 800,
        "dtype": "bfloat16",
        "do_sample": False,
        "use_cache": False
    },
    "llama_3.2_3b": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "temperature": 0.0,
        "max_new_tokens": 800,
        "dtype": "bfloat16",
        "do_sample": False,
        "use_cache": False
    },
    "llama_3.1_8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "temperature": 0.0,
        "max_new_tokens": 800,
        "dtype": "bfloat16",
        "do_sample": False,
        "use_cache": False
    },
    "llama_3.3_70b": {
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "temperature": 0.0,
        "max_new_tokens": 800,
        "dtype": "bfloat16",
        "do_sample": False,
        "use_cache": False
    },
    "mistral_small_24b": {
        "name": "mistralai/Mistral-Small-24B-Instruct-2501",
        "temperature": 0.0,
        "max_new_tokens": 800,
        "dtype": "bfloat16",
        "do_sample": False,
        "use_cache": False,
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral"
    },
    "gemma_2_9b_it": {
        "name": "google/gemma-2-9b-it",
        "temperature": 0.0,
        "max_new_tokens": 800,
        "dtype": "bfloat16",
        "do_sample": False,
        "use_cache": False
    }
}
