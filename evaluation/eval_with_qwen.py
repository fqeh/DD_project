import json
import re
import argparse
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# =========================================================
# Config
# =========================================================
EVALUATOR_MODEL = "Qwen/Qwen2.5-32B-Instruct"
SEED = 42
MAX_NEW_TOKENS = 512

CONSTRAINT_WEIGHT = 0.6
LOGICAL_WEIGHT = 0.4

MAX_RUNS_PER_MODEL = 10


# =========================================================
# Few-shot examples
# =========================================================
FEW_SHOTS = [
    {
        "user": """Baseline:
PromptID: P1
Text: Write a Python function named initials(test_input) that takes a string as input and returns a new string consisting of the first letters of each word in the input if the first letter is an uppercase letter.
Formalization:
C1: Python function
C2: named initials(test_input)
C3: takes a string as input
C4: returns a new string
C5: the first letters
C6: each word
C7: if the first letter is an uppercase letter
LogicalExpression: <P1> P1 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6 ∧ C7) </P1>

PromptID: P2
Text: Write a Python function named initials(test_input) that takes a string as input and returns a new list containing the first letters of each word in the input if the first letter is a lowercase letter.
Formalization:
C1: Python function
C2: named initials(test_input)
C3: takes a string as input
C4: returns a new list
C5: the first letters
C6: each word
C8: if the first letter is a lowercase letter
LogicalExpression: <P2> P2 → (C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6 ∧ C8) </P2>

Candidate:
(same format as above)

Return JSON only.""",
        "assistant": """{"p1":{"explanation":"0 mistakes found. Precision=7/7=1.0, Recall=7/7=1.0 -> constraint_f1=1.0. Logical structure matches perfectly (Base=1.0) -> logical_score=1.0. final_accuracy = 0.6*1.0 + 0.4*1.0 = 1.0","constraint_f1":1.0,"logical_score":1.0,"final_accuracy":1.0},"p2":{"explanation":"0 mistakes found. Valid refinement correctly applied. Precision=7/7=1.0, Recall=7/7=1.0 -> constraint_f1=1.0. Logical structure matches perfectly (Base=1.0) -> logical_score=1.0. final_accuracy=1.0","constraint_f1":1.0,"logical_score":1.0,"final_accuracy":1.0},"overall_accuracy":1.0}"""
    },
    {
        "user": """Baseline:
PromptID: P1
Text: Write a Python function named repeat that prints each number in a list for the number of times it appears.
Formalization:
C1: Python function
C2: named repeat
C3: print each number in the list
C4: for the number of times it appears

PromptID: P2
Text: Write a Python function named repeat that prints each number in a list for the number of times it appears and ensures numbers cannot be repeated.
Formalization:
C1: Python function
C2: named repeat
C3: print each number in the list
C4: for the number of times it appears
C5: input numbers cannot be repeated

Candidate:
(same format as above)

Return JSON only.""",
        "assistant": """{"p1":{"explanation":"0 mistakes. Precision=4/4=1.0, Recall=4/4=1.0 -> constraint_f1=1.0. Logical structure matches perfectly (Base=1.0) -> logical_score=1.0. final_accuracy=1.0","constraint_f1":1.0,"logical_score":1.0,"final_accuracy":1.0},"p2":{"explanation":"0 mistakes. Correct addition of new constraint without breaking consistency. Precision=5/5=1.0, Recall=5/5=1.0 -> constraint_f1=1.0. Logical matches perfectly (Base=1.0) -> logical_score=1.0. final_accuracy=1.0","constraint_f1":1.0,"logical_score":1.0,"final_accuracy":1.0},"overall_accuracy":1.0}"""
    }
]

SYSTEM_PROMPT = """You are a strict evaluator for semantic constraint extraction and logical formalization of Python prompts.

Your task is to compare a BASELINE (gold reference) and a CANDIDATE output and compute structured evaluation scores.

You must focus ONLY on semantic correctness. Ignore wording, formatting, ordering, and style.

---

# =========================
# 1. CONSTRAINT CORRECTNESS
# =========================

A constraint is correct if it matches the baseline meaning.

Binary rule:
match(C_candidate, C_baseline) ∈ {0,1}

1 = correct meaning  
0 = incorrect meaning  

---

# =========================
# 2. LOGICAL SCORE
# =========================

Evaluate correctness of logical structure + semantics.

Score ∈ [0,1]:

- 1.0 → perfect match
- 0.7–0.9 → minor missing/extra constraints
- 0.3–0.6 → partial mismatch
- 0.0–0.2 → incorrect or invalid expression

---

# =========================
# 3. REFINEMENT RULE (KEY CONCEPT)
# =========================

A constraint change from P1 → P2 is a VALID REFINEMENT if:

semantic_similarity ≥ τ AND transformation ∈ {
    type_change,
    scope_change,
    restriction_change,
    abstraction_change
}

Examples:
- string → list
- return string → return list
- uppercase → lowercase condition

---

# =========================
# 4. C-NUMBER CONSISTENCY (WEAK SIGNAL)
# =========================

C-numbers are NOT identifiers of correctness.

Rules:

- If meaning is unchanged → C-number should ideally remain the same
- If valid refinement occurs → renumbering is allowed
- If meaning is unchanged but C-number changes without reason → apply penalty

IMPORTANT:
Semantic correctness is always more important than numbering consistency.

---

# =========================
# 5. SCORING FORMULAS
# =========================

constraint_f1:

Precision = correct_mapped / candidate_total  
Recall = correct_mapped / baseline_total  

constraint_f1 = 2 * (Precision * Recall) / (Precision + Recall)
if (Precision + Recall) > 0 else 0.0

---

logical_score:
Base = 1.0
Deductions:
- Subtract 0.2 for each minor missing/extra constraint in the logic.
- Subtract 0.5 for partial structural mismatch.
- Set to 0.0 if invalid expression.
logical_score = Base - Deductions (min 0.0)

---

final_accuracy:

final_accuracy = 0.6 * constraint_f1 + 0.4 * logical_score

---

# =========================
# 6. OUTPUT FORMAT (STRICT)
# =========================

Return ONLY JSON:

{
  "p1": {
    "explanation": "List ALL mistakes, the exact deduction for each error. Show constraint fraction math. Specify Logical Base=1.0 minus specific deductions. Show final accuracy calculation.",
    "constraint_f1": 0.0-1.0,
    "logical_score": 0.0-1.0,
    "final_accuracy": 0.0-1.0
  },
  "p2": {
    "explanation": "List ALL mistakes, the exact deduction for each error. Show constraint fraction math. Specify Logical Base=1.0 minus specific deductions. Show final accuracy calculation.",
    "constraint_f1": 0.0-1.0,
    "logical_score": 0.0-1.0,
    "final_accuracy": 0.0-1.0
  },
  "overall_accuracy": 0.0-1.0
}

Return JSON only.
"""


# =========================================================
# JSON helpers
# =========================================================
def safe_json_loads(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return {}


def load_dataset(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected the JSON root to be a list of records.")
    return data


def group_by_model(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for rec in records:
        model_name = rec.get("baseline") or rec.get("execution_model") or rec.get("example_model")
        if not model_name:
            continue
        grouped[model_name].append(rec)
    return dict(grouped)


def sort_by_timestamp(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(r):
        ts = r.get("timestamp", "")
        return ts
    return sorted(records, key=key_fn)


def get_inquiry_keys(record: Dict[str, Any]) -> List[str]:
    keys = [k for k in record.keys() if re.fullmatch(r"inquiry_\d+", k)]
    return sorted(keys, key=lambda x: int(x.split("_")[1]))


# =========================================================
# Prompt construction
# =========================================================
def build_eval_messages(baseline_raw: str, candidate_raw: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in FEW_SHOTS:
        messages.append({"role": "user", "content": shot["user"]})
        messages.append({"role": "assistant", "content": shot["assistant"]})

    user_prompt = f"""Compare the following baseline and candidate outputs.

Baseline:
{baseline_raw}

Candidate:
{candidate_raw}

Return JSON only using the schema from the instructions."""
    messages.append({"role": "user", "content": user_prompt})
    return messages


def prepare_chat_prompts(tokenizer, pairs: List[Tuple[str, str]]) -> List[str]:
    prompts = []
    for baseline_raw, candidate_raw in pairs:
        messages = build_eval_messages(baseline_raw, candidate_raw)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


# =========================================================
# Parsing judge output
# =========================================================
def normalize_score(v: Any) -> float:
    try:
        x = float(v)
        return max(0.0, min(1.0, x))
    except Exception:
        return 0.0


def parse_judge_output(text: str) -> Dict[str, Any]:
    obj = safe_json_loads(text)
    if not obj:
        return {
            "p1": {"explanation": "", "constraint_f1": 0.0, "logical_score": 0.0, "final_accuracy": 0.0},
            "p2": {"explanation": "", "constraint_f1": 0.0, "logical_score": 0.0, "final_accuracy": 0.0},
            "overall_accuracy": 0.0,
            "_raw": text,
            "_parse_error": True,
        }

    def fix_section(section: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "explanation": section.get("explanation", ""),
            "constraint_f1": normalize_score(section.get("constraint_f1", 0.0)),
            "logical_score": normalize_score(section.get("logical_score", 0.0)),
            "final_accuracy": normalize_score(section.get("final_accuracy", 0.0)),
        }

    p1 = fix_section(obj.get("p1", {}))
    p2 = fix_section(obj.get("p2", {}))
    overall = normalize_score(obj.get("overall_accuracy", (p1["final_accuracy"] + p2["final_accuracy"]) / 2.0))

    return {
        "p1": p1,
        "p2": p2,
        "overall_accuracy": overall,
        "_raw": text,
        "_parse_error": False,
    }


# =========================================================
# vLLM judge
# =========================================================
def build_llm(model_name: str, tensor_parallel_size: int | None = None) -> LLM:
    if tensor_parallel_size is None:
        tensor_parallel_size = max(1, torch.cuda.device_count())

    return LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.95,
        seed=SEED,
    )


def build_sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_NEW_TOKENS,
        seed=SEED,
    )


def judge_pairs(llm: LLM, tokenizer, sampling_params: SamplingParams, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    prompts = prepare_chat_prompts(tokenizer, pairs)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    results = []
    for out in outputs:
        raw = out.outputs[0].text.strip()
        results.append(parse_judge_output(raw))
    return results


# =========================================================
# Evaluation
# =========================================================
def evaluate_model_group(
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
    records: List[Dict[str, Any]],
    model_name: str,
) -> Dict[str, Any]:
    records = sort_by_timestamp(records)

    if len(records) < 2:
        return {
            "model_name": model_name,
            "num_runs": 0,
            "avg_accuracy": 0.0,
            "runs": [],
            "error": "Not enough records in this model group.",
        }

    # Assumption: first record in the sorted group is the baseline,
    # and the next up to 10 records are the runs.
    baseline_record = records[0]
    run_records = records[1 : 1 + MAX_RUNS_PER_MODEL]

    baseline_inquiries = get_inquiry_keys(baseline_record)
    if not baseline_inquiries:
        return {
            "model_name": model_name,
            "num_runs": 0,
            "avg_accuracy": 0.0,
            "runs": [],
            "error": "No inquiry_* keys found in baseline record.",
        }

    run_results = []
    run_scores = []

    for ridx, run_record in enumerate(run_records, start=1):
        common_inquiries = [k for k in baseline_inquiries if k in run_record]
        if not common_inquiries:
            continue

        pairs = []
        inquiry_names = []
        for ikey in common_inquiries:
            base_raw = baseline_record[ikey].get("GPT_raw_output", "")
            cand_raw = run_record[ikey].get("GPT_raw_output", "")
            pairs.append((base_raw, cand_raw))
            inquiry_names.append(ikey)

        inquiry_judgments = judge_pairs(llm, tokenizer, sampling_params, pairs)

        inquiry_details = {}
        final_scores = []
        for ikey, judgment in zip(inquiry_names, inquiry_judgments):
            inquiry_details[ikey] = judgment
            final_scores.append(judgment["overall_accuracy"])

        run_avg = mean(final_scores) if final_scores else 0.0
        run_scores.append(run_avg)

        run_results.append({
            "run_index": ridx,
            "timestamp": run_record.get("timestamp"),
            "avg_accuracy": run_avg,
            "inquiries": inquiry_details,
        })

    model_avg = mean(run_scores) if run_scores else 0.0

    return {
        "model_name": model_name,
        "baseline_timestamp": baseline_record.get("timestamp"),
        "num_runs": len(run_results),
        "avg_accuracy": model_avg,
        "runs": run_results,
    }


def evaluate_dataset(path: str, tensor_parallel_size: int | None = None) -> Dict[str, Any]:
    records = load_dataset(path)
    grouped = group_by_model(records)

    tokenizer = AutoTokenizer.from_pretrained(EVALUATOR_MODEL, trust_remote_code=True)
    llm = build_llm(EVALUATOR_MODEL, tensor_parallel_size=tensor_parallel_size)
    sampling_params = build_sampling_params()

    all_model_results = []
    model_avgs = []

    for model_name, group_records in grouped.items():
        print(f"Evaluating model group: {model_name} ({len(group_records)} records)")
        result = evaluate_model_group(llm, tokenizer, sampling_params, group_records, model_name)
        all_model_results.append(result)
        model_avgs.append(result["avg_accuracy"])
        print(f"  avg_accuracy = {result['avg_accuracy']:.4f}")

    overall_avg = mean(model_avgs) if model_avgs else 0.0

    return {
        "judge_model": EVALUATOR_MODEL,
        "overall_avg_accuracy_across_models": overall_avg,
        "models": all_model_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the JSON dataset")
    parser.add_argument("--output", default="evaluation_results.json", help="Where to save the results")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Tensor parallel size for vLLM")
    args = parser.parse_args()

    results = evaluate_dataset(args.input, tensor_parallel_size=args.tensor_parallel_size)
    Path(args.output).write_text(json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"\nOverall average accuracy across models: {results['overall_avg_accuracy_across_models']:.4f}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()