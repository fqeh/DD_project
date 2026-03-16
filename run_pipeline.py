import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import experiments, inquiries, models
from dd_engine import run_inference, save_dd_record
from dd_classes import StructuralStepDD, LogicalStepDD, ExampleTextDD


# -----------------------------------------------------------
# Utility
# -----------------------------------------------------------

def get_candidates(dd_instance):
    """Return non-empty unique candidates sorted by length (shortest first)."""
    return sorted(
        [c for c in set(dd_instance.successful_candidates) if len(c.strip()) > 0],
        key=len
    )


def run_ddmin_with_full_check(
    dd_class,
    candidate_text,
    bA, bB,
    pA, pB,
    model, tokenizer, m_cfg,
    log_file,
    tier_name
):
    """
    Runs ddmin and performs FULL candidate verification first.
    Returns:
        minimal_candidate (shortest) OR None
        dd_instance OR None
    """

    dd_instance = dd_class(
        candidate_text,
        bA, bB,
        pA, pB,
        model, tokenizer,
        m_cfg,
        log_file
    )

    dd_instance.ddmin(dd_instance.deltas)

    if not dd_instance.successful_candidates:
        save_dd_record(
            log_file,
            candidate_text,
            pA, pB,
            None, None,
            f"{tier_name}_REJECTED_FULL",
            "Full candidate did NOT reproduce baseline"
        )
        return None, None

    save_dd_record(
        log_file,
        candidate_text,
        pA, pB,
        None, None,
        f"{tier_name}_ACCEPTED_FULL",
        "Full candidate reproduced baseline"
    )

    minimal_candidate = min(dd_instance.successful_candidates, key=len)
    return minimal_candidate, dd_instance


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():

    # 1. Setup
    exp_id = 1
    inquiry = "initials"
    model_id = "qwen_32b"

    exp = next(e for e in experiments if e["id"] == exp_id)
    pA, pB = inquiries[inquiry]
    m_cfg = models[model_id]

    # 2. Load Model
    print("🚀 Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(m_cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        m_cfg["name"],
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # 3. Establish Baseline
    print("🔍 Establishing Baseline...")
    bA, bB, _ = run_inference(
        f"{exp['example']}\n\nPrompt 1: {pA}\nPrompt 2: {pB}\n",
        model, tokenizer, m_cfg
    )

    # =======================================================
    # Tier 1: Structural
    # =======================================================
    print("\n--- Tier 1: Structural ---")

    s_dd = StructuralStepDD(
        exp['example'],
        bA, bB,
        pA, pB,
        model, tokenizer,
        m_cfg,
        "log_struct.json"
    )

    s_dd.ddmin(s_dd.deltas)
    struct_candidates = get_candidates(s_dd)

    if not struct_candidates:
        raise Exception("Tier 1 failed: No structural candidates found.")

    # =======================================================
    # Tier 2: Logical (STOP after first ACCEPTED_FULL)
    # =======================================================
    print("\n--- Tier 2: Logical ---")

    all_logical_candidates = []

    for candidate in struct_candidates:

        print(f"\n🔁 Trying structural candidate (len={len(candidate.split())})")

        minimal, dd_obj = run_ddmin_with_full_check(
            LogicalStepDD,
            candidate,
            bA, bB,
            pA, pB,
            model, tokenizer,
            m_cfg,
            "log_logical.json",
            "TIER2"
        )

        # If rejected → try next
        if dd_obj is None:
            continue

        # If accepted → collect logical minima and STOP
        all_logical_candidates = get_candidates(dd_obj)
        print("✅ Tier 2 accepted first valid structural candidate. Stopping Tier 2 search.")
        break

    if not all_logical_candidates:
        raise Exception("Tier 2 failed: No structural candidate survived logical minimization.")

    # =======================================================
    # Tier 3: Lexical (STOP after first ACCEPTED_FULL)
    # =======================================================
    print("\n--- Tier 3: Lexical ---")

    final_out = None

    for candidate in all_logical_candidates:

        print(f"\n🔁 Trying logical candidate (len={len(candidate.split())})")

        minimal, dd_obj = run_ddmin_with_full_check(
            ExampleTextDD,
            candidate,
            bA, bB,
            pA, pB,
            model, tokenizer,
            m_cfg,
            "log_lexical.json",
            "TIER3"
        )

        # If rejected → try next
        if minimal is None:
            continue

        # If accepted → STOP
        final_out = minimal
        print("✅ Tier 3 accepted first valid logical candidate. Stopping Tier 3 search.")
        break

    if not final_out:
        raise Exception("Tier 3 failed: No logical candidate survived lexical minimization.")

    # =======================================================
    # Done
    # =======================================================
    print("\n🎉 Pipeline Complete.")
    print("\nFinal Minimized Prompt:\n")
    print(final_out)


if __name__ == "__main__":
    main()