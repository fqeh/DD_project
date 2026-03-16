import re
from dd_engine import run_inference, save_dd_record, normalize
from DD import DD


class BaseDD(DD):
    """Common setup for all tiers, collecting successful candidates."""

    def __init__(self, example_text, baselineA, baselineB,
                 promptA, promptB, model, tokenizer, config, log_file):

        super().__init__()

        self.example_text = example_text
        self.baselineA = baselineA
        self.baselineB = baselineB
        self.promptA = promptA
        self.promptB = promptB
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.log_file = log_file

        # collected successful reduced examples (strings)
        self.successful_candidates = []

        # flag used by some control flows to avoid double-checking the full candidate
        self.full_candidate_checked = False

        # optional abort flag (not required with pre-check approach, but harmless)
        self.abort = False

    def _test(self, c):
        """
        Called by ddmin to evaluate subset 'c' (list of indices).
        Returns one of DD.PASS / DD.FAIL / DD.UNRESOLVED.
        """

        # If an external condition set abort, immediately return PASS (not interesting)
        if getattr(self, "abort", False):
            return DD.UNRESOLVED

        reduced_example = self.rebuild(c)
        full_input = (
            f"{reduced_example}\n\n"
            f"Prompt 1: {self.promptA}\n"
            f"Prompt 2: {self.promptB}\n"
        )

        exprA, exprB, raw = run_inference(
            full_input,
            self.model,
            self.tokenizer,
            self.config
        )

        # 1️⃣ Incomplete extraction (model couldn't return both expressions)
        if exprA is None or exprB is None:
            save_dd_record(
                self.log_file,
                reduced_example,
                self.promptA,
                self.promptB,
                exprA,
                exprB,
                "PASS (<2)",
                raw
            )
            return DD.PASS

        # 2️⃣ Full candidate check (kept here as a safety; primary full-check is done outside ddmin)
        # If full candidate hasn't been checked by the wrapper, do a one-time check.
        # If it fails, mark abort so ddmin will not produce spurious results.
        if not self.full_candidate_checked and len(c) == len(self.deltas):
            self.full_candidate_checked = True
            if normalize(exprA) != normalize(self.baselineA) or normalize(exprB) != normalize(self.baselineB):
                # mark abort to prevent ddmin from continuing to shrink this candidate
                self.abort = True
                save_dd_record(
                    self.log_file,
                    reduced_example,
                    self.promptA,
                    self.promptB,
                    exprA,
                    exprB,
                    "FULL_CANDIDATE_FAILED",
                    "Full candidate did NOT reproduce baseline"
                )
                # Return PASS so the ddmin algorithm treats this as not interesting
                # (we rely on the pipeline wrapper to skip this candidate entirely)
                return DD.UNRESOLVED

        # 3️⃣ Baseline preserved -> interesting; collect candidate and report FAIL
        if normalize(exprA) == normalize(self.baselineA) and normalize(exprB) == normalize(self.baselineB):
            # avoid duplicates: only append if not already present
            if reduced_example not in self.successful_candidates:
                self.successful_candidates.append(reduced_example)

            save_dd_record(
                self.log_file,
                reduced_example,
                self.promptA,
                self.promptB,
                exprA,
                exprB,
                "FAIL (same)",
                raw
            )
            return DD.FAIL

        # 4️⃣ Logic mismatch
        save_dd_record(
            self.log_file,
            reduced_example,
            self.promptA,
            self.promptB,
            exprA,
            exprB,
            "PASS (diff)",
            raw
        )
        return DD.UNRESOLVED


# ==========================================================
# =================== STRUCTURAL ===========================
# ==========================================================

class StructuralStepDD(BaseDD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens = re.split(r'\n\n+', self.example_text)
        self.deltas = list(range(len(self.tokens)))

    def rebuild(self, c):
        return "\n\n".join([self.tokens[i] for i in c])


# ==========================================================
# =================== LOGICAL ==============================
# ==========================================================

class LogicalStepDD(BaseDD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens = self.example_text.splitlines(keepends=True)
        self.deltas = list(range(len(self.tokens)))

    def rebuild(self, c):
        return "".join([self.tokens[i] for i in c])


# ==========================================================
# =================== LEXICAL ==============================
# ==========================================================

class ExampleTextDD(BaseDD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # lexical tokens include newlines as separate tokens
        self.tokens = re.findall(r"\n|[^\s]+", self.example_text)
        self.deltas = list(range(len(self.tokens)))

    def rebuild(self, c):
        out = []
        prev_token = False
        for i in c:
            token = self.tokens[i]
            if token == "\n":
                out.append("\n")
                prev_token = False
            else:
                if prev_token:
                    out.append(" ")
                out.append(token)
                prev_token = True
        return "".join(out)