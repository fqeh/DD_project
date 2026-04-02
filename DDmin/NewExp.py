import os
import json
import torch
import random
import numpy as np
import re
from datetime import datetime
from DD import DD
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams
from config import experiments, inquiries, models
 
# Environment & Seeding
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)
 
# torch.use_deterministic_algorithms(True)  # Commented out for vLLM Triton autotuning
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
 
# Configuration
SELECTED_EXPERIMENT_ID = 1
INQUIRY_1 = "initials"
INQUIRY_2 = "repeat"
SELECTED_MODEL = "qwen_72b"

 
exp = next(e for e in experiments if e["id"] == SELECTED_EXPERIMENT_ID)
example_text = exp["example"]
promptA_1, promptB_1 = inquiries[INQUIRY_1]
promptA_2, promptB_2 = inquiries[INQUIRY_2]
 
model_config = models[SELECTED_MODEL]
MODEL_NAME = model_config["name"]
model_tag = MODEL_NAME.split("/")[-1].replace("-", "_")
 
LOG_FILE = f"dd_log_exp{SELECTED_EXPERIMENT_ID}_{INQUIRY_1}_{INQUIRY_2}_{model_tag}_T{model_config['temperature']}.json"
 
# Model Loading
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Loading vLLM model {MODEL_NAME} with tensor_parallel_size={torch.cuda.device_count()}")
llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=torch.cuda.device_count(),
    dtype=model_config["dtype"],
    seed=seed,
    trust_remote_code=True,
    max_model_len=8192,
    gpu_memory_utilization=0.95
)

is_sampling = model_config.get("do_sample", False)
sampling_params = SamplingParams(
    temperature=model_config.get("temperature", 0.0) if is_sampling else 0.0,
    top_p=model_config.get("top_p", 1.0) if is_sampling else 1.0,
    top_k=model_config.get("top_k", -1) if is_sampling else -1,
    repetition_penalty=model_config.get("repetition_penalty", 1.0),
    max_tokens=model_config["max_new_tokens"],
    seed=seed,
    detokenize=True
)
 
# --- UTILS ---
 
def save_dd_record(example_text, verdict,
                   promptA_1, promptB_1, exprA_1, exprB_1, gpt_raw_1,
                   promptA_2=None, promptB_2=None, exprA_2=None, exprB_2=None, gpt_raw_2=None):
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "verdict": verdict,
        "example_length": len(example_text.split()),
        "example_text": example_text,
        "inquiry_1": {
            "promptA": promptA_1, "promptB": promptB_1,
            "LogicalExpression_A": exprA_1, "LogicalExpression_B": exprB_1,
            "GPT_raw_output": gpt_raw_1
        },
        "inquiry_2": {
            "promptA": promptA_2, "promptB": promptB_2,
            "LogicalExpression_A": exprA_2, "LogicalExpression_B": exprB_2,
            "GPT_raw_output": gpt_raw_2
        } if promptA_2 is not None else None
    }
    try:
        data = json.load(open(LOG_FILE, "r", encoding="utf-8"))
    except:
        data = []
    data.append(record)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_baseline_record(example_text, bA1, bB1, raw1, bA2=None, bB2=None, raw2=None):
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "example_length": len(example_text.split()),
        "example_text": example_text,
        "inquiry_1": {
            "promptA": promptA_1, "promptB": promptB_1,
            "LogicalExpression_A": bA1, "LogicalExpression_B": bB1,
            "GPT_raw_output": raw1
        },
        "inquiry_2": {
            "promptA": promptA_2, "promptB": promptB_2,
            "LogicalExpression_A": bA2, "LogicalExpression_B": bB2,
            "GPT_raw_output": raw2
        } if raw2 is not None else None
    }
    try:
        data = json.load(open(LOG_FILE, "r", encoding="utf-8"))
    except:
        data = []
    data.append(record)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
 
def extract_expressions(text):
    if not text: return None, None
    p1_match = re.search(r'<(?:P1|Prompt 1)>\s*(.*?)\s*<\/(?:P1|Prompt 1)>', text, re.DOTALL)
    if not p1_match:
        p1_match = re.search(r'<(?:P1|Prompt 1)>\s*(.*?\))', text, re.DOTALL)
        
    p2_match = re.search(r'<(?:P2|Prompt 2)>\s*(.*?)\s*<\/(?:P2|Prompt 2)>', text, re.DOTALL)
    if not p2_match:
        p2_match = re.search(r'<(?:P2|Prompt 2)>\s*(.*?\))', text, re.DOTALL)
        
    return (p1_match.group(1).strip() if p1_match else None,
            p2_match.group(1).strip() if p2_match else None)
 
def normalize(expr):
    return re.sub(r"\s+", "", expr) if expr else None

def compare_expressions(e1, e2):
    if e1 is None or e2 is None:
        return e1 == e2
    if normalize(e1) == normalize(e2):
        return True
        
    def clean_and_split(e):
        # Remove common prefix like 'P1 -> ' or 'P2 \rightarrow '
        e = re.sub(r'P\d+\s*(?:->|→|\\rightarrow)\s*', '', e)
        e = e.strip()
        # Remove outer parentheses
        while e.startswith('(') and e.endswith(')'):
            e = e[1:-1].strip()
            
        # Split by various AND representations: 'and', '∧', '&', '\land'
        parts = re.split(r'(?i)\s+and\s+|\s*∧\s*|\s*&\s*|\s*\\land\s*', e)
        return sorted([normalize(p) for p in parts if p])
        
    return clean_and_split(e1) == clean_and_split(e2)
 
def run_inference_batch(full_text_list):
    """Single source of truth for batched model generation."""
    try:
        # Instruct models (DeepSeek, Mixtral) require their specific chat templates to generate responses
        chat_prompts = []
        for text in full_text_list:
            messages = [{"role": "user", "content": text}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            chat_prompts.append(formatted_prompt)
            
        outputs = llm.generate(chat_prompts, sampling_params, use_tqdm=False)
        exprA_list, exprB_list, raw_list = [], [], []
        for out in outputs:
            raw = out.outputs[0].text.strip()
            exprA, exprB = extract_expressions(raw)
            exprA_list.append(exprA)
            exprB_list.append(exprB)
            raw_list.append(raw)
        return exprA_list, exprB_list, raw_list
    except Exception as e:
        print(f"⚠️ vLLM GPT error: {e}")
        return [None]*len(full_text_list), [None]*len(full_text_list), [None]*len(full_text_list)

def run_inference(full_text):
    """Legacy single inference hook for establishing the baseline."""
    eA, eB, raw = run_inference_batch([full_text])
    return eA[0], eB[0], raw[0]
 
# --- DELTA DEBUGGING ---
 
class ExampleTextDD(DD):
    def __init__(self, example_text, bA1, bB1, bA2, bB2, pA1, pB1, pA2, pB2):
        super().__init__()
        self.bA1, self.bB1 = bA1, bB1
        self.bA2, self.bB2 = bA2, bB2
        self.pA1, self.pB1 = pA1, pB1
        self.pA2, self.pB2 = pA2, pB2
        
        # Split text into elements (words + whitespaces), preserving exact structure
        self.text_elements = re.split(r'(\s+)', example_text)
        
        # Only non-whitespace words are subject to Delta Debugging
        self.deltas = [i for i, el in enumerate(self.text_elements) if el.strip() != ""]
        # Whitespace and newlines are preserved as is
        self.fixed_indices = [i for i, el in enumerate(self.text_elements) if el.strip() == ""]
 
    def construct_text(self, c):
        keep_indices = sorted(c + self.fixed_indices)
        return "".join([self.text_elements[i] for i in keep_indices])

    def _test(self, c):
        return self._test_batch([c])[0]
        
    def _test_batch(self, configs):
        full_inputs_1 = []
        reduced_examples = []
        for c in configs:
            reduced_example = self.construct_text(c)
            reduced_examples.append(reduced_example)
            full_input = f"{reduced_example}\n\nPrompt 1: {self.pA1}\nPrompt 2: {self.pB1}\n"
            full_inputs_1.append(full_input)
            
        print(f"\n[Parallel DD - Stage 1] ⚡ Sending batch of {len(configs)} generation prompts (Inq1) to vLLM...")
        exprA_list1, exprB_list1, raw_list1 = run_inference_batch(full_inputs_1)
        
        outcomes = [None] * len(configs)
        stage2_indices = []
        full_inputs_2 = []
        
        for idx, (c, reduced_example, exprA, exprB, raw_output) in enumerate(zip(configs, reduced_examples, exprA_list1, exprB_list1, raw_list1)):
            if exprA is None or exprB is None:
                save_dd_record(reduced_example, "PASS (<2 expressions Inq1)", self.pA1, self.pB1, exprA, exprB, raw_output)
                outcomes[idx] = DD.PASS
            elif compare_expressions(exprA, self.bA1) and compare_expressions(exprB, self.bB1):
                stage2_indices.append(idx)
                full_input2 = f"{reduced_example}\n\nPrompt 1: {self.pA2}\nPrompt 2: {self.pB2}\n"
                full_inputs_2.append(full_input2)
            else:
                save_dd_record(reduced_example, "PASS (diff Inq1)", self.pA1, self.pB1, exprA, exprB, raw_output)
                outcomes[idx] = DD.UNRESOLVED

        stage3_indices = []
        full_inputs_3 = []
        if stage2_indices:
            print(f"\n[Parallel DD - Stage 2] ⚡ Sending {len(stage2_indices)} candidates (Inq2) to vLLM...")
            exprA_list2, exprB_list2, raw_list2 = run_inference_batch(full_inputs_2)
            
            for i, idx in enumerate(stage2_indices):
                reduced_example = reduced_examples[idx]
                eA2, eB2, raw2 = exprA_list2[i], exprB_list2[i], raw_list2[i]
                eA1, eB1, raw1 = exprA_list1[idx], exprB_list1[idx], raw_list1[idx]
                
                if eA2 is None or eB2 is None:
                    save_dd_record(reduced_example, "PASS (<2 expressions Inq2)", 
                                   self.pA1, self.pB1, eA1, eB1, raw1,
                                   self.pA2, self.pB2, eA2, eB2, raw2)
                    outcomes[idx] = DD.PASS
                elif compare_expressions(eA2, self.bA2) and compare_expressions(eB2, self.bB2):
                    # Matched both original stages! Prepare for double verification (Stage 3).
                    stage3_indices.append(idx)
                    full_input3 = f"{reduced_example}\n\nPrompt 1: {self.pA1}\nPrompt 2: {self.pB1}\n"
                    full_inputs_3.append(full_input3)
                else:
                    save_dd_record(reduced_example, "PASS (diff Inq2)", 
                                   self.pA1, self.pB1, eA1, eB1, raw1,
                                   self.pA2, self.pB2, eA2, eB2, raw2)
                    outcomes[idx] = DD.UNRESOLVED
                
        # --- VERIFICATION STAGES ---
        stage4_indices = []
        full_inputs_4 = []

        if stage3_indices:
            print(f"\n[Parallel DD - Stage 3] ⚡ VERIFICATION: Re-testing {len(stage3_indices)} matching fail candidates (Inq1) to eliminate flakiness...")
            exprA_list3, exprB_list3, raw_list3 = run_inference_batch(full_inputs_3)

            for i, idx in enumerate(stage3_indices):
                reduced_example = reduced_examples[idx]
                eA3, eB3, raw3 = exprA_list3[i], exprB_list3[i], raw_list3[i]
                
                if eA3 is None or eB3 is None:
                    save_dd_record(reduced_example, "UNRESOLVED - Flaky Inq1 (<2 expressions Inq1 Verif)", 
                                   self.pA1, self.pB1, eA3, eB3, raw3)
                    outcomes[idx] = DD.UNRESOLVED
                elif compare_expressions(eA3, self.bA1) and compare_expressions(eB3, self.bB1):
                    stage4_indices.append(idx)
                    full_input4 = f"{reduced_example}\n\nPrompt 1: {self.pA2}\nPrompt 2: {self.pB2}\n"
                    full_inputs_4.append(full_input4)
                else:
                    save_dd_record(reduced_example, "UNRESOLVED - Flaky Inq1 (diff Inq1 Verif)", 
                                   self.pA1, self.pB1, eA3, eB3, raw3)
                    outcomes[idx] = DD.UNRESOLVED

        if stage4_indices:
            print(f"\n[Parallel DD - Stage 4] ⚡ VERIFICATION: Re-testing {len(stage4_indices)} matching fail candidates (Inq2) to eliminate flakiness...")
            exprA_list4, exprB_list4, raw_list4 = run_inference_batch(full_inputs_4)

            for i, idx in enumerate(stage4_indices):
                reduced_example = reduced_examples[idx]
                eA4, eB4, raw4 = exprA_list4[i], exprB_list4[i], raw_list4[i]
                
                stage3_i = stage3_indices.index(idx)
                eA3, eB3, raw3 = exprA_list3[stage3_i], exprB_list3[stage3_i], raw_list3[stage3_i]

                if eA4 is None or eB4 is None:
                    save_dd_record(reduced_example, "UNRESOLVED - Flaky Inq2 (<2 expressions Inq2 Verif)", 
                                   self.pA1, self.pB1, eA3, eB3, raw3,
                                   self.pA2, self.pB2, eA4, eB4, raw4)
                    outcomes[idx] = DD.UNRESOLVED
                elif compare_expressions(eA4, self.bA2) and compare_expressions(eB4, self.bB2):
                    save_dd_record(reduced_example, "FAIL (doubly verified match)", 
                                   self.pA1, self.pB1, eA3, eB3, raw3,
                                   self.pA2, self.pB2, eA4, eB4, raw4)
                    outcomes[idx] = DD.FAIL
                else:
                    save_dd_record(reduced_example, "UNRESOLVED - Flaky Inq2 (diff Inq2 Verif)", 
                                   self.pA1, self.pB1, eA3, eB3, raw3,
                                   self.pA2, self.pB2, eA4, eB4, raw4)
                    outcomes[idx] = DD.UNRESOLVED

        return outcomes
 
# --- EXECUTION ---
 
def main():
    print("🔍 Establishing Baseline for Inquiry 1...")
    full_prompt1 = f"{example_text}\n\nPrompt 1: {promptA_1}\nPrompt 2: {promptB_1}\n"
    bA1, bB1, raw1 = run_inference(full_prompt1)
 
    if bA1 is None or bB1 is None:
        print("❌ Failed baseline extraction for Inq1.")
        save_baseline_record(example_text, bA1, bB1, raw1)
        return

    print("🔍 Establishing Baseline for Inquiry 2...")
    full_prompt2 = f"{example_text}\n\nPrompt 1: {promptA_2}\nPrompt 2: {promptB_2}\n"
    bA2, bB2, raw2 = run_inference(full_prompt2)
 
    if bA2 is None or bB2 is None:
        print("❌ Failed baseline extraction for Inq2.")
        save_baseline_record(example_text, bA1, bB1, raw1, bA2, bB2, raw2)
        return
 
    save_baseline_record(example_text, bA1, bB1, raw1, bA2, bB2, raw2)
    print(f"Baseline Inq1 A: {bA1}\nBaseline Inq1 B: {bB1}")
    print(f"Baseline Inq2 A: {bA2}\nBaseline Inq2 B: {bB2}\n🚀 Starting DD...")
    
    dd = ExampleTextDD(example_text, bA1, bB1, bA2, bB2, promptA_1, promptB_1, promptA_2, promptB_2)
    min_indices = dd.ddmin(dd.deltas)
 
    final_text = dd.construct_text(min_indices)
    print(f"\n=== MINIMIZED EXAMPLE ===\n{final_text}")
    print("=========================")
    
    print("\n🔍 Verifying Minimized Example Output deterministic reproduction...")
    min_prompt1 = f"{final_text}\n\nPrompt 1: {promptA_1}\nPrompt 2: {promptB_1}\n"
    mA1, mB1, mRaw1 = run_inference(min_prompt1)
    
    if mA1 is None or mB1 is None:
        print("❌ Verification Failed: Minimized example failed to extract tags for Inq1.")
        return
        
    min_prompt2 = f"{final_text}\n\nPrompt 1: {promptA_2}\nPrompt 2: {promptB_2}\n"
    mA2, mB2, mRaw2 = run_inference(min_prompt2)
    
    if mA2 is None or mB2 is None:
        print("❌ Verification Failed: Minimized example failed to extract tags for Inq2.")
        return
        
    print(f"Minimized Inq1 A: {mA1}\nMinimized Inq1 B: {mB1}")
    print(f"Minimized Inq2 A: {mA2}\nMinimized Inq2 B: {mB2}")
        
    if compare_expressions(mA1, bA1) and compare_expressions(mB1, bB1) and \
       compare_expressions(mA2, bA2) and compare_expressions(mB2, bB2):
        print("\n✅ VERIFIED: Minimized example perfectly reproduces BOTH baseline formalizations!")
    else:
        print("\n❌ Verification Failed: Minimized output diverged from baselines.")

if __name__ == "__main__":
    main()
