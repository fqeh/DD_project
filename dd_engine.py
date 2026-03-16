import json
import re
from datetime import datetime
from DD import DD

def save_dd_record(log_file, example_text, promptA, promptB, exprA, exprB, verdict, gpt_raw):
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "verdict": verdict,
        "example_length": len(example_text.split()),
        "example_text": example_text,
        "promptA": promptA, "promptB": promptB,
        "LogicalExpression_A": exprA, "LogicalExpression_B": exprB,
        "GPT_raw_output": gpt_raw,
    }
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        data = []
    data.append(record)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def run_inference(full_text, model, tokenizer, model_config):
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=model_config["max_new_tokens"],
        temperature=model_config["temperature"],
        do_sample=model_config["do_sample"],
        use_cache=False,
        num_beams=1
    )
    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # Extract
    p1_match = re.search(r'<*1>\s*(.*?\))', raw, re.DOTALL)
    p2_match = re.search(r'<*2>\s*(.*?\))', raw, re.DOTALL)
    exprA = p1_match.group(1).strip() if p1_match else None
    exprB = p2_match.group(1).strip() if p2_match else None
    
    return exprA, exprB, raw

def normalize(expr):
    return re.sub(r"\s+", "", expr) if expr else None