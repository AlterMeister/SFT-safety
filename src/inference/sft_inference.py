import argparse
import json
import torch
import yaml

from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def build_prompt(user_prompt):
    return f"User: {user_prompt}\nAssistant:"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_model_name = cfg["base_model_name"]
    adapter_path = cfg["adapter_path"]
    input_file = cfg["input_file"]
    output_file = cfg["output_file"]

    max_new_tokens = cfg.get("max_new_tokens", 256)
    temperature = cfg.get("temperature", 0.3)
    top_p = cfg.get("top_p", 0.8)
    do_sample = cfg.get("do_sample", True)

    print(f"Loading tokenizer from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )

    print(f"Loading base model from: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype="auto"
    ).cuda()

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    eval_data = load_jsonl(input_file)
    print(f"Loaded {len(eval_data)} eval samples from {input_file}")

    outputs = []
    for item in tqdm(eval_data, desc="SFT Inference", ncols=100):
        user_prompt = item["messages"][0]["content"]

        text = build_prompt(user_prompt)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        outputs.append({
            "id": item["id"],
            "category": item.get("category", ""),
            "source": item.get("source", ""),
            "prompt": user_prompt,
            "response": response,
            "model": adapter_path
        })

    save_jsonl(outputs, output_file)
    print(f"Saved outputs to {output_file}")


if __name__ == "__main__":
    main()

# python src/inference/sft_inference.py --config configs/sft_infer.yaml