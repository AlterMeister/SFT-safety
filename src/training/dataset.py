import json
import random
from datasets import Dataset

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def build_text(example):
    user_text = example["messages"][0]["content"].strip()
    assistant_text = example["messages"][1]["content"].strip()
    text = f"User: {user_text}\nAssistant: {assistant_text}"
    return {"text": text}

def load_sft_datasets(train_file, val_ratio=0.05, seed=42):
    data = load_jsonl(train_file)
    random.seed(seed)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - val_ratio))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_dataset = Dataset.from_list(train_data).map(build_text)
    val_dataset = Dataset.from_list(val_data).map(build_text)

    return train_dataset, val_dataset