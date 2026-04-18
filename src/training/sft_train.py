import argparse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from src.training.utils import load_yaml, set_seed
from src.training.dataset import load_sft_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    model_name = cfg["model_name"]
    train_file = cfg["train_file"]
    output_dir = cfg["output_dir"]
    max_length = cfg.get("max_length", 512)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    print("Loading datasets...")
    train_dataset, val_dataset = load_sft_datasets(
        train_file,
        val_ratio=cfg.get("val_ratio", 0.05),
        seed=cfg.get("seed", 42),
    )

    peft_config = LoraConfig(
        r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    model = get_peft_model(model, peft_config)

    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        eval_steps=cfg.get("eval_steps", 100),
        eval_strategy="steps",
        save_strategy="steps",
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        bf16=cfg.get("bf16", False),
        fp16=cfg.get("fp16", True),
        report_to="none",
        load_best_model_at_end=False,

        # 原来放在 SFTTrainer 里的参数，挪到这里
        dataset_text_field="text",
        max_length=max_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )

    print("Start training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
