# Mini SFT Safety Project

This is a small hands-on project for learning the basics of **Supervised Fine-Tuning (SFT)** on large language models.

The goal of this project is not to build a production-ready alignment system, but to **understand the full SFT workflow in practice** through a compact, reproducible engineering pipeline.

We use a small base model and a lightweight dataset to explore a simple but important question:

> Can SFT make a small language model behave more safely, and what trade-offs does it introduce?

## What this project covers

This project walks through the main steps of a beginner-friendly SFT workflow:

- building a custom SFT dataset from open-source data
- creating a small evaluation set with safety-related categories
- running baseline inference on a base model
- training a LoRA-based SFT model
- evaluating model behavior before and after fine-tuning
- analyzing trade-offs such as:
  - harmful compliance
  - proper refusal
  - jailbreak robustness
  - over-refusal
  - helpfulness on benign prompts

## Why this project exists

This repo is designed as a **learning project** for understanding:

- what SFT actually does to model behavior
- how to organize a small LLM fine-tuning project in an engineering-oriented way
- how to build a minimal but complete pipeline:
  - data preprocessing
  - training
  - inference
  - evaluation

Instead of focusing on large-scale experiments, this project focuses on:

- clarity
- reproducibility
- modular code structure
- practical understanding

## Project setting

- **Base model**: Qwen2.5-1.5B
- **Training method**: LoRA-based SFT
- **Training data**: custom English-only SFT dataset built from public sources
- **Evaluation data**: prompts grouped into:
  - benign
  - harmful
  - borderline
  - jailbreak
  - over-refusal

## What this project is not

This is **not** intended to be:

- a state-of-the-art safety benchmark
- a full RLHF implementation
- a production deployment system
- a rigorous research paper

Instead, it is a **small educational project** for learning how SFT works end-to-end.

## Main learning goals

By working through this project, the goal is to understand:

1. how to format instruction-style data for SFT
2. how to fine-tune a base LLM with LoRA
3. how model behavior changes after SFT
4. how safety improvements may come with side effects such as template refusal or over-refusal
5. how to structure an LLM project in a more engineering-oriented way

## Repository structure

configs/        # yaml configs for training and inference
data/           # sft data, eval data, reward data
scripts/        # shell scripts for running pipelines
src/
  training/     # SFT training code
  inference/    # base / sft inference
  eval/         # simple evaluation scripts
  models/       # model-related helpers
  pipelines/    # pipeline entry points

## Current workflow

The current pipeline is:

preprocess raw open-source data
build sft_train_1000.jsonl
build sft_eval_150.jsonl
run base model inference
score baseline outputs
train SFT model
run SFT model inference
compare base vs SFT behavior
Notes

This project is intentionally kept small and simple so that the full process remains easy to understand and modify.

It is best viewed as a starter project for learning SFT, rather than a final system.