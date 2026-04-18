#!/bin/bash
source /data/liujiajun/miniconda3/etc/profile.d/conda.sh
conda activate NewJulian

python src/inference/base_inference.py --config configs/base_infer.yaml