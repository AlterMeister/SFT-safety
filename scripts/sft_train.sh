#!/bin/bash
source /data/liujiajun/miniconda3/etc/profile.d/conda.sh
conda activate NewJulian

python -m src.training.sft_train --config configs/sft_train.yaml