#!/bin/bash

python3 main.py --downstream_task LP --dataset Pubmed --hidden_features 32 --embed_features 16 --lr_Riemann 0.01 --lr_gating 0.001 --w_decay 0.0 --w_decay_gating 0.0 \
    --coef_dis 1 --exp_iters 10
