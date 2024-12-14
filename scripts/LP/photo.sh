#!/bin/bash

python3 main.py --downstream_task LP --dataset photo --hidden_features 32 --embed_features 16 --lr_Riemann 0.001 --lr_gating 0.001 --w_decay 0.000 --w_decay_gating 0.000 \
    --coef_dis 0.1 --exp_iters 10 --sample_hop 1

