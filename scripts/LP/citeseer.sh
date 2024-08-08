#!/bin/bash

python3 main.py --downstream_task LP --dataset Citeseer --hidden_features 64 --embed_features 32 --lr_Riemann 0.01 --lr_gating 0.01 --w_decay 0.0005 --w_decay_gating 0.0005 \
    --coef_dis 0.5 --exp_iters 10 --sample_hop 1
