#!/bin/bash


python3 main.py --downstream_task NC --dataset Citeseer --backbone gcn --hidden_features 64 --embed_features 32 --hidden_features_cls 32 --lr_Riemann 0.01 --lr_gating 0.01 --w_decay 0.0005 --w_decay_gating 0.0005 \
    --lr_cls 0.01 \
    --w_decay_cls 0.0005 \
    --drop_cls 0.8 \
    --drop_edge_cls 0.8 \
    --sample_hop 1 \
    --coef_dis 0.5 \
    --exp_iters 10
