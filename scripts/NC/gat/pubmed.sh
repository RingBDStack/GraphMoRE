#!/bin/bash


python3 main.py --downstream_task NC --dataset Pubmed --backbone gat --hidden_features 32 --embed_features 32 --hidden_features_cls 32 --lr_Riemann 0.01 --lr_gating 0.01 --w_decay 0.0005 --w_decay_gating 0.0005 --min_epoch_cls 200 \
    --lr_cls 0.01 \
    --w_decay_cls 0.0005 \
    --drop_cls 0.3 \
    --drop_edge_cls 0.1 \
    --coef_dis 1 \
    --exp_iters 10
