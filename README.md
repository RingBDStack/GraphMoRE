# GraphMoRE

This repository is the implementation of GraphMoRE in PyTorch.

# Environment

```
python==3.8.0
pytorch==1.13.1
pyg==2.3.1
cuda==11.7
```


# Examples

Running GraphMoRE for link prediction on the Pubmed
```
python3 main.py --downstream_task LP --dataset Pubmed --hidden_features 32 --embed_features 16 --lr_Riemann 0.01 --lr_gating 0.001 --w_decay 0.0 --w_decay_gating 0.0 --coef_dis 1 --exp_iters 10
```

Running GraphMoRE for link prediction on the Cora
```
python3 main.py --downstream_task LP --dataset Cora --hidden_features 64 --embed_features 32 --lr_Riemann 0.01 --lr_gating 0.01 --w_decay 0.0005 --w_decay_gating 0.0005 --coef_dis 1 --exp_iters 10 --sample_hop 1 2 3
```

Running GraphMoRE for node classification on the Citeseer
```
python3 main.py --downstream_task NC --dataset Citeseer --backbone gcn --hidden_features 64 --embed_features 32 --hidden_features_cls 32 --lr_Riemann 0.01 --lr_gating 0.01 --w_decay 0.0005 --w_decay_gating 0.0005 --min_epoch_cls 200 --lr_cls 0.01 --w_decay_cls 0.0005 --drop_cls 0.8 --drop_edge_cls 0.8 --sample_hop 1 --coef_dis 0.5 --exp_iters 10
```

Running GraphMoRE for node classification on the Photo
```
python3 main.py --downstream_task NC --dataset photo --backbone sage --hidden_features 32 --embed_features 32 --hidden_features_cls 32 --lr_Riemann 0.01 --lr_gating 0.01 --w_decay 0.0005 --w_decay_gating 0.0005 --min_epoch_cls 200 --lr_cls 0.01 --w_decay_cls 0.0005 --drop_cls 0.0 --drop_edge_cls 0.0 --sample_hop 1 --exp_iters 10
```
