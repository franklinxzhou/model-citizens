# Import
import torch
import torch.nn as nn
import optuna
import pandas as pd

from TabM_save_load import TabM_save_load
device = 'cpu'

# make_model function
def build_model(n_num_features: int,
                cat_cardinalities,
                trial: optuna.Trial) -> nn.Module:
    # EDIT: tighten but beef up architecture search space
    n_blocks = trial.suggest_int("n_blocks", 3, 5)                 # was 2–5
    d_block  = trial.suggest_int("d_block", 384, 1024, log=True)   # was 256–1024
    dropout  = trial.suggest_float("dropout", 0.0, 0.4)
    k        = trial.suggest_int("k", 8, 32, step=8)               # keep 8–32

    num_emb = LinearReLUEmbeddings(n_num_features)

    model = TabM.make(
        n_num_features=n_num_features,
        num_embeddings=num_emb,
        cat_cardinalities=cat_cardinalities,
        d_out=1,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
        k=k,
    )

    return model.to(device)

# Loading pipeline
loaded_pipeline = TabM_save_load.load(
    path="results/tabm_full_pipeline_5938",
    build_model_fn=build_model,  # your function that reconstructs TabM
    device=device
)
