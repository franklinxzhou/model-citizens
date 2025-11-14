import os
import json
import pickle
import torch
import optuna

class TabM_save_load:
    """
    Save and load a complete TabM pipeline:
    - model weights (.pth)
    - preprocessor (.pkl)
    - metadata (.json): threshold, hyperparams, num_cols, cat_cols
    """

    def __init__(self, model=None, preprocessor=None, threshold=None,
                 best_params=None, num_cols=None, cat_cols=None,
                 device="cpu"):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold
        self.best_params = best_params
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.device = device

    # ============================
    #      SAVING PIPELINE
    # ============================
    def save(self, path):
        """
        Save model, preprocessor, metadata into a folder.
        path: folder path (created if doesn't exist)
        """

        os.makedirs(path, exist_ok=True)

        # 1. Save model weights
        torch.save(self.model.state_dict(), os.path.join(path, "tabm_model.pth"))

        # 2. Save preprocessor
        with open(os.path.join(path, "tabm_preprocessor.pkl"), "wb") as f:
            pickle.dump(self.preprocessor, f)

        # 3. Save metadata (threshold, hyperparams, feature lists)
        metadata = {
            "best_threshold": float(self.threshold),
            "best_params": self.best_params,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
        }

        with open(os.path.join(path, "tabm_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"TabM pipeline saved successfully to {path}")

    # ============================
    #      LOADING PIPELINE
    # ============================
    @staticmethod
    def load(path, build_model_fn, device="cpu"):
        """
        Load a TabM pipeline from folder.
        Requires build_model_fn(n_num_features, cat_cardinalities, trial)

        Returns:
            TabM_save_load instance with model + preprocessor ready.
        """

        # ---- Load metadata ----
        with open(os.path.join(path, "tabm_metadata.json"), "r") as f:
            metadata = json.load(f)

        best_threshold = metadata["best_threshold"]
        best_params = metadata["best_params"]
        num_cols = metadata["num_cols"]
        cat_cols = metadata["cat_cols"]

        # ---- Load preprocessor ----
        with open(os.path.join(path, "tabm_preprocessor.pkl"), "rb") as f:
            preproc = pickle.load(f)

        # ---- Load state_dict FIRST to infer numeric feature count ----
        state_dict = torch.load(
            os.path.join(path, "tabm_model.pth"),
            map_location=device,
        )

        # Prefer the true number of numeric features from the checkpoint
        if "num_module.linear.weight" in state_dict:
            n_num_features = state_dict["num_module.linear.weight"].shape[0]
        else:
            # fallback to metadata length
            n_num_features = len(num_cols)

        # Cat cardinalities from the saved preprocessor and cat_cols list
        cat_cardinalities = [preproc.cat_cardinalities_[c] for c in cat_cols]

        # ---- Rebuild model with fixed_trial (same architecture/hparams) ----
        fixed_trial = optuna.trial.FixedTrial(best_params)

        model = build_model_fn(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            trial=fixed_trial,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # ---- Return ready object ----
        return TabM_save_load(
            model=model,
            preprocessor=preproc,
            threshold=best_threshold,
            best_params=best_params,
            num_cols=num_cols,
            cat_cols=cat_cols,
            device=device,
        )

    # ============================
    #   APPLYING THE PIPELINE
    # ============================
    def predict(self, df):
        """
        Apply full pipeline:
        - transform with preprocessor
        - run TabM
        - average ensemble
        - threshold using stored best_threshold
        """

        # Preprocess
        X_num, X_cat, _ = self.preprocessor.transform(df)

        Xn = torch.from_numpy(X_num).to(self.device)
        Xc = torch.from_numpy(X_cat).to(self.device)

        with torch.no_grad():
            logits = self.model(Xn, Xc)             # (N, k, 1)
            probs = torch.sigmoid(logits).mean(dim=1)
            probs = probs.squeeze(-1).cpu().numpy()

        preds = (probs >= self.threshold).astype(int)

        return probs, preds
