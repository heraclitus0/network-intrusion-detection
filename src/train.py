"""
Training entrypoint for the network intrusion CNN.

This script:
  1) Loads & preprocesses CICIDS-2018 data.
  2) Builds a 1D-CNN classifier.
  3) Trains with a Cognize-driven adaptive controller (required).
  4) Saves the trained model and label artifacts.

Run:
    python -m src.train --data ../data/02-14-2018.csv --epochs 30

Notes:
- Cognize is embedded architecturally via `src.callbacks.AdaptivePlateau`.
- Surface optics remain standard Keras; Cognize acts as a control-plane
  for drift/plateau handling and bounded stability nudges.
"""

from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data_utils import load_cicids_dataset
from src.model import build_cnn
from src.callbacks import AdaptivePlateau


# ------------------------ utils ------------------------ #
def set_seeds(seed: int = 42) -> None:
    """Reproducibility helpers."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_datasets(X_train, y_train, X_val, y_val, batch_size: int = 256):
    """
    Wrap numpy arrays into tf.data.Datasets for performance.
    """
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_train = ds_train.shuffle(min(10000, len(X_train))).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    ds_val = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_val


# ------------------------ main ------------------------ #
def main(args):
    set_seeds(args.seed)

    # IO setup
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "artifacts").mkdir(exist_ok=True)

    # 1) Load + preprocess data
    (X_train, X_test, y_train, y_test), le = load_cicids_dataset(
        path=args.data,
        sample_frac=args.sample_frac,
        random_state=args.seed
    )

    # Reshape for Conv1D: (N, F, 1)
    n_features = X_train.shape[1]
    X_train = X_train.reshape((-1, n_features, 1)).astype(np.float32)
    X_test  = X_test.reshape((-1, n_features, 1)).astype(np.float32)

    # 2) Build CNN (+ Cognize hooks inside the architecture)
    num_classes = int(len(np.unique(y_train)))
    model, cog_graph = build_cnn(input_shape=(n_features, 1), num_classes=num_classes)

    # 3) tf.data
    ds_train, ds_val = make_datasets(X_train, y_train, X_test, y_test, batch_size=args.batch_size)

    # 4) Callbacks (Cognize-driven adaptive training is REQUIRED)
    log_csv = keras.callbacks.CSVLogger(str(out_dir / "logs" / "train.csv"), append=True)
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "artifacts" / "best.keras"),
        monitor="val_loss",
        save_best_only=True
    )
    adaptive = AdaptivePlateau(
        monitor="val_loss",
        factor=0.7,
        patience=2,
        cooldown=1,
        min_lr=1e-5,
        stop_patience=6,
        clipnorm_start=1.0,
        cooling_bias_max=0.06,
    )

    # 5) Train
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs,
        callbacks=[log_csv, ckpt, adaptive],
        verbose=1,
    )

    # 6) Persist artifacts
    #    - final model (even if not best)
    model.save(str(out_dir / "artifacts" / "final.keras"))
    #    - label classes
    with open(out_dir / "artifacts" / "label_classes.json", "w") as f:
        json.dump(list(le.classes_), f)

    # 7) Quick eval on holdout
    eval_metrics = model.evaluate(ds_val, verbose=0)
    metric_names = [m.name if hasattr(m, "name") else m for m in model.metrics]
    report = {name: float(val) for name, val in zip(["loss"] + metric_names, eval_metrics)}
    with open(out_dir / "artifacts" / "eval.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved artifacts to:", str(out_dir.resolve()))
    print("Eval:", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CICIDS-2018 CSV file.")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--sample_frac", type=float, default=1.0, help="Fraction of data to use (0<..<=1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    main(args)
