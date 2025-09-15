"""
Evaluation script for trained CNN on CICIDS-2018.

This script:
  1) Loads the trained model + label encoder.
  2) Runs predictions on holdout/test data.
  3) Saves classification report, confusion matrix, and ROC curves.
  4) Ensures Cognize graph stays active during inference (for drift logging).

Run:
    python -m src.evaluate --data ../data/02-14-2018.csv --model results/artifacts/best.keras
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

from src.data_utils import load_cicids_dataset
from src.model import build_cnn


def plot_confusion(y_true, y_pred, labels, out_path: Path):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset (full test split)
    (X_train, X_test, y_train, y_test), le = load_cicids_dataset(
        path=args.data, sample_frac=args.sample_frac, random_state=42
    )
    n_features = X_test.shape[1]
    X_test = X_test.reshape((-1, n_features, 1)).astype(np.float32)

    # 2) Load trained model + cognize graph
    #    The architecture ensures Cognize graph is part of build_cnn
    num_classes = int(len(np.unique(y_train)))
    model, cog_graph = build_cnn(input_shape=(n_features, 1), num_classes=num_classes)
    model.load_weights(args.model)

    # 3) Predict
    y_prob = model.predict(X_test, batch_size=256, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    # 4) Reports
    labels = list(le.classes_)
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    plot_confusion(y_test, y_pred, labels, out_dir / "confusion.png")

    # ROC for each class (only if multi-class)
    if y_prob.shape[1] > 2:
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "roc.png")
        plt.close()

    print("Saved evaluation results to:", str(out_dir.resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CICIDS-2018 CSV file.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .keras weights.")
    parser.add_argument("--out_dir", type=str, default="results/eval", help="Output directory.")
    parser.add_argument("--sample_frac", type=float, default=0.2, help="Fraction of data to use for eval.")
    args = parser.parse_args()
    main(args)
