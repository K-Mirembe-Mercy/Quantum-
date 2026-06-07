"""
train.py — Train the Quantum Sentiment Classifier and save weights.

Usage:
  python train.py
  python train.py --epochs 30 --lr 0.05 --batch-size 4
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import load_demo_dataset
from quantum_classifier import train, evaluate, predict
from visualize import plot_training_history, plot_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Train Quantum Sentiment Classifier")
    p.add_argument("--epochs",     type=int,   default=25,  help="Training epochs")
    p.add_argument("--lr",         type=float, default=0.1, help="Learning rate")
    p.add_argument("--batch-size", type=int,   default=6,   help="Mini-batch size")
    p.add_argument("--seed",       type=int,   default=42,  help="Random seed")
    p.add_argument("--output-dir", type=str,   default=".",
                   help="Directory to save weights + plots")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  ⚛  Quantum Sentiment Classifier — Training")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    print("\n📦  Loading dataset …")
    X_tr, X_te, y_tr, y_te, pipeline = load_demo_dataset(seed=args.seed)
    print(f"    Train: {len(X_tr)} samples  |  Test: {len(X_te)} samples")

    # ── Train ────────────────────────────────────────────────────
    print(f"\n🚀  Training for {args.epochs} epochs …\n")
    weights, history = train(
        X_tr, y_tr,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        verbose=True,
    )

    # ── Evaluate ─────────────────────────────────────────────────
    print("\n📊  Evaluating on test set …")
    metrics = evaluate(X_te, y_te, weights)
    print(f"\n  Accuracy  : {metrics['accuracy']:.1%}")
    print(f"  Precision : {metrics['precision']:.1%}")
    print(f"  Recall    : {metrics['recall']:.1%}")
    print(f"  F1 Score  : {metrics['f1']:.1%}")

    # ── Save weights ─────────────────────────────────────────────
    weights_path = os.path.join(args.output_dir, "weights.npy")
    np.save(weights_path, np.array(weights))
    print(f"\n💾  Weights saved → {weights_path}")

    # ── Plots ────────────────────────────────────────────────────
    print("\n🎨  Generating plots …")
    plot_training_history(
        history,
        path=os.path.join(args.output_dir, "training_curve.png"),
    )
    preds = [predict(x, weights) for x in X_te]
    plot_confusion_matrix(
        y_te, preds,
        path=os.path.join(args.output_dir, "confusion_matrix.png"),
    )

    print("\n✅  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
