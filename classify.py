"""
classify.py — Classify sentiment of text using a trained quantum model.

Usage (demo mode, no weights needed):
  python classify.py "I love this product!"
  python classify.py "Terrible quality, never buying again."

Usage (with saved weights):
  python classify.py --weights weights.npy "This is amazing!"

Interactive mode:
  python classify.py --interactive
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import load_demo_dataset, QuantumFeaturePipeline
from quantum_classifier import train, predict_proba, predict, init_weights
from visualize import render_prediction_bar


def get_pipeline_and_weights(weights_path=None):
    """
    Return (pipeline, weights).
    If no weights file provided, train quickly on demo data first.
    """
    # Always build pipeline from demo data so vocabulary is defined
    X_tr, X_te, y_tr, y_te, pipeline = load_demo_dataset()

    if weights_path and os.path.exists(weights_path):
        weights = np.load(weights_path, allow_pickle=True)
        weights_array = weights
        import pennylane.numpy as pnp
        weights = pnp.array(weights_array, requires_grad=False)
        print(f"✅  Loaded weights from {weights_path}")
    else:
        if weights_path:
            print(f"⚠️  Weights file '{weights_path}' not found — training quickly …")
        else:
            print("⚡  No weights file provided — training on demo data (fast mode) …\n")
        weights, _ = train(X_tr, y_tr, n_epochs=15, verbose=True)

    return pipeline, weights


def classify_text(text, pipeline, weights):
    features = pipeline.transform_single(text)
    prob = predict_proba(features, weights)
    render_prediction_bar(text, prob)
    return prob


def interactive_mode(pipeline, weights):
    print("\n" + "=" * 60)
    print("  ⚛  Quantum Sentiment Classifier — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    while True:
        try:
            text = input("\n  Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        classify_text(text, pipeline, weights)


def parse_args():
    p = argparse.ArgumentParser(description="Quantum Sentiment Classifier — Inference")
    p.add_argument("text", nargs="?", default=None,
                   help="Text to classify (wrap in quotes)")
    p.add_argument("--weights", type=str, default=None,
                   help="Path to saved weights .npy file")
    p.add_argument("--interactive", action="store_true",
                   help="Launch interactive REPL")
    return p.parse_args()


def main():
    args = parse_args()

    if args.text is None and not args.interactive:
        print("Usage: python classify.py \"Your text here\"")
        print("       python classify.py --interactive")
        sys.exit(0)

    pipeline, weights = get_pipeline_and_weights(args.weights)

    if args.interactive:
        interactive_mode(pipeline, weights)
    else:
        classify_text(args.text, pipeline, weights)


if __name__ == "__main__":
    main()
