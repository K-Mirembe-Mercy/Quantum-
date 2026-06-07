"""
Quantum-Enhanced Sentiment Classifier
Uses a variational quantum circuit as the classification layer.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# ── Quantum Device ──────────────────────────────────────────────────────────

N_QUBITS = 4
N_LAYERS = 2

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    """
    Variational quantum circuit for binary classification.

    Architecture:
      1. AngleEmbedding  — encodes classical features as qubit rotation angles
      2. BasicEntanglerLayers — trainable rotations + CNOT entanglement
      3. Measure PauliZ on qubit 0 → expectation value in [-1, 1]
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


def predict_proba(inputs, weights):
    """Map circuit output [-1,1] to probability [0,1]."""
    raw = quantum_circuit(inputs, weights)
    return float((raw + 1) / 2)


def predict(inputs, weights, threshold=0.5):
    """Return 1 (positive) or 0 (negative)."""
    return int(predict_proba(inputs, weights) >= threshold)


# ── Weight Initialisation ───────────────────────────────────────────────────

def init_weights(seed=42):
    """Random initial weights for the variational layers."""
    rng = np.random.default_rng(seed)
    return pnp.array(
        rng.uniform(0, 2 * np.pi, (N_LAYERS, N_QUBITS)),
        requires_grad=True,
    )


# ── Training ────────────────────────────────────────────────────────────────

def binary_cross_entropy(weights, X_batch, y_batch):
    """Differentiable loss over a mini-batch."""
    loss = 0.0
    for x, y in zip(X_batch, y_batch):
        prob = predict_proba(x, weights)
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        loss += -(y * np.log(prob) + (1 - y) * np.log(1 - prob))
    return loss / len(y_batch)


def train(X_train, y_train, n_epochs=20, batch_size=8, lr=0.1, seed=42,
          verbose=True):
    """
    Train the quantum classifier using gradient descent (Adam).

    Returns
    -------
    weights : trained parameter array
    history : list of (epoch, loss, accuracy) tuples
    """
    weights = init_weights(seed)
    opt = qml.AdamOptimizer(stepsize=lr)
    history = []

    n = len(X_train)
    rng = np.random.default_rng(seed)

    for epoch in range(1, n_epochs + 1):
        # Shuffle
        idx = rng.permutation(n)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            xb = X_shuf[start: start + batch_size]
            yb = y_shuf[start: start + batch_size]
            weights, loss_val = opt.step_and_cost(
                lambda w: binary_cross_entropy(w, xb, yb), weights
            )
            epoch_loss += float(loss_val) * len(xb)

        epoch_loss /= n
        preds = [predict(x, weights) for x in X_train]
        acc = np.mean(np.array(preds) == y_train)
        history.append((epoch, epoch_loss, acc))

        if verbose:
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"Epoch {epoch:>3}/{n_epochs}  loss={epoch_loss:.4f}  "
                  f"acc=[{bar}] {acc:.1%}")

    return weights, history


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(X_test, y_test, weights):
    """Return accuracy, precision, recall, F1."""
    preds = np.array([predict(x, weights) for x in X_test])
    y = np.array(y_test)

    acc = np.mean(preds == y)
    tp = np.sum((preds == 1) & (y == 1))
    fp = np.sum((preds == 1) & (y == 0))
    fn = np.sum((preds == 0) & (y == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1}
