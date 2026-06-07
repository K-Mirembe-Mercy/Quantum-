"""
Visualisation helpers — circuit diagrams, training curves, confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pennylane as qml

from quantum_classifier import quantum_circuit, N_QUBITS, N_LAYERS, init_weights


# ── Colour Palette ───────────────────────────────────────────────────────────

QUANTUM_BLUE   = "#0D1B2A"
NEON_CYAN      = "#00F5FF"
NEON_PURPLE    = "#BF5FFF"
WARM_WHITE     = "#F0F4F8"
ACCENT_GREEN   = "#39FF14"
ACCENT_ORANGE  = "#FF6B35"


# ── Circuit Diagram ──────────────────────────────────────────────────────────

def save_circuit_diagram(path="circuit.png"):
    """Render the quantum circuit and save as PNG."""
    weights = init_weights()
    dummy_input = np.zeros(N_QUBITS)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(QUANTUM_BLUE)
    ax.set_facecolor(QUANTUM_BLUE)

    # Draw using PennyLane's draw_mpl
    fig_qml, ax_qml = qml.draw_mpl(quantum_circuit, decimals=2)(dummy_input, weights)
    fig_qml.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=QUANTUM_BLUE)
    plt.close(fig_qml)
    print(f"Circuit diagram saved → {path}")


# ── Training Curve ───────────────────────────────────────────────────────────

def plot_training_history(history, path="training_curve.png"):
    """
    Plot loss and accuracy over epochs.

    history: list of (epoch, loss, accuracy) tuples
    """
    epochs = [h[0] for h in history]
    losses = [h[1] for h in history]
    accs   = [h[2] * 100 for h in history]   # percent

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(QUANTUM_BLUE)

    for ax in (ax1, ax2):
        ax.set_facecolor("#0A1520")
        ax.tick_params(colors=WARM_WHITE, labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E3A5F")

    # Loss
    ax1.plot(epochs, losses, color=NEON_CYAN, linewidth=2.5,
             marker="o", markersize=5, markerfacecolor=NEON_PURPLE)
    ax1.fill_between(epochs, losses, alpha=0.15, color=NEON_CYAN)
    ax1.set_title("Training Loss", color=WARM_WHITE, fontsize=14, pad=12)
    ax1.set_xlabel("Epoch", color=WARM_WHITE)
    ax1.set_ylabel("Binary Cross-Entropy", color=WARM_WHITE)
    ax1.grid(alpha=0.2, color="#1E3A5F")

    # Accuracy
    ax2.plot(epochs, accs, color=ACCENT_GREEN, linewidth=2.5,
             marker="o", markersize=5, markerfacecolor=ACCENT_ORANGE)
    ax2.fill_between(epochs, accs, alpha=0.15, color=ACCENT_GREEN)
    ax2.set_ylim(0, 105)
    ax2.axhline(50, color=ACCENT_ORANGE, linestyle="--",
                linewidth=1, alpha=0.5, label="Random baseline")
    ax2.set_title("Training Accuracy", color=WARM_WHITE, fontsize=14, pad=12)
    ax2.set_xlabel("Epoch", color=WARM_WHITE)
    ax2.set_ylabel("Accuracy (%)", color=WARM_WHITE)
    ax2.grid(alpha=0.2, color="#1E3A5F")
    ax2.legend(facecolor="#0A1520", labelcolor=WARM_WHITE, fontsize=9)

    fig.suptitle("Quantum Sentiment Classifier — Training",
                 color=NEON_CYAN, fontsize=16, y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=QUANTUM_BLUE)
    plt.close(fig)
    print(f"Training curve saved → {path}")


# ── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, path="confusion_matrix.png"):
    """Draw a styled 2×2 confusion matrix."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    matrix = np.array([[tn, fp], [fn, tp]])
    labels = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(QUANTUM_BLUE)
    ax.set_facecolor(QUANTUM_BLUE)

    colors = [[NEON_CYAN, ACCENT_ORANGE], [ACCENT_ORANGE, ACCENT_GREEN]]

    for i in range(2):
        for j in range(2):
            ax.add_patch(mpatches.FancyBboxPatch(
                (j + 0.05, 1 - i + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=colors[i][j], alpha=0.25,
                edgecolor=colors[i][j], linewidth=2,
            ))
            ax.text(j + 0.5, 1 - i + 0.55, str(matrix[i][j]),
                    ha="center", va="center",
                    color=WARM_WHITE, fontsize=22, fontweight="bold")
            ax.text(j + 0.5, 1 - i + 0.2, labels[i][j],
                    ha="center", va="center",
                    color=colors[i][j], fontsize=11)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted\nNegative", "Predicted\nPositive"],
                       color=WARM_WHITE, fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual\nPositive", "Actual\nNegative"],
                       color=WARM_WHITE, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Confusion Matrix", color=NEON_CYAN, fontsize=14, pad=12)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=QUANTUM_BLUE)
    plt.close(fig)
    print(f"Confusion matrix saved → {path}")


# ── Probability Bar ──────────────────────────────────────────────────────────

def render_prediction_bar(text: str, prob: float):
    """Print a colour-coded probability bar to the terminal."""
    label = "POSITIVE 😊" if prob >= 0.5 else "NEGATIVE 😞"
    pct   = int(prob * 40)
    bar   = "█" * pct + "░" * (40 - pct)
    print(f"\n  Text   : {text[:80]}")
    print(f"  Result : {label}")
    print(f"  Prob   : [{bar}] {prob:.1%}\n")
