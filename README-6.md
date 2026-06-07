# ⚛ Quantum Sentiment Classifier

> *What if your AI model ran on a quantum computer?*

This project replaces the classical classification layer of a sentiment analysis pipeline with a **variational quantum circuit** — trained end-to-end using gradient descent, just like a neural network.

```
"I love this product!"  →  [TF-IDF + PCA]  →  ⚛ Quantum Circuit  →  😊 POSITIVE (87%)
"Broke after one day."  →  [TF-IDF + PCA]  →  ⚛ Quantum Circuit  →  😞 NEGATIVE (91%)
```

---

## How It Works

### Architecture

```
Raw Text
   │
   ▼
TF-IDF Vectorizer (500 vocab, bigrams)
   │
   ▼
PCA → 4 dimensions
   │
   ▼
MinMax Scale → [0, π]          ← angles for quantum encoding
   │
   ▼
┌─────────────────────────────────────────┐
│       Variational Quantum Circuit       │
│                                         │
│  q0: ──RY(x₀)──●──RY(θ₀₀)──●──────── │
│  q1: ──RY(x₁)──X──RY(θ₀₁)──│──●───── │
│  q2: ──RY(x₂)──●──RY(θ₀₂)──X──│───── │
│  q3: ──RY(x₃)──X──RY(θ₀₃)────X───── │
│                                         │
│  ⟨Z₀⟩ ∈ [-1, 1] → sigmoid → prob      │
└─────────────────────────────────────────┘
   │
   ▼
Binary Cross-Entropy Loss + Adam Optimiser
   │
   ▼
POSITIVE / NEGATIVE
```

### Key Concepts

| Component | Classical Equivalent | Quantum Version |
|---|---|---|
| Feature encoding | Input layer | `AngleEmbedding` (RY gates) |
| Trainable layer | Dense layer weights | `BasicEntanglerLayers` (rotation + CNOT) |
| Nonlinearity | ReLU/sigmoid | Quantum interference |
| Output | Logit | PauliZ expectation value |

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/quantum-sentiment.git
cd quantum-sentiment
pip install -r requirements.txt
```

### 2. Classify text immediately (trains in ~30 sec)

```bash
python classify.py "This product is absolutely amazing!"
# → 😊 POSITIVE (84%)

python classify.py "Terrible quality, avoid at all costs."
# → 😞 NEGATIVE (91%)
```

### 3. Interactive mode

```bash
python classify.py --interactive
```

### 4. Full training run with plots

```bash
python train.py --epochs 25 --lr 0.1
```

This saves `weights.npy`, `training_curve.png`, and `confusion_matrix.png`.

### 5. Use saved weights

```bash
python classify.py --weights weights.npy "I'm so happy with this!"
```

---

## Results

Training on the built-in 30-sentence demo dataset (25 epochs):

| Metric | Score |
|---|---|
| Accuracy | ~80–90% |
| Precision | ~82–92% |
| Recall | ~80–90% |
| F1 | ~81–91% |

> Results vary slightly due to quantum randomness in measurement simulation.

---

## Project Structure

```
quantum-sentiment/
├── src/
│   ├── quantum_classifier.py   # Quantum circuit + training loop
│   ├── preprocessing.py        # TF-IDF → PCA → angle scaling
│   └── visualize.py            # Plots: circuit, loss curve, confusion matrix
├── tests/
│   └── test_classifier.py      # pytest unit tests
├── train.py                    # Full training script
├── classify.py                 # CLI inference tool
├── requirements.txt
└── README.md
```

---

## Run on Real Quantum Hardware

IBM offers free access to real quantum computers. To run this on actual hardware:

1. Create a free account at [quantum.ibm.com](https://quantum.ibm.com)
2. Get your API token
3. Change the device in `src/quantum_classifier.py`:

```python
# Simulator (default)
dev = qml.device("default.qubit", wires=N_QUBITS)

# Real IBM quantum hardware
import qiskit_ibm_runtime
dev = qml.device("qiskit.ibm", wires=N_QUBITS,
                 backend="ibm_brisbane",
                 ibmq_token="YOUR_TOKEN")
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Extending This Project

Ideas to make it your own:

- **Larger dataset** — plug in IMDB reviews or Twitter sentiment data
- **More qubits** — increase `N_QUBITS` for richer feature encoding
- **Hybrid model** — add classical dense layers before/after the circuit
- **Multi-class** — extend to positive / neutral / negative with 2 output qubits
- **Web app** — wrap `classify.py` in a FastAPI endpoint

---

## Tech Stack

- [PennyLane](https://pennylane.ai) — quantum ML framework
- [scikit-learn](https://scikit-learn.org) — TF-IDF, PCA, preprocessing
- [NumPy](https://numpy.org) — numerical computing
- [Matplotlib](https://matplotlib.org) — visualisations

---

## What is Quantum ML?

Classical machine learning uses matrix multiplications and nonlinear activations. Quantum ML encodes data into **quantum states** and uses **unitary transformations** (quantum gates) to process it. The key advantages being researched:

- **Superposition** — a qubit can represent 0 and 1 simultaneously, enabling parallel computation
- **Entanglement** — correlations between qubits that have no classical equivalent
- **Interference** — amplifying correct answers and cancelling wrong ones

This project uses the **variational quantum eigensolver (VQE)** paradigm: a parameterised circuit trained by a classical optimiser — the best of both worlds.

---

## Licence

MIT — do whatever you want with it.

---

*Built with ⚛ PennyLane and a healthy curiosity about the future of computing.*
