# ⚛️ Quantum Algorithm Benchmark Suite

A Python project that implements, simulates, and benchmarks three fundamental
quantum algorithms under both ideal and realistic noisy conditions using **Qiskit** and **Qiskit Aer**.

---

## Algorithms

| Algorithm | Quantum Speedup | Classical Queries | Quantum Queries |
|-----------|----------------|-------------------|-----------------|
| **Grover's Search** | Quadratic | O(N) | O(√N) |
| **Deutsch-Jozsa** | Exponential | O(2^(n-1)+1) | O(1) |
| **Bernstein-Vazirani** | Linear | O(n) | O(1) |

---

## Project Structure

```
quantum-benchmark/
├── algorithms/
│   ├── __init__.py
│   ├── grovers.py            # Grover's Search with phase oracle + diffusion
│   ├── deutsch_jozsa.py      # Deutsch-Jozsa with constant/balanced oracles
│   └── bernstein_vazirani.py # Bernstein-Vazirani dot-product oracle
├── results/
│   ├── benchmark_results.png # Generated benchmark plots
│   └── raw_results.json      # Raw data from all runs
├── benchmark.py              # Main runner: runs all benchmarks, saves plots
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/quantum-benchmark.git
cd quantum-benchmark

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

```bash
python benchmark.py
```

This will:
1. Run each algorithm for **n = 2–6 qubits**, both ideal and noisy
2. Save raw JSON results to `results/raw_results.json`
3. Generate a 6-panel benchmark plot at `results/benchmark_results.png`

---

## Noise Model

The noisy simulation uses a realistic hardware-inspired model built with `qiskit-aer`:

- **Depolarizing errors** — 0.1% on single-qubit gates, 1% on two-qubit gates
- **Thermal relaxation** — T1 = 50 µs, T2 = 70 µs (typical superconducting qubit values)

This lets you observe how circuit depth and qubit count affect algorithm fidelity
as noise accumulates — a core concern in near-term (NISQ) quantum computing.

---

## What the Plots Show

| Panel | What to look for |
|-------|-----------------|
| Success Rate (each algo) | Ideal ≈ 1.0; noisy degrades with qubit count |
| Circuit Depth vs Qubits | Grover grows fastest — more iterations needed |
| Noise Degradation Bar | Algorithms with deeper circuits suffer more |
| Simulation Time | Exponential classical overhead of state-vector sim |

---

## Theory Notes

### Grover's Search
Uses **amplitude amplification** to boost the probability of measuring the target
state. Applies the oracle–diffusion cycle ⌊π/4 · √(2^n)⌋ times. Too few or too
many iterations reduces success probability.

### Deutsch-Jozsa
A single quantum query determines if a black-box function is *constant* (same
output for all inputs) or *balanced* (equal 0s and 1s). Classical algorithms
need up to 2^(n-1)+1 queries in the worst case.

### Bernstein-Vazirani
Recovers a hidden n-bit string *s* from the function f(x) = s·x (mod 2) using
phase kickback. Classically requires n separate queries; quantum needs just one.

---

## Requirements

- Python 3.10+
- qiskit ≥ 1.0
- qiskit-aer ≥ 0.14
- numpy, matplotlib

---

## License

MIT
