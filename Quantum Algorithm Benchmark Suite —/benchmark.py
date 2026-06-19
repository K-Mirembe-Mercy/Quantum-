"""
Quantum Algorithm Benchmark Suite
Compares Grover's Search, Deutsch-Jozsa, and Bernstein-Vazirani
under ideal and noisy simulation conditions.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from algorithms.grovers import build_grover_circuit
from algorithms.deutsch_jozsa import build_dj_circuit
from algorithms.bernstein_vazirani import build_bv_circuit

SHOTS = 4096
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Noise model ──────────────────────────────────────────────────────────────

def build_noise_model(p1q=0.001, p2q=0.01, t1=50e3, t2=70e3, gate_time=50):
    """
    Realistic noise model with:
      - Depolarizing errors on 1- and 2-qubit gates
      - Thermal relaxation (T1/T2) on all gates
    """
    nm = NoiseModel()
    # Thermal relaxation
    tr_1q = thermal_relaxation_error(t1, t2, gate_time)
    tr_2q = tr_1q.expand(thermal_relaxation_error(t1, t2, gate_time))
    # Depolarizing
    dp_1q = depolarizing_error(p1q, 1)
    dp_2q = depolarizing_error(p2q, 2)
    # Compose and add
    err_1q = dp_1q.compose(tr_1q)
    err_2q = dp_2q.compose(tr_2q)
    for gate in ["u1", "u2", "u3", "id", "rz", "sx", "x"]:
        nm.add_all_qubit_quantum_error(err_1q, gate)
    for gate in ["cx", "cz", "ecr"]:
        nm.add_all_qubit_quantum_error(err_2q, gate)
    return nm


# ── Simulators ───────────────────────────────────────────────────────────────

ideal_sim = AerSimulator()
noisy_sim = AerSimulator(noise_model=build_noise_model())


def run(circuit: QuantumCircuit, noisy: bool = False) -> dict:
    sim = noisy_sim if noisy else ideal_sim
    t = transpile(circuit, sim)
    t0 = time.perf_counter()
    counts = sim.run(t, shots=SHOTS).result().get_counts()
    elapsed = time.perf_counter() - t0
    return {"counts": counts, "time": elapsed}


# ── Success-rate helper ───────────────────────────────────────────────────────

def success_rate(counts: dict, target: str) -> float:
    return counts.get(target, 0) / SHOTS


# ── Benchmark each algorithm ─────────────────────────────────────────────────

def benchmark_grover():
    records = []
    for n in range(2, 7):
        target = format(np.random.randint(0, 2**n), f"0{n}b")
        circ = build_grover_circuit(n, target)
        for noisy in (False, True):
            res = run(circ, noisy)
            records.append({
                "n_qubits": n,
                "target": target,
                "noisy": noisy,
                "success_rate": success_rate(res["counts"], target),
                "time": res["time"],
                "depth": circ.depth(),
            })
        print(f"  Grover n={n} target={target} ✓")
    return records


def benchmark_dj():
    records = []
    for n in range(2, 7):
        for oracle_type in ("constant", "balanced"):
            circ = build_dj_circuit(n, oracle_type)
            expected = "0" * n if oracle_type == "constant" else None
            for noisy in (False, True):
                res = run(circ, noisy)
                if expected:
                    sr = success_rate(res["counts"], expected)
                else:
                    # Balanced: any non-zero string counts as correct
                    total_nonzero = sum(
                        v for k, v in res["counts"].items() if k != "0" * n
                    )
                    sr = total_nonzero / SHOTS
                records.append({
                    "n_qubits": n,
                    "oracle": oracle_type,
                    "noisy": noisy,
                    "success_rate": sr,
                    "time": res["time"],
                    "depth": circ.depth(),
                })
        print(f"  Deutsch-Jozsa n={n} ✓")
    return records


def benchmark_bv():
    records = []
    for n in range(2, 7):
        secret = format(np.random.randint(1, 2**n), f"0{n}b")
        circ = build_bv_circuit(n, secret)
        for noisy in (False, True):
            res = run(circ, noisy)
            records.append({
                "n_qubits": n,
                "secret": secret,
                "noisy": noisy,
                "success_rate": success_rate(res["counts"], secret),
                "time": res["time"],
                "depth": circ.depth(),
            })
        print(f"  Bernstein-Vazirani n={n} secret={secret} ✓")
    return records


# ── Plotting ─────────────────────────────────────────────────────────────────

ALGO_META = {
    "grover":   {"label": "Grover's Search",        "color": ("#4C72B0", "#C44E52")},
    "dj":       {"label": "Deutsch-Jozsa",           "color": ("#55A868", "#DD8452")},
    "bv":       {"label": "Bernstein-Vazirani",      "color": ("#8172B2", "#937860")},
}

def plot_results(data: dict):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0D1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for ax in axes:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#8B949E")
        ax.spines[:].set_color("#30363D")
        ax.xaxis.label.set_color("#8B949E")
        ax.yaxis.label.set_color("#8B949E")
        ax.title.set_color("#E6EDF3")

    plot_idx = 0
    for algo, records in data.items():
        meta = ALGO_META[algo]
        ideal = [r for r in records if not r["noisy"]]
        noisy = [r for r in records if r["noisy"]]
        ns = sorted({r["n_qubits"] for r in ideal})

        # Success rate plot
        ax = axes[plot_idx]
        ideal_sr = [np.mean([r["success_rate"] for r in ideal if r["n_qubits"] == n]) for n in ns]
        noisy_sr = [np.mean([r["success_rate"] for r in noisy if r["n_qubits"] == n]) for n in ns]
        ax.plot(ns, ideal_sr, "o-", color=meta["color"][0], label="Ideal", lw=2, ms=6)
        ax.plot(ns, noisy_sr, "s--", color=meta["color"][1], label="Noisy", lw=2, ms=6)
        ax.set_title(meta["label"])
        ax.set_xlabel("Qubits")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1.05)
        ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="#E6EDF3", fontsize=8)
        ax.grid(True, color="#21262D", lw=0.8)
        plot_idx += 1

    # Circuit depth comparison
    ax = axes[3]
    for algo, records in data.items():
        meta = ALGO_META[algo]
        ideal = [r for r in records if not r["noisy"]]
        ns = sorted({r["n_qubits"] for r in ideal})
        depths = [np.mean([r["depth"] for r in ideal if r["n_qubits"] == n]) for n in ns]
        ax.plot(ns, depths, "o-", label=meta["label"], lw=2, ms=5)
    ax.set_title("Circuit Depth vs Qubits")
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Gate Depth")
    ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="#E6EDF3", fontsize=7)
    ax.grid(True, color="#21262D", lw=0.8)

    # Noise degradation bar chart
    ax = axes[4]
    degradations, labels, colors = [], [], []
    for algo, records in data.items():
        meta = ALGO_META[algo]
        for n in sorted({r["n_qubits"] for r in records})[-3:]:
            ideal_sr = np.mean([r["success_rate"] for r in records if not r["noisy"] and r["n_qubits"] == n])
            noisy_sr = np.mean([r["success_rate"] for r in records if r["noisy"] and r["n_qubits"] == n])
            degradations.append(max(0, ideal_sr - noisy_sr))
            labels.append(f"{meta['label'][:3]}\nn={n}")
            colors.append(meta["color"][1])
    bars = ax.bar(range(len(degradations)), degradations, color=colors, edgecolor="#30363D")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("Noise Degradation (Ideal − Noisy)")
    ax.set_ylabel("Δ Success Rate")
    ax.grid(True, color="#21262D", lw=0.8, axis="y")

    # Runtime comparison
    ax = axes[5]
    for algo, records in data.items():
        meta = ALGO_META[algo]
        ideal = [r for r in records if not r["noisy"]]
        ns = sorted({r["n_qubits"] for r in ideal})
        times = [np.mean([r["time"] for r in ideal if r["n_qubits"] == n]) * 1000 for n in ns]
        ax.plot(ns, times, "o-", label=meta["label"], lw=2, ms=5)
    ax.set_title("Simulation Time (Ideal)")
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Time (ms)")
    ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="#E6EDF3", fontsize=7)
    ax.grid(True, color="#21262D", lw=0.8)

    fig.suptitle("Quantum Algorithm Benchmark Suite", color="#E6EDF3", fontsize=16, fontweight="bold", y=0.98)
    out = os.path.join(RESULTS_DIR, "benchmark_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Plot saved → {out}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Quantum Algorithm Benchmark Suite ===\n")

    print("[1/3] Running Grover's Search...")
    grover_data = benchmark_grover()

    print("\n[2/3] Running Deutsch-Jozsa...")
    dj_data = benchmark_dj()

    print("\n[3/3] Running Bernstein-Vazirani...")
    bv_data = benchmark_bv()

    all_data = {"grover": grover_data, "dj": dj_data, "bv": bv_data}

    with open(os.path.join(RESULTS_DIR, "raw_results.json"), "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\n  Raw data saved → {RESULTS_DIR}/raw_results.json")

    print("\nGenerating plots...")
    plot_results(all_data)

    print("\n=== Done! Results in ./results/ ===")
