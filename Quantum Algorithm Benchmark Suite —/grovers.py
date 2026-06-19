"""
Grover's Search Algorithm
-------------------------
Quadratic speedup for unstructured search: O(√N) vs classical O(N).

Given a target bitstring, this module builds a Grover circuit with:
  - A phase-oracle that marks the target state
  - The Grover diffusion operator (inversion about the mean)
  - ⌊π/4 · √(2^n)⌋ iterations for optimal amplification
"""

import numpy as np
from qiskit import QuantumCircuit


def phase_oracle(n: int, target: str) -> QuantumCircuit:
    """
    Marks the target state with a phase flip (|target⟩ → -|target⟩).
    Uses multi-controlled Z gate with X gates to handle '0' bits.
    """
    qc = QuantumCircuit(n, name="Oracle")
    # Flip qubits where target bit is '0' (so all |1⟩ when target is present)
    for i, bit in enumerate(reversed(target)):
        if bit == "0":
            qc.x(i)
    # Multi-controlled Z: phases the |11...1⟩ state
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    # Undo the X flips
    for i, bit in enumerate(reversed(target)):
        if bit == "0":
            qc.x(i)
    return qc


def diffusion_operator(n: int) -> QuantumCircuit:
    """
    Grover diffusion: 2|s⟩⟨s| - I  (inversion about the uniform superposition).
    Amplifies the amplitude of the marked state.
    """
    qc = QuantumCircuit(n, name="Diffusion")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(n: int, target: str) -> QuantumCircuit:
    """
    Builds a full Grover circuit for an n-qubit search with a single marked target.

    Args:
        n      : number of qubits (search space = 2^n)
        target : target bitstring, e.g. '1011'

    Returns:
        Measured QuantumCircuit
    """
    assert len(target) == n, "target length must equal n"
    iterations = max(1, int(np.floor(np.pi / 4 * np.sqrt(2**n))))

    qc = QuantumCircuit(n, n)
    # Initialise uniform superposition
    qc.h(range(n))
    qc.barrier()
    # Grover iterations
    oracle = phase_oracle(n, target)
    diffusion = diffusion_operator(n)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
        qc.barrier()
    qc.measure(range(n), range(n))
    return qc
