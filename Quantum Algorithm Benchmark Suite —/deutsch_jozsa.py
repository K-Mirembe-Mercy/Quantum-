"""
Deutsch-Jozsa Algorithm
-----------------------
Determines whether a function f: {0,1}^n → {0,1} is constant or balanced
in a single query — exponentially faster than any deterministic classical approach.

  Constant : f(x) = 0 for all x, or f(x) = 1 for all x
  Balanced : f(x) = 0 for exactly half of all inputs

Measurement outcome:
  |00...0⟩ → constant
  anything else → balanced
"""

import numpy as np
from qiskit import QuantumCircuit


def _constant_oracle(n: int, output_bit: int = 0) -> QuantumCircuit:
    """
    Constant oracle: always outputs output_bit.
    output_bit=0 → identity; output_bit=1 → X on ancilla.
    """
    qc = QuantumCircuit(n + 1, name="Const-Oracle")
    if output_bit == 1:
        qc.x(n)  # flip ancilla
    return qc


def _balanced_oracle(n: int) -> QuantumCircuit:
    """
    Balanced oracle implemented as a parity function:
      f(x) = x_0 XOR x_1 XOR ... XOR x_{n-1}
    Each input qubit is CNOT-ed onto the ancilla.
    """
    qc = QuantumCircuit(n + 1, name="Bal-Oracle")
    # Random subset of qubits to flip before/after CNOTs
    # so f is balanced but not trivially the identity parity
    flip_mask = np.random.randint(0, 2, n)
    for i, flip in enumerate(flip_mask):
        if flip:
            qc.x(i)
    for i in range(n):
        qc.cx(i, n)
    for i, flip in enumerate(flip_mask):
        if flip:
            qc.x(i)
    return qc


def build_dj_circuit(n: int, oracle_type: str = "balanced") -> QuantumCircuit:
    """
    Builds the Deutsch-Jozsa circuit.

    Args:
        n           : number of input qubits
        oracle_type : 'constant' or 'balanced'

    Returns:
        Measured QuantumCircuit (n classical bits; ancilla not measured)
    """
    assert oracle_type in ("constant", "balanced"), \
        "oracle_type must be 'constant' or 'balanced'"

    qc = QuantumCircuit(n + 1, n)

    # Ancilla starts in |1⟩ so the phase kickback works
    qc.x(n)
    qc.h(range(n + 1))
    qc.barrier()

    # Apply oracle
    if oracle_type == "constant":
        oracle = _constant_oracle(n, output_bit=np.random.randint(0, 2))
    else:
        oracle = _balanced_oracle(n)
    qc.compose(oracle, inplace=True)
    qc.barrier()

    # Hadamard on input register, then measure
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc
