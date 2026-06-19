"""
Bernstein-Vazirani Algorithm
----------------------------
Recovers a hidden bitstring s from a linear function f(x) = s·x (mod 2)
in a single quantum query — classical requires n queries.

The dot product oracle phase-kicks each input qubit conditioned on
the corresponding bit of s, so a single round of Hadamards reveals s exactly.
"""

from qiskit import QuantumCircuit


def _bv_oracle(n: int, secret: str) -> QuantumCircuit:
    """
    Phase-kickback oracle for f(x) = secret · x (mod 2).
    CNOT from qubit i to ancilla for each '1' bit in secret.
    """
    assert len(secret) == n
    qc = QuantumCircuit(n + 1, name="BV-Oracle")
    for i, bit in enumerate(reversed(secret)):
        if bit == "1":
            qc.cx(i, n)
    return qc


def build_bv_circuit(n: int, secret: str) -> QuantumCircuit:
    """
    Builds the Bernstein-Vazirani circuit.

    Args:
        n      : number of input qubits
        secret : hidden bitstring to recover, e.g. '1011'

    Returns:
        Measured QuantumCircuit; measuring input register gives secret directly.
    """
    assert len(secret) == n, "secret length must equal n"

    qc = QuantumCircuit(n + 1, n)
    # Ancilla in |−⟩ for phase kickback
    qc.x(n)
    qc.h(range(n + 1))
    qc.barrier()

    qc.compose(_bv_oracle(n, secret), inplace=True)
    qc.barrier()

    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc
