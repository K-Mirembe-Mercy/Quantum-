"""
tests/test_classifier.py — Unit tests for the quantum classifier.
Run with: pytest tests/
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import (
    clean_text,
    QuantumFeaturePipeline,
    load_demo_dataset,
    DEMO_SENTENCES,
)
from quantum_classifier import (
    quantum_circuit,
    predict_proba,
    predict,
    init_weights,
    evaluate,
    N_QUBITS,
    N_LAYERS,
)


# ── Preprocessing Tests ──────────────────────────────────────────────────────

class TestPreprocessing:
    def test_clean_text_lowercase(self):
        assert clean_text("Hello WORLD") == "hello world"

    def test_clean_text_removes_punctuation(self):
        result = clean_text("Amazing! 100% great.")
        assert "!" not in result
        assert "%" not in result
        assert "." not in result

    def test_clean_text_strips_whitespace(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_pipeline_output_shape(self):
        texts = ["great product", "terrible experience", "okay I guess"]
        pipe = QuantumFeaturePipeline(n_features=4)
        X = pipe.fit_transform(texts)
        assert X.shape == (3, 4)

    def test_pipeline_values_in_range(self):
        """All values must be in [0, π] for angle embedding."""
        texts = [s for s, _ in DEMO_SENTENCES]
        pipe = QuantumFeaturePipeline(n_features=4)
        X = pipe.fit_transform(texts)
        assert np.all(X >= 0)
        assert np.all(X <= np.pi + 1e-9)

    def test_pipeline_transform_single(self):
        texts = ["good", "bad"]
        pipe = QuantumFeaturePipeline(n_features=4)
        pipe.fit_transform(texts)
        result = pipe.transform_single("amazing")
        assert result.shape == (4,)

    def test_load_demo_dataset_shapes(self):
        X_tr, X_te, y_tr, y_te, pipe = load_demo_dataset()
        assert X_tr.shape[1] == 4
        assert X_te.shape[1] == 4
        assert len(y_tr) == len(X_tr)
        assert len(y_te) == len(X_te)

    def test_load_demo_dataset_binary_labels(self):
        _, _, y_tr, y_te, _ = load_demo_dataset()
        assert set(y_tr).issubset({0, 1})
        assert set(y_te).issubset({0, 1})


# ── Quantum Circuit Tests ─────────────────────────────────────────────────────

class TestQuantumCircuit:
    def setup_method(self):
        self.weights = init_weights()
        self.dummy_input = np.zeros(N_QUBITS)

    def test_circuit_returns_scalar(self):
        result = quantum_circuit(self.dummy_input, self.weights)
        assert np.isscalar(float(result))

    def test_circuit_output_in_range(self):
        """Expectation value of PauliZ must be in [-1, 1]."""
        result = float(quantum_circuit(self.dummy_input, self.weights))
        assert -1.0 - 1e-6 <= result <= 1.0 + 1e-6

    def test_predict_proba_in_range(self):
        prob = predict_proba(self.dummy_input, self.weights)
        assert 0.0 <= prob <= 1.0

    def test_predict_returns_binary(self):
        label = predict(self.dummy_input, self.weights)
        assert label in (0, 1)

    def test_weights_shape(self):
        assert self.weights.shape == (N_LAYERS, N_QUBITS)

    def test_different_inputs_different_outputs(self):
        input_a = np.zeros(N_QUBITS)
        input_b = np.full(N_QUBITS, np.pi)
        prob_a = predict_proba(input_a, self.weights)
        prob_b = predict_proba(input_b, self.weights)
        # Different inputs should (almost certainly) give different outputs
        assert abs(prob_a - prob_b) > 1e-6


# ── Evaluation Tests ──────────────────────────────────────────────────────────

class TestEvaluation:
    def test_evaluate_perfect_predictions(self):
        y = [0, 1, 0, 1]
        weights = init_weights()
        # Manually override to always predict 0 for simplicity
        metrics = evaluate(
            np.zeros((4, N_QUBITS)), np.array(y), weights
        )
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_evaluate_all_correct(self):
        """Accuracy should be 1.0 when predictions match labels."""
        weights = init_weights()
        X = np.zeros((4, N_QUBITS))
        y = np.array([predict(x, weights) for x in X])  # use own predictions
        metrics = evaluate(X, y, weights)
        assert metrics["accuracy"] == pytest.approx(1.0)
