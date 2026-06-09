"""Tests for SCSB ensemble models."""

import numpy as np
import pytest
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from scsb import SCSBClassifier, SCSBRegressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_data():
    """Synthetic classification dataset with a fitted base ensemble."""
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    base = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=30,
        random_state=42,
        bootstrap=True,
    )
    base.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, base


@pytest.fixture
def regression_data():
    """Synthetic regression dataset with a fitted base ensemble."""
    X, y = make_regression(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    base = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=3),
        n_estimators=30,
        random_state=42,
        bootstrap=True,
    )
    base.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, base


# ---------------------------------------------------------------------------
# SCSBClassifier
# ---------------------------------------------------------------------------

class TestSCSBClassifier:

    def test_fit_compresses_ensemble(self, classification_data):
        """SCSB should prune at least some estimators."""
        X_train, _, y_train, _, base = classification_data
        model = SCSBClassifier(base_ensemble=base, lam=0.05)
        model.fit(X_train, y_train)

        assert model.n_active_estimators_ < model.n_estimators_
        assert model.compression_ratio_ > 0.0

    def test_weights_on_simplex(self, classification_data):
        """Optimized weights must be non-negative and sum to 1."""
        X_train, _, y_train, _, base = classification_data
        model = SCSBClassifier(base_ensemble=base, lam=0.05)
        model.fit(X_train, y_train)

        assert np.all(model.weights_ >= -1e-10)
        assert abs(np.sum(model.weights_) - 1.0) < 1e-4

    def test_active_weights_sum_to_one(self, classification_data):
        """Active (renormalized) weights must sum to 1."""
        X_train, _, y_train, _, base = classification_data
        model = SCSBClassifier(base_ensemble=base, lam=0.05)
        model.fit(X_train, y_train)

        np.testing.assert_allclose(np.sum(model.active_weights_), 1.0, atol=1e-10)

    def test_predict_returns_valid_classes(self, classification_data):
        """Predictions must be members of the training classes."""
        X_train, X_test, y_train, _, base = classification_data
        model = SCSBClassifier(base_ensemble=base, lam=0.05)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        assert all(p in model.classes_ for p in preds)

    def test_predict_proba_sums_to_one(self, classification_data):
        """Class probabilities must sum to 1 for each sample."""
        X_train, X_test, y_train, _, base = classification_data
        model = SCSBClassifier(base_ensemble=base, lam=0.05)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_accuracy_not_catastrophic(self, classification_data):
        """SCSB accuracy should not drop more than 10% from the base."""
        X_train, X_test, y_train, y_test, base = classification_data
        model = SCSBClassifier(base_ensemble=base, lam=0.05)
        model.fit(X_train, y_train)

        base_acc = base.score(X_test, y_test)
        scsb_acc = model.score(X_test, y_test)
        assert scsb_acc >= base_acc - 0.10

    def test_auto_fits_unfitted_ensemble(self):
        """SCSB should auto-fit an unfitted base ensemble."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        unfitted = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=10,
            random_state=42,
            bootstrap=True,
        )
        model = SCSBClassifier(base_ensemble=unfitted, lam=0.05)
        model.fit(X, y)

        assert model.n_active_estimators_ > 0
        assert hasattr(model, "weights_")

    def test_compression_ratio_is_fraction_pruned(self, classification_data):
        """compression_ratio_ == fraction of estimators *pruned*."""
        X_train, _, y_train, _, base = classification_data
        model = SCSBClassifier(base_ensemble=base, lam=0.05)
        model.fit(X_train, y_train)

        expected = 1.0 - model.n_active_estimators_ / model.n_estimators_
        assert abs(model.compression_ratio_ - expected) < 1e-10

    def test_no_oob_arrays_stored(self, classification_data):
        """OOB arrays should NOT be stored on the fitted model."""
        X_train, _, y_train, _, base = classification_data
        model = SCSBClassifier(base_ensemble=base, lam=0.05)
        model.fit(X_train, y_train)

        assert not hasattr(model, "oob_predictions_")
        assert not hasattr(model, "oob_indicators_")


# ---------------------------------------------------------------------------
# SCSBRegressor
# ---------------------------------------------------------------------------

class TestSCSBRegressor:

    def test_fit_compresses_ensemble(self, regression_data):
        """SCSB should prune at least some estimators."""
        X_train, _, y_train, _, base = regression_data
        model = SCSBRegressor(base_ensemble=base, lam=0.1)
        model.fit(X_train, y_train)

        assert model.n_active_estimators_ < model.n_estimators_
        assert model.compression_ratio_ > 0.0

    def test_weights_on_simplex(self, regression_data):
        """Optimized weights must be non-negative and sum to 1."""
        X_train, _, y_train, _, base = regression_data
        model = SCSBRegressor(base_ensemble=base, lam=0.1)
        model.fit(X_train, y_train)

        assert np.all(model.weights_ >= -1e-10)
        assert abs(np.sum(model.weights_) - 1.0) < 1e-4

    def test_predict_shape(self, regression_data):
        """Predictions should have shape (n_samples,)."""
        X_train, X_test, y_train, _, base = regression_data
        model = SCSBRegressor(base_ensemble=base, lam=0.1)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        assert preds.shape == (X_test.shape[0],)

    def test_r2_not_catastrophic(self, regression_data):
        """SCSB R² should not drop more than 20% from the base."""
        X_train, X_test, y_train, y_test, base = regression_data
        model = SCSBRegressor(base_ensemble=base, lam=0.1)
        model.fit(X_train, y_train)

        base_r2 = base.score(X_test, y_test)
        scsb_r2 = model.score(X_test, y_test)
        assert scsb_r2 >= base_r2 - 0.20

    def test_auto_fits_unfitted_ensemble(self):
        """SCSB should auto-fit an unfitted base ensemble."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        unfitted = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=2),
            n_estimators=10,
            random_state=42,
            bootstrap=True,
        )
        model = SCSBRegressor(base_ensemble=unfitted, lam=0.1)
        model.fit(X, y)

        assert model.n_active_estimators_ > 0
        assert hasattr(model, "weights_")

    def test_no_oob_arrays_stored(self, regression_data):
        """OOB arrays should NOT be stored on the fitted model."""
        X_train, _, y_train, _, base = regression_data
        model = SCSBRegressor(base_ensemble=base, lam=0.1)
        model.fit(X_train, y_train)

        assert not hasattr(model, "oob_predictions_")
        assert not hasattr(model, "oob_indicators_")
