"""Simplex-Constrained Sparse Bagging (SCSB) ensemble models.

This module implements model-agnostic post-training compression and calibration
for bagging ensembles via simplex-constrained optimization of estimator weights.
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MetaEstimatorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from scipy.optimize import minimize


class _BaseSCSB(BaseEstimator, MetaEstimatorMixin):
    """Base class for Simplex-Constrained Sparse Bagging (SCSB) models.

    Parameters
    ----------
    base_ensemble : estimator
        A fitted (or unfitted) scikit-learn bagging ensemble that exposes
        ``estimators_samples_`` after fitting (e.g., ``BaggingClassifier``
        or ``BaggingRegressor`` with ``bootstrap=True``).
    lam : float, default=0.05
        Regularization strength for the concave sparsity-inducing penalty
        (``-lam * ||w||_2^2``). Higher values induce more sparsity.
    max_iter : int, default=100
        Maximum number of iterations for the SLSQP optimizer.
    tol : float, default=1e-6
        Convergence tolerance for the SLSQP optimizer.
    """

    def __init__(self, base_ensemble, lam=0.05, max_iter=100, tol=1e-6):
        self.base_ensemble = base_ensemble
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def _get_oob_indicators(self, base_ensemble, n_samples):
        """Extract OOB indicator matrix of shape (n_samples, n_estimators).

        Handles both index-array and boolean-mask formats of
        ``estimators_samples_`` across scikit-learn versions.
        """
        n_estimators = len(base_ensemble.estimators_)
        oob_indicators = np.ones((n_samples, n_estimators), dtype=bool)

        if not hasattr(base_ensemble, "estimators_samples_"):
            raise ValueError(
                "The base ensemble must expose `estimators_samples_`. "
                "Please use a scikit-learn `BaggingClassifier` or "
                "`BaggingRegressor` with bootstrap=True."
            )

        for j, samples in enumerate(base_ensemble.estimators_samples_):
            # Build a boolean mask for in-bag samples, handling both
            # index arrays and boolean masks across sklearn versions.
            in_bag_mask = np.zeros(n_samples, dtype=bool)
            in_bag_mask[samples] = True
            oob_indicators[:, j] = ~in_bag_mask

        return oob_indicators

    def _apply_sparsity(self):
        """Extract active estimators and compute compression statistics.

        Sets the following attributes:
            active_idx_, active_weights_, active_estimators_,
            n_active_estimators_, compression_ratio_
        """
        self.active_idx_ = np.where(self.weights_ > 1e-5)[0]
        self.active_weights_ = self.weights_[self.active_idx_]

        if len(self.active_weights_) > 0:
            self.active_weights_ = self.active_weights_ / np.sum(self.active_weights_)
        else:
            # Fallback: keep the single best estimator
            self.active_idx_ = np.array([np.argmax(self.weights_)])
            self.active_weights_ = np.array([1.0])

        self.active_estimators_ = [self.estimators_[idx] for idx in self.active_idx_]
        self.n_active_estimators_ = len(self.active_estimators_)
        self.compression_ratio_ = 1.0 - (self.n_active_estimators_ / self.n_estimators_)

    def _check_convergence(self, result):
        """Emit a warning if the optimizer did not converge."""
        if not result.success:
            warnings.warn(
                f"SCSB weight optimization did not converge: {result.message}. "
                f"Consider increasing `max_iter` (current: {self.max_iter}) "
                f"or relaxing `tol` (current: {self.tol}).",
                stacklevel=3,
            )


class SCSBClassifier(_BaseSCSB, ClassifierMixin):
    """Simplex-Constrained Sparse Bagging Classifier.

    Compresses and calibrates a bagging classifier ensemble by finding optimal
    sparse weights over the probability simplex using OOB predictions.

    Parameters
    ----------
    base_ensemble : estimator
        A fitted (or unfitted) scikit-learn ``BaggingClassifier`` (or
        compatible) with ``bootstrap=True``.
    lam : float, default=0.05
        Regularization strength for the concave penalty.
    max_iter : int, default=100
        Maximum iterations for the SLSQP optimizer.
    tol : float, default=1e-6
        Convergence tolerance for the optimizer.

    Attributes
    ----------
    weights_ : ndarray of shape (n_estimators,)
        Optimized simplex weights for all base estimators.
    active_estimators_ : list
        Base estimators with non-zero weight.
    n_active_estimators_ : int
        Number of active (non-pruned) estimators.
    compression_ratio_ : float
        Fraction of estimators pruned (``1 - active / total``).
    """

    def __init__(self, base_ensemble, lam=0.05, max_iter=100, tol=1e-6):
        super().__init__(
            base_ensemble=base_ensemble,
            lam=lam,
            max_iter=max_iter,
            tol=tol,
        )

    def fit(self, X, y):
        """Fit the SCSB classifier weights.

        Fits the underlying bagging ensemble if not already fitted, extracts
        OOB predictions, and solves the simplex-constrained optimization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        check_classification_targets(y)

        # Fit base ensemble if needed
        try:
            check_is_fitted(self.base_ensemble)
        except Exception:
            self.base_ensemble.fit(X, y)

        self.estimators_ = self.base_ensemble.estimators_
        self.n_estimators_ = len(self.estimators_)
        self.classes_ = self.base_ensemble.classes_
        self.n_classes_ = len(self.classes_)

        n_samples = X.shape[0]
        oob_indicators = self._get_oob_indicators(self.base_ensemble, n_samples)

        # Precompute OOB predictions: shape (n_samples, n_estimators, n_classes)
        oob_predictions = np.zeros((n_samples, self.n_estimators_, self.n_classes_))

        for j, estimator in enumerate(self.estimators_):
            oob_idx = np.where(oob_indicators[:, j])[0]
            if len(oob_idx) > 0:
                probs = estimator.predict_proba(X[oob_idx])

                # Handle estimators trained on a subset of classes
                if probs.shape[1] < self.n_classes_:
                    mapped = np.zeros((len(oob_idx), self.n_classes_))
                    idx_map = [np.where(self.classes_ == c)[0][0]
                               for c in estimator.classes_]
                    mapped[:, idx_map] = probs
                    oob_predictions[oob_idx, j, :] = mapped
                else:
                    oob_predictions[oob_idx, j, :] = probs

        # Optimize weights with analytical gradient
        self.weights_ = self._optimize_weights(y, oob_indicators, oob_predictions)
        self._apply_sparsity()

        return self

    def _optimize_weights(self, y, oob_indicators, oob_predictions):
        """Solve the simplex-constrained optimization for classifier weights."""
        n_samples = len(y)

        # One-hot encode targets
        y_one_hot = np.zeros((n_samples, self.n_classes_))
        for c_idx, c in enumerate(self.classes_):
            y_one_hot[y == c, c_idx] = 1.0

        oob_ind_float = oob_indicators.astype(np.float64)

        def objective_and_grad(w):
            # Forward pass
            denom = oob_ind_float @ w                                    # (M,)
            numer = np.einsum('j,ijc->ic', w, oob_predictions)           # (M, C)
            denom_safe = denom[:, np.newaxis] + 1e-15                    # (M, 1)
            oob_pred = numer / denom_safe                                # (M, C)
            oob_pred = np.clip(oob_pred, 1e-15, 1.0 - 1e-15)

            # Log-Loss
            log_loss = -np.mean(np.sum(y_one_hot * np.log(oob_pred), axis=1))
            penalty = -self.lam * np.sum(w ** 2)
            obj = log_loss + penalty

            # Analytical gradient
            # dL/dw_k = -1/M * sum_i [I_ik/D_i * (sum_c y_ic*P_ikc/p_ic - 1)] - 2*lam*w_k
            ratio = y_one_hot / oob_pred                                 # (M, C)
            weighted_preds = np.einsum('ic,ijc->ij', ratio, oob_predictions)  # (M, N)
            term = (weighted_preds - 1.0) * oob_ind_float / (denom[:, np.newaxis] + 1e-15)
            grad = -np.mean(term, axis=0) - 2.0 * self.lam * w          # (N,)

            return obj, grad

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * self.n_estimators_
        w0 = np.ones(self.n_estimators_) / self.n_estimators_

        res = minimize(
            objective_and_grad,
            w0,
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=constraints,
            tol=self.tol,
            options={'maxiter': self.max_iter},
        )

        self._check_convergence(res)
        return res.x

    def predict_proba(self, X):
        """Predict class probabilities for X using only active estimators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        probs : ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self.n_classes_))

        for weight, estimator in zip(self.active_weights_, self.active_estimators_):
            est_probs = estimator.predict_proba(X)

            if est_probs.shape[1] < self.n_classes_:
                mapped = np.zeros((n_samples, self.n_classes_))
                idx_map = [np.where(self.classes_ == c)[0][0]
                           for c in estimator.classes_]
                mapped[:, idx_map] = est_probs
                probs += weight * mapped
            else:
                probs += weight * est_probs

        # Ensure probabilities are bounded in [0, 1] and sum to 1
        probs = np.clip(probs, 0.0, 1.0)
        sum_probs = np.sum(probs, axis=1, keepdims=True)
        sum_probs = np.where(sum_probs == 0, 1e-15, sum_probs)
        probs = probs / sum_probs
        return probs

    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class SCSBRegressor(_BaseSCSB, RegressorMixin):
    """Simplex-Constrained Sparse Bagging Regressor.

    Compresses a bagging regressor ensemble by finding optimal sparse weights
    over the probability simplex using OOB predictions.

    Parameters
    ----------
    base_ensemble : estimator
        A fitted (or unfitted) scikit-learn ``BaggingRegressor`` (or
        compatible) with ``bootstrap=True``.
    lam : float, default=0.05
        Regularization strength for the concave penalty.
    max_iter : int, default=100
        Maximum iterations for the SLSQP optimizer.
    tol : float, default=1e-6
        Convergence tolerance for the optimizer.

    Attributes
    ----------
    weights_ : ndarray of shape (n_estimators,)
        Optimized simplex weights for all base estimators.
    active_estimators_ : list
        Base estimators with non-zero weight.
    n_active_estimators_ : int
        Number of active (non-pruned) estimators.
    compression_ratio_ : float
        Fraction of estimators pruned (``1 - active / total``).
    """

    def __init__(self, base_ensemble, lam=0.05, max_iter=100, tol=1e-6):
        super().__init__(
            base_ensemble=base_ensemble,
            lam=lam,
            max_iter=max_iter,
            tol=tol,
        )

    def fit(self, X, y):
        """Fit the SCSB regressor weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)

        # Fit base ensemble if needed
        try:
            check_is_fitted(self.base_ensemble)
        except Exception:
            self.base_ensemble.fit(X, y)

        self.estimators_ = self.base_ensemble.estimators_
        self.n_estimators_ = len(self.estimators_)

        n_samples = X.shape[0]
        oob_indicators = self._get_oob_indicators(self.base_ensemble, n_samples)

        # Precompute OOB predictions: shape (n_samples, n_estimators)
        oob_predictions = np.zeros((n_samples, self.n_estimators_))

        for j, estimator in enumerate(self.estimators_):
            oob_idx = np.where(oob_indicators[:, j])[0]
            if len(oob_idx) > 0:
                oob_predictions[oob_idx, j] = estimator.predict(X[oob_idx])

        # Optimize weights with analytical gradient
        self.weights_ = self._optimize_weights(y, oob_indicators, oob_predictions)
        self._apply_sparsity()

        return self

    def _optimize_weights(self, y, oob_indicators, oob_predictions):
        """Solve the simplex-constrained optimization for regressor weights."""
        oob_ind_float = oob_indicators.astype(np.float64)

        def objective_and_grad(w):
            # Forward pass
            denom = oob_ind_float @ w                              # (M,)
            numer = oob_predictions @ w                             # (M,)
            denom_safe = denom + 1e-15
            oob_pred = numer / denom_safe                           # (M,)

            # MSE
            residual = oob_pred - y
            mse = np.mean(residual ** 2)
            penalty = -self.lam * np.sum(w ** 2)
            obj = mse + penalty

            # Analytical gradient
            # dL/dw_k = 2/M * sum_i (p_i - y_i) * I_ik * (f_k - p_i) / D_i - 2*lam*w_k
            diff = oob_predictions - oob_pred[:, np.newaxis]        # (M, N)
            term = residual[:, np.newaxis] * oob_ind_float * diff / denom_safe[:, np.newaxis]
            grad = 2.0 * np.mean(term, axis=0) - 2.0 * self.lam * w  # (N,)

            return obj, grad

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * self.n_estimators_
        w0 = np.ones(self.n_estimators_) / self.n_estimators_

        res = minimize(
            objective_and_grad,
            w0,
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=constraints,
            tol=self.tol,
            options={'maxiter': self.max_iter},
        )

        self._check_convergence(res)
        return res.x

    def predict(self, X):
        """Predict regression target for X using only active estimators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        n_samples = X.shape[0]
        preds = np.zeros(n_samples)

        for weight, estimator in zip(self.active_weights_, self.active_estimators_):
            preds += weight * estimator.predict(X)

        return preds
