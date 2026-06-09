import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MetaEstimatorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from scipy.optimize import minimize

class _BaseSCSB(BaseEstimator, MetaEstimatorMixin):
    """Base class for Simplex-Constrained Sparse Bagging (SCSB) models."""

    def __init__(self, base_ensemble, lam=0.05, max_iter=100, tol=1e-6):
        self.base_ensemble = base_ensemble
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def _get_oob_indicators(self, base_ensemble, n_samples):
        """Extract OOB indicator matrix of shape (n_samples, n_estimators)."""
        n_estimators = len(base_ensemble.estimators_)
        oob_indicators = np.ones((n_samples, n_estimators), dtype=bool)
        
        if hasattr(base_ensemble, "estimators_samples_"):
            for j, samples in enumerate(base_ensemble.estimators_samples_):
                oob_indicators[samples, j] = False
        else:
            raise ValueError(
                "The base ensemble must expose `estimators_samples_`. "
                "Please use a scikit-learn `BaggingClassifier` or `BaggingRegressor` with bootstrap=True."
            )
        return oob_indicators


class SCSBClassifier(_BaseSCSB, ClassifierMixin):
    """Simplex-Constrained Sparse Bagging Classifier.

    Compresses and calibrates a bagging classifier ensemble by finding optimal
    sparse weights over the probability simplex using OOB predictions.
    """

    def __init__(self, base_ensemble, lam=0.05, max_iter=100, tol=1e-6):
        super().__init__(
            base_ensemble=base_ensemble,
            lam=lam,
            max_iter=max_iter,
            tol=tol
        )

    def fit(self, X, y):
        """Fit the SCSB classifier weights.

        Fits the underlying bagging ensemble if not already fitted, extracts
        OOB predictions, and solves the simplex-constrained optimization.
        """
        # Validate training data
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
        self.oob_indicators_ = self._get_oob_indicators(self.base_ensemble, n_samples)
        
        # Precompute OOB predictions of base estimators: shape (n_samples, n_estimators, n_classes)
        self.oob_predictions_ = np.zeros((n_samples, self.n_estimators_, self.n_classes_))
        
        for j, estimator in enumerate(self.estimators_):
            oob_idx = np.where(self.oob_indicators_[:, j])[0]
            if len(oob_idx) > 0:
                probs = estimator.predict_proba(X[oob_idx])
                
                # Handle cases where base estimator predicted a subset of classes
                if probs.shape[1] < self.n_classes_:
                    mapped_probs = np.zeros((len(oob_idx), self.n_classes_))
                    est_classes = estimator.classes_
                    class_idx_map = [np.where(self.classes_ == c)[0][0] for c in est_classes]
                    mapped_probs[:, class_idx_map] = probs
                    self.oob_predictions_[oob_idx, j, :] = mapped_probs
                else:
                    self.oob_predictions_[oob_idx, j, :] = probs
                    
        # Optimize weights
        self.weights_ = self._optimize_weights(y)
        
        # Sparse inference engine: extract estimators with weights above threshold
        self.active_idx_ = np.where(self.weights_ > 1e-5)[0]
        self.active_weights_ = self.weights_[self.active_idx_]
        
        if len(self.active_weights_) > 0:
            self.active_weights_ /= np.sum(self.active_weights_)
        else:
            # Fallback if all weights are thresholded to zero
            self.active_idx_ = np.array([np.argmax(self.weights_)])
            self.active_weights_ = np.array([1.0])
            
        self.active_estimators_ = [self.estimators_[idx] for idx in self.active_idx_]
        self.n_active_estimators_ = len(self.active_estimators_)
        self.compression_ratio_ = self.n_active_estimators_ / self.n_estimators_
        
        return self

    def _optimize_weights(self, y):
        n_samples = len(y)
        
        # One-hot encode targets
        y_one_hot = np.zeros((n_samples, self.n_classes_))
        for c_idx, c in enumerate(self.classes_):
            y_one_hot[y == c, c_idx] = 1.0
            
        def objective(w):
            # Calculate denominator for OOB average: sum_j w_j * I_{i,j}
            denom = np.dot(self.oob_indicators_, w)
            
            # Calculate numerator: sum_j w_j * P_{i,j,c}
            numer = np.einsum('j,ijc->ic', w, self.oob_predictions_)
            
            # Compute OOB prediction probabilities
            denom_clipped = denom[:, np.newaxis] + 1e-15
            oob_pred = numer / denom_clipped
            
            # Clip probabilities to avoid log(0)
            oob_pred = np.clip(oob_pred, 1e-15, 1.0 - 1e-15)
            
            # Calculate Log-Loss
            log_loss = -np.mean(np.sum(y_one_hot * np.log(oob_pred), axis=1))
            
            # Concave penalty to induce sparsity
            penalty = -self.lam * np.sum(w ** 2)
            
            return log_loss + penalty

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.0, 1.0) for _ in range(self.n_estimators_)]
        w0 = np.ones(self.n_estimators_) / self.n_estimators_
        
        res = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            tol=self.tol,
            options={'maxiter': self.max_iter}
        )
        
        return res.x

    def predict_proba(self, X):
        """Predict class probabilities for X using only active estimators."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self.n_classes_))
        
        for weight, estimator in zip(self.active_weights_, self.active_estimators_):
            est_probs = estimator.predict_proba(X)
            
            if est_probs.shape[1] < self.n_classes_:
                mapped_probs = np.zeros((n_samples, self.n_classes_))
                est_classes = estimator.classes_
                class_idx_map = [np.where(self.classes_ == c)[0][0] for c in est_classes]
                mapped_probs[:, class_idx_map] = est_probs
                probs += weight * mapped_probs
            else:
                probs += weight * est_probs
                
        return probs

    def predict(self, X):
        """Predict class labels for X."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class SCSBRegressor(_BaseSCSB, RegressorMixin):
    """Simplex-Constrained Sparse Bagging Regressor.

    Compresses and calibrates a bagging regressor ensemble by finding optimal
    sparse weights over the probability simplex using OOB predictions.
    """

    def __init__(self, base_ensemble, lam=0.05, max_iter=100, tol=1e-6):
        super().__init__(
            base_ensemble=base_ensemble,
            lam=lam,
            max_iter=max_iter,
            tol=tol
        )

    def fit(self, X, y):
        """Fit the SCSB regressor weights.

        Fits the underlying bagging ensemble if not already fitted, extracts
        OOB predictions, and solves the simplex-constrained optimization.
        """
        # Validate training data
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Fit base ensemble if needed
        try:
            check_is_fitted(self.base_ensemble)
        except Exception:
            self.base_ensemble.fit(X, y)
            
        self.estimators_ = self.base_ensemble.estimators_
        self.n_estimators_ = len(self.estimators_)
        
        n_samples = X.shape[0]
        self.oob_indicators_ = self._get_oob_indicators(self.base_ensemble, n_samples)
        
        # Precompute OOB predictions of base estimators: shape (n_samples, n_estimators)
        self.oob_predictions_ = np.zeros((n_samples, self.n_estimators_))
        
        for j, estimator in enumerate(self.estimators_):
            oob_idx = np.where(self.oob_indicators_[:, j])[0]
            if len(oob_idx) > 0:
                self.oob_predictions_[oob_idx, j] = estimator.predict(X[oob_idx])
                
        # Optimize weights
        self.weights_ = self._optimize_weights(y)
        
        # Sparse inference engine: extract estimators with weights above threshold
        self.active_idx_ = np.where(self.weights_ > 1e-5)[0]
        self.active_weights_ = self.weights_[self.active_idx_]
        
        if len(self.active_weights_) > 0:
            self.active_weights_ /= np.sum(self.active_weights_)
        else:
            # Fallback if all weights are thresholded to zero
            self.active_idx_ = np.array([np.argmax(self.weights_)])
            self.active_weights_ = np.array([1.0])
            
        self.active_estimators_ = [self.estimators_[idx] for idx in self.active_idx_]
        self.n_active_estimators_ = len(self.active_estimators_)
        self.compression_ratio_ = self.n_active_estimators_ / self.n_estimators_
        
        return self

    def _optimize_weights(self, y):
        def objective(w):
            # Calculate denominator for OOB average: sum_j w_j * I_{i,j}
            denom = np.dot(self.oob_indicators_, w)
            
            # Calculate numerator: sum_j w_j * P_{i,j}
            numer = np.dot(self.oob_predictions_, w)
            
            # Compute OOB predictions
            denom_clipped = denom + 1e-15
            oob_pred = numer / denom_clipped
            
            # Calculate Mean Squared Error
            mse = np.mean((y - oob_pred) ** 2)
            
            # Concave penalty to induce sparsity
            penalty = -self.lam * np.sum(w ** 2)
            
            return mse + penalty

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.0, 1.0) for _ in range(self.n_estimators_)]
        w0 = np.ones(self.n_estimators_) / self.n_estimators_
        
        res = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            tol=self.tol,
            options={'maxiter': self.max_iter}
        )
        
        return res.x

    def predict(self, X):
        """Predict regression target for X using only active estimators."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        
        n_samples = X.shape[0]
        preds = np.zeros(n_samples)
        
        for weight, estimator in zip(self.active_weights_, self.active_estimators_):
            preds += weight * estimator.predict(X)
            
        return preds
