"""Baseline estimators for comparison with SCSB.

This module implements naive Lasso-pruned bagging classifiers and regressors.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MetaEstimatorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.utils.multiclass import check_classification_targets

class LassoPrunedRegressor(BaseEstimator, RegressorMixin, MetaEstimatorMixin):
    """Lasso-pruned Bagging Regressor baseline.
    
    Fits a LassoCV model on Out-of-Bag (OOB) predictions of base estimators
    to find a sparse combination of estimators.
    """
    def __init__(self, base_ensemble, cv=5, max_iter=1000):
        self.base_ensemble = base_ensemble
        self.cv = cv
        self.max_iter = max_iter
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        # Fit base ensemble if needed
        try:
            check_is_fitted(self.base_ensemble)
        except Exception:
            self.base_ensemble.fit(X, y)
            
        self.estimators_ = self.base_ensemble.estimators_
        self.n_estimators_ = len(self.estimators_)
        
        n_samples = X.shape[0]
        # Extract OOB indicators
        oob_indicators = np.ones((n_samples, self.n_estimators_), dtype=bool)
        for j, samples in enumerate(self.base_ensemble.estimators_samples_):
            in_bag_mask = np.zeros(n_samples, dtype=bool)
            in_bag_mask[samples] = True
            oob_indicators[:, j] = ~in_bag_mask
            
        # Get OOB predictions: shape (n_samples, n_estimators)
        oob_predictions = np.zeros((n_samples, self.n_estimators_))
        for j, estimator in enumerate(self.estimators_):
            oob_idx = np.where(oob_indicators[:, j])[0]
            if len(oob_idx) > 0:
                oob_predictions[oob_idx, j] = estimator.predict(X[oob_idx])
                
        # Fit LassoCV with positive=True to force non-negative weights
        self.lasso_ = LassoCV(cv=self.cv, max_iter=self.max_iter, positive=True)
        self.lasso_.fit(oob_predictions, y)
        
        self.weights_ = self.lasso_.coef_
        sum_weights = np.sum(self.weights_)
        if sum_weights > 1e-5:
            self.weights_ = self.weights_ / sum_weights
        else:
            # Fallback to uniform weights if Lasso selects nothing
            self.weights_ = np.ones(self.n_estimators_) / self.n_estimators_
            
        self.active_idx_ = np.where(self.weights_ > 1e-5)[0]
        self.active_weights_ = self.weights_[self.active_idx_]
        if len(self.active_weights_) > 0:
            self.active_weights_ /= np.sum(self.active_weights_)
        else:
            self.active_idx_ = np.array([0])
            self.active_weights_ = np.array([1.0])
            
        self.active_estimators_ = [self.estimators_[idx] for idx in self.active_idx_]
        self.n_active_estimators_ = len(self.active_estimators_)
        self.compression_ratio_ = 1.0 - (self.n_active_estimators_ / self.n_estimators_)
        
        return self
        
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        preds = np.zeros(X.shape[0])
        for weight, estimator in zip(self.active_weights_, self.active_estimators_):
            preds += weight * estimator.predict(X)
        return preds


class LassoPrunedClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """Lasso-pruned Bagging Classifier baseline.
    
    Fits a Logistic Regression with L1 penalty on the Out-of-Bag (OOB) probability
    predictions of base estimators to find a sparse combination.
    """
    def __init__(self, base_ensemble, cv=5, max_iter=1000, C=1.0):
        self.base_ensemble = base_ensemble
        self.cv = cv
        self.max_iter = max_iter
        self.C = C
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
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
        # Extract OOB indicators
        oob_indicators = np.ones((n_samples, self.n_estimators_), dtype=bool)
        for j, samples in enumerate(self.base_ensemble.estimators_samples_):
            in_bag_mask = np.zeros(n_samples, dtype=bool)
            in_bag_mask[samples] = True
            oob_indicators[:, j] = ~in_bag_mask
            
        # Get OOB probability predictions: shape (n_samples, n_estimators * n_classes)
        # We flatten the probabilities of all estimators to form features.
        oob_predictions = np.zeros((n_samples, self.n_estimators_ * self.n_classes_))
        for j, estimator in enumerate(self.estimators_):
            oob_idx = np.where(oob_indicators[:, j])[0]
            if len(oob_idx) > 0:
                probs = estimator.predict_proba(X[oob_idx])
                if probs.shape[1] < self.n_classes_:
                    mapped = np.zeros((len(oob_idx), self.n_classes_))
                    idx_map = [np.where(self.classes_ == c)[0][0] for c in estimator.classes_]
                    mapped[:, idx_map] = probs
                    probs = mapped
                
                # Fill features corresponding to estimator j
                start_col = j * self.n_classes_
                end_col = start_col + self.n_classes_
                oob_predictions[oob_idx, start_col:end_col] = probs
                
        # Fit L1-regularized Logistic Regression (LogisticL1)
        # Using saga solver for multiclass L1 support
        self.lr_ = LogisticRegression(
            penalty='l1',
            C=self.C,
            solver='saga',
            max_iter=self.max_iter,
            random_state=42
        )
        self.lr_.fit(oob_predictions, y)
        
        # Check coefficients to identify active estimators
        # coef_ shape is (1, n_features) for binary, or (n_classes, n_features) for multiclass
        coefs = np.abs(self.lr_.coef_)
        if coefs.ndim > 1 and coefs.shape[0] > 1:
            # For multiclass, sum coefficients across all target classes
            estimator_coefs = np.zeros(self.n_estimators_)
            for j in range(self.n_estimators_):
                start_col = j * self.n_classes_
                end_col = start_col + self.n_classes_
                estimator_coefs[j] = np.sum(coefs[:, start_col:end_col])
        else:
            # Binary classification (coef_ shape is (1, n_features))
            coefs = coefs.ravel()
            estimator_coefs = np.zeros(self.n_estimators_)
            for j in range(self.n_estimators_):
                start_col = j * self.n_classes_
                end_col = start_col + self.n_classes_
                estimator_coefs[j] = np.sum(coefs[start_col:end_col])
                
        # Weights are proportional to the sum of absolute coefficients
        self.weights_ = estimator_coefs
        sum_weights = np.sum(self.weights_)
        if sum_weights > 1e-5:
            self.weights_ = self.weights_ / sum_weights
        else:
            self.weights_ = np.ones(self.n_estimators_) / self.n_estimators_
            
        self.active_idx_ = np.where(self.weights_ > 1e-5)[0]
        self.active_weights_ = self.weights_[self.active_idx_]
        if len(self.active_weights_) > 0:
            self.active_weights_ /= np.sum(self.active_weights_)
        else:
            self.active_idx_ = np.array([0])
            self.active_weights_ = np.array([1.0])
            
        self.active_estimators_ = [self.estimators_[idx] for idx in self.active_idx_]
        self.n_active_estimators_ = len(self.active_estimators_)
        self.compression_ratio_ = 1.0 - (self.n_active_estimators_ / self.n_estimators_)
        
        return self
        
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        probs = np.zeros((X.shape[0], self.n_classes_))
        for weight, estimator in zip(self.active_weights_, self.active_estimators_):
            est_probs = estimator.predict_proba(X)
            if est_probs.shape[1] < self.n_classes_:
                mapped = np.zeros((X.shape[0], self.n_classes_))
                idx_map = [np.where(self.classes_ == c)[0][0] for c in estimator.classes_]
                mapped[:, idx_map] = est_probs
                est_probs = mapped
            probs += weight * est_probs
        
        # Ensure probabilities are bounded in [0, 1] and sum to 1
        probs = np.clip(probs, 0.0, 1.0)
        sum_probs = np.sum(probs, axis=1, keepdims=True)
        sum_probs = np.where(sum_probs == 0, 1e-15, sum_probs)
        probs = probs / sum_probs
        return probs
        
    def predict(self, X):
        check_is_fitted(self)
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
