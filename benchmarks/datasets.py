"""Dataset loaders for classification and regression benchmarks.

This module provides standard loaders for real-world datasets from scikit-learn
and OpenML to evaluate the calibration and compression of SCSB.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_california_housing, fetch_openml
from sklearn.preprocessing import LabelEncoder

def load_classification_datasets():
    """Load standard classification datasets.

    Returns
    -------
    dict
        Dictionary of dataset name to tuple of (X, y, task_type), where task_type
        is 'binary' or 'multiclass'.
    """
    datasets = {}
    
    # 1. Breast Cancer (Binary, 569 samples, 30 features)
    print("Loading Breast Cancer dataset...")
    X, y = load_breast_cancer(return_X_y=True)
    datasets['breast_cancer'] = (X, y, 'binary')
    
    # 2. Diabetes (Binary, 768 samples, 8 features)
    print("Loading Diabetes dataset from OpenML...")
    try:
        data = fetch_openml(name='diabetes', version=1, as_frame=False, parser='auto')
        X, y = data.data, data.target
        # Encode target 'tested_positive'/'tested_negative' to 1/0
        if y.dtype.kind in {'O', 'U', 'S'}:
            y = (y == 'tested_positive').astype(int)
        else:
            y = y.astype(int)
        # Ensure any float/object columns are numeric
        X = X.astype(float)
        datasets['diabetes_clf'] = (X, y, 'binary')
    except Exception as e:
        print(f"Warning: Could not fetch diabetes from OpenML: {e}")
        
    # 3. Spambase (Binary, 4601 samples, 57 features)
    print("Loading Spambase dataset from OpenML...")
    try:
        data = fetch_openml(name='spambase', version=1, as_frame=False, parser='auto')
        X = data.data.astype(float)
        y = data.target.astype(int)
        datasets['spambase'] = (X, y, 'binary')
    except Exception as e:
        print(f"Warning: Could not fetch spambase from OpenML: {e}")

    # 4. Segment (Multiclass, 2310 samples, 19 features)
    print("Loading Segment dataset from OpenML...")
    try:
        data = fetch_openml(name='segment', version=1, as_frame=False, parser='auto')
        X = data.data.astype(float)
        y = data.target
        if y.dtype.kind in {'O', 'U', 'S', 'V'}:
            y = LabelEncoder().fit_transform(y)
        else:
            y = y.astype(int)
        datasets['segment'] = (X, y, 'multiclass')
    except Exception as e:
        print(f"Warning: Could not fetch segment from OpenML: {e}")
        
    return datasets

def load_regression_datasets():
    """Load standard regression datasets.

    Returns
    -------
    dict
        Dictionary of dataset name to tuple of (X, y).
    """
    datasets = {}
    
    # 1. Diabetes (Regression, 442 samples, 10 features)
    print("Loading Diabetes Regression dataset...")
    X, y = load_diabetes(return_X_y=True)
    datasets['diabetes_reg'] = (X, y)
    
    # 2. California Housing (Regression, 20640 samples, 8 features)
    print("Loading California Housing dataset...")
    # Use a subset of 5000 samples to keep benchmarks fast
    X, y = fetch_california_housing(return_X_y=True)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), size=5000, replace=False)
    datasets['california_housing'] = (X[indices], y[indices])
    
    # 3. CPU Activity / Act (Regression, 8192 samples, 21 features)
    print("Loading CPU Activity dataset from OpenML...")
    try:
        data = fetch_openml(name='cpu_act', version=1, as_frame=False, parser='auto')
        X = data.data.astype(float)
        y = data.target.astype(float)
        # Use a subset of 5000 samples to keep benchmarks fast
        if len(X) > 5000:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(X), size=5000, replace=False)
            X, y = X[indices], y[indices]
        datasets['cpu_act'] = (X, y)
    except Exception as e:
        print(f"Warning: Could not fetch cpu_act from OpenML: {e}")
        
    return datasets
