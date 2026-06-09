"""Benchmark runner to compare SCSB, standard bagging, Lasso-pruned bagging, and XGBoost.

This script runs classification and regression benchmarks, computes metrics,
generates reliability diagrams, and saves the results.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

# Local imports
from benchmarks.datasets import load_classification_datasets, load_regression_datasets
from scsb.models.ensemble import SCSBClassifier, SCSBRegressor
from experiments.baselines import LassoPrunedClassifier, LassoPrunedRegressor

# Create output directories
os.makedirs("experiments/plots", exist_ok=True)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if y_prob.ndim == 1:
        probs = y_prob
        preds = (probs >= 0.5).astype(int)
    elif y_prob.shape[1] == 2:
        probs = y_prob[:, 1]
        preds = (probs >= 0.5).astype(int)
    else:
        probs = np.max(y_prob, axis=1)
        preds = np.argmax(y_prob, axis=1)
        
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        count = np.sum(in_bin)
        
        if count > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == preds[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += (count / total_samples) * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece


def plot_reliability_diagram(y_true, y_prob, title, save_path, n_bins=10):
    """Plot and save a reliability diagram."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if y_prob.ndim == 1:
        probs = y_prob
        preds = (probs >= 0.5).astype(int)
    elif y_prob.shape[1] == 2:
        probs = y_prob[:, 1]
        preds = (probs >= 0.5).astype(int)
    else:
        probs = np.max(y_prob, axis=1)
        preds = np.argmax(y_prob, axis=1)
        
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        count = np.sum(in_bin)
        bin_counts.append(count)
        
        if count > 0:
            accuracy = np.mean(y_true[in_bin] == preds[in_bin])
            confidence = np.mean(probs[in_bin])
            bin_accuracies.append(accuracy)
            bin_confidences.append(confidence)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
            
    plt.figure(figsize=(6, 6))
    plt.bar(bin_centers, bin_accuracies, width=1.0/n_bins, edgecolor='black', color='#1f77b4', alpha=0.7, label='Outputs')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    
    # Draw gaps/gaps lines
    for center, acc, conf in zip(bin_centers, bin_accuracies, bin_confidences):
        if acc > 0 or conf > 0:
            plt.plot([center, center], [acc, conf], color='red', linestyle='-', alpha=0.5)
            
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f"{title}\nECE: {ece:.4f}")
    plt.legend(loc='upper left')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_classification_benchmarks(datasets):
    """Run all classification benchmarks."""
    results = []
    
    for name, (X, y, task_type) in datasets.items():
        print(f"\n==================== Classification: {name} ====================")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # We test two types of base estimators: DecisionTree and LogisticRegression
        base_configs = [
            ("DecisionTree", BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, bootstrap=True, random_state=42)),
            ("LogisticRegression", BaggingClassifier(estimator=LogisticRegression(max_iter=1000), n_estimators=50, bootstrap=True, random_state=42))
        ]
        
        for base_name, base_ensemble in base_configs:
            print(f"\n--- Base Ensemble: {base_name} ---")
            
            # 1. Standard Bagging (Uniform Weights)
            print("Fitting Standard Bagging...")
            t0 = time.perf_counter()
            base_ensemble.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            probs_uniform = base_ensemble.predict_proba(X_test)
            preds_uniform = base_ensemble.predict(X_test)
            inference_time_uniform = (time.perf_counter() - t0) * 1000.0  # ms
            
            acc_uniform = accuracy_score(y_test, preds_uniform)
            loss_uniform = log_loss(y_test, probs_uniform)
            ece_uniform = expected_calibration_error(y_test, probs_uniform)
            
            results.append({
                "Dataset": name,
                "Base Estimator": base_name,
                "Model": "Standard Bagging",
                "Accuracy": acc_uniform,
                "Log-Loss": loss_uniform,
                "ECE": ece_uniform,
                "Active Estimators": len(base_ensemble.estimators_),
                "Compression Ratio": 0.0,
                "Train Time (s)": train_time,
                "Inference Time per 1000 (ms)": (inference_time_uniform / len(X_test)) * 1000.0
            })
            
            plot_reliability_diagram(
                y_test, probs_uniform, 
                f"Standard Bagging ({base_name}) on {name}", 
                f"experiments/plots/clf_{name}_{base_name}_uniform.png"
            )
            
            # 2. Lasso-Pruned Bagging
            print("Fitting Lasso-Pruned Bagging...")
            lasso_model = LassoPrunedClassifier(base_ensemble=base_ensemble, max_iter=2000)
            t0 = time.perf_counter()
            lasso_model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            probs_lasso = lasso_model.predict_proba(X_test)
            preds_lasso = lasso_model.predict(X_test)
            inference_time_lasso = (time.perf_counter() - t0) * 1000.0  # ms
            
            acc_lasso = accuracy_score(y_test, preds_lasso)
            loss_lasso = log_loss(y_test, probs_lasso)
            ece_lasso = expected_calibration_error(y_test, probs_lasso)
            
            results.append({
                "Dataset": name,
                "Base Estimator": base_name,
                "Model": "Lasso-Pruned Bagging",
                "Accuracy": acc_lasso,
                "Log-Loss": loss_lasso,
                "ECE": ece_lasso,
                "Active Estimators": int(lasso_model.n_active_estimators_),
                "Compression Ratio": float(lasso_model.compression_ratio_),
                "Train Time (s)": train_time,
                "Inference Time per 1000 (ms)": (inference_time_lasso / len(X_test)) * 1000.0
            })
            
            plot_reliability_diagram(
                y_test, probs_lasso, 
                f"Lasso-Pruned ({base_name}) on {name}", 
                f"experiments/plots/clf_{name}_{base_name}_lasso.png"
            )
            
            # 3. SCSB
            print("Fitting SCSB...")
            scsb_model = SCSBClassifier(base_ensemble=base_ensemble, lam=0.05, max_iter=100)
            t0 = time.perf_counter()
            scsb_model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            probs_scsb = scsb_model.predict_proba(X_test)
            preds_scsb = scsb_model.predict(X_test)
            inference_time_scsb = (time.perf_counter() - t0) * 1000.0  # ms
            
            acc_scsb = accuracy_score(y_test, preds_scsb)
            loss_scsb = log_loss(y_test, probs_scsb)
            ece_scsb = expected_calibration_error(y_test, probs_scsb)
            
            results.append({
                "Dataset": name,
                "Base Estimator": base_name,
                "Model": "SCSB",
                "Accuracy": acc_scsb,
                "Log-Loss": loss_scsb,
                "ECE": ece_scsb,
                "Active Estimators": int(scsb_model.n_active_estimators_),
                "Compression Ratio": float(scsb_model.compression_ratio_),
                "Train Time (s)": train_time,
                "Inference Time per 1000 (ms)": (inference_time_scsb / len(X_test)) * 1000.0
            })
            
            plot_reliability_diagram(
                y_test, probs_scsb, 
                f"SCSB ({base_name}) on {name}", 
                f"experiments/plots/clf_{name}_{base_name}_scsb.png"
            )
            
        # 4. XGBoost (standalone baseline)
        print("\nFitting XGBoost...")
        # Map target label types for multiclass or binary
        xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        t0 = time.perf_counter()
        xgb_model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        probs_xgb = xgb_model.predict_proba(X_test)
        preds_xgb = xgb_model.predict(X_test)
        inference_time_xgb = (time.perf_counter() - t0) * 1000.0  # ms
        
        acc_xgb = accuracy_score(y_test, preds_xgb)
        loss_xgb = log_loss(y_test, probs_xgb)
        ece_xgb = expected_calibration_error(y_test, probs_xgb)
        
        results.append({
            "Dataset": name,
            "Base Estimator": "N/A",
            "Model": "XGBoost",
            "Accuracy": acc_xgb,
            "Log-Loss": loss_xgb,
            "ECE": ece_xgb,
            "Active Estimators": 100,
            "Compression Ratio": 0.0,
            "Train Time (s)": train_time,
            "Inference Time per 1000 (ms)": (inference_time_xgb / len(X_test)) * 1000.0
        })
        
        plot_reliability_diagram(
            y_test, probs_xgb, 
            f"XGBoost on {name}", 
            f"experiments/plots/clf_{name}_xgb.png"
        )
        
    return results


def run_regression_benchmarks(datasets):
    """Run all regression benchmarks."""
    results = []
    
    for name, (X, y) in datasets.items():
        print(f"\n==================== Regression: {name} ====================")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # We test two types of base estimators: DecisionTree and Ridge
        base_configs = [
            ("DecisionTree", BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=100, bootstrap=True, random_state=42)),
            ("Ridge", BaggingRegressor(estimator=Ridge(), n_estimators=50, bootstrap=True, random_state=42))
        ]
        
        for base_name, base_ensemble in base_configs:
            print(f"\n--- Base Ensemble: {base_name} ---")
            
            # 1. Standard Bagging (Uniform Weights)
            print("Fitting Standard Bagging...")
            t0 = time.perf_counter()
            base_ensemble.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            preds_uniform = base_ensemble.predict(X_test)
            inference_time_uniform = (time.perf_counter() - t0) * 1000.0  # ms
            
            mse_uniform = mean_squared_error(y_test, preds_uniform)
            r2_uniform = r2_score(y_test, preds_uniform)
            
            results.append({
                "Dataset": name,
                "Base Estimator": base_name,
                "Model": "Standard Bagging",
                "MSE": mse_uniform,
                "R2": r2_uniform,
                "Active Estimators": len(base_ensemble.estimators_),
                "Compression Ratio": 0.0,
                "Train Time (s)": train_time,
                "Inference Time per 1000 (ms)": (inference_time_uniform / len(X_test)) * 1000.0
            })
            
            # 2. Lasso-Pruned Bagging
            print("Fitting Lasso-Pruned Bagging...")
            lasso_model = LassoPrunedRegressor(base_ensemble=base_ensemble, max_iter=2000)
            t0 = time.perf_counter()
            lasso_model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            preds_lasso = lasso_model.predict(X_test)
            inference_time_lasso = (time.perf_counter() - t0) * 1000.0  # ms
            
            mse_lasso = mean_squared_error(y_test, preds_lasso)
            r2_lasso = r2_score(y_test, preds_lasso)
            
            results.append({
                "Dataset": name,
                "Base Estimator": base_name,
                "Model": "Lasso-Pruned Bagging",
                "MSE": mse_lasso,
                "R2": r2_lasso,
                "Active Estimators": int(lasso_model.n_active_estimators_),
                "Compression Ratio": float(lasso_model.compression_ratio_),
                "Train Time (s)": train_time,
                "Inference Time per 1000 (ms)": (inference_time_lasso / len(X_test)) * 1000.0
            })
            
            # 3. SCSB
            print("Fitting SCSB...")
            scsb_model = SCSBRegressor(base_ensemble=base_ensemble, lam=0.05, max_iter=100)
            t0 = time.perf_counter()
            scsb_model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            preds_scsb = scsb_model.predict(X_test)
            inference_time_scsb = (time.perf_counter() - t0) * 1000.0  # ms
            
            mse_scsb = mean_squared_error(y_test, preds_scsb)
            r2_scsb = r2_score(y_test, preds_scsb)
            
            results.append({
                "Dataset": name,
                "Base Estimator": base_name,
                "Model": "SCSB",
                "MSE": mse_scsb,
                "R2": r2_scsb,
                "Active Estimators": int(scsb_model.n_active_estimators_),
                "Compression Ratio": float(scsb_model.compression_ratio_),
                "Train Time (s)": train_time,
                "Inference Time per 1000 (ms)": (inference_time_scsb / len(X_test)) * 1000.0
            })
            
        # 4. XGBoost (standalone baseline)
        print("\nFitting XGBoost...")
        xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        t0 = time.perf_counter()
        xgb_model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        preds_xgb = xgb_model.predict(X_test)
        inference_time_xgb = (time.perf_counter() - t0) * 1000.0  # ms
        
        mse_xgb = mean_squared_error(y_test, preds_xgb)
        r2_xgb = r2_score(y_test, preds_xgb)
        
        results.append({
            "Dataset": name,
            "Base Estimator": "N/A",
            "Model": "XGBoost",
            "MSE": mse_xgb,
            "R2": r2_xgb,
            "Active Estimators": 100,
            "Compression Ratio": 0.0,
            "Train Time (s)": train_time,
            "Inference Time per 1000 (ms)": (inference_time_xgb / len(X_test)) * 1000.0
        })
        
    return results


def main():
    print("Starting Experimental Pipeline Benchmarks...")
    
    # Load Datasets
    clf_datasets = load_classification_datasets()
    reg_datasets = load_regression_datasets()
    
    # Run Classification
    clf_results = run_classification_benchmarks(clf_datasets)
    df_clf = pd.DataFrame(clf_results)
    
    # Run Regression
    reg_results = run_regression_benchmarks(reg_datasets)
    df_reg = pd.DataFrame(reg_results)
    
    # Display Results
    print("\n\n==================== FINAL CLASSIFICATION RESULTS ====================")
    print(df_clf.to_markdown(index=False))
    
    print("\n\n==================== FINAL REGRESSION RESULTS ====================")
    print(df_reg.to_markdown(index=False))
    
    # Save Results to JSON/CSV
    df_clf.to_csv("experiments/classification_results.csv", index=False)
    df_reg.to_csv("experiments/regression_results.csv", index=False)
    
    raw_results = {
        "classification": clf_results,
        "regression": reg_results
    }
    with open("experiments/benchmark_results.json", "w") as f:
        json.dump(raw_results, f, indent=4)
        
    print("\nBenchmarks completed successfully. Results saved to experiments/")


if __name__ == "__main__":
    main()
