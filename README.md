# Simplex-Constrained Sparse Bagging (SCSB)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

Simplex-Constrained Sparse Bagging (**SCSB**) is a mathematically rigorous post-training model compression and calibration framework for bagging ensembles. 

By transitioning from a naive **Uniform Prior** (where every base estimator has equal voting power, $w_i = 1/N$) to an optimal, **Sparse Posterior** distribution over the probability simplex ($w_i \ge 0, \sum w_i = 1$), SCSB significantly compresses ensemble size and improves probability calibration while preserving or enhancing generalization performance.

SCSB is **completely model-agnostic** and can be applied post-training to any bagging ensemble (including Random Forests, Bagged SVMs, Bagged Neural Networks, and Bagged KNNs) that utilizes bootstrap sampling.

---

## Key Features

* ⚙️ **Model-Agnostic Compression**: Works out-of-the-box with any standard `scikit-learn` bagging ensemble or custom bootstrap-based model list.
* 🎯 **Calibration Superiority**: Directly minimizes Out-of-Bag (OOB) Log-Loss (classification) or Mean Squared Error (regression) over the probability simplex, correcting ensemble over-confidence and lowering Expected Calibration Error (ECE).
* ⚡ **True Sparsity via Simplex Geometry**: Utilizes a concave quadratic penalty (negative $L_2$ norm) to force weights to the vertices of the simplex, yielding exact zero weights for non-essential estimators.
* 🚀 **Inference Acceleration**: The sparse inference engine bypasses zero-weighted estimators completely, yielding a linear speedup in prediction time directly proportional to the compression ratio.
* 🧩 **Scikit-Learn API**: Fully compatible with the `scikit-learn` estimator API (`fit`, `predict`, `predict_proba`, `score`).

---

## Mathematical Formulation

Let $I_{i,j}$ be the Out-Of-Bag (OOB) indicator variable for sample $i$ and base model $j$:
$$I_{i,j} = \begin{cases} 1 & \text{if sample } i \text{ is Out-of-Bag for model } j \\ 0 & \text{otherwise} \end{cases}$$

The weighted OOB prediction of the ensemble for sample $i$ is:
$$\hat{y}_i^{OOB}(w) = \frac{\sum_{j=1}^N w_j I_{i,j} f_j(x_i)}{\sum_{j=1}^N w_j I_{i,j}}$$

SCSB solves the following constrained optimization problem:
$$\min_{w} \frac{1}{M} \sum_{i=1}^M \text{Loss}\left(y_i, \hat{y}_i^{OOB}(w)\right) + \lambda R(w)$$

**Subject to the Simplex Constraints:**
1. $w_j \ge 0 \quad \forall j \in \{1, \dots, N\}$ (Non-negativity)
2. $\sum_{j=1}^N w_j = 1$ (Sum to one)

### Concave Sparsity-Inducing Penalty $R(w)$
Since the $L_1$ norm is constant (exactly $1$) on the probability simplex, standard Lasso-style regularization fails to induce sparsity. SCSB resolves this paradox by employing a **concave quadratic penalty** (negative $L_2$ norm) that drives weights to the boundaries of the simplex:
$$R(w) = -\|w\|_2^2 = -\sum_{j=1}^N w_j^2$$

---

## Quickstart

### Installation

Clone the repository and install using [uv](https://docs.astral.sh/uv/):
```bash
git clone https://github.com/your-username/simplex-constrained-sparse-bagging.git
cd simplex-constrained-sparse-bagging
uv venv && uv pip install -e .
```

Or with pip:
```bash
pip install -e .
```

### Classification Example

Here is a quick example showing how to calibrate and compress a standard `scikit-learn` ensemble using `SCSBClassifier`:

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scsb.models.ensemble import SCSBClassifier

# 1. Create a synthetic dataset
X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train a standard Bagging Classifier (e.g., 100 trees)
base_ensemble = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42,
    oob_score=True,
)
base_ensemble.fit(X_train, y_train)

# 3. Fit SCSB to find sparse optimal weights
scsb_model = SCSBClassifier(
    base_ensemble=base_ensemble,
    lam=0.05  # Regularization strength for concave penalty
)
scsb_model.fit(X_train, y_train)

# 4. Make fast, calibrated predictions
preds = scsb_model.predict(X_test)
probs = scsb_model.predict_proba(X_test)

# Check compression ratio
print(f"Original estimators: {len(base_ensemble.estimators_)}")
print(f"Active estimators after SCSB: {scsb_model.n_active_estimators_}")
print(f"Estimators pruned: {scsb_model.compression_ratio_:.2%}")
```

---

## Hypotheses and Research Focus

Our experimental pipeline is designed to validate three core hypotheses:
1. **H1 (Posterior Collapse)**: The combination of the concave penalty and the simplex boundary geometry forces a majority of base model weights to exactly zero, compressing model size with minimal loss in accuracy.
2. **H2 (Calibration Superiority)**: Minimizing Log-Loss over OOB predictions corrects the over-confidence of standard bagging models, leading to significantly lower Expected Calibration Error (ECE).
3. **H3 (Inference Acceleration)**: Bypassing zero-weighted estimators yields linear speedups in inference latency proportional to the fraction of active estimators.

---

## Project Structure

```text
simplex-constrained-sparse-bagging/
│
├── scsb/                     # Main source library
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       └── ensemble.py       # SCSBClassifier & SCSBRegressor
│
├── tests/                    # Test suite (pytest)
│   └── test_ensemble.py
│
├── benchmarks/               # Benchmark configurations (Phase 2)
├── experiments/              # Experiment scripts (Phase 2)
├── paper/                    # Paper drafting files (Markdown / LaTeX)
│
├── pyproject.toml            # Package configuration and dependencies
├── LICENSE                   # MIT License
├── README.md                 # This file
└── .gitignore
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
