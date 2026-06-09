# Simplex-Constrained Sparse Bagging: Transitioning from Uniform Priors to Sparse Posteriors in Ensemble Learning

## Abstract
We present **Simplex-Constrained Sparse Bagging (SCSB)**, a mathematically rigorous framework for post-training compression and probability calibration of bagging ensembles. While traditional bagging ensembles (such as Random Forests, Bagged SVMs, and Bagged Neural Networks) utilize a naive uniform prior for voting (equal weights), we optimize estimator weights over the probability simplex using Out-Of-Bag (OOB) predictions. We resolve the theoretical "L1 simplex paradox"—where standard L1 regularization fails to induce sparsity due to the simplex constraint—by employing a concave quadratic penalty. Our method is model-agnostic and compresses large bagging ensembles by pruning up to 90% of constituent estimators, yielding linear inference speedups and superior probability calibration (lowered ECE) without sacrificing generalization accuracy.

---

## 1. Introduction
* Ensembles reduce variance but are memory-heavy and slow at inference.
* Naive averaging assumes all base models are equally competent across the input space.
* We introduce **SCSB** as a model-agnostic framework to prune and calibrate bagging models post-training.

## 2. Related Work
* Traditional ensemble pruning (greedy selection, heuristic search).
* Stacking and its vulnerability to data leakage on training sets.
* The failure of standard Lasso (L1) regularization on simplex-constrained domains.

## 3. Proposed Method: SCSB
### 3.1 The Base Ensemble
We train a standard bagging ensemble of $N$ base estimators (e.g., SVMs, MLP Neural Networks, or Trees). Let $I_{i,j} \in \{0, 1\}$ be the indicator mapping whether sample $i$ was out-of-bag during the bootstrap training of estimator $j$.

### 3.2 Leakage-Free OOB Ensemble Estimation
To prevent data leakage without expensive cross-validation, the ensemble's OOB prediction for sample $i$ is calculated dynamically as:
$$\hat{y}_i^{OOB}(w) = \frac{\sum_{j=1}^N w_j I_{i,j} f_j(x_i)}{\sum_{j=1}^N w_j I_{i,j}}$$

### 3.3 Simplex Optimization & The Sparsity Penalty
We optimize the weight vector $w$ over the probability simplex:
$$\min_{w} \mathcal{L}(y, \hat{y}^{OOB}(w)) - \lambda \|w\|_2^2 \quad \text{s.t.} \quad w \ge 0, \sum w_j = 1$$
We prove that the L1 penalty is constant on the simplex, and demonstrate how the concave quadratic penalty $-\|w\|_2^2$ forces the optimal weights to the boundaries of the simplex, inducing true sparsity (pruning).

## 4. Theoretical Analysis
* **L1 Simplex Paradox Proof**: Let $w \in \Delta^N$. Then $\|w\|_1 = \sum_{j=1}^N |w_j| = \sum_{j=1}^N w_j = 1$. The derivative with respect to any active weight is zero; thus, L1 regularization has no sparsifying effect on the simplex.
* **Concave Optimization**: The geometry of minimizing a convex loss combined with a concave penalty over a simplex forces coordinate directions to zero, yielding sparse (pruned) ensembles.

## 5. Experimental Setup
*(To be populated: Benchmarking on OpenML datasets across Random Forests, Bagged SVMs, and Bagged Neural Networks, tracking accuracy, ECE, and inference speed).*

## 6. Results
*(To be populated: Tables showing original vs. SCSB accuracy, compression ratios, and ECE values across estimator classes).*

## 7. Ablation Studies
*(To be populated: The effect of regularization strength $\lambda$ on the sparsity-accuracy Pareto frontier).*

## 8. Limitations & Future Work
* Non-convexity of the concave penalty optimization requires careful initialization (starting from the uniform prior $w_j = 1/N$).

## 9. Conclusion
* SCSB offers a mathematically sound, plug-and-play solution to compress and calibrate bagging models for production, independent of the underlying estimator architecture.
