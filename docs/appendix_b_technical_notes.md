# Appendix B: Technical Explanations ("In My Own Words")

This document fulfills Section 9.2 of the Semester Project Guide, providing concise technical explanations of model principles, leakage prevention strategies, and unexpected findings.

---

## 1. Selected Machine Learning Methods & Critical Hyperparameters

### A. $L_1$-Penalized Logistic Regression (Lasso / ElasticNet)
- **Principle:** Adds the sum of absolute coefficient values ($L_1$ norm) as a regularization penalty to the log-loss objective function. Geometrically, the $L_1$ constraint ball has sharp corners; as loss contours expand, they intersect these vertices, driving the coefficients of uninformative features **strictly to zero**. Consequently, the model functions simultaneously as an embedded sparse biomarker selector.
- **Critical Hyperparameter 1 ($C$ — Inverse Regularization Strength):**
  - *Bias-Variance Impact:* Small $C$ (strong penalty) heavily constrains model capacity, zeroing out non-essential gene weights, increasing bias but drastically curbing variance in $p \gg n$ regimes. Large $C$ decreases bias but leads to severe memorization of noise.
- **Critical Hyperparameter 2 (`l1_ratio` / Solver Selection):**
  - *Bias-Variance Impact:* In ElasticNet mode, determines the trade-off between $L_1$ and $L_2$ penalties. While pure Lasso selects arbitrarily from collinear gene groups, ElasticNet encourages grouped selection, stabilizing variance across correlated transcriptomic networks.

### B. Support Vector Machines (Linear and RBF Kernels)
- **Principle:** Identifies the optimal decision hyperplane that maximizes the geometric margin between classes (ALL and AML). The Radial Basis Function (RBF) kernel projects input samples into an infinite-dimensional Hilbert feature space to resolve non-linear transcriptomic patterns. The decision function relies exclusively on boundary instances (support vectors).
- **Critical Hyperparameter 1 ($C$ — Soft Margin Penalty):**
  - *Bias-Variance Impact:* Low $C$ tolerates margin violations to enforce a wider, smoother margin (higher bias, lower variance, better generalization). High $C$ penalizes misclassifications aggressively, seeking a tighter separation (lower bias, higher variance, risk of overfitting).
- **Critical Hyperparameter 2 ($\gamma$ — Kernel Coefficient):**
  - *Bias-Variance Impact:* Small $\gamma$ broadens the Gaussian influence of each support vector (smoother decision boundaries, lower variance). High $\gamma$ creates localized decision boundaries around individual training points, producing extreme variance.

### C. Random Forest
- **Principle:** An ensemble of de-correlated decision trees combining bootstrap aggregating (bagging) and the random subspace method. Each tree is trained on a distinct bootstrap sample, and at each split, the best feature is chosen from a random subset of $\sqrt{p}$ candidate genes.
- **Critical Hyperparameter 1 (`max_depth` or `min_samples_leaf`):**
  - *Bias-Variance Impact:* Restricting tree depth introduces slight bias to individual trees but significantly dampens ensemble variance, preventing the trees from over-indexing on rare transcriptomic outliers.
- **Critical Hyperparameter 2 (`n_estimators` — Number of Trees):**
  - *Bias-Variance Impact:* Increasing the ensemble size does not alter asymptotic bias, but reduces variance monotonically by averaging uncorrelated predictions, converging stably without overfitting.

---

## 2. Preventing Data Leakage: Code References

1. **Patient Identifier Alignment vs. Column Ordering:**
   - [`src/data/processors/golub.py`](file:///c:/Users/Ozbay/Desktop/leukemia-subtype-benchmark/src/data/processors/golub.py): Raw CSV files list patient columns out of numerical sequence. Labels are resolved strictly via `sample_id` from `actual.csv`. Disjoint sample splits ($n=38$ train vs. $n=34$ test) are verified in [`tests/test_make_dataset.py`](file:///c:/Users/Ozbay/Desktop/leukemia-subtype-benchmark/tests/test_make_dataset.py#L110).
2. **Leak-Free Probe Collapsing (`ProbeCollapser`):**
   - [`src/features/collapser.py`](file:///c:/Users/Ozbay/Desktop/leukemia-subtype-benchmark/src/features/collapser.py#L70): Selecting the most informative probe targeting a HUGO gene based on variance is executed **strictly on training fold samples**. `ProbeCollapser.fit()` computes probe variances solely on `X_train` and freezes selected probe indices. `transform()` maps the test fold blindly using those frozen indices. Verified in [`tests/test_gene_mapping.py`](file:///c:/Users/Ozbay/Desktop/leukemia-subtype-benchmark/tests/test_gene_mapping.py#L220).
3. **Encapsulated Preprocessing & Feature Selection:**
   - Preprocessing steps (`StandardScaler`, `SelectKBest`, SMOTE) are encapsulated within `imblearn.pipeline.Pipeline` objects, ensuring the outer test fold is never exposed to fold-internal statistics during nested cross-validation.

---

## 3. Unexpected Findings and Biological Insights

- **Observation (PCA vs. Non-Linear Manifold Separation):**
  In the exploratory data analysis (Figure 2), the first two linear principal components (PC1: 11.9%, PC2: 8.6%) exhibit substantial spatial overlap between ALL and AML samples. Conversely, unsupervised non-linear manifold techniques (t-SNE and UMAP) partition the two leukemia subtypes into distinct clusters without any access to class labels.
- **Interpretation:**
  In a high-dimensional space of $p = 7{,}129$ features, the majority of global variance arises from uncoordinated baseline transcription, metabolic background, and experimental noise rather than subtype-specific pathology. Because PCA captures directions of maximum total variance, it is easily dominated by this non-specific variance. In contrast, t-SNE and UMAP preserve local neighborhood affinities, effectively isolating subtle non-linear transcriptomic signatures from background noise.

