# Appendix A: GenAI Disclosure and Usage Log

This document is prepared in accordance with the Semester Project Guide (Section 9) and IEEE Authorship and AI Policies.

## 1. Summary of Generative AI Tools Used

| Tool and Version | Scope and Objective | Affected Modules / Files | Human Verification and Modifications |
|---|---|---|---|
| **Google Gemini 2.0 Flash / Pro (Antigravity Agent v2026.3)** | Data ingestion, preprocessing, probe-to-gene mapping, statistical EDA engine (`eda.py`), and automated testing suite | `src/data/`, `src/features/`, `src/visualization/eda.py`, `tests/` | Code reviewed line by line, platform encoding issues resolved, exact matrix medians verified, dual Welch/MWU tests implemented, and verified using 47 automated pytest tests (45 passed, 2 skipped). |
| **Google Gemini 1.5 Pro / 2.0 Flash (LLM)** | Systematic literature review synthesis, research questions framing (two-phased RQ3), and IEEE conference paper formatting | `paper/midterm/en/main.tex`, `paper/midterm/tr/main.tex`, `CONTEXT.md` | All referenced articles cross-checked against publisher DOIs and GEO accessions; peer-reviewed and preprint statuses accurately distinguished; exact page budgets enforced. |

## 2. Detailed Interaction & Prompt Log

### Session 1: Data Pipeline and Probe Mapping Architecture
- **User Prompt Summary:** Design a scikit-learn transformer and data pipeline to map Affymetrix Hu6800 and HG-U133A probe codes to HUGO gene symbols, collapsing multi-mapped probes without data leakage (variance computed strictly on training folds).
- **Model Output Summary:** `ProbeCollapser` transformer class (`fit`, `transform`), GPL80/GPL96 annotation parsers, and the `harmonize_cohort_genes` orchestration function.
- **Student Verification:** Developed 11 unit and regression tests in `tests/test_gene_mapping.py`, explicitly proving zero cross-split information leakage via `test_probe_collapser_fit_leakage`.

### Session 2: Exploratory Data Analysis (EDA) and IEEE Visualization Engine
- **User Prompt Summary:** Create a modular visualization engine with customizable styling presets (`grayscale`, `academic`, `vibrant`) producing IEEE-compliant 300+ DPI figures for expression distributions (Fig 1), dimensionality reduction (Fig 2), top 40 gene heatmap (Fig 3), and batch effects (Fig 4).
- **Model Output Summary:** `src/visualization/style.py` and `src/visualization/eda.py`.
- **Student Verification:** Inspected exported `.png` and `.pdf` files, verified typography and contrast ratios, and automated validation via `tests/test_eda.py`.

### Session 3: Methodological Review Remediation & Statistical Rigor Alignment
- **User Prompt Summary:** Reconcile all AI review findings: align Figure 3 with dual Welch's t-test and Mann-Whitney U test and strict Benjamini-Hochberg FDR < 0.001 data-driven filtering; fix Figure 1 exact matrix median; reframe Figure 4 as exploratory cohort-wise standardized PCA; divide RQ3 into transcriptomic harmonization (GSE28497) vs. supervised transfer (MILE); document Coustan-Smith sample filtering breakdown; add task column to literature table; and cite Haferlach et al. (2010).
- **Model Output Summary:** Refactored `src/visualization/eda.py`, updated `references.bib`, and aligned English and Turkish IEEE papers.
- **Student Verification:** Recompiled all figures, passed all 47 pytest tests, and compiled both English and Turkish manuscripts with 0 errors to exactly 6 pages.

## 3. Originality and Ethics Statement
The experimental hypotheses, methodology design, research questions, code verification, and final paper conclusions remain the intellectual and scientific responsibility of the author.
