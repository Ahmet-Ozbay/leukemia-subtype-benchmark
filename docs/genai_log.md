# Appendix A: GenAI Disclosure and Usage Log

This document is prepared in accordance with the Semester Project Guide (Section 9) and IEEE Authorship and AI Policies.

## 1. Summary of Generative AI Tools Used

| Tool and Version | Scope and Objective | Affected Modules / Files | Human Verification and Modifications |
|---|---|---|---|
| **Google Gemini 2.0 Flash / Pro (Antigravity Agent v2026.3)** | Data ingestion, preprocessing, probe-to-gene mapping, statistical EDA engine (`eda.py`), and automated testing suite | `src/data/`, `src/features/`, `src/visualization/eda.py`, `tests/` | Code reviewed line by line, platform encoding issues resolved, exact matrix medians verified, dual Welch/MWU tests implemented, GSE51082 RMA log2 adaptive preprocessing added, and verified using 54 automated pytest tests (49 passed, 5 skipped). |
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

### Session 4: External Validation Cohort Replacement (GSE51082 Integration)
- **User Prompt Summary:** Integrate GSE51082 (Agarwal et al. 2014, GPL96 / HG-U133A) as the primary similar dataset for ALL vs AML classification to replace the AML-deficient GSE28497 cohort. Analyze tissue confounding, handle RMA log2 normalization differences without linear clipping, organize dual outputs (66-sample modeling subset vs. 139-sample 6-class exploratory cohort), collapse to HUGO genes, and update tests.
- **Model Output Summary:** Implemented `GSE51082Processor` (`src/data/processors/gse51082.py`), updated download pipeline (`download_gse51082`), adaptive normalization in `BaseDatasetProcessor`, GPL96 annotation extraction and cross-platform harmonizer, updated `eda.py` and `style.py`, and expanded unit test suite.
- **Student Verification:** Verified RMA log2 scaling invariant ([0.0, 25.0]), confirmed balanced tissue distribution (86.2% BM in ALL vs. 81.1% BM in AML, Fisher exact p=0.744), generated 5,426 shared genes with 100% biomarker preservation, and verified all 54 pytest test cases (49 passed, 5 skipped, 0 failed).

### Session 5: Roadmap Overhaul, Literature Citation Remediation & Dual Manuscript 6-Page Finalization
- **User Prompt Summary:** Update `ROADMAP.md` to reflect architectural changes; maintain scientific direction investigating feature selection methods (ANOVA $F$-test vs. Mutual Information vs. Monte Carlo random subsets) across sample-to-feature ratio bounds ($k \in \{3, 6, 15, 50\}$); integrate fundamental analysis of MILE study (GSE13159; $n=2{,}096$) for Bonus B4; purge obsolete Coustan-Smith numbers and text; incorporate foundational methodology citations (Ambroise & McLachlan 2002, Varma & Simon 2006, Cawley & Talbot 2010, Dudoit et al. 2002, Arber et al. 2016); soften literature critiques on data leakage; fix Table I entries (Gillariose review status, Golub 29/34 prediction clarity); expand EDA visualization engine with tissue confounding diagnostics and 6-class global landscape; and compile English and Turkish IEEE manuscripts to strictly 6 pages.
- **Model Output Summary:** Updated `ROADMAP.md`, expanded `src/visualization/eda.py` with comprehensive 4-panel diagnostic Figure 4 (train/test shift, cross-platform PCA, tissue confounding, and 6-class global landscape) and standalone Figure 5, updated `tests/test_eda.py`, updated `references.bib` with 21 verified citations, and fully rewrote `paper/midterm/en/main.tex` and `paper/midterm/tr/main.tex`.
- **Student Verification:** Tested all figures at 300 DPI, verified 49 unit tests passing (0 failures), compiled both English and Turkish PDF manuscripts with 0 errors, 0 undefined references, and verified strict 6-page limits via `pdftotext`.

## 3. Originality and Ethics Statement
The experimental hypotheses, methodology design, research questions, code verification, and final paper conclusions remain the intellectual and scientific responsibility of the author.
