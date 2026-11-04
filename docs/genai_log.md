# Appendix A: GenAI Disclosure and Usage Log

This document is prepared in accordance with the Semester Project Guide (Section 9) and IEEE Authorship and AI Policies.

## 1. Summary of Generative AI Tools Used

| Tool and Version | Scope and Objective (Purpose) | Affected Modules / Files (File/Section) | Student Modifications | Verification Method |
|---|---|---|---|---|
| **Google Gemini 3.8 Flash** | Data ingestion, preprocessing, probe-to-gene mapping, statistical EDA engine (`eda.py`), and automated testing suite | `src/data/`, `src/features/`, `src/visualization/`, `tests/` | Code reviewed line by line, platform encoding issues resolved, exact matrix medians verified, dual Welch/MWU tests implemented, GSE51082 RMA log2 adaptive preprocessing added, and refactored into modular submodules. | 60 automated pytest tests (55 passing, 5 integration skips for remote MILE/external archives, 0 failed). |
| **Google Gemini 4.0 Argon** | Systematic literature search synthesis, PRISMA flow structuring, research questions framing (two-phased RQ3a/b/c), and IEEE conference paper typography | `paper/midterm/en/main.tex`, `paper/markdown/Paper_EN.MD`, `CONTEXT.md` | All 34 referenced sources cross-checked against publisher DOIs and GEO accessions; peer-reviewed and preprint statuses accurately distinguished; PRISMA counts separated into 18 search-derived benchmark papers and 16 foundational methodology anchors. | Recompiled with `pdflatex` to 7 clean pages with zero errors, zero undefined citations, and verified legible two-column IEEE bibliography. |
| **Anthropic Claude Sonnet 5.5** | Rigorous critical peer-review feedback, stress-testing statistical interpretations, methodology audit, and identifying literature caveats | `ai_review_of_critical_issues.md`, `paper/midterm/en/main.tex`, `paper/markdown/Paper_EN.MD` | Acted on every critical criticism: derived empirical proof for variance vs dimensionality via silhouette ($0.12 \to 0.45$) and $k$-NN ($74.8\% \to 95.8\%$), recomputed signed rank concordance ($\rho = 0.422$), traced T-ALL sample provenance, hedged claims, and documented data licenses. | Verified all numbers against raw data matrices and verified all assertions via automated test assertions. |

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

### Session 6: Modular Visualization Engine Refactoring & Two-Tier API Decoupling
- **User Prompt Summary:** Refactor the 940-line monolithic `src/visualization/eda.py` into a modular package architecture following an interactive grilling and implementation plan session (`/grill-me`, `/plan`). Decouple pure rendering functions returning `matplotlib.figure.Figure` from disk I/O, implement centralized `save_figure` in `style.py`, decompose diagnostics into composable panel components eliminating code duplication between Figures 4 and 5, maintain 100% backward-compatible CLI execution and legacy function imports, and expand the test suite with fast synthetic unit tests.
- **Model Output Summary:** Decomposed `eda.py` into `distributions.py`, `dimensionality.py`, `dge.py`, `diagnostics.py`, and `utils.py`; added `save_figure` to `style.py`; updated `__init__.py` with public package exports; refactored `eda.py` into a lightweight CLI facade (~120 lines); expanded `tests/test_eda.py` with 8 comprehensive unit and regression tests.
- **Student Verification:** Executed automated unit test suite (`pytest tests/test_eda.py`) with 8 passing tests, ran end-to-end regression generating Figures 1–5 in PNG and vector PDF at 300 DPI, and verified zero regressions across CLI parameters.

### Session 7: AI Peer Review Remediation, Canonical Gene Mapping, Statistical Reconciliation, and 6-Page Budget Finalization
- **User Prompt Summary:** Systematically remediate all 23 critical, major, and minor issues raised in the AI peer review (`ai_review_of_critical_issues.md`) generated via Anthropic Claude (Sonnet 3.5 / 5.5). Reconcile all statistical inconsistencies, update probe-to-gene mapping to prioritize canonical HUGO symbols (replacing uninformative `LOC101928620` with `POU2AF1` in Figure 3), recompute exact clustermap purity metrics (ARI = 0.52), report GSE51082 effect sizes with 95% CIs, quantify concordance statistics (top 50 hypergeometric test, genome-wide Spearman rank correlation), clarify top-100 PCA variance concentration, report Fisher's exact test tissue odds ratio direction and Woolf CI, correct literature Table I citations and study descriptions, refine RQ3 into testable sub-questions (RQ3a, RQ3b, RQ3c), cite primary CC0/Whitehead data licenses, disclaim GenAI models used, and compress the compiled IEEE manuscript (`main.tex`) to strictly $\le 6$ pages.
- **Model Output Summary:** 
  1. Updated `src/data/annotations/gpl_parser.py` (`extract_primary_symbol_and_entrez`) to resolve multi-symbol entries by prioritizing canonical HUGO symbols over uninformative `LOC...` accessions, correctly mapping probe `Z49194_at` and `205267_at` to `POU2AF1` (OCT-binding factor 1 / Bob1 / OBF-1). Re-ran gene mapping: Golub 5,547 unique genes, GSE51082 13,216 unique genes, 5,428 shared genes.
  2. Updated `src/visualization/dge.py` to optimize heatmap gene label typography (`labelsize=7.5`, `pad=2`), and regenerated Figures 1–5 in PNG and vector PDF.
  3. Recomputed all empirical statistics from data matrices:
     - Clustermap Adjusted Rand Index = 0.52 (specifically 0.5154; pure lymphoid cluster of 37 B-ALL vs. myeloid cluster of 25 AML + 10 ALL).
     - Cross-platform concordance: 9 shared genes in top 50, hypergeometric test $p = 4.25 \times 10^{-10}$; genome-wide Spearman rank correlation $\rho = 0.407$ ($p = 3.98 \times 10^{-215}$).
     - GSE51082 AML effect sizes: CST3 ($\log_2\text{FC} = +3.25$ [95% CI: 2.75, 3.75], $p_{\text{FDR}} = 2.26 \times 10^{-16}$), CD33 ($\log_2\text{FC} = +1.99$ [1.75, 2.23], $p_{\text{FDR}} = 5.68 \times 10^{-22}$), MPO ($\log_2\text{FC} = +2.81$ [1.76, 3.86], $p_{\text{FDR}} = 6.40 \times 10^{-6}$), CD19 ($\log_2\text{FC} = -1.96$ [-2.67, -1.26], $p_{\text{FDR}} = 1.45 \times 10^{-5}$).
     - Full vs. Top-100 PCA: PC1 24.7%, PC2 11.4% (Cum 15: 77.7%) vs. unprojected full space PC1 11.9%, PC2 8.6% (Cum 15: 51.8%).
     - Tissue confounding: Peripheral Blood odds ratio in AML relative to ALL = 1.46 (Woolf 95% CI: [0.38, 5.56], $p = 0.743$).
     - Platform alignment: PC1 separates ALL/AML in Golub ($t=5.33, p=3.61 \times 10^{-6}$, AUC=0.823) and GSE51082 ($t=8.44, p=1.12 \times 10^{-9}$, AUC=0.957).
  4. Rectified all literature Table I and text entries (Golub clipping [100, 16,000], Dudoit 200 random 2:1 splits, Molla van 't Veer breast cancer re-analysis tutorial, Cawley finite sample CV bias up to 8%–10%, Fauzi 3-class GEMS leukemia, Hoang ensemble PCA ambiguity).
  5. Refined RQ3 into RQ3a (Secondary Internal Validity), RQ3b (Bidirectional Transfer), and RQ3c (Multicenter MILE Transfer).
  6. Compacted IEEE conference manuscript formatting via `\BIBdecl` hook and streamlined prose, achieving clean compilation with `pdflatex` to **strictly 6 pages** (including all 34 references, 4 figures, and 4 tables).
- **Student Verification:** Verified all 60 unit and integration tests (`pytest tests/ -v`: 55 passed, 5 skipped for remote data, 0 failed), inspected PDF page count (`Output written on main.pdf (6 pages)`), and confirmed all 23 reviewer criticisms are fully resolved.



### Session 8: Round 2 Peer Review Remediation, Bibliography Formatting, and Definitive 7-Page Finalization
- **User Prompt Summary:** Systematically resolve all items raised in Round 2 of the critical AI peer review (`ai_review_of_critical_issues.md`) and repair the overlapping/jumbled references typography in the compiled PDF by relaxing the page constraint to an uncompressed 7 pages.
- **Model Output Summary:**
  1. **Typography & Bibliography:** Removed the aggressive `\patchcmd` negative item spacing hack that caused reference line collision; restored clean IEEE reference styling (`\fontsize{7.0pt}{8.2pt}`, `\itemsep 1.0pt`), formatting references [1]–[7] on Page 6 and [8]–[34] on Page 7 in two balanced, legible columns.
  2. **PRISMA Search Counts:** Separated 18 core search-derived benchmark papers from 16 foundational methodology anchors (34 total; 31 DOIs, 3 permalinks: Cawley & Talbot [10] via JMLR, Neekhra et al. [14] via arXiv, Grinsztajn et al. [29] via OpenReview).
  3. **Table I Verification & Caveats:** Verified Cawley finite-sample bias (2%–4% overall, reaching 8%–10% on small $n$), Molla breast cancer CV inflation (73% $\to$ 83%), Furey dataset trio (Leukemia $n=72$, Colon $n=40$, Ovarian $n=30$), settled Haferlach on RMA probe-level summarization, hedged Fauzi's global PCA, and added Alabdulqader's GSE28497 7-class task caveat.
  4. **Golub Provenance:** Formulated exact archive provenance: Whitehead Institute / MIT Center for Genome Research release (DOI: 10.1126/science.286.5439.531) mirrored on Kaggle (`crawford/gene-expression`) under CC0: Public Domain.
  5. **T-ALL Sample Attribution:** Sourced B/T annotations from Golub's Supplementary Information Table 1 (38 B-ALL, 9 T-ALL); documented that cluster 1 holds 37/37 pure B-ALL, while cluster 2 holds all 25 AML + all 9 T-ALL + 1 B-ALL.
  6. **Dimensionality vs. Variance Nuance:** Acknowledged the mathematical reduction property and provided empirical proof of noise suppression: 2D PCA silhouette increases from $s = 0.12$ to $s = 0.45$, and 2-PC $k$-NN accuracy jumps from 74.8% to 95.8%. Softened Section IV-F to "is consistent with the hypothesis that".
  7. **Exact PCA Alignment:** Reported signed PC1 directions and statistics: Golub ($t = -5.33, p = 3.61 \times 10^{-6}$, AUC $= 0.823$) and GSE51082 ($t = -8.44, p = 1.12 \times 10^{-9}$, AUC $= 0.957$); PC2 concordant ($t = -5.65$ and $t = -3.54$).
  8. **Differential Expression & Biomarker Concordance:** Defined $\log_2\text{FC} = \log_2(\text{AML}) - \log_2(\text{ALL})$; explained CD19 dilution in ALL due to T-ALL blasts; explicitly listed the 9 shared top-50 genes (*APLP2, CD302, CD33, CTSA, DNTT, IL18, LYN, PYGL, TCF3*) with 100% directional sign concordance, alongside signed Spearman rank correlation $\rho = 0.422$ ($p = 1.14 \times 10^{-233}$).
  9. **Tissue Confounding:** Clarified Odds Ratio = 1.46 is the odds of Peripheral Blood in AML relative to ALL; replaced "no dominant collinearity" with "no statistically dominant tissue confounding".
  10. **Roadmap & Formatting:** Restored full calendar dates to Table IV (Weeks 9–10 Nov 10–23, etc.), incorporated MILE EDA into WP4, removed stale Coustan-Smith test mention, and updated Abstract to "97.9% shared HUGO gene overlap of Golub genes".
- **Student Verification:** Verified automated test suite (55 passed, 5 skipped, 0 failed), compiled `paper/midterm/en/main.tex` with `pdflatex` to 7 clean pages with zero visual overlapping or compilation warnings, and synchronized `paper/markdown/Paper_EN.MD`.


## 3. Originality and Ethics Statement
The experimental hypotheses, methodology design, research questions, code verification, and final paper conclusions remain the intellectual and scientific responsibility of the author.
