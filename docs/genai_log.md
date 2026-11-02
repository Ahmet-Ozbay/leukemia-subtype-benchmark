# Appendix A: GenAI Disclosure and Usage Log

This document is prepared in accordance with the Semester Project Guide (Section 9) and IEEE Authorship and AI Policies.

## 1. Summary of Generative AI Tools Used

| Tool and Version | Scope and Objective | Affected Modules / Files | Human Verification and Modifications |
|---|---|---|---|
| **Google Gemini (Antigravity Agent)** | Data ingestion, preprocessing, probe-to-gene mapping, and publication-grade EDA visualization architecture | `src/data/`, `src/features/`, `src/visualization/`, `tests/` | Code reviewed line by line, platform encoding issues resolved, and verified using 45+ automated pytest tests. |
| **Google Gemini (LLM)** | Literature review synthesis, research questions framing (RQ1-RQ3), and IEEE paper structure formatting | `CONTEXT.md`, `PROJECT_REQUIREMENTS.md` | All 15 referenced peer-reviewed articles checked against original publisher records; verified DOIs and citation metadata. |

## 2. Detailed Interaction & Prompt Log

### Session 1: Data Pipeline and Probe Mapping Architecture
- **User Prompt Summary:** Design a scikit-learn transformer and data pipeline to map Affymetrix Hu6800 and HG-U133A probe codes to HUGO gene symbols, collapsing multi-mapped probes without data leakage (variance computed strictly on training folds).
- **Model Output Summary:** `ProbeCollapser` transformer class (`fit`, `transform`), GPL80/GPL96 annotation parsers, and the `harmonize_cohort_genes` orchestration function.
- **Student Verification:** Developed 11 unit and regression tests in `tests/test_gene_mapping.py`, explicitly proving zero cross-split information leakage via `test_probe_collapser_fit_leakage`.

### Session 2: Exploratory Data Analysis (EDA) and IEEE Visualization Engine
- **User Prompt Summary:** Create a modular visualization engine with customizable styling presets (`grayscale`, `academic`, `vibrant`) producing IEEE-compliant 300+ DPI figures for expression distributions (Fig 1), dimensionality reduction (Fig 2), top 40 gene heatmap (Fig 3), and batch effects (Fig 4).
- **Model Output Summary:** `src/visualization/style.py` and `src/visualization/eda.py`.
- **Student Verification:** Inspected exported `.png` and `.pdf` files, verified typography and contrast ratios, and automated validation via `tests/test_eda.py`.

## 3. Originality and Ethics Statement
The experimental hypotheses, methodology design, research questions, code verification, and final paper conclusions remain the intellectual and scientific responsibility of the author.
