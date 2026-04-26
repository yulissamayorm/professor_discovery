# Professors Summary

A rolled-up view of every professor evaluated, with each paper, its dataset situation, and the overall ranking. The table below is sorted in descending order by overall score. Click a professor's name to jump to their section.

## Ranking Table

| Rank | Professor | Affiliation | Dominant Modality | Overall Score | Status |
|------|-----------|-------------|-------------------|---------------|--------|
| 1 | [Dwight S. Seferos](#dwight-s-seferos) | University of Toronto | Tabular molecular descriptors (RDKit from SMILES) | 6.3 / 10 | Provisional (7 of 8 variables; personal interest pending) |

---

## Dwight S. Seferos

**Affiliation:** Department of Chemistry & Department of Chemical Engineering and Applied Chemistry, University of Toronto

**Lab focus (one sentence):** Design and synthesis of functional organic and polymer materials for energy storage, with emphasis on aqueous zinc-ion batteries and sustainable electrode components.

**Lab webpage / Scholar:** dwight.seferos@utoronto.ca (lab webpage not yet confirmed — to be verified)

### Article 1: Stable, High-Rate Organic Zinc-Ion Batteries via Marine-Inspired Binder

- **Full citation:** Battaglia, A. M., Liu, J. T., Lotocki, V., Perry, K. L., & Seferos, D. S. (2025). Stable, high-rate, organic zinc-ion batteries accomplished using an ion-conducting marine-inspired binder. *EES Batteries*, 1, 1173–1183. DOI: 10.1039/d5eb00125k
- **Problem in plain English:** Aqueous zinc-ion batteries are a cheaper, safer alternative to Li-ion, but their organic cathodes lose capacity due to poor conductivity and high solubility in water. The standard PVDF binder is insulating, weakly adhesive, and a PFAS pollutant. The paper tests whether a marine-inspired binder (POxaPG) can fix both problems at once.
- **ML task type:** No ML in the paper. Reframed as multi-output regression: molecular features of binder/cathode → electrochemical performance metrics (discharge capacity in mAh g⁻¹, capacity retention %).
- **Input representation:**
  - Modality: Tabular molecular descriptors derived from SMILES (molecular weight, functional group counts, logP, ring counts, etc.)
  - Preprocessing: Look up SMILES for each material name via PubChem; compute descriptors with RDKit
- **Output representation:** Discharge capacity (mAh g⁻¹) at one or more current densities; optionally capacity retention (%) and Zn²⁺ diffusion coefficient
- **Model architecture (brief):** Gradient boosted regressor (XGBoost or LightGBM), with ridge regression as linear baseline
- **Dataset used in the paper:**
  - Name: Author-generated experimental data (no named public dataset)
  - Size: 4 binder variants tested; ~8 organic cathode materials compared via supplementary Table S2
  - Source: Wet-chemistry synthesis and electrochemical testing at University of Toronto
  - Public? No
  - Link: Not provided in paper
- **Reported headline metric:** POxaPG + NTCDI achieves ~350 mAh g⁻¹ at 0.01 A g⁻¹ and 183 mAh g⁻¹ at 20 A g⁻¹; 78% retention after 8,000 cycles at 1 A g⁻¹ (vs. 26% for PVDF control)
- **Public alternatives to replicate or approximate this:**
  - ChemDataExtractor Battery Database v2.0 — 210,416 records — match: loose (covers all battery chemistries; filter to organic cathodes) — access: easy (direct Figshare CSV/JSON/SQLite, no account) — https://doi.org/10.6084/m9.figshare.18154715 (v1.0 with 17,354 chemicals: https://doi.org/10.6084/m9.figshare.11888115)
  - AI-Driven Polymer Electrode Dataset — 771 organic electrode materials with voltage and capacity labels — match: close — access: medium (preprint, dataset link not yet confirmed) — https://arxiv.org/abs/2502.13899
  - Materials Project Battery Explorer — ~2,986 electrode materials with computed electrochemical properties — match: loose analogy (mostly inorganic/intercalation) — access: easy (free account, mp-api Python package) — https://materialsproject.org/batteries
- **Data access difficulty (for the paper's own dataset):** Very hard — author-generated, not released
- **Tooling I would need to learn:** RDKit (descriptor computation from SMILES), pubchempy (name → SMILES lookup), XGBoost or LightGBM. pandas + scikit-learn already familiar.
- **Portfolio angle:** Headline: *"Predicting discharge capacity of organic battery cathodes from molecular features — a literature-mined ML benchmark."* Filter ChemDataExtractor v2.0 to organic cathodes, look up SMILES via PubChem, compute ~10–20 RDKit descriptors, train XGBoost vs. ridge baseline. Single Jupyter notebook with k-fold CV, predicted-vs-actual scatter, and feature importance plot — chemistry context explained inline.

### Ranking for this professor

| Variable | Score (1–10) | One-line justification |
|---|---|---|
| Data availability | 6 | No public AZIB binder dataset; loose analogues (ChemDataExtractor) cover capacity broadly but not binder-specific |
| Data access ease | 8 | ChemDataExtractor v2.0 is a direct Figshare download in CSV/JSON — no account, no API needed |
| Task tractability on laptop / free GPU | 8 | Tabular regression with XGBoost trains in seconds on CPU; no GPU required |
| Tooling alignment with my current skills | 5 | Core is sklearn/XGBoost (familiar), but RDKit and pubchempy are new; GNNs not required |
| Problem clarity | 7 | Inputs (molecular features) and output (capacity in mAh g⁻¹) are well-defined; main ambiguity is which current density to target |
| Reproducibility signals | 3 | No code released; paper's dataset is author-generated and not public |
| Portfolio impact | 7 | Battery materials + ML is a hot area with clear industry relevance (EV, grid storage); maps onto a well-understood ML problem class |
| My personal interest | TBD | Awaiting student score |

**Overall score computation:** Provisional (7 of 8 variables): (6 + 8 + 8 + 5 + 7 + 3 + 7) / 7 = **6.3 / 10**

**One-sentence verdict:** A credible GitHub project is achievable using ChemDataExtractor as a proxy dataset, framed as structure→capacity regression for organic cathodes — the main limitation is that the paper's own experimental data is not public, so the project approximates rather than replicates the paper's findings.

### Summary for this professor

- **Common thread across their work:** TBD — only one article processed so far.
- **Dominant data modality:** Tabular molecular descriptors derived from SMILES (RDKit-based featurization).
- **Biggest obstacle for me to replicate their work publicly:** Data access — the lab's primary outputs are wet-chemistry experimental measurements, not released datasets.
- **Skills I would gain by working with them:** Polymer/organic electrode chemistry; structure→property regression workflows; literature-mined dataset curation; RDKit-based featurization.

### Open items / notes

- Lab webpage URL for Prof. Seferos not yet confirmed — verify before finalizing.
- Personal interest score for Article 1 not yet provided by student.
- arXiv:2502.13899 dataset availability needs confirmation — check supplementary or contact authors before relying on it.
- Common-thread summary to be completed once additional articles for this professor are processed.
