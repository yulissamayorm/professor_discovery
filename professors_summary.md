# Professors Summary

A rolled-up view of every professor evaluated, with each paper, its dataset situation, and the overall ranking. The table below is sorted in descending order by the **professor-level overall score** — the simple arithmetic mean of every article's article-level overall score, per the rule defined in `Claude.md`. Click a professor's name to jump to their section.

## Ranking Table

| Rank | Professor | Affiliation | Articles | Dominant Modality | Professor-Level Overall | Status |
|------|-----------|-------------|----------|-------------------|--------------------------|--------|
| 1 | [Andrew A. Beharry](#andrew-a-beharry) | University of Toronto Mississauga | 2 | Small-molecule descriptors (RDKit) for SAR; cohort RNA-seq + clinical features for biomarker stratification | 6.6 / 10 | Provisional — both articles awaiting personal-interest scores |
| 2 | [Dwight S. Seferos](#dwight-s-seferos) | University of Toronto | 3 | Tabular molecular descriptors (RDKit from SMILES) | 6.2 / 10 | Provisional — all three articles awaiting personal-interest scores |

---

## Andrew A. Beharry

**Affiliation:** Department of Chemical and Physical Sciences, University of Toronto Mississauga (cross-appointed to Department of Chemistry, University of Toronto)

**Lab focus (one sentence):** Chemical biology and medicinal chemistry — two intersecting threads: (i) **fluorescent chemosensors** for clinically actionable enzyme activity (e.g., ratiometric probes for human carboxylesterase 2 to predict Irinotecan response in pancreatic cancer), and (ii) **photoactivatable / photocaged small molecules** (e.g., coumarin-based DEAC photocages on natural-product drugs) for spatially and temporally controlled release of bioactive compounds in cancer and antimicrobial settings.

**Lab webpage / Scholar:** andrew.beharry@utoronto.ca (lab webpage not yet confirmed — to be verified)

### Article 1: Antimicrobial Efficacy of Photocaged β-Lapachone in B. subtilis Biofilms

- **Full citation:** Hudson, E., Faylinn, C., Lopez-Miranda, I. R., Milstein, J. N., & Beharry, A. A. (2024). Antimicrobial efficacy of photocaged β-lapachone in *Bacillus subtilis* biofilms. *ChemPhotoChem*, 8(12), e202400164. DOI: 10.1002/cptc.202400164
- **Problem in plain English:** Bacterial biofilms (on catheters, implants, wounds) require 100–1000× higher antimicrobial doses than free-floating bacteria. β-lapachone (a natural product from the lapacho tree) inhibits biofilms by blocking catalase, but has bad off-target toxicity (methemoglobinemia from iron oxidation in blood) and a ~20-min half-life. The paper photocages β-lapachone with a diethylaminocoumarin (DEAC) so it is biologically inert until 420 nm violet light cleaves the cage, giving spatially and temporally controlled antibiotic activity.
- **ML task type:** No ML in the paper. Reframed as **structure → antimicrobial activity regression**: Morgan fingerprints / RDKit descriptors from SMILES → log(MIC) or log(EC₅₀) for B. subtilis (or, more broadly, biofilm-forming Gram-positive bacteria). Alternative reframe (noted, not chosen for the headline): ML surrogate for KatA (catalase) docking scores, mirroring the paper's GOLD docking analysis.
- **Input representation:**
  - Modality: Tabular molecular descriptors and/or Morgan fingerprints derived from SMILES
  - Preprocessing: PubChem name → SMILES lookup; RDKit descriptor + Morgan fingerprint computation. Optional: a "photocaged vs. native" boolean feature for the small subset where photocaged analogues exist.
- **Output representation:** log(MIC) or log(EC₅₀) for B. subtilis (primary target, distinct from the Seferos articles' battery-performance targets); units in mol L⁻¹ or μg mL⁻¹.
- **Model architecture (brief):** Gradient boosted regressor (XGBoost or LightGBM) with ridge regression as a linear baseline.
- **Dataset used in the paper:**
  - Name: Author-generated experimental data on photocaged β-lapachone (compound 1) and native β-lapachone
  - Size: 1 photocaged compound + 1 native control + 1 coumarin byproduct control (DEAC-OH); 5-point concentration range per compound
  - Source: Wet-lab synthesis, 96-well plate biofilm assays, agar well diffusion, catalase bubble assay, and GOLD docking against KatA (PDB 4e37) at University of Toronto Mississauga
  - Public? **Explicitly no** — the paper's Data Availability Statement reads "Research data are not shared."
  - Link: Not provided in paper
- **Reported headline metric:** Irradiated compound 1 inhibits B. subtilis biofilm growth with EC₅₀ = 0.14 mM, matching native β-lapachone (EC₅₀ = 0.15 mM). Dark control = no inhibition (no dark toxicity). 75% photo-uncaging at 20 min / 0.5 W·cm⁻² (420 nm); full uncaging at ~70 min.
- **Public alternatives to replicate or approximate this:**
  - CO-ADD (Community for Open Antimicrobial Drug Discovery) — ~300K compound-bacteria activity records including B. subtilis MIC values — match: **close** (direct B. subtilis activity data with the same Gram-positive context as the paper) — access: medium (free academic registration required) — https://www.co-add.org
  - ChEMBL antibacterial bioactivity subset — millions of bioactivity records, filter by target organism (B. subtilis) and assay type (MIC) — match: **medium-to-close** (broader, but the filtered subset overlaps strongly with the paper's setting) — access: easy (direct download / Python `chembl_webresource_client`) — https://www.ebi.ac.uk/chembl/
  - PubChem BioAssay — keyword search for biofilm-inhibition assays; broader still — match: **medium** — access: easy (pubchempy / REST API) — https://pubchem.ncbi.nlm.nih.gov/
- **Data access difficulty (for the paper's own dataset):** Very hard — author explicitly states data is not shared.
- **Tooling I would need to learn:** RDKit (descriptors + Morgan fingerprints), pubchempy (name → SMILES lookup), XGBoost or LightGBM, ChEMBL or CO-ADD API client. pandas + scikit-learn already familiar. Optional for the alternative docking-surrogate angle: AutoDock Vina + a PDB structure of KatA (PDB 4e37 used in the paper).
- **Portfolio angle:** Headline: *"Predicting antimicrobial activity against biofilm-forming bacteria from molecular structure — a B. subtilis MIC benchmark."* Filter CO-ADD or ChEMBL to compounds with B. subtilis MIC values, featurize via RDKit Morgan fingerprints + descriptors, train XGBoost vs. ridge baseline on log(MIC). Single Jupyter notebook with k-fold CV, predicted-vs-actual scatter, feature-importance plot. Stretch goal: pull the catalase-inhibitor sub-class (using known KatA inhibitors as a label set) and probe whether they cluster in feature space — a small mechanistic nod to the paper's catalase-inhibition finding.

#### Article 1 ranking

| Variable | Score (1–10) | One-line justification |
|---|---|---|
| Data availability | 7 | CO-ADD specifically covers B. subtilis MIC values; ChEMBL is a broad fallback covering antimicrobial assays generally |
| Data access ease | 7 | CO-ADD requires free academic registration; ChEMBL is a direct download with a well-supported Python client |
| Task tractability on laptop / free GPU | 8 | Small-molecule SAR via tabular regression trains in seconds on CPU; no GPU required |
| Tooling alignment with my current skills | 5 | Core stack is RDKit + sklearn/XGBoost (familiar core; RDKit + ChEMBL client are new pieces); GNNs not required for the headline angle |
| Problem clarity | 7 | log(MIC) regression is textbook-clean; main ambiguity is which assay format and bacterial strain panel to standardize on |
| Reproducibility signals | 3 | Paper explicitly states "Research data are not shared"; no code released; the GOLD docking workflow is reproducible in principle but not published as a dataset |
| Portfolio impact | 8 | Antibiotic-resistance + photocaged drugs is highly recruiter-legible; antimicrobial discovery is a hot ML application area with both academic and industry pull |
| My personal interest | TBD | Awaiting student score |

**Article 1 overall score:** Provisional (7 of 8 variables): (7 + 7 + 8 + 5 + 7 + 3 + 8) / 7 = 45 / 7 = **6.4 / 10**

**Article 1 verdict:** A credible GitHub project is achievable using CO-ADD or ChEMBL as the data source, framed as structure→MIC regression for B. subtilis (or biofilm-forming Gram-positive bacteria more broadly). The paper's actual experimental data is unavailable, so the project approximates the activity-prediction task that motivates the photocaging strategy rather than reproducing the paper's specific findings.

### Article 2: Ratiometric Fluorescent Chemosensor for hCES2 Activity in Pancreatic Cancer Patient-Derived Xenografts

- **Full citation:** Kailass, K., Sadovski, O., Capello, M., Kang, Y., Fleming, J. B., Hanash, S. M., & Beharry, A. A. (2019). Measuring human carboxylesterase 2 activity in pancreatic cancer patient-derived xenografts using a ratiometric fluorescent chemosensor. *Chemical Science*, 10(36), DOI: 10.1039/c9sc00283a
- **Problem in plain English:** Pancreatic cancer patients are commonly treated with FOLFIRINOX, which contains **Irinotecan** — a prodrug that only becomes active once **human carboxylesterase 2 (hCES2)** hydrolyzes it into SN-38, a topoisomerase 1 inhibitor. Patients with low hCES2 activity respond poorly to Irinotecan. Existing assays for measuring hCES2 (p-NPA colorimetric, immunoblot, mass spectrometry) are inaccurate, slow, or cross-reactive with the closely related hCES1, making them impractical in a clinical setting. The paper develops a single-step, hCES2-selective ratiometric fluorescent probe that can be applied directly to live cells and patient-derived tissue.
- **ML task type:** No ML in the paper. The natural ML reframe is *not* probe-design (data-poor) but the **downstream clinical biomarker question** the probe enables: a **survival-regression / drug-response-prediction** task — given CES2 expression (and other clinical / molecular features), predict Irinotecan response in pancreatic cancer patients. This is genuinely distinct from Article 1's small-molecule SAR angle.
- **Input representation:**
  - Modality: Patient-cohort tabular data — RNA-seq-derived gene expression (CES2 and selected covariates) plus clinical features (age, stage, KRAS mutation status, treatment regimen). Optional: small-molecule descriptors of the probe itself for a secondary probe-design subtask.
  - Preprocessing: TCGA / GDC tabular pull (cBioPortal API or `cBioPortalData` / `pyTCGA`-style clients); RNA-seq normalization to log₂(TPM + 1) or DESeq2 variance-stabilized counts.
- **Output representation:** Survival outcome (overall survival or progression-free survival) for FOLFIRINOX-treated pancreatic adenocarcinoma patients; or, for the cell-line side validation, log(IC₅₀) of Irinotecan in pancreatic cancer cell lines.
- **Model architecture (brief):** Cox proportional-hazards regression with CES2 expression as the primary feature + clinical covariates; compared against random-forest survival (`scikit-survival`) and a simple gradient-boosted IC₅₀ regressor for the cell-line external validation.
- **Dataset used in the paper:**
  - Name: Author-generated experimental data (probe synthesis + characterization, SU.86.86 cell line ± CES2 overexpression vector, two patient-derived xenografts: PDX112 and PDX121)
  - Size: 1 probe synthesized + 2 controls; 1 cell line ± vector pair; 2 PDXs
  - Source: University of Toronto Mississauga (probe development); MD Anderson Cancer Center (PDX models)
  - Public? No — paper does not release data; PDX tissues themselves are biobanked at MD Anderson
  - Link: Not provided in paper
- **Reported headline metric:** 40-fold fluorescence turn-on at 605 nm with hCES2 (110 nM); selective for hCES2 over hCES1 (no measurable activity on hCES1 even at 440 nM); Km = 6.6 ± 3.7 μM, kcat = 0.26 ± 0.04 s⁻¹; PDX112 vs PDX121 ratiometric Fred/Fyellow medians of 0.98 vs 0.34 (p < 0.0001), tracking Western-blot CES2 expression differences.
- **Public alternatives to replicate or approximate this:**
  - **TCGA-PAAD (Pancreatic Adenocarcinoma cohort)** — ~185 patients with RNA-seq, mutations, treatment metadata, and survival outcomes — match: **close** for the downstream Irinotecan-response biomarker question — access: easy (free via GDC Data Portal / cBioPortal; Python access via `cBioPortalData` or REST) — https://portal.gdc.cancer.gov/projects/TCGA-PAAD
  - **DepMap (Cancer Cell Line Encyclopedia + PRISM/CTRP drug-sensitivity)** — pancreatic cancer cell lines with CES2 expression and Irinotecan/SN-38 IC₅₀ values — match: **medium** (cell-line proxy for patient response, but directly addresses the "does CES2 expression predict Irinotecan sensitivity?" question) — access: easy (CSV downloads + Python `depmap` client) — https://depmap.org/portal/
  - **GDSC (Genomics of Drug Sensitivity in Cancer)** — drug sensitivity across ~1,000 cell lines including pancreatic — match: **medium** (broader cell-line coverage; useful for sanity check) — access: easy (https://www.cancerrxgene.org/)
- **Data access difficulty (for the paper's own dataset):** Very hard — the patient-derived xenograft tissue is biobanked at MD Anderson and not released; probe characterization data is in the paper / SI but not as a structured release.
- **Tooling I would need to learn:** **lifelines** or **scikit-survival** for Cox PH and random-forest survival models (new but Python-native and well-documented), **cBioPortalData** / GDC API client (new but standard REST), pandas + sklearn already familiar. Optional: pyDESeq2 for principled RNA-seq normalization. The probe-design alternative angle would also use RDKit (carry-over from Article 1).
- **Portfolio angle (distinct from Article 1):** Headline: *"Predicting pancreatic cancer Irinotecan response from CES2 expression — a TCGA-PAAD biomarker-stratification benchmark."* Pull TCGA-PAAD via cBioPortal, filter to patients with FOLFIRINOX or Irinotecan-containing regimens, fit a Cox proportional-hazards model with CES2 expression as the primary feature plus age / stage / KRAS as covariates. Compare against random-forest survival (`scikit-survival`). Kaplan-Meier curves stratified by CES2 high vs low. Predicted-vs-actual hazard ratios with 95% CIs; concordance index reported. **Stretch goal:** External validation on DepMap — does CES2 expression predict Irinotecan IC₅₀ in pancreatic cancer cell lines? README's headline frames the project as "what would be possible if Beharry's probe were deployed clinically — answered with the public cohort that already exists."

#### Article 2 ranking

| Variable | Score (1–10) | One-line justification |
|---|---|---|
| Data availability | 7 | TCGA-PAAD + DepMap directly support the downstream biomarker-stratification question; not the paper's own data, but a strong proxy benchmark |
| Data access ease | 8 | TCGA is one of the most accessible benchmarks in ML — well-documented Python clients (cBioPortalData, GDC API), CSV downloads from DepMap |
| Task tractability on laptop / free GPU | 7 | Cohort survival modeling adds a small piece beyond plain regression; still trains in seconds-to-minutes on CPU |
| Tooling alignment with my current skills | 6 | sklearn + pandas already familiar; lifelines / scikit-survival are new but Python-native and well-documented |
| Problem clarity | 7 | Biomarker-to-response is a textbook clinical-ML task; main ambiguity is which survival endpoint and treatment-subgroup filter to use |
| Reproducibility signals | 4 | Paper itself releases no code/data; but TCGA + DepMap function as the field's gold-standard reproducibility benchmark for this question |
| Portfolio impact | 8 | Precision oncology + pancreatic cancer biomarker prediction is highly recruiter-legible across pharma, biotech, and academic ML/health-AI roles |
| My personal interest | TBD | Awaiting student score |

**Article 2 overall score:** Provisional (7 of 8 variables): (7 + 8 + 7 + 6 + 7 + 4 + 8) / 7 = 47 / 7 = **6.7 / 10**

**Article 2 verdict:** A credible — and meaningfully different — GitHub project is achievable by treating the paper's chemical-biology contribution as the *motivation* for a precision-oncology biomarker-prediction task, then using TCGA-PAAD and DepMap (both public, both well-tooled) as the actual data source. The project is data-richer than Article 1's antimicrobial SAR angle and bridges chemistry to clinical ML, which is a strong story for advisor and recruiter audiences.

### Professor-level rollup

- **Article 1 overall:** 6.4 (provisional)
- **Article 2 overall:** 6.7 (provisional)
- **Professor overall** = (6.4 + 6.7) / 2 = 13.1 / 2 = 6.55 ≈ **6.6 / 10** (provisional — both articles awaiting personal-interest scores)

### Summary for this professor

- **Common thread across their work:** Both articles use rationally designed small molecules as **functional reporters or actuators on a clinically relevant biological target**. Article 1 (photocaged β-lapachone) uses light to *actuate* — releasing a catalase-inhibiting antimicrobial only where and when irradiated. Article 2 (Benz-AP / probe 2) uses fluorescence to *report* — turning on a ratiometric signal in the presence of hCES2 to read out a clinical biomarker for Irinotecan response. The unifying philosophy is "tunable small molecules whose biological activity (or detectability) is controlled by an external trigger" — light in Article 1, enzymatic hydrolysis in Article 2.
- **Dominant data modalities:** (i) Small-molecule descriptors / Morgan fingerprints from SMILES for SAR work (Article 1's portfolio angle); (ii) cohort-level RNA-seq + clinical features for biomarker-stratification work (Article 2's portfolio angle). The two articles point to two different ML toolchains rather than one shared one.
- **Biggest obstacle for me to replicate their work publicly:** Data access — both papers either explicitly state data is not shared (Article 1) or rely on biobanked PDX tissue that isn't publicly distributable (Article 2). Both portfolio projects rely on *adjacent* public benchmarks (CO-ADD / ChEMBL for Article 1; TCGA-PAAD / DepMap for Article 2) rather than reproducing the papers' specific measurements.
- **Skills I would gain by working with them:** Photopharmacology / photocage chemistry; fluorescent chemosensor design; small-molecule antimicrobial SAR; biofilm biology; molecular docking against bacterial enzyme targets (e.g., KatA); standard medicinal-chemistry ML featurization (RDKit + Morgan fingerprints); ChEMBL / CO-ADD querying; **clinical-cohort biomarker stratification (TCGA-PAAD), survival analysis with lifelines / scikit-survival, cell-line drug-sensitivity benchmarking with DepMap**. Article 2 specifically gives an entry point into clinical / health-AI tooling that Article 1 doesn't.

### Open items / notes

- Lab webpage URL for Prof. Beharry not yet confirmed — verify before finalizing.
- Personal interest score for Article 1 not yet provided by student.
- Personal interest score for Article 2 not yet provided by student.
- CO-ADD academic registration not yet completed — confirm dataset accessibility before relying on it for the Article 1 portfolio project.
- TCGA-PAAD subset filter (which patients received Irinotecan-containing regimens vs. gemcitabine-based) needs to be pinned down from the GDC clinical metadata — confirm cohort size after filtering before committing to the Cox PH design.
- The two Beharry articles point to genuinely different ML toolchains (small-molecule SAR vs. clinical-cohort survival modeling). Worth noting in any advisor conversation that the lab itself spans both threads, so a thesis would likely concentrate on one of the two directions rather than both.

---

## Dwight S. Seferos

**Affiliation:** Department of Chemistry & Department of Chemical Engineering and Applied Chemistry, University of Toronto

**Lab focus (one sentence):** Design and synthesis of functional organic and polymer materials for energy storage, with emphasis on aqueous zinc-ion batteries and sustainable, bio-derived electrode components.

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

#### Article 1 ranking

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

**Article 1 overall score:** Provisional (7 of 8 variables): (6 + 8 + 8 + 5 + 7 + 3 + 7) / 7 = **6.3 / 10**

**Article 1 verdict:** A credible GitHub project is achievable using ChemDataExtractor as a proxy dataset, framed as structure→capacity regression for organic cathodes — the main limitation is that the paper's own experimental data is not public, so the project approximates rather than replicates the paper's findings.

### Article 2: Bio-Derived Pendant Flavin Polymer Cathode for Sustainable Lithium-Ion Batteries

- **Full citation:** Schon, T. B., Tilley, A. J., Bridges, C. R., Miltenburg, M. B., & Seferos, D. S. (2016). Bio-derived polymers for sustainable lithium-ion batteries. *Adv. Funct. Mater.*, 26(38), 6896–6903. DOI: 10.1002/adfm.201602114
- **Problem in plain English:** State-of-the-art Li-ion cathodes use transition metals (Co, Ni) that are expensive (~30% of cell cost) and environmentally costly to mine. Riboflavin (vitamin B2) carries a flavin redox core that accepts two electrons near ~2.5 V — a promising bio-derived alternative — but small organics dissolve into the electrolyte and fade fast. The paper asks whether tethering flavins as pendant groups on a polymer backbone preserves their redox chemistry while suppressing dissolution. They also use DFT to identify *why* the resulting polymer fades on cycling.
- **ML task type:** No ML in the paper (experimental synthesis + coin-cell electrochemistry + DFT mechanism study). Reframed as multi-output regression: tabular molecular descriptors of organic polymer cathode candidates → discharge capacity (mAh g⁻¹) and **capacity retention (%) after N cycles** (the distinctive Article-2 target). A complementary classification reframe — "will this polymer's redox geometry change destabilize the electrode?" — is interesting but data-poor without external DFT generation.
- **Input representation:**
  - Modality: Tabular molecular descriptors derived from SMILES (molecular weight, functional group counts, ring counts, logP, donor/acceptor counts, etc.)
  - Preprocessing: PubChem name → SMILES lookup, then RDKit descriptor computation. Optional: a "bio-derived" boolean feature sourced from PubChem / ChEBI annotations.
- **Output representation:** Capacity retention (%) after a fixed cycle count (primary target, distinct from Article 1); discharge capacity (mAh g⁻¹) at a specified C-rate (secondary target).
- **Model architecture (brief):** Gradient boosted regressor (XGBoost or LightGBM) with ridge regression as a linear baseline. Same shape as Article 1 with a different label.
- **Dataset used in the paper:**
  - Name: Author-generated experimental data on a single polymer (P3), plus a literature comparison table in the Supporting Information
  - Size: 1 polymer synthesized in-house (P3); SI lists ~4–6 prior bio-derived cathodes for context
  - Source: Wet-chemistry synthesis, coin-cell electrochemistry, DFT geometry optimization, and TEM/SEM/Raman/XPS/FTIR characterization at University of Toronto
  - Public? No
  - Link: Not provided in paper
- **Reported headline metric:** 125 mAh g⁻¹ at 0.1 C (87% of theoretical max), two voltage plateaus near 2.55 V and 1.85 V; ~110% capacity retention over 250 cycles when cycled in a limited window (3.25–2.0 V) that excludes the second reduction. Highest capacity reported for any bio-derived polymeric cathode at the time.
- **Public alternatives to replicate or approximate this:**
  - AI-Driven Polymer Electrode Dataset — 771 organic polymer electrode materials with voltage and capacity labels — match: **close** (specifically organic polymer electrodes; closest public analogue to this paper's setting) — access: medium (preprint; supplementary download link to be verified) — https://arxiv.org/abs/2502.13899
  - ChemDataExtractor Battery Database v2.0 — 210,416 records — match: loose-to-medium (filter to organic cathodes for Li-ion) — access: easy (direct Figshare CSV/JSON/SQLite) — https://doi.org/10.6084/m9.figshare.18154715
  - Materials Project Battery Explorer — ~2,986 electrode materials, mostly inorganic intercalation — match: loose analogy (regression-pipeline warm-up only) — access: easy (free account, mp-api Python package) — https://materialsproject.org/batteries
- **Data access difficulty (for the paper's own dataset):** Very hard — author-generated, single-polymer experimental measurements, not released.
- **Tooling I would need to learn:** RDKit (SMILES → descriptors), pubchempy (name → SMILES lookup), XGBoost or LightGBM. Optional: PubChem / ChEBI annotation lookups for the bio-derived flag. pandas + scikit-learn already familiar.
- **Portfolio angle (distinct from Article 1):** Headline: *"Predicting cycling stability of organic battery cathodes from molecular features — a literature-mined ML benchmark for capacity retention."* Same featurization pipeline as Article 1, but the regression target is **capacity retention (%) after N cycles** rather than initial capacity. Optional sustainability axis: bio-derived vs. fully-synthetic split using PubChem / ChEBI annotations. Single Jupyter notebook with k-fold CV, predicted-vs-actual scatter for retention, and a feature-importance plot showing which descriptors predict stability. README frames the project against the paper's DFT-derived insight that *geometry change on the second reduction* is the dominant failure mode for flavin-bearing polymers.

#### Article 2 ranking

| Variable | Score (1–10) | One-line justification |
|---|---|---|
| Data availability | 7 | Closer public match than Article 1 — AI-Driven Polymer Electrode Dataset specifically covers organic polymer electrodes with capacity, mirroring this paper's cathode setting |
| Data access ease | 8 | ChemDataExtractor v2.0 remains a one-click Figshare download as a fallback workhorse, even if the closer dataset proves access-limited |
| Task tractability on laptop / free GPU | 8 | Tabular regression with XGBoost trains in seconds on CPU; no GPU required |
| Tooling alignment with my current skills | 5 | Same core stack as Article 1 (sklearn/XGBoost familiar; RDKit + pubchempy new; GNNs not required) |
| Problem clarity | 7 | Inputs (molecular features) and outputs (capacity, retention) well-defined; main ambiguity is which current density and cycle count to standardize on |
| Reproducibility signals | 3 | No code released; paper's data is one polymer's worth of in-house experimental measurements |
| Portfolio impact | 7 | Sustainability + bio-derived materials is a strong narrative angle for a recruiter audience; maps onto a recognized regression problem class |
| My personal interest | TBD | Awaiting student score |

**Article 2 overall score:** Provisional (7 of 8 variables): (7 + 8 + 8 + 5 + 7 + 3 + 7) / 7 = 45 / 7 = **6.4 / 10**

**Article 2 verdict:** A credible GitHub project is achievable using AI-Driven Polymer Electrode Dataset (or ChemDataExtractor as fallback) framed as structure→retention regression for organic polymer cathodes — the bio-derived sustainability angle adds narrative weight without changing the technical pipeline, and the project is meaningfully distinct from the Article 1 capacity-prediction angle.

### Article 3: Organic Magnesium-Ion Battery Cathodes (Future Energy Perspective / Review)

- **Full citation:** McAllister, B. T., Kyne, L. T., Schon, T. B., & Seferos, D. S. (2019). Potential for disruption with organic magnesium-ion batteries. *Joule*, 3(3), 620–630. DOI: 10.1016/j.joule.2018.12.005
- **Article type:** Review / *Future Energy* perspective — not primary experimental research. The paper's contribution is a structured argument for organic Mg-ion cathodes plus a curated comparison (Table 1) of published examples.
- **Problem in plain English:** Li-ion batteries dominate but face cobalt/lithium scarcity, dendrite safety risks, and flammability. Mg-ion is a leading post-Li candidate — Mg is abundant, dendrite-free, and offers ~2× volumetric capacity from two-electron oxidation. The bottleneck is the cathode: inorganic intercalation cathodes form MgO passivation layers and force two-electron redox per metal site. The authors argue organic cathodes side-step these problems via resonance stabilization, rotational flexibility, and ortho-carbonyl chelation, and review ~8 published examples (DMBQ, PAQS, 26PAQ, 14PAQ, NP, DMcT, CFx, C₆₀).
- **ML task type:** No ML in the paper. Reframed as **transfer-learning regression** — pre-train on abundant Li-ion organic cathode data, fine-tune on the small Mg-ion subset (the AI-Driven Polymer Electrode Dataset is mostly Li-ion; Mg-ion data is sparse). Alternative reframe (noted, not chosen for the headline angle): interpretable design-rule extraction (decision tree + SHAP) to recover the paper's claimed rules — rotational flexibility, ortho-carbonyl chelation, electron-withdrawing groups → higher voltage.
- **Input representation:**
  - Modality: Tabular molecular descriptors derived from SMILES (RDKit), same featurization stack as Articles 1 and 2.
  - Preprocessing: PubChem name → SMILES lookup; RDKit descriptor computation. Optional: a "Mg-ion vs. Li-ion" cell-chemistry feature so a single model can be trained jointly.
- **Output representation:** Mg-ion discharge capacity (mAh g⁻¹), operating voltage (V) vs. Mg/Mg²⁺, and capacity retention (%) after N cycles.
- **Model architecture (brief):** Two-stage. Stage 1: XGBoost (or ridge baseline) trained on Li-ion organic cathode data. Stage 2: fine-tune on the Mg-ion subset. Compare against an Mg-only baseline trained from scratch.
- **Dataset used in the paper:**
  - Name: Hand-curated literature comparison (Table 1)
  - Size: ~8 organic Mg-ion cathodes
  - Source: Aggregated from primary research articles cited in the review
  - Public? The table itself is published in the paper; underlying primary papers are individually accessible.
  - Link: Table 1 of the paper — no machine-readable release
- **Reported headline metric:** N/A (review paper). Notable cited result: 14PAQ retains 99% capacity after 100 cycles and 90% after 1,000 cycles at 1 C — surpassing Chevrel-phase inorganic cathodes for cycling.
- **Public alternatives to replicate or approximate this:**
  - Hand-curated Table 1 from this paper — ~8 organic Mg-ion cathodes — match: **direct** (this *is* the data the paper uses) — access: hard (manual extraction from the PDF, requires SMILES lookup per entry)
  - ChemDataExtractor Battery Database v2.0 filtered to Mg-ion — match: **medium** (Mg-ion is sparse compared to Li-ion in the corpus) — access: easy (https://doi.org/10.6084/m9.figshare.18154715)
  - AI-Driven Polymer Electrode Dataset — 771 organic polymer electrodes, predominantly Li-ion — match: **loose for Mg-ion directly, but ideal as the source domain for transfer learning** — access: medium (https://arxiv.org/abs/2502.13899)
- **Data access difficulty (for the paper's own dataset):** Hard — a literature-mined comparison table that has to be extracted by hand from the PDF, with SMILES sourced separately from PubChem.
- **Tooling I would need to learn:** Same core stack as Articles 1 & 2 (RDKit, pubchempy, XGBoost). Transfer learning adds no new tooling — pretrain → fine-tune is just two consecutive `.fit()` calls in sklearn/XGBoost. Optional: SHAP for the alternative interpretable-rules angle.
- **Portfolio angle (distinct from Articles 1 & 2):** Headline: *"Predicting Mg-ion organic cathode performance via transfer learning from Li-ion data — a tiny-data benchmark."* Hand-curate Table 1 (~8 entries) plus the Mg-ion slice of ChemDataExtractor v2.0 to land at ~10–30 Mg-ion datapoints. Featurize with RDKit. Compare three models: (a) Mg-only baseline (small data, will underperform), (b) Li-only model evaluated zero-shot on Mg, (c) Li → Mg fine-tuned model. Report RMSE on a held-out Mg test split, plus a learning-curve plot showing how the transfer benefit scales with Mg training fraction. Single notebook; the README's headline is the transfer-learning lift over the from-scratch baseline.

#### Article 3 ranking

| Variable | Score (1–10) | One-line justification |
|---|---|---|
| Data availability | 4 | Mg-ion organic cathode data is genuinely sparse — the paper's Table 1 is ~8 entries; ChemDataExtractor's Mg-ion slice is small; no dedicated public benchmark exists |
| Data access ease | 7 | What exists is either hand-curatable from this PDF or downloadable from Figshare; the bottleneck is the size of the data, not access friction |
| Task tractability on laptop / free GPU | 8 | Tiny tabular dataset trains in seconds; transfer learning adds zero compute overhead |
| Tooling alignment with my current skills | 5 | Same RDKit / XGBoost / pubchempy stack as the other two articles — no new ML tooling needed for the headline angle |
| Problem clarity | 6 | Review paper rather than a single experimental study; the transfer-learning reframe adds clarity but also a layer of interpretation |
| Reproducibility signals | 4 | Review articles cite primary papers but produce no original dataset or code; the paper itself isn't reproducing anything |
| Portfolio impact | 8 | "Tiny-data + transfer learning" for post-Li-ion energy storage is a strong, recruiter-legible story; Mg-ion is a hot pre-commercial space |
| My personal interest | TBD | Awaiting student score |

**Article 3 overall score:** Provisional (7 of 8 variables): (4 + 7 + 8 + 5 + 6 + 4 + 8) / 7 = 42 / 7 = **6.0 / 10**

**Article 3 verdict:** A credible — and distinctive — GitHub project is achievable by hand-curating Table 1, supplementing with the Mg-ion slice of ChemDataExtractor, and demonstrating transfer learning from a Li-ion source domain. The data scarcity is genuine but is itself the point of the project: this is the "tiny-data + smart pre-training" story rather than yet another structure→property regression.

### Professor-level rollup

- **Article 1 overall:** 6.3 (provisional)
- **Article 2 overall:** 6.4 (provisional)
- **Article 3 overall:** 6.0 (provisional)
- **Professor overall** = (6.3 + 6.4 + 6.0) / 3 = 18.7 / 3 = 6.23 ≈ **6.2 / 10** (provisional — all three articles awaiting personal-interest scores)

### Summary for this professor

- **Common thread across their work:** All three articles target organic / polymeric materials for next-generation batteries across three different cell chemistries — aqueous zinc-ion (Article 1, binder), lithium-ion (Article 2, bio-derived cathode), and magnesium-ion (Article 3, organic cathodes generally) — with a consistent focus on the structure → electrochemical-performance relationship and on sustainability (PFAS-free binders, vitamin-B2-derived polymers, post-Li chemistries). Outputs are predominantly wet-chemistry experimental measurements, supplemented by DFT (Article 2) and literature synthesis (Article 3).
- **Dominant data modality:** Tabular molecular descriptors derived from SMILES (RDKit-based featurization). Article 3's tiny-data setting opens a transfer-learning angle layered on the same featurization stack.
- **Biggest obstacle for me to replicate their work publicly:** Data access — the lab's primary outputs are wet-chemistry experimental measurements on small numbers of bespoke polymers (Articles 1, 2) and a curated literature comparison (Article 3); none are released as machine-readable datasets.
- **Skills I would gain by working with them:** Polymer / organic electrode chemistry across multiple cell chemistries; structure→property regression workflows; literature-mined dataset curation; RDKit-based featurization; bio-derived / semi-synthetic polymer chemistry; DFT-informed feature engineering for redox stability; tiny-data transfer-learning pipelines (Li → Mg).

### Open items / notes

- Lab webpage URL for Prof. Seferos not yet confirmed — verify before finalizing.
- Personal interest score for Article 1 not yet provided by student.
- Personal interest score for Article 2 not yet provided by student.
- Personal interest score for Article 3 not yet provided by student.
- arXiv:2502.13899 dataset availability needs confirmation — check supplementary or contact authors before relying on it for any of the three articles' projects.
- For Article 3: hand-curating Table 1 to machine-readable form (~8 entries with SMILES) is a one-off step that could be open-sourced as part of the GitHub project itself.
- `professors_survey.md` was previously the working draft document but no longer exists in the project directory; this summary is now the single source of truth.
