# Professors Summary

A rolled-up view of every professor evaluated, with each paper, its dataset situation, and the overall ranking. The table below is sorted in descending order by the **professor-level overall score** — the simple arithmetic mean of every article's article-level overall score, per the rule defined in `Claude.md`. Click a professor's name to jump to their section.

## Ranking Table

| Rank | Professor | Affiliation | Articles | Dominant Modality | Professor-Level Overall | Status |
|------|-----------|-------------|----------|-------------------|--------------------------|--------|
| 1 | [Benjamín Sánchez-Lengeling](#benjamín-sánchez-lengeling) | University of Toronto (department to be confirmed) | 3 | SMILES sequences (organic, QM9 / ZINC / MOSES) + 3D voxel images (inorganic crystals, Materials Project); generative ML (VAE family) + standardized benchmarking | 7.7 / 10 | Provisional — all three articles are middle-authored PhD-era papers; awaiting personal-interest scores and confirmation against independent U of T group's first-author work |
| 2 | [Seyed Mohamad Moosavi](#seyed-mohamad-moosavi) | University of Toronto (department to be confirmed) | 1 | MOF crystal structures + Monte Carlo adsorption properties from a hypothetical 325K-MOF database (Materials Cloud); graph-similarity / data mining + property prediction | 7.3 / 10 | Provisional — single middle-authored EPFL-postdoc-era paper; awaiting personal-interest score and confirmation against independent U of T group's first-author work |
| 3 | [Andrew A. Beharry](#andrew-a-beharry) | University of Toronto Mississauga | 3 | Small-molecule descriptors (RDKit) for SAR; cohort RNA-seq + clinical features for biomarker stratification | 6.4 / 10 | Provisional — all three articles awaiting personal-interest scores |
| 4 | [Dwight S. Seferos](#dwight-s-seferos) | University of Toronto | 3 | Tabular molecular descriptors (RDKit from SMILES) | 6.2 / 10 | Provisional — all three articles awaiting personal-interest scores |

---

## Benjamín Sánchez-Lengeling

**Affiliation:** University of Toronto (specific department to be confirmed — likely Chemistry and/or Computer Science based on his research area). At the time of Article 1 below he was a PhD student in Alán Aspuru-Guzik's group at Harvard.

**Lab focus (one sentence):** **Inferred** from Article 1 + Sánchez-Lengeling's broader publication history (Harvard PhD → Google Brain → U of T faculty): machine learning for molecular and materials design, with emphasis on generative models (VAEs, GANs, graph neural nets) for chemical-space exploration, property prediction, and inverse design. **Verify against the lab's current Toronto-era publications before treating this as authoritative.**

**Lab webpage / Scholar:** Not yet provided — needs to be sourced.

### Article 1: Variational Autoencoder for Continuous Molecular Representation (Foundational Generative-Chemistry Paper)

- **Full citation:** Gómez-Bombarelli, R., Wei, J. N., Duvenaud, D., Hernández-Lobato, J. M., **Sánchez-Lengeling, B.**, Sheberla, D., Aguilera-Iparraguirre, J., Hirzel, T. D., Adams, R. P., & Aspuru-Guzik, A. (2018). Automatic chemical design using a data-driven continuous representation of molecules. *ACS Central Science*, 4(2), 268–276. DOI: 10.1021/acscentsci.7b00572
- **Authorship caveat:** Sánchez-Lengeling is a **middle author** (5th of 10), authored during his PhD at Harvard in Alán Aspuru-Guzik's group — **not from his current independent group at University of Toronto**. The four co-first authors (marked #) are Gómez-Bombarelli, Wei, Duvenaud, and Hernández-Lobato; Aspuru-Guzik is sole corresponding author. **Treat this as evidence of the kind of research Sánchez-Lengeling was trained on rather than as representative of his independent research direction.**
- **Problem in plain English:** Drug and material design is an optimization problem in molecular space — but molecular space is **vast** (10²³ to 10⁶⁰ drug-like molecules), **discrete** (atoms are atoms; you can't take half-steps), and **unstructured** (no natural distance metric). Existing methods (virtual screening of fixed libraries, genetic algorithms with hand-crafted mutation rules) can't use gradient-based optimization because there are no gradients in discrete space. The paper learns a *continuous* real-valued representation of molecules that supports gradient-based optimization, interpolation, and standard generative-modeling techniques.
- **ML task type:** This **IS** a primary ML paper — no reframing needed. Generative modeling (VAE) + property prediction + Bayesian optimization in latent space, demonstrated on molecular design.
- **Input representation:**
  - Modality: SMILES strings, one-hot encoded over a small character vocabulary (35 chars for ZINC, 22 for QM9), padded to fixed length (120 for ZINC, 34 for QM9)
  - Preprocessing: Canonicalize SMILES via RDKit; one-hot encode; pad
- **Output representation:** Three coupled outputs — (i) reconstructed SMILES via the decoder, (ii) a continuous latent vector (156 dim for QM9, 196 dim for ZINC), (iii) property predictions (logP / QED / SAS for ZINC; HOMO / LUMO / electronic spatial extent R² for QM9) from the latent vector via an MLP head
- **Model architecture (brief):** **Variational autoencoder.** Encoder = three 1D convolutional layers + one fully-connected layer. Decoder = three GRU layers (488-dim hidden for ZINC, 500 for QM9). Property predictor = MLP (2 × 1000 neurons, or 3 × 67 neurons) on the latent vector. Jointly trained on reconstruction loss + variational KL penalty + property regression loss. A Gaussian process is later trained on the latent space for downstream Bayesian optimization of (5 × QED − SAS).
- **Dataset used in the paper:**
  - Name: **QM9** (134K small molecules with ≤9 heavy atoms + DFT-computed properties) and **ZINC250k** (250K drug-like molecules randomly sampled from ZINC)
  - Size: 108K (QM9 training subset) and 250K (ZINC)
  - Source: Public benchmarks — QM9 from Ramakrishnan et al. *Sci. Data* 2014; ZINC from Irwin et al. *J. Chem. Inf. Model.* 2012
  - Public? **Yes — both are gold-standard open benchmarks.**
  - Link: QM9 → http://quantum-machine.org/datasets/ ; ZINC → https://zinc.docking.org/ ; both also one-line loadable via DeepChem / MoleculeNet
- **Reported headline metric:** ~73–79% SMILES validity rate near training-set points (4% for randomly-selected latent points). Joint property training organizes the latent space so molecules cluster by property value (Figure 3 PCA plots). Latent-space Bayesian optimization on (5×QED − SAS) consistently beats genetic-algorithm and random-search baselines, finding molecules in the top decile when seeded from the bottom decile. Property-prediction MAE on QM9 LUMO: 0.16 eV (competitive with graph convolutions at 0.15 eV).
- **Public alternatives to replicate or approximate this:**
  - **Direct match — released code + data:** https://github.com/aspuru-guzik-group/chemical_vae — match: **direct, perfect** — access: easy (clone + run)
  - **MoleculeNet (DeepChem)** benchmark suite — wraps QM9, ZINC, with standardized splits + evaluation — match: **direct** — access: easy (`pip install deepchem`) — https://moleculenet.org/
  - **Modern baselines for comparison:** JT-VAE (Jin et al. 2018, https://github.com/wengong-jin/icml18-jtnn), GraphAF (Shi et al. 2020), MoFlow (Zang & Wang 2020), Chemformer (Irwin et al. 2022) — match: close (newer architectures for the same generative-chemistry task) — access: easy (all on GitHub)
- **Data access difficulty (for the paper's own dataset):** **Easy** — this is the rare paper where the authors release everything publicly.
- **Tooling I would need to learn:** **PyTorch (or TensorFlow / Keras as the paper uses)**, **VAE architecture and training** (encoder + decoder + variational KL loss + reparameterization trick — a substantial new piece if you've only done supervised learning before), **RDKit** (already in the toolchain from prior professors' projects), **Gaussian processes** for the latent-space Bayesian optimization step (scikit-learn or GPyTorch). Optional for a stretch goal: graph neural networks (PyTorch Geometric or DGL) for a graph-VAE comparison.
- **Portfolio angle:** This is meaningfully different from any prior portfolio angle in the survey because it's **primary ML** (no reframing needed). Three viable framings:
  1. **Reproduction-with-extension:** *"Reproducing the molecular VAE (Gómez-Bombarelli et al. 2018) and benchmarking against modern baselines (JT-VAE, GraphAF, Chemformer)."* Use the released code as a starting point, train on a subsampled ZINC for compute reasons, then add one modern baseline. Single notebook with reconstruction accuracy, validity %, novelty %, and property-optimization comparison.
  2. **Domain transfer:** *"Applying the molecular VAE framework to [a specific chemistry domain you care about]."* Train the VAE on a domain-specific subset and use latent-space optimization to propose new candidates.
  3. **Architecture comparison:** *"SMILES-VAE vs. graph-VAE vs. transformer-VAE on a single benchmark"* — three small models, one notebook, head-to-head on QM9-subset reconstruction and property prediction.

#### Article 1 ranking

| Variable | Score (1–10) | One-line justification |
|---|---|---|
| Data availability | 10 | QM9 and ZINC are gold-standard public benchmarks; the paper uses them directly and releases its training subsets via GitHub |
| Data access ease | 9 | DeepChem / MoleculeNet wrap both datasets with one-line loaders; only minor friction is canonicalization preprocessing |
| Task tractability on laptop / free GPU | 5 | Full VAE training on 250K ZINC across 120 epochs takes hours on a free Colab GPU; viable with subsampling to ~25K and smaller latent dim, but not laptop-CPU friendly |
| Tooling alignment with my current skills | 4 | VAE architecture + variational KL loss + reparameterization + GP-based latent-space Bayesian optimization is a substantial step up from the tabular RDKit-XGBoost pipelines used in prior professors' projects |
| Problem clarity | 8 | Generative-chemistry benchmarks (reconstruction accuracy, validity %, novelty %, uniqueness, property optimization) are well-established and standardized |
| Reproducibility signals | 10 | Code AND data both released on GitHub with full hyperparameters; this is the rare 10-out-of-10 reproducibility case |
| Portfolio impact | 8 | Foundational generative-chemistry paper (~3000 citations); pure 2018 replication risks reading tutorial-level in 2026, but extension or modern-baseline comparison reads strongly to recruiters in pharma / biotech / ML roles |
| My personal interest | TBD | Awaiting student score |

**Article 1 overall score:** Provisional (7 of 8 variables): (10 + 9 + 5 + 4 + 8 + 10 + 8) / 7 = 54 / 7 = **7.7 / 10**

**Article 1 verdict:** A **highly credible** GitHub project is achievable using the released code as a starting point, with the primary cost being VAE training compute (hours on free Colab GPU) and the new ML tooling (variational autoencoders, GP-based Bayesian optimization). The strongest portfolio framings are reproduction-with-extension (compare against a modern baseline like JT-VAE) or domain transfer (apply the VAE framework to a chemistry domain you care about) — pure replication risks reading as tutorial-level given the paper's age.

### Article 2: Variational Autoencoder for Inverse Design of Inorganic Crystal Structures (iMatGen — Vanadium Oxides)

- **Full citation:** Noh, J., Kim, J., Stein, H. S., **Sanchez-Lengeling, B.**, Gregoire, J. M., Aspuru-Guzik, A., & Jung, Y. (2019). Inverse design of solid-state materials via a continuous representation. *Matter*, 1(5), 1370–1384. DOI: 10.1016/j.matt.2019.08.017
- **Authorship caveat:** Sánchez-Lengeling is again a **middle author** (4th of 7), this time listed under his Harvard affiliation (Department of Chemistry and Chemical Biology). Per the explicit author contributions: *"B.S.-L. and A.A.-G. assisted with the machine-learning model construction."* First author is Juhwan Noh (KAIST); co-corresponding authors are Aspuru-Guzik and Yousung Jung (KAIST). **Like Article 1, this is from Sánchez-Lengeling's PhD-era affiliation, not his independent University of Toronto research direction.** Pattern across both articles processed so far: middle-authored Harvard-era contributions to ML-for-chemistry papers led by other groups. Pulling at least one first-author or corresponding-author paper from his independent U of T group remains the single most important next step for advisor evaluation.
- **Problem in plain English:** Article 1 demonstrated VAEs for *organic molecules* using SMILES strings. Inorganic solid-state crystals (periodic 3D structures with unit cell + atomic basis) are much harder because **no invertible representation existed**. SMILES is invertible (string → molecule); fingerprints, Coulomb matrices, and crystal graphs used by prior crystal-ML work are *not* — you can predict properties from them but you can't reconstruct a crystal back from them. This paper invents an **invertible 3D image-based representation** for crystals (one image for the unit cell, one image per element type for atomic positions on a 3D Gaussian-convolved grid) and trains a VAE on it. Demonstrated on vanadium oxides (V-O); rediscovers 25 of 31 known V-O structures and generates 40+ new metastable polymorphs.
- **ML task type:** **Primary ML** (no reframing needed). Generative model for inorganic crystal structures via a two-step hierarchical autoencoder: (i) convolutional autoencoder compresses 3D voxel images of cell + basis to a fingerprint vector, (ii) VAE on the fingerprint with an auxiliary stability classifier head (semi-supervised VAE per Snoek et al. 2012). Latent-space sampling via Slerp interpolation and random Gaussian sampling.
- **Input representation:**
  - Modality: **3D voxel images of crystal structures.** Each crystal → one cell image (Gaussian on a 32³ grid representing unit-cell vectors) + one basis image per element type (64³ grid with Gaussians at atomic positions). For V-O materials: two basis images (V + O) per structure.
  - Preprocessing: Pull crystals from Materials Project via pymatgen; map non-cubic unit cells onto cubic 32³ grids; rasterize atomic positions to 64³ grids with Gaussian smoothing; constrain to N_sites ≤ 20 atoms per cell and unit-cell length ≤ 10 Å for the V-O subset
- **Output representation:** Three coupled outputs — (i) reconstructed 3D voxel images that decode back to crystal coordinates + lattice parameters via inverse Gaussian transform + an image filter for atomic-position recovery, (ii) latent vector in the materials-design space, (iii) binary stability label (stable vs unstable, threshold formation energy E_f ≤ −0.5 eV/atom) from the auxiliary classifier head
- **Model architecture (brief):** **Two-step hierarchical autoencoder.** Step 1 = convolutional autoencoder for image compression (cell + basis separately). Step 2 = VAE ("Materials Generator") on the compressed fingerprints + a feed-forward classifier mapping latent → binary stability label. Trained on a custom VO dataset of 10,981 V-O structures with stability labels. Latent-space sampling via Slerp (interpolating between known structures) and random Gaussian sampling around the data mean.
- **Dataset used in the paper:**
  - Name: **Materials Project (MP)** as the source database (83,989 inorganic structures total, 112 V-O entries) + a custom **VO dataset** of 10,981 V-O structures constructed by substituting V and O into 7,472 binary AxBy compounds from MP
  - Size: 10,981 structures total (90% train / 10% test); 31 of these correspond to known experimental V-O materials in MP
  - Source: Materials Project + author-constructed substitutional dataset
  - Public? **Yes** — MP is gold-standard public; the constructed VO dataset + trained model parameters are released on GitHub
  - Link: Code + data at https://github.com/kaist-amsg/imatgen.git ; Materials Project at https://materialsproject.org/
- **Reported headline metric:** Reconstruction: 10,079 of 10,981 input structures fully reconstructed (cell within 0.1 Å and 2°, atomic positions within 0.2 Å avg). Rediscovery: 25 of 31 known V-O structures + 8 of 10 experimentally-reported ICSD V-O structures regenerated when the model was trained without them. Generation: 52 new V-O compositions and 40 new metastable polymorphs with E_hull ≤ 80 meV/atom (the empirical threshold for synthesizability per Singh et al. 2019). Comparison vs CALYPSO genetic algorithm: lowest-energy V₅O₈ and V₃O₄ predictions match between iMatGen and GA, but iMatGen explores compositional space ~7× more efficiently when target compositions are a priori unknown.
- **Public alternatives to replicate or approximate this:**
  - **Direct match — released code + data:** https://github.com/kaist-amsg/imatgen.git — match: **direct, perfect** — access: easy (clone + run; needs PyTorch / TensorFlow + pymatgen)
  - **Materials Project Python API + pymatgen** — for pulling alternative crystal subsets to train an inverse-design model in a different chemistry — match: **direct** — access: easy (`pip install mp-api pymatgen`; free MP account required) — https://materialsproject.org/api
  - **Modern crystal-VAE / generative baselines for comparison:** CGCNN (Xie & Grossman 2018, structure → property), CDVAE (Xie et al. 2022 ICLR — equivariant diffusion for crystals), GNoME (DeepMind 2023 — graph-network exploration of materials chemistry) — match: close (newer architectures for the same generative-crystals task) — access: easy (all on GitHub)
- **Data access difficulty (for the paper's own dataset):** **Easy** — like Article 1, the authors release code, data, and trained parameters publicly.
- **Tooling I would need to learn:** **Substantially more than Article 1.** **3D convolutional neural networks** (PyTorch / TensorFlow with 3D conv layers — non-trivial to debug), **the custom invertible voxel-image representation pipeline** (Gaussian rasterization + inverse transform with image filters via SciPy), **pymatgen + Materials Project API** for pulling and parsing crystal structures (new domain library — handles space groups, lattice systems, formation energies), **two-step hierarchical autoencoder training** (more complex than a single-stage VAE), and (ideally) **DFT post-validation** via VASP / Quantum ESPRESSO / GPAW (this last step is realistically out of reach for a laptop / free-Colab portfolio project; you'd skip it or rely on existing MP-computed energies for sanity checks).
- **Portfolio angle (distinct from Article 1):** Article 1 is the foundational organic-molecule SMILES VAE; Article 2 extends the same idea to **inorganic crystals** with a novel invertible 3D-voxel representation. Three viable framings:
  1. **Reproduction-with-extension on a different chemistry:** *"Reproducing iMatGen and applying it to [a different binary system from Materials Project]."* Use the released code as a starting point; pull a different binary chemistry from MP (e.g., Mn-O for a manganese-cathode echo of Seferos's work, or Ti-O for photocatalysis); train the same VAE; report rediscovery rate + new metastable candidates.
  2. **Architecture comparison on the same task:** *"3D-voxel VAE vs. crystal-graph VAE vs. equivariant diffusion (CDVAE) on V-O rediscovery."* Three small models, one notebook, head-to-head on the same VO dataset.
  3. **Bridge between Articles 1 and 2:** *"Generative chemistry across modalities — molecular VAE (SMILES) vs. materials VAE (3D voxels) on a shared evaluation framework."* Reproduce both Article 1 and Article 2 at small scale, report a unified table comparing reconstruction / validity / novelty / property-conditioning behaviour. Strong story for a portfolio because it shows mastery of *both* generative-chemistry sub-fields — and explicitly traces the lineage of Sánchez-Lengeling's PhD trajectory.

#### Article 2 ranking

| Variable | Score (1–10) | One-line justification |
|---|---|---|
| Data availability | 9 | Materials Project is gold-standard public benchmark for inorganic chemistry; the constructed VO dataset is also released on GitHub. Slightly less universal than Article 1's QM9 / ZINC because the VO subset is custom-built, but MP itself is the field standard |
| Data access ease | 8 | mp-api / pymatgen Python clients give one-line access; only friction is the free MP account registration. Slightly more friction than Article 1's DeepChem one-liner |
| Task tractability on laptop / free GPU | 4 | 3D voxel CNNs + two-step hierarchical autoencoder training + (ideally) DFT post-validation is substantially more compute-heavy than Article 1's 1D SMILES VAE; on free Colab the autoencoder training is feasible with subsampling but DFT validation of generated candidates is realistically out of reach |
| Tooling alignment with my current skills | 3 | 3D CNNs + custom invertible-representation pipeline + pymatgen + Materials Project API + (ideally) DFT toolchain is a substantial step UP from Article 1's already-difficult 1D-SMILES VAE stack — new domain-specific libraries on top of new ML architecture |
| Problem clarity | 7 | Generative chemistry for inorganic materials is well-defined (validity = chemically valid composition + valid oxidation state; novelty = not in training set; stability = E_hull from DFT or pretrained predictor); slightly less clean than the organic-molecule case because there's no universal validity metric for crystals |
| Reproducibility signals | 10 | Code, data, AND trained parameters all released on GitHub with a clear DATA AND CODE AVAILABILITY statement; same perfect 10 as Article 1 |
| Portfolio impact | 8 | Inverse materials design is a hot field (battery materials, photovoltaics, catalysts); the inorganic-VAE here is less foundational than Article 1's molecular VAE but still highly cited and recruiter-legible for materials-science / chem-AI roles |
| My personal interest | TBD | Awaiting student score |

**Article 2 overall score:** Provisional (7 of 8 variables): (9 + 8 + 4 + 3 + 7 + 10 + 8) / 7 = 49 / 7 = **7.0 / 10**

**Article 2 verdict:** A credible — and meaningfully different — GitHub project is achievable using the released iMatGen code as a starting point, with the primary cost being **substantially more complex ML tooling than Article 1** (3D CNNs + custom invertible voxel representation + pymatgen) and the realistic compromise of skipping DFT post-validation (relying on pre-computed Materials Project energies instead). The strongest portfolio framings are domain-transfer to a different binary chemistry (e.g., Mn-O echoing Seferos), or the cross-modality bridge between Articles 1 and 2 (molecular SMILES VAE vs. crystal voxel VAE on a unified evaluation table) — the latter is particularly strong because it explicitly traces the lineage of Sánchez-Lengeling's own PhD trajectory.

### Article 3: MOSES — A Standardized Benchmarking Platform for Molecular Generative Models

- **Full citation:** Polykovskiy, D., Zhebrak, A., **Sanchez-Lengeling, B.**, Golovanov, S., Tatanov, O., Belyaev, S., Kurbanov, R., Artamonov, A., Aladinskiy, V., Veselov, M., Kadurin, A., Johansson, S., Chen, H., Nikolenko, S., Aspuru-Guzik, A., & Zhavoronkov, A. (2020). Molecular Sets (MOSES): A benchmarking platform for molecular generation models. *Frontiers in Pharmacology*, 11, 565644. DOI: 10.3389/fphar.2020.565644 (originally arXiv:1811.12823, 2018)
- **Authorship caveat:** Sánchez-Lengeling is again a **middle author** (3rd of 16), still listed under his Harvard affiliation (Department of Chemistry and Chemical Biology). Per the explicit author contributions: *"DP and AZhe, BS-L, VA, MV, SJ, HC, SN, AA-G, AZha wrote the manuscript"* — i.e., his contribution was **manuscript writing**, not designing or conducting the experiments. Lead author Daniil Polykovskiy is at Insilico Medicine; co-corresponding authors are Polykovskiy + Zhavoronkov (Insilico) + Aspuru-Guzik (U Toronto) + Nikolenko (HSE / Neuromation). **Same pattern as Articles 1 and 2: middle-authored Harvard-PhD-era contribution to a project led by another group.** With three articles processed, the pattern is now unambiguous — none of the survey's data points come from his independent University of Toronto group.
- **Article type:** **Benchmark / platform paper** — not a new generative model, but the standardized *infrastructure* (dataset + metrics + baseline implementations + open-source package) for evaluating molecular generation models. Conceptually analogous to ImageNet for vision or GLUE for NLP, but for molecular generative ML.
- **Problem in plain English:** By 2018-2020 there were dozens of molecular generative models (VAEs, GANs, autoregressive RNNs, junction-tree VAEs, etc.), but no standardized way to compare them. Each paper used different datasets, different splits, and different evaluation metrics, making "our model is better than X" claims hard to verify. The paper provides everything needed for fair comparison: a curated dataset, train / test / scaffold-test splits, ten evaluation metrics with reference implementations, and reference implementations of seven baseline models. Released as `pip install molsets`.
- **ML task type:** **Primary ML infrastructure** (not a new model). The contribution is the benchmark itself + the implementations of the baselines + the metric library.
- **Input representation:**
  - Modality: SMILES strings, same as Article 1
  - Preprocessing: Filter ZINC Clean Leads (4.59M molecules) by molecular weight (250–350 Da), rotatable bonds (≤ 7), XlogP (≤ 3.5), removing charged atoms; apply custom medicinal-chemistry filters (MCF) and PAINS filters; final dataset = 1,936,962 molecules
- **Output representation:** Models are evaluated on ten metrics — **Validity** (RDKit-parseable), **Uniqueness@1k/@10k**, **Novelty** (fraction not in training set), **Filters** (fraction passing MCF + PAINS), **Fragment similarity (Frag)** (cosine similarity of BRICS-fragment frequencies), **Scaffold similarity (Scaf)** (cosine similarity of Bemis-Murcko-scaffold frequencies on Test and TestSF splits), **SNN** (Tanimoto similarity to nearest-neighbor on Morgan fingerprints), **Internal diversity (IntDiv1, IntDiv2)** (avg pairwise Tanimoto distance), **Fréchet ChemNet Distance (FCD)** (Fréchet distance in ChemNet activation space), and **property-distribution Wasserstein distances** for MW / logP / QED / SA
- **Model architecture (brief):** Multiple — the paper benchmarks **CharRNN** (character-level RNN on SMILES), **VAE** (SMILES-based), **AAE** (Adversarial Autoencoder), **JTN-VAE** (Junction Tree VAE on graphs), **LatentGAN** (latent-vector GAN on autoencoder embeddings), plus three non-neural baselines: **HMM**, **NGram**, and **Combinatorial** (BRICS fragment recombination). All implementations are PyTorch-based, hyperparameters tuned via random search, all bundled in the `molsets` package.
- **Dataset used in the paper:**
  - Name: **MOSES dataset** (derived from ZINC Clean Leads)
  - Size: 1,936,962 molecules total → 1,584,664 train / 176,075 random test / 176,226 scaffold test (TestSF — held-out scaffolds for assessing generalization to unseen chemotypes)
  - Source: ZINC Clean Leads + custom filtering pipeline (MCF + PAINS + atom-type restrictions)
  - Public? **Yes — the dataset, splits, metric implementations, baseline weights, and platform code are all open source.**
  - Link: https://github.com/molecularsets/moses + `pip install molsets`
- **Reported headline metric:** Across all metrics, **CharRNN performs best overall** (best FCD/Test = 0.073, best Frag = 1.0, best Scaff/Test = 0.924), suggesting that a simple character-level language model on SMILES is hard to beat for distribution-learning. **VAE has the highest SNN (0.626 vs. train baseline 0.642) but lowest novelty (0.695)** — strong overfitting indicator. **Combinatorial generator has the highest diversity** (IntDiv1 = 0.873). All neural models implicitly learn to avoid PAINS / MCF filters even though such restrictions were never explicitly imposed during training.
- **Public alternatives to replicate or approximate this:**
  - **Direct match — the platform itself:** https://github.com/molecularsets/moses + `pip install molsets` — match: **direct, perfect** — access: easy (one pip install)
  - **GuacaMol** (Brown et al. 2019) — competing benchmark for molecular generation, focuses on goal-directed tasks rather than distribution learning — match: complementary — access: easy (https://github.com/BenevolentAI/guacamol)
  - **MoleculeNet** (Wu et al. 2018) — benchmark for *predictive* molecular ML (regression / classification), not generative — match: orthogonal — access: easy (`pip install deepchem`)
- **Data access difficulty (for the paper's own dataset):** **Trivial** — `pip install molsets` and the dataset, splits, metrics, and baseline weights are all there. This is the lowest-friction data access in the entire survey.
- **Tooling I would need to learn:** PyTorch (carry-over from Articles 1 and 2), the `molsets` package (one new domain library, but well-documented and the metric implementations are fully abstracted), RDKit (carry-over). **Substantially less new tooling than Article 2** — the platform handles the evaluation harness for you.
- **Portfolio angle (distinct from Articles 1 and 2):** This article is **infrastructure**, not a model — which makes it the **easiest on-ramp** for a portfolio project that aims to be directly comparable to the published literature. Three viable framings:
  1. **Train one new model and report MOSES metrics vs. all baselines:** *"Implementing a transformer-based molecular generator and benchmarking with MOSES against CharRNN, VAE, JTN-VAE, and LatentGAN."* This is the highest-leverage angle — you build one model and the platform gives you a publication-grade comparison table for free.
  2. **Use MOSES as the evaluation harness for property-conditioned generation:** Train a generative model that conditions on QED or SA, evaluate the property-distribution Wasserstein metric to quantitatively show that conditioning works.
  3. **Bridge across the three Sánchez-Lengeling articles:** Use MOSES (Article 3) as the evaluation harness to compare a SMILES-VAE (Article 1's family) against a graph-based generator. Trace the lineage of the lab's generative-chemistry work in a single notebook.

#### Article 3 ranking

| Variable | Score (1–10) | One-line justification |
|---|---|---|
| Data availability | 10 | MOSES *is* the gold-standard public benchmark for molecular generative ML; train / test / scaffold-test splits all curated and released |
| Data access ease | 10 | `pip install molsets` installs the dataset, baselines, metrics, and trained weights — the lowest-friction data access in the entire survey |
| Task tractability on laptop / free GPU | 6 | Training a CharRNN baseline on 1.94M molecules takes hours on a free Colab GPU but is feasible; the platform's baselines are explicitly designed to be runnable; better than Article 1 because the harness is bundled |
| Tooling alignment with my current skills | 5 | PyTorch + RDKit (carry-over from Articles 1 and 2) + the `molsets` package (one new domain library, well-abstracted); substantially less new tooling than Article 2 |
| Problem clarity | 9 | Benchmark protocol is unambiguous — train on MOSES train, generate 30K, compute metrics with provided functions; the ten metrics are formally defined; very clean |
| Reproducibility signals | 10 | Entire platform is on GitHub + PyPI; baseline implementations included; trained weights bundled; hyperparameters documented |
| Portfolio impact | 9 | Using MOSES gives instant comparability with all published baselines — a "I trained model X and here are MOSES metrics vs. CharRNN, VAE, JTN-VAE, LatentGAN" framing is publication-grade evaluation that recruiters can verify directly against the literature |
| My personal interest | TBD | Awaiting student score |

**Article 3 overall score:** Provisional (7 of 8 variables): (10 + 10 + 6 + 5 + 9 + 10 + 9) / 7 = 59 / 7 = **8.4 / 10**

**Article 3 verdict:** This is the **highest-scoring article in the entire survey**, and rightly so — it's a standardized benchmark platform with a `pip install` interface, full code release, and the lowest-friction evaluation pipeline available for molecular generative ML. The strongest portfolio framing is *"train one new model and report MOSES metrics vs. all the published baselines"* — that gives you a publication-grade evaluation table for the cost of one model implementation, which is the best leverage in the entire survey for a portfolio project.

### Professor-level rollup

- **Article 1 overall:** 7.7 (provisional)
- **Article 2 overall:** 7.0 (provisional)
- **Article 3 overall:** 8.4 (provisional)
- **Professor overall** = (7.7 + 7.0 + 8.4) / 3 = 23.1 / 3 = **7.7 / 10** (provisional — all three articles are middle-authored PhD-era papers; awaiting personal-interest scores)

### Summary for this professor

- **Common thread across their work:** Only one article processed so far, and it's a co-authored PhD-era paper rather than independent-PI work. Apparent direction (per Article 1 + Sánchez-Lengeling's broader publication history at Harvard / Google Brain / Toronto): **machine learning for molecular and materials design** — generative models, latent-space optimization, property prediction across chemistry. **Verify the lab's current Toronto-era research direction by pulling first-author or corresponding-author papers from his independent group.**
- **Dominant data modality:** SMILES strings (sequence-based) and large public chemistry benchmarks (QM9, ZINC). Likely also graph representations and quantum-chemistry features in his more recent work — to be confirmed.
- **Biggest obstacle for me to replicate their work publicly:** **Not data** (Article 1 uses fully public datasets and releases its code) but **ML tooling step-up** — variational autoencoders + variational training + Bayesian optimization in latent space is a substantial complexity jump from the tabular RDKit-XGBoost pipelines used for prior professors.
- **Skills I would gain by working with them:** Variational autoencoders for chemistry; sequence-to-sequence learning on SMILES; latent-space Bayesian optimization with Gaussian processes; modern generative chemistry (likely also graph VAEs, transformer-based molecular language models, diffusion models for molecular design); rigorous benchmarking on QM9 / ZINC / MoleculeNet; PyTorch / TensorFlow at a more advanced level than tabular sklearn pipelines.

### Open items / notes

- Affiliation, department, and lab webpage for Prof. Sánchez-Lengeling at University of Toronto **not yet confirmed** — verify (likely Chemistry and/or Computer Science) before finalizing.
- **Critical gap:** Article 1 is a **middle-authored PhD-era paper from Harvard**, not first-author or corresponding-author work from his independent University of Toronto group. The portfolio rubric scores the *article's* potential; the *advisor* rubric should also weigh whether his current Toronto group does similar work. Before relying on this professor-level score, pull at least one first-author or corresponding-author paper from his independent group at U of T and re-rank.
- Personal interest score for Article 1 not yet provided by student.
- Article 1's score of 7.7 is substantially higher than any prior professor's (Beharry 6.4, Seferos 6.2). This is a real signal — primary ML papers with released code on gold-standard benchmarks score very differently from experimental chemistry papers that have to be creatively reframed for ML. Worth keeping in mind when comparing across the survey: the rubric is the same but the gap reflects genuine portfolio-feasibility differences between primary-ML and experimental-chemistry research outputs.
- Personal interest score for Article 2 not yet provided by student.
- **Pattern emerging across both articles processed:** Both are middle-authored PhD-era contributions from Sánchez-Lengeling's Harvard / Aspuru-Guzik-group years (Article 1 = 5th of 10 authors, Article 2 = 4th of 7 authors), *not* first-author or corresponding-author work from his independent University of Toronto group. The "critical gap" flagged after Article 1 is **stronger** after Article 2 — pulling at least one first-author or corresponding-author paper from his independent U of T group is the single most important next step before treating his professor-level score as representative of his current research direction.
- Article 2's score of 7.0 is lower than Article 1's 7.7 primarily because of harder ML tooling (3D CNNs + custom voxel pipeline + pymatgen) and harder compute (DFT post-validation realistically out of reach for a portfolio project). Both articles retain perfect-10 reproducibility because both author groups released code + data on GitHub.
- The Summary section's existing bullets reference "only one article processed so far" and "SMILES strings (sequence-based)" as the dominant modality — these are now slightly stale after Article 2 (which adds 3D voxel images of crystals as a second modality, and Materials Project as a second public benchmark). Left unchanged on purpose per the no-rewrite preference; flag this for possible rephrasing.
- Personal interest score for Article 3 not yet provided by student.
- **Pattern is now confirmed across all three articles processed:** All three are middle-authored PhD-era contributions from Sánchez-Lengeling's Harvard / Aspuru-Guzik-group years (Article 1 = 5th of 10, Article 2 = 4th of 7, Article 3 = 3rd of 16). Author contributions on Article 3 specifically: *"writing the manuscript"*. None of the survey's three data points come from his independent University of Toronto group. This remains the **single most important next step** for advisor evaluation: pull at least one first-author or corresponding-author paper from his independent U of T group and re-rank.
- Article 3's score of 8.4 is the **highest in the entire survey**. The boost reflects MOSES being a packaged benchmark platform (`pip install molsets`), not just a model — using MOSES gives publication-grade evaluation comparability for the cost of one model implementation. The rubric is treating this fairly: gold-standard public benchmarks with bundled baselines + reference implementations are genuinely the easiest possible portfolio target.
- Updated professor-level mean: (7.7 + 7.0 + 8.4) / 3 = **7.7 / 10** — Sánchez-Lengeling stays rank 1 over Beharry (6.4) and Seferos (6.2). The gap to Beharry has widened from 1.0 (after Article 2) to **1.3** after Article 3.
- Across the three articles, a coherent generative-chemistry trajectory emerges that the student could leverage as a single integrated portfolio narrative: Article 1 (foundational organic-molecule SMILES VAE) → Article 2 (extension to inorganic crystals via invertible 3D-voxel representation) → Article 3 (standardized benchmarking infrastructure for the whole field). A "bridge" project that ties all three together using MOSES as the evaluation harness is the single highest-leverage portfolio framing flagged in the survey so far.

---

## Seyed Mohamad Moosavi

**Affiliation:** University of Toronto (specific department to be confirmed — likely Chemistry and/or Chemical Engineering and Applied Chemistry, based on his research area). At the time of Article 1 below he was a postdoc / PhD student in Berend Smit's group at École Polytechnique Fédérale de Lausanne (EPFL), Switzerland.

**Lab focus (one sentence):** **Inferred** from Article 1 + Moosavi's broader publication trajectory (EPFL → U of T faculty): **machine learning and data-driven design for porous materials**, particularly metal–organic frameworks (MOFs) — including high-throughput computational screening, structural-motif identification ("adsorbaphores"), and ML-based property prediction for gas storage / separation / carbon capture applications. **Verify against the lab's current Toronto-era publications before treating this as authoritative.**

**Lab webpage / Scholar:** Not yet provided — needs to be sourced.

### Article 1: Data-Driven Design of MOFs for Wet Flue Gas CO₂ Capture (Adsorbaphore Identification)

- **Full citation:** Boyd, P. G., Chidambaram, A., García-Díez, E., Ireland, C. P., Daff, T. D., Bounds, R., Gładysiak, A., Schouwink, P., **Moosavi, S. M.**, Maroto-Valer, M. M., Reimer, J. A., Navarro, J. A. R., Woo, T. K., García, S., Stylianou, K. C., & Smit, B. (2019). Data-driven design of metal–organic frameworks for wet flue gas CO₂ capture. *Nature*, 576, 253–256. DOI: 10.1038/s41586-019-1798-7
- **Authorship caveat:** Moosavi is a **middle author** (9th of 16), listed under his EPFL affiliation (Laboratory of Molecular Simulation, Berend Smit's group) — **not from his current independent group at University of Toronto**. Per the explicit author contributions: *"S.M.M. carried out the similarity analysis"* — i.e., his specific contribution was the data-mining / clustering step that identified the three adsorbaphore motifs from the 106,680 CO₂-binding sites discovered by the screening pipeline. Senior corresponding author is Berend Smit (EPFL); the synthesis + experimental work was led by Stylianou (EPFL). **Same pattern as the three Sánchez-Lengeling articles processed earlier:** middle-authored pre-independent-PI contribution to a project led by another group. Pulling at least one first-author or corresponding-author paper from his independent U of T group remains the single most important next step for advisor evaluation.
- **Problem in plain English:** Carbon capture from flue gas is a key climate-mitigation technology. **Metal–organic frameworks (MOFs)** are highly tunable solid adsorbents that can selectively bind CO₂. But realistic flue gas contains water (H₂O), and water competes with CO₂ for the same binding sites in most MOFs — destroying the selectivity. Drying flue gas first is prohibitively expensive. **Can data mining of a huge computational MOF database identify binding-site motifs whose CO₂/N₂ selectivity survives in wet flue gas?**
- **ML / data-driven task type:** **Data-driven materials design** — high-throughput Monte Carlo screening + graph-similarity clustering for binding-site motif identification, followed by experimental synthesis of the predicted material. **Not modern deep learning** (no neural networks). The "similarity analysis" step (Moosavi's specific contribution) uses **maximum-clique detection on chemical graphs** to find common substructure motifs across 106,680 CO₂-binding sites — a classical computational-chemistry technique applied at scale. For ML reframing (portfolio purposes), the strongest angle is to use the released 325K-MOF database as training data for a neural property predictor, rather than reproducing the original adsorbaphore-identification pipeline.
- **Input representation:**
  - Modality: MOF crystal structures (CIF format) + computed adsorption properties (CO₂/N₂ Henry coefficients, working capacity, water Henry coefficient)
  - Preprocessing: Pull MOFs from Materials Cloud release; parse via pymatgen; for property-prediction reframe → compute MOF descriptors (porosity, surface area, pore size distribution, building-block fingerprints) or train a graph neural network directly on the MOF crystal graph
- **Output representation:** For the paper's actual task: identified binding-site motifs ("adsorbaphores") + predicted CO₂/N₂ selectivity in dry vs. wet conditions. For the ML reframe: regression target = CO₂/N₂ selectivity, CO₂ working capacity, or H₂O Henry coefficient.
- **Model architecture (brief):** Paper itself = no neural network. Pipeline = (i) ToBasCCo MOF-generation algorithm, (ii) RASPA Monte Carlo for adsorption isotherms, (iii) ABSL binding-site locator, (iv) maximum-clique-based graph similarity clustering (MCQD library), (v) DFT validation. For ML reframe: gradient-boosted regressor (XGBoost / LightGBM) on hand-engineered MOF descriptors, or graph neural network (PyTorch Geometric / DGL) on the MOF crystal graph.
- **Dataset used in the paper:**
  - Name: **Hypothetical 325K-MOF database** (constructed via the ToBasCCo MOF-generation algorithm) + computed adsorption properties (CO₂, N₂, H₂O at flue-gas conditions)
  - Size: 325,000 hypothetical MOFs; 8,325 top performers; 106,680 CO₂-binding sites identified
  - Source: ToBasCCo-generated structures + RASPA Monte Carlo simulations at EPFL
  - Public? **Yes — released on Materials Cloud with the computed adsorption data**
  - Link: https://doi.org/10.24435/materialscloud:2018.0016/v3 (data) + https://github.com/peteboyd/tobascco (MOF generation) + https://github.com/peteboyd/adsorbaphore (similarity analysis) + https://bitbucket.org/tdaff/automation (binding-site locator)
- **Reported headline metric:** 8,325 of 325,000 MOFs achieve CO₂ working capacity > 2 mmol g⁻¹ AND CO₂/N₂ selectivity > 50 (better than commercial zeolite 13X under dry conditions). Three adsorbaphore classes identified: **A1** (parallel aromatic rings ~7 Å apart, hydrophobic — preferred for wet flue gas, 31% of binding sites), **A2** (metal–oxygen bridges, 32%), **A3** (open metal sites, 21%). Two predicted MOFs (Al-PMOF, Al-PyrMOF) were synthesized and **maintain CO₂ capacity at 85% relative humidity, outperforming commercial zeolite 13X and activated carbon** in wet flue gas.
- **Public alternatives to replicate or approximate this:**
  - **Direct match — released code + data:** https://doi.org/10.24435/materialscloud:2018.0016/v3 + the three GitHub / BitBucket repos — match: **direct, perfect** — access: medium (multi-repo with a C library dependency for MCQD; not as `pip install`-bundled as Sánchez-Lengeling Article 3)
  - **CoRE-MOF database** — ~14K experimentally-realized MOFs with computed properties — match: **close** (alternative MOF dataset for the ML property-prediction reframe) — access: easy (https://mof.tech.northwestern.edu/)
  - **ARC-MOF / QMOF / MOFid** — additional curated MOF databases with quantum-chemistry properties — match: medium-to-close — access: easy (Zenodo / GitHub releases)
- **Data access difficulty (for the paper's own dataset):** **Easy-to-medium** — Materials Cloud is a direct download; the multi-repo code structure is moderate friction.
- **Tooling I would need to learn:** **pymatgen** for MOF parsing (carry-over from Sánchez-Lengeling Article 2), RDKit (carry-over), sklearn / XGBoost (familiar). For the ML property-prediction reframe: optionally **PyTorch Geometric** or **DGL** for a graph-neural-network baseline (new piece — substantial). For the original similarity-analysis pipeline: the **MCQD** C library + Python bindings (specialized, niche). Realistically out of reach for a portfolio: **RASPA Monte Carlo** for generating new adsorption data (hours per MOF on a CPU; you'd rely on the paper's pre-computed data instead).
- **Portfolio angle:** This is **data-driven materials design**, not deep ML — which means the strongest portfolio framings reframe the paper rather than reproducing it. Three options:
  1. **ML property prediction on the released 325K-MOF database:** *"Predicting CO₂/N₂ selectivity of metal–organic frameworks from structural descriptors — a benchmark on the EPFL 325K-MOF database."* Pull the released data, compute MOF descriptors (porosity, surface area, building-block fingerprints), train XGBoost vs. GNN baseline. Single notebook, predicted-vs-actual scatter, feature importance plot showing which structural features predict selectivity. **This is the highest-leverage angle** because it converts a Monte-Carlo / data-mining paper into a clean ML regression task with substantial public data.
  2. **Reproduce the adsorbaphore-identification pipeline on a different chemistry:** *"Apply the maximum-clique-based binding-site clustering to the H₂-storage subset of CoRE-MOF — what motifs emerge?"* More faithful to the paper but more niche; requires the MCQD C library.
  3. **Bridge to Sánchez-Lengeling's iMatGen (Article 2 of his section):** Use the 325K-MOF database as training data for a generative model for MOFs (analogous to iMatGen for V-O crystals but for the broader MOF design space). Strong story because it explicitly connects two faculty candidates' research areas.

#### Article 1 ranking

| Variable | Score (1–10) | One-line justification |
|---|---|---|
| Data availability | 9 | 325K-MOF database publicly released on Materials Cloud; not as universal a benchmark as Sánchez-Lengeling's QM9 / ZINC / MOSES, but a strong domain-specific dataset for MOF ML |
| Data access ease | 8 | Materials Cloud direct download + pymatgen-compatible CIF format; only friction is the multi-repo code structure for the original adsorbaphore-identification pipeline |
| Task tractability on laptop / free GPU | 5 | Training an ML property predictor on 325K MOFs is feasible on free Colab; reproducing the full Monte Carlo screening pipeline is realistically out of reach (RASPA / GCMC needs hours per material) |
| Tooling alignment with my current skills | 5 | pymatgen + RDKit + sklearn / XGBoost is a familiar core; the GNN angle (PyTorch Geometric / DGL) adds one new piece; the original maximum-clique-graph-similarity pipeline (MCQD C library) is a specialized, niche dependency |
| Problem clarity | 7 | Predicting CO₂/N₂ selectivity from MOF structure is a clean regression task; the original adsorbaphore-identification framing is more open-ended; the ML reframe is the cleaner version |
| Reproducibility signals | 9 | Code AND data both released across multiple repos with the C library dependency for MCQD; slightly lower than Sánchez-Lengeling's perfect 10 because nothing is `pip install`-bundled |
| Portfolio impact | 8 | Climate-tech / carbon-capture is highly recruiter-legible across cleantech, energy, and materials-AI roles; MOF property prediction is a recognized ML application area |
| My personal interest | TBD | Awaiting student score |

**Article 1 overall score:** Provisional (7 of 8 variables): (9 + 8 + 5 + 5 + 7 + 9 + 8) / 7 = 51 / 7 = **7.3 / 10**

**Article 1 verdict:** A **highly credible** portfolio project is achievable using the released 325K-MOF database as training data for an ML property predictor (XGBoost on MOF descriptors, or GNN on the MOF crystal graph) — this reframes the paper's data-mining contribution into a clean ML regression task with substantial public data. The strongest portfolio framing is the property-prediction reframe rather than reproducing the original adsorbaphore-identification pipeline (which uses specialized graph-similarity tooling that's harder to scale into a portfolio narrative). Climate-tech / carbon-capture framing makes the project highly recruiter-legible.

### Professor-level rollup

- **Article 1 overall:** 7.3 (provisional)
- **Professor overall** = 7.3 / 1 = **7.3 / 10** (provisional — single middle-authored EPFL-postdoc-era paper; awaiting personal-interest score)

### Summary for this professor

- **Common thread across their work:** Only one article processed so far, and like the three Sánchez-Lengeling articles it's a co-authored pre-independent-PI paper rather than independent-faculty work. Apparent direction (per Article 1 + Moosavi's broader publication trajectory at EPFL → U of T): **machine learning and data-driven design for porous materials**, particularly MOFs — high-throughput screening, structural-motif identification, ML-based property prediction for gas storage / separation / carbon-capture applications. **Verify the lab's current Toronto-era research direction by pulling first-author or corresponding-author papers from his independent group.**
- **Dominant data modality:** MOF crystal structures (CIF format) + computed adsorption properties from large hypothetical-materials databases (Materials Cloud, CoRE-MOF, QMOF, etc.). Likely also graph representations and quantum-chemistry features in his more recent work — to be confirmed.
- **Biggest obstacle for me to replicate their work publicly:** **Not data** (Article 1 uses a fully released 325K-MOF database) but **the original methodology relies on specialized non-ML tooling** (RASPA Monte Carlo, MCQD C library for graph clique detection) that's harder to scale into a portfolio than standard PyTorch / sklearn pipelines. The portfolio reframe (ML property prediction) sidesteps this.
- **Skills I would gain by working with them:** MOF structural representation and parsing (pymatgen), high-throughput computational screening of porous materials, graph-similarity / motif-identification methods (maximum clique detection), Monte Carlo adsorption simulations (RASPA), ML property prediction for materials (sklearn / GNN), and the broader carbon-capture / gas-separation application domain.

### Open items / notes

- Affiliation, department, and lab webpage for Prof. Moosavi at University of Toronto **not yet confirmed** — verify (likely Chemistry and/or Chemical Engineering and Applied Chemistry) before finalizing.
- **Critical gap:** Article 1 is a **middle-authored EPFL-postdoc-era paper** (9th of 16, contribution = "carried out the similarity analysis"), **not first-author or corresponding-author work** from his independent University of Toronto group. **Same pattern as the three Sánchez-Lengeling articles.** Before relying on this professor-level score, pull at least one first-author or corresponding-author paper from his independent group at U of T and re-rank.
- Personal interest score for Article 1 not yet provided by student.
- Article 1's score of 7.3 places Moosavi at rank 2 (between Sánchez-Lengeling at 7.7 and Beharry at 6.4). The data-driven materials-design framing is genuinely closer to "ML for chemistry" than the experimental chemistry papers from Beharry / Seferos, but less directly an ML paper than Sánchez-Lengeling's three.
- **Cross-professor synergy worth flagging:** The 325K-MOF database from Article 1 + Sánchez-Lengeling's iMatGen (Article 2 of his section, V-O voxel VAE) + Sánchez-Lengeling's MOSES (Article 3, generative-chemistry benchmarking) form a natural triple for a portfolio project on **generative models for MOFs evaluated with MOSES-style benchmarks**. This is a stronger cross-faculty portfolio narrative than any single-professor angle in the survey so far.

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

### Article 3: Tellurophene-Appended BODIPY Photosensitizers for Photodynamic Therapy

- **Full citation:** Campbell, J. W., Tung, M. T., Taylor, B. B., Beharry, A. A., & Thompson, A. (2024). A series of potent BODIPY photosensitisers featuring tellurophene motifs at boron. *Organic & Biomolecular Chemistry*, 22(20), 4157–4162. DOI: 10.1039/d4ob00546e
- **Article type:** Primary research article in collaboration with Alison Thompson's group at Dalhousie. Beharry is co-corresponding author; the Beharry-lab student (Tung) handled photophysics and HeLa cell-toxicity measurements while the Thompson group handled synthesis.
- **Problem in plain English:** Photodynamic therapy (PDT) kills cancer cells by exciting a photosensitizer (PS) with light, generating cytotoxic singlet oxygen (¹O₂). BODIPY dyes are popular PS scaffolds because they're tunable, but their natural state favors fluorescence over the triplet state required for ¹O₂ generation. Adding a heavy atom near the chromophore promotes intersystem crossing → triplet → ¹O₂. Halogens work; this paper uses **tellurium** (heavier than the halogens, plus eight stable isotopes that double as a mass-cytometry mass label) appended via tellurophene at the BODIPY boron center. The paper expands the synthetic scope across 11 BODIPY substituents and reports an 8-fold improvement in light IC₅₀ vs prior tellurophene-BODIPY designs.
- **ML task type:** No ML in the paper. Reframed as **structure → photophysics regression** on a tiny dataset: tabular molecular descriptors of [Te]-BODIPY analogues → singlet-oxygen quantum yield (ΦΔ), light IC₅₀ (μM), or phototoxicity index. Genuinely distinct from Article 1 (antimicrobial MIC) and Article 2 (clinical-cohort survival).
- **Input representation:**
  - Modality: Tabular molecular descriptors and Morgan fingerprints from SMILES; same RDKit-based stack as Article 1.
  - Preprocessing: Hand-curated SMILES for each BODIPY analogue (Tables 1 and 3 of the paper); RDKit descriptor + fingerprint computation. Optional domain-knowledge feature: presence and proximity of a heavy atom relative to the chromophore.
- **Output representation:** Singlet-oxygen quantum yield ΦΔ (primary, distinct from prior articles' targets); light IC₅₀ (μM) for HeLa as a secondary target; phototoxicity index (light IC₅₀ / dark IC₅₀) as a stretch-goal target.
- **Model architecture (brief):** Tiny-data tabular regression — ridge as the linear baseline, XGBoost as the non-linear comparator. With ~9 in-paper compounds + literature-mined BODIPY photosensitizers (~30–50 total), the model is small but interpretable.
- **Dataset used in the paper:**
  - Name: In-paper Table 1 (synthetic scope, 11 successful + 6 failed substrates 4j–4o) and Table 3 (photophysical + cell-viability data for the 9 successful compounds).
  - Size: 9 [Te]-BODIPYs with ΦΔ, ε, λ_max, dark IC₅₀, light IC₅₀, and phototoxicity index reported.
  - Source: Wet-lab synthesis at Dalhousie University; photophysics (DPBF / Eosin Y standard) and HeLa MTT assays at University of Toronto Mississauga.
  - Public? The data itself is published in Table 3 of the paper (unusually transparent vs Articles 1 and 2), but no separate machine-readable release.
  - Link: Table 3 of the paper / ESI; no separate dataset URL.
- **Reported headline metric:** ΦΔ values 0.17–0.85 across the series; light IC₅₀ as low as 5 nM (compounds 4g and 4h); phototoxicity index up to >2640. Under similar HeLa conditions, **clinical photosensitizers Photofrin (PI > 4.3) and Hypericin (PI > 12) are dramatically less potent** — the [Te]-BODIPYs are competitive with state-of-the-art clinical PSs.
- **Public alternatives to replicate or approximate this:**
  - Hand-curated Table 3 from this paper — 9 compounds — match: **direct** — access: medium (PDF table extraction + SMILES drawing for each analogue)
  - Literature BODIPY photophysics from Loudet & Burgess *Chem. Rev.* 2007 and the photosensitizer papers cited in refs 4–14 / 25–28 of this article — extends the dataset to ~30–50 compounds — match: **medium-to-close** — access: hard (manual literature mining; no aggregated public release)
  - PhotochemCAD — small-molecule photophysics database — match: **loose** (broader chromophore set; few BODIPYs specifically) — access: easy (https://www.photochemcad.com/)
- **Data access difficulty (for the paper's own dataset):** Medium — Table 3 is published directly in the paper, but it requires hand-curation (text extraction + SMILES drawing for each analogue). No machine-readable release.
- **Tooling I would need to learn:** Same RDKit + sklearn / XGBoost stack as Article 1 — no new ML tooling required for the headline angle. Optional for a stretch goal: descriptor-based predictors of singlet-oxygen quantum yield from quantum-chemistry literature, or a small TDDFT calculation pipeline if extending beyond pure ML descriptors.
- **Portfolio angle (distinct from Articles 1 and 2):** Headline: *"Predicting singlet-oxygen quantum yield (ΦΔ) of BODIPY photosensitizers from molecular structure — a tiny-data photophysics ML benchmark."* Hand-curate Table 3 (9 compounds), then extend via Loudet & Burgess and the cited photophysics papers (~30–50 compounds total). Featurize with RDKit. Train ridge vs XGBoost on ΦΔ. Single notebook with predicted-vs-actual scatter, a learning curve as the dataset is extended via literature mining, and a feature-importance plot. **Stretch:** binary "successful synthesis" classifier on the failed substrates 4j–4o to predict which BODIPY backbones tolerate the nucleophilic-substitution route. README's headline frames the project as "medicinal chemistry SAR on extreme small-data — when literature mining IS the data pipeline."

#### Article 3 ranking

| Variable | Score (1–10) | One-line justification |
|---|---|---|
| Data availability | 5 | ~9 compounds in this paper directly; extendable to ~30–50 via literature mining of cited photophysics papers; no aggregated public BODIPY photophysics dataset |
| Data access ease | 6 | Paper Table 3 is hand-curatable; literature mining is moderate manual effort; no API or download for an aggregated source |
| Task tractability on laptop / free GPU | 8 | Tiny tabular regression; trains in seconds on CPU |
| Tooling alignment with my current skills | 5 | Same RDKit + sklearn / XGBoost stack as Article 1; no new ML tooling required |
| Problem clarity | 7 | ΦΔ and phototoxicity index are well-defined, standardized targets; main ambiguity is which solvent / irradiation condition to standardize on across literature sources |
| Reproducibility signals | 5 | Table 3 publishes the full photophysical + cell-viability dataset directly in the paper — unusually transparent vs Articles 1 and 2; no machine-readable release though |
| Portfolio impact | 7 | Photodynamic therapy + cancer is recruiter-legible; tiny-data + literature-mining narrative adds story richness; somewhat narrower than precision oncology (Article 2) or antibiotic resistance (Article 1) |
| My personal interest | TBD | Awaiting student score |

**Article 3 overall score:** Provisional (7 of 8 variables): (5 + 6 + 8 + 5 + 7 + 5 + 7) / 7 = 43 / 7 = **6.1 / 10**

**Article 3 verdict:** A credible — and meaningfully different — GitHub project is achievable by hand-curating Table 3 and extending via literature mining of cited BODIPY photophysics papers, then training a tiny-data regressor for ΦΔ. The data scarcity is genuine but is itself the point: this is the "medicinal chemistry SAR on extreme small-data" story rather than another large-cohort regression. Distinct from Article 1 (antimicrobial MIC) and Article 2 (clinical-cohort survival), giving Beharry's portfolio three genuinely different ML angles.

### Professor-level rollup

- **Article 1 overall:** 6.4 (provisional)
- **Article 2 overall:** 6.7 (provisional)
- **Article 3 overall:** 6.1 (provisional)
- **Professor overall** = (6.4 + 6.7 + 6.1) / 3 = 19.2 / 3 = **6.4 / 10** (provisional — all three articles awaiting personal-interest scores)

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
- Personal interest score for Article 3 not yet provided by student.
- Article 3 dataset is extreme tiny-data (~9 in-paper, ~30–50 with literature mining) — confirm willingness to do the manual literature curation as part of the project before committing to the photophysics angle.
- The above "two Beharry articles point to two ML toolchains" bullet now slightly understates the picture: Article 3 adds a third toolchain flavor (small-data SAR with literature-mined data extension). The earlier bullet has been left unchanged on purpose — flag this to the user for possible rephrasing.

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
