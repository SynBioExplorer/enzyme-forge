# Technical Requirements: Computational Enzyme Engineering Pipeline for Metabolic Pathway Optimization

**Version:** 0.2 — Revised
**Date:** March 2026
**Author:** Felix — Australian Genome Foundry, Macquarie University

**Changelog (v0.2):** Upgraded RFdiffusion → RFdiffusion3 (10× faster, all-atom). Added product inhibition screening, zero-shot fitness ensemble, MD simulations (optional), N-glycosylation screening. Replaced Chai-1 with Protenix-v1. Softened pLDDT filtering (evidence: AUC only 0.60 for design success). Replaced arbitrary composite scoring with Pareto-front ranking. Added few-shot active learning for Round 2 DBTL. Fixed MMseqs2 license (now MIT). Noted ESMFold repo archived. **License audit:** All core pipeline tools are commercially usable (MIT/BSD/Apache 2.0). Non-commercial tools (CLEAN, EVOLVEpro) flagged with open-source alternatives.

---

## 1. Project scope

A four-enzyme metabolic pathway (A → B → C → D) expressed in *Saccharomyces cerevisiae* produces a target compound. The goal is to maximize pathway throughput by computationally engineering each enzyme for improved catalytic efficiency, stability, and expression, then experimentally validating top candidates at the AGF.

The pipeline chains structure-based generative protein design tools into an end-to-end Design-Build-Test-Learn (DBTL) workflow, running entirely on a single GPU workstation.

**Target enzymes:**

| Enzyme | Length (aa) | Pathway position | Substrate → Product |
|--------|------------|-----------------|---------------------|
| A | 428 | Entry | Substrate₁ → Intermediate₁ |
| B | 512 | Step 2 | Intermediate₁ → Intermediate₂ |
| C | 348 | Step 3 | Intermediate₂ → Intermediate₃ |
| D | 295 | Terminal | Intermediate₃ → Final product |

---

## 2. Hardware

**Primary workstation:**

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 6000 Pro, 96 GB GDDR7, Blackwell architecture |
| CPU | Intel Core i9, 24 cores |
| RAM | 128 GB DDR5 |
| Storage | ≥2 TB NVMe SSD (databases + intermediate files) |
| OS | Ubuntu 22.04 or 24.04 LTS |

**Storage breakdown estimate:**

| Item | Size |
|------|------|
| ColabFoldDB (UniRef30 + ColabFoldDB) | ~1.2 TB |
| MMseqs2-GPU indexed databases | ~200 GB |
| Boltz-2 model weights | ~10 GB |
| RFdiffusion3 weights (via Foundry) | ~5 GB |
| LigandMPNN weights | ~1 GB |
| ESMFold weights | ~6 GB |
| Protenix-v1 weights | ~5 GB |
| CatPred model + data | ~5 GB |
| MD trajectories (per campaign round, optional) | ~50–100 GB |
| Pipeline outputs (per campaign round) | ~50–100 GB |
| **Total recommended** | **≥2 TB** |

**Critical VRAM constraints:** The 96 GB VRAM accommodates all pipeline stages for proteins up to ~950 aa. The longest target enzyme (B, 512 aa) is well within limits for Boltz-2 complex prediction (~40–60 GB peak), RFdiffusion3 partial diffusion (~20–30 GB), Protenix-v1 (~40–60 GB), and LigandMPNN (~8–12 GB).

---

## 3. Software stack

### 3.1 Core pipeline tools

| Tool | Version | Purpose | License | Source |
|------|---------|---------|---------|--------|
| **Boltz-2** | ≥2.2.1 | Structure prediction + binding affinity scoring | MIT | [github.com/jwohlwend/boltz](https://github.com/jwohlwend/boltz) |
| **RFdiffusion3** | latest (via Foundry ≥0.1.9) | All-atom backbone diversification (partial diffusion) | BSD-3 | [github.com/RosettaCommons/foundry](https://github.com/RosettaCommons/foundry) (under `models/rfd3/`) |
| **LigandMPNN** | latest | Ligand-aware inverse folding / sequence design | MIT | [github.com/dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN) |
| **ProteinMPNN** | v1.0.1 | Inverse folding (protein-only contexts) | MIT | [github.com/dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) |
| **ESMFold** | 2.0 | Rapid structure pre-screening | MIT | [github.com/facebookresearch/esm](https://github.com/facebookresearch/esm) (archived Aug 2024, still functional) |
| **MMseqs2-GPU** | ≥Release 18 | GPU-accelerated homology search / MSA generation | MIT | [github.com/soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2) |
| **ColabFold** | ≥1.6.0 | MSA pipeline wrapper | MIT | [github.com/sokrypton/ColabFold](https://github.com/sokrypton/ColabFold) |
| **Protenix-v1** | ≥1.0.5 | Cross-validation of structure predictions (replaces Chai-1) | Apache 2.0 | [github.com/bytedance/Protenix](https://github.com/bytedance/Protenix) |

**Note on RFdiffusion3:** RFdiffusion3 replaces both RFdiffusion (v1) and RFdiffusion2. It uses an all-atom representation (individual atoms as fundamental units), outperforms RFdiffusion2 on 37/41 active-site benchmarks (90%), and is 10× faster. In experimental validation, 35/190 designs (18%) showed catalytic activity (Watson et al., 2025). Installed via the Rosetta Foundry CLI (`foundry install`).

**Note on Protenix-v1:** Protenix-v1 (ByteDance, Feb 2026) is the first fully open-source model to outperform AlphaFold3 on several structure prediction benchmarks. It replaces Chai-1 for cross-validation in Stage 7. Apache 2.0 license allows commercial use.

### 3.2 Auxiliary tools

| Tool | Purpose | License | Source |
|------|---------|---------|--------|
| **CatPred** | Enzyme kinetics prediction (kcat, Km, Ki) for variant ranking | MIT | [github.com/maranasgroup/CatPred](https://github.com/maranasgroup/CatPred) |
| **RealKcat** | Mutation-aware kcat prediction (complements CatPred) | MIT | [github.com/TKAI-LAB-Mali/RealKcat](https://github.com/TKAI-LAB-Mali/RealKcat) |
| **UniKP** | Alternative kinetics prediction for ensemble scoring | GPL-3.0 | [github.com/Luo-SynBioLab/UniKP](https://github.com/Luo-SynBioLab/UniKP) |
| **CLEAN** | EC number assignment and functional annotation | Research use only ⚠️ | [github.com/tttianhao/CLEAN](https://github.com/tttianhao/CLEAN) — non-commercial; use InterProScan or DeepEC (MIT) as commercial-friendly alternative |
| **COBRApy** | Flux balance analysis with yeast-GEM for bottleneck ID | GPL-2.0 / LGPL-2.0 | [github.com/opencobra/cobrapy](https://github.com/opencobra/cobrapy) |
| **IQ-TREE 3** | Phylogenetic tree construction for ASR | GPL-2.0 | [github.com/iqtree/iqtree3](https://github.com/iqtree/iqtree3) |
| **MAFFT** | Multiple sequence alignment for ASR | BSD | [mafft.cbrc.jp](https://mafft.cbrc.jp/alignment/software/) |
| **GRASP** | Ancestral sequence reconstruction | AGPL-3.0 | [github.com/bodenlab/GRASP](https://github.com/bodenlab/GRASP) |
| **FireProt-ASR** | Automated ASR from a single query sequence | Web service | [loschmidt.chemi.muni.cz/fireprotasr](https://loschmidt.chemi.muni.cz/fireprotasr/) |
| **PyMOL / ChimeraX** | Structure visualization and manual inspection | — | — |
| **TranceptEVE** | Zero-shot fitness prediction (MSA + autoregressive) | MIT | [github.com/OATML-Markslab/Tranception](https://github.com/OATML-Markslab/Tranception) + [EVE](https://github.com/OATML-Markslab/EVE) |
| **GEMME** | Zero-shot fitness prediction (MSA-based, fast) | MIT | [lcqb.upmc.fr/GEMME](https://www.lcqb.upmc.fr/GEMME/) or [Docker](https://hub.docker.com/r/elodielaine/gemme) |
| **OpenMM** | GPU-accelerated MD simulations (optional Stage 6.5) | MIT / LGPL | [github.com/openmm/openmm](https://github.com/openmm/openmm) |
| **N-glycosylation scan** | N-glycosylation site prediction | N/A (regex) | Simple Asn-Xaa-Ser/Thr sequon pattern matching (no external tool required; NetNGlyc is proprietary) |
| **PLACER** | Active site preorganization scoring (GNN) | BSD | [github.com/baker-laboratory/PLACER](https://github.com/baker-laboratory/PLACER) |
| **EVOLVEpro** | Few-shot active learning for Round 2+ optimization | Non-commercial ⚠️ | [github.com/mat10d/EvolvePro](https://github.com/mat10d/EvolvePro) — academic use only; for commercial use, implement equivalent few-shot active learning with scikit-learn GP regression + ESM-2 embeddings (MIT) |

**Pending tools (code not yet released):**
- **ResiDPO-LigandMPNN** (Baker Lab, arXiv 2506.00297): Fine-tuned LigandMPNN achieving 17.6% designability success vs 6.6% for base LigandMPNN (~3× improvement). Add to Stage 4 when code is released.
- **AiCE** (Cell, July 2025): Inverse folding with structural + evolutionary constraints. Outperforms other AI methods by 36-90% on 60 DMS benchmarks. Contact authors for code.

### 3.3 Environment and dependencies

| Dependency | Version |
|-----------|---------|
| CUDA | ≥12.4 (13.0 for TensorRT-optimized inference) |
| Python | 3.10–3.11 |
| PyTorch | ≥2.2 |
| TensorRT | ≥10.13 (for OpenFold-TRT acceleration) |
| JAX | ≥0.4.20 (for ColabFold, AlphaFold baselines) |
| Conda / Mamba | Package management |

Each tool should be installed in an isolated conda environment to avoid dependency conflicts. A `Snakemake` or `Nextflow` workflow manager orchestrates the pipeline stages.

---

## 4. Pipeline architecture

### 4.1 Stage 0 — Bottleneck identification (CPU-bound)

**Objective:** Determine which enzyme(s) exert the most flux control over pathway output before committing GPU time to design.

**Method:**
1. Construct a kinetic ODE model of the four-enzyme pathway using literature or predicted kinetic parameters.
2. Predict kcat and Km for all four wild-type enzymes using CatPred (primary) and UniKP (secondary).
3. Run flux balance analysis with COBRApy using yeast-GEM to estimate intracellular flux distributions.
4. Calculate flux control coefficients (FCCs) by perturbing each enzyme's Vmax ±10% and measuring pathway output sensitivity.

**Output:** Ranked priority list of enzymes by FCC. Enzymes with FCC > 0.3 are primary computational engineering targets. Enzymes with FCC < 0.1 may benefit more from promoter/expression optimization than protein engineering.

**Compute:** ~1 day CPU time. No GPU required.

### 4.2 Stage 1 — MSA generation (GPU-bound)

**Objective:** Generate multiple sequence alignments for all target enzymes.

**Method:** ColabFold-search with MMseqs2-GPU backend searching UniRef30 2023_02 + ColabFoldDB 2021_08.

**Parameters:**
- Sensitivity: default ColabFold-search settings (cascaded search)
- Database: pre-indexed for GPU on local NVMe

**Output:** MSA files (.a3m) for each enzyme.

**Compute:** 4 sequences × ~10s = ~40 seconds total.

### 4.3 Stage 2 — Wild-type structure prediction (GPU-bound)

**Objective:** Predict holo structures of all four enzymes complexed with substrates, products, and cofactors.

**Method:** Boltz-2 with ligand SMILES input.

**Parameters:**
- Recycling iterations: 10 (default)
- Number of seeds: 5 per complex (select by highest confidence)
- Input format: FASTA + ligand SMILES + cofactor definitions
- **Predict complexes with both substrate AND product SMILES** (product structures needed for Stage 6 product inhibition screening)

**Output:** PDB files of enzyme-substrate and enzyme-product complexes with confidence metrics (pLDDT, iPTM, PAE). These serve as the starting scaffolds for all downstream design.

**Compute:** ~20 complexes × 60s = ~20 minutes (doubled from v0.1 to include product complexes).

### 4.4 Stage 3 — Backbone diversification with RFdiffusion3 (GPU-bound)

**Objective:** Generate diverse backbone conformations around the active site while preserving the global fold.

**Method:** RFdiffusion3 partial diffusion via Rosetta Foundry. RFdiffusion3 uses an all-atom representation, treating individual atoms as fundamental units. It handles proteins, small molecules, metals, and cofactors in a unified framework.

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `partial_T` | 10–20 (of 50 total timesteps) | 10 for conservative loop remodeling; 20 for aggressive diversification |
| `contigs` | Define fixed core + designable regions per enzyme | Preserve catalytic residues and key secondary structure |
| `hotspot_residues` | Active site residues (enzyme-specific) | Focus sampling around functionally relevant regions |
| Designs per enzyme | 200–500 | Balance diversity vs compute time |

**Design strategy per enzyme:**
- **High-FCC enzymes** (primary targets): 500 backbones at partial_T=15, plus 100 at partial_T=20 for aggressive exploration.
- **Low-FCC enzymes** (secondary targets): 200 backbones at partial_T=10, conservative redesign only.

**Output:** PDB backbone files with all-atom coordinates. Pre-filter by radius of gyration, ligand SASA, and active site pocket volume before proceeding to sequence design.

**Compute:** ~1,600 backbones × ~15s average = **~7 hours** (10× faster than RFdiffusion v1).

### 4.5 Stage 4 — Sequence design with LigandMPNN (GPU-bound)

**Objective:** Design amino acid sequences for each backbone that are compatible with the substrate-bound active site geometry.

**Method:** LigandMPNN with ligand context conditioning. When ResiDPO-LigandMPNN code is released, run it as a secondary pass on all backbones and merge outputs.

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--temperature` | 0.1 (primary), 0.2 (exploratory) | 0.1 for high-confidence recovery; 0.2 for broader diversity |
| `--number_of_batches` | 8–16 per backbone | Standard sampling depth |
| `--fixed_residues` | Catalytic residues (enzyme-specific) | Preserve catalytic mechanism |
| `--pack_side_chains` | 1 | Enable side-chain packing |
| `--pack_with_ligand_context` | 1 | Account for substrate during packing |
| `--omit_aa` | "C" (optional) | Suppress free cysteines to avoid misfolding in yeast cytoplasm |

**Parallel run — Direct WT diversification (Strategy B):**
Run LigandMPNN directly on wild-type Boltz-2 structures at T=0.1, 0.2, and 0.4, generating 250 sequences per enzyme per temperature. This produces 3,000 conservative variants as a baseline library.

**Output:** FASTA files of designed sequences with per-residue confidence scores.

**Compute:**
- RFdiffusion3 backbones: 1,600 × 10 seqs = 16,000 sequences × 3s = ~13 hours
- WT direct: 3,000 sequences × 3s = ~2.5 hours
- **Total: ~15.5 hours**

### 4.6 Stage 4.5 — Zero-shot fitness ensemble (CPU/GPU-bound)

**Objective:** Score all designed sequences for evolutionary plausibility using an orthogonal signal independent of structural metrics.

**Method:** Ensemble of three zero-shot fitness predictors, each capturing different aspects of sequence fitness:
1. **TranceptEVE L** — autoregressive + EVE variational model (best overall on ProteinGym; Spearman ~0.46-0.50 with experimental fitness)
2. **ESM-2 log-likelihood** — protein language model pseudo-likelihood
3. **GEMME** — MSA-based evolutionary conservation (fast, strong on non-human proteins)

**Rationale:** The ProteinGym benchmark (Notin et al., NeurIPS 2023) shows that ensembles across modalities outperform single large models, and scaling pLMs beyond 1-4B parameters does not improve zero-shot fitness prediction.

**Output:** Per-sequence fitness scores from each predictor. Averaged ensemble score used as a soft weighting factor in Stage 7 composite ranking (NOT a hard filter).

**Compute:** ~30 minutes for ~19,000 sequences.

### 4.7 Stage 5 — Pre-screening with ESMFold (GPU-bound)

**Objective:** Rapidly eliminate sequences that are clearly misfolded.

**Method:** Predict structures with ESMFold (no MSA required, ~1s per sequence).

**Filtering (softened from v0.1):**
- **Hard floor:** Discard sequences with mean pLDDT < 60 (catch only clearly misfolded designs)
- **Soft signal:** Record ESMFold pLDDT as a continuous score for Stage 7 ranking
- **Hard filter:** Discard sequences with Cα RMSD > 5 Å to WT (significant topology deviation)

**Rationale for softened pLDDT filtering:** Garcia et al. (*Protein Science* 2025) showed that AF2/Boltz pLDDT has AUC of only 0.60 for discriminating design success from failure — barely better than random. ESMFold pLDDT is a better discriminator (AUC 0.72) but still insufficient as a hard gate. Active sites are often in dynamic regions where pLDDT is inherently low; hard cutoffs bias against catalytically competent designs. The previous threshold of pLDDT > 70 has been lowered to > 60 to retain more potentially functional designs.

**Output:** Filtered set with continuous pLDDT scores, typically ~60–70% of input retained (more retained due to softened filter).

**Compute:** ~19,000 sequences × 1s = ~5.3 hours.

### 4.8 Stage 6 — Full structural screening with Boltz-2 (GPU-bound)

**Objective:** Predict enzyme-substrate complex structures, binding affinity, and product inhibition potential for all surviving variants.

**Method:** Boltz-2 with substrate SMILES and cofactor context. Use the affinity prediction head for ranking. **Additionally, run each candidate with product SMILES to assess product inhibition risk.**

**Structural filtering cascade:**

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Cα RMSD to WT | < 2.0 Å | Fold preservation |
| Active site heavy-atom RMSD | < 1.5 Å | Catalytic geometry |
| Ligand RMSD | < 2.0 Å | Substrate positioning |

**Continuous scores (passed to Stage 7 ranking, NOT hard-filtered):**

| Metric | Purpose |
|--------|---------|
| Boltz-2 pLDDT | Overall fold confidence (used as score, not gate) |
| Boltz-2 substrate affinity score | Substrate binding strength |
| Boltz-2 product affinity score | Product binding strength (for inhibition ratio) |
| LigandMPNN confidence | Sequence-structure compatibility |

**Product inhibition screening (new in v0.2):** For each candidate, calculate the ratio of substrate affinity to product affinity. Variants where product binds more tightly than substrate are flagged as high product inhibition risk. CatPred Ki predictions are run on top 200 candidates as additional validation.

**Output:** Ranked list of designs passing structural thresholds, with continuous scores for downstream ranking.

**Compute:** ~12,000 surviving sequences × 18s × 2 (substrate + product) = ~120 hours (~5 days). Can be reduced by running product docking only on top 50th percentile of substrate results (~2.5 days for product pass).

### 4.9 Stage 6.5 — Short MD simulations (GPU-bound, OPTIONAL)

**Objective:** Filter for catalytically relevant conformational dynamics. Addresses the central failure mode: designs that fold correctly but lack catalytic activity due to incorrect dynamics.

**Method:** Run 10–50 ns molecular dynamics simulations on top ~200 candidates (50 per enzyme) using OpenMM with AMBER ff14SB or CHARMM36m force fields. GPU-accelerated MD on the RTX 6000 Pro achieves >1 ns/day for enzyme-sized systems.

**Filtering criteria:**

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Active site catalytic residue RMSD | < 2.0 Å (over trajectory) | Catalytic geometry stability |
| Substrate RMSD in pocket | < 3.0 Å | Substrate retention (doesn't drift out) |
| Backbone RMSD plateau | < 3.0 Å | Global fold stability (doesn't unfold) |
| Active site RMSF | Lower is better (scored) | Preorganization for catalysis |

**Rationale:** The GRACE workflow (Hu et al., *ACS Synthetic Biology* 2024) includes MD as an explicit screening step and successfully identified active enzymes. NAC4ED uses MD-based near-attack conformations with 92.5% prediction accuracy for mutation effects.

**Output:** MD-validated candidates with dynamic stability scores.

**Compute:** ~200 candidates × 20 ns × ~12 hours/ns = ~4–8 days. Mark as **OPTIONAL for Round 1 fast-track; RECOMMENDED for Round 2+** when Stage 6 compute savings from learned filters offset the MD cost.

### 4.10 Stage 7 — Cross-validation and final ranking (GPU-bound)

**Objective:** Validate top candidates with an independent structure predictor and compose a final ranking using multi-metric Pareto-front selection.

**Method:**
1. Predict top 200 candidates (50 per enzyme) with **Protenix-v1**. Discard any design where Boltz-2 and Protenix-v1 disagree on fold or active site geometry (Cα RMSD between predictions > 2.0 Å).
2. Score all surviving designs with CatPred (primary) and RealKcat (secondary, better for predicting mutation effects) for predicted kcat and Km. Use rankings (not absolute values) for prioritization.
3. **Multi-metric Pareto-front ranking** (replaces arbitrary weighted composite):

**Ranking metrics:**

| Metric | Source | Signal |
|--------|--------|--------|
| ESMFold pLDDT | Stage 5 | Fold confidence (better discriminator than Boltz pLDDT; AUC 0.72 vs 0.60) |
| Active site heavy-atom RMSD | Stage 6 | Catalytic geometry preservation |
| LigandMPNN confidence | Stage 4 | Sequence-structure compatibility |
| Zero-shot fitness ensemble score | Stage 4.5 | Evolutionary plausibility |
| Product inhibition ratio (substrate aff. / product aff.) | Stage 6 | Pathway-specific: penalize high product binding |
| CatPred predicted kcat (ranking) | Stage 7 | Predicted catalytic rate |
| Boltz-2 substrate affinity | Stage 6 | Substrate binding (downweighted — one signal among many) |
| MD stability metrics (if Stage 6.5 run) | Stage 6.5 | Dynamic catalytic competence |

**Ranking strategy:**
- **Round 1:** Do NOT assign fixed weights. Rank by Pareto front across all metrics. Select diverse designs spanning the Pareto front to maximize information gain from the first experimental round. Include WT-adjacent designs (low-risk) and Pareto-optimal designs (high-potential) in every batch.
- **Round 2+:** Learn optimal weights from Round 1 experimental data using random forest or Gaussian process regression. Recalibrate the composite scoring function.

**Rationale:** The previous composite score (0.2×pLDDT + 0.3×RMSD + 0.3×affinity + 0.2×LigandMPNN) had no evidence base. Boltz-2 affinity is a poor proxy for catalytic activity — Km and kcat are independent parameters (R ~ 0.57). Johnson et al. (*Nature Biotechnology* 2024) showed that a 20-metric composite (COMPSS) outperformed any single metric. Pareto-front selection avoids the need for a priori weight assignment while maximizing experimental information gain.

**Output:** Final ranked list of 48–96 designs per enzyme for experimental testing.

**Compute:** ~200 Protenix-v1 predictions × 2 min = ~7 hours. CatPred + RealKcat scoring: ~2 hours CPU.

### 4.11 Stage 7.5 — N-glycosylation screening (CPU-bound)

**Objective:** Identify and mitigate N-glycosylation sites that could impair enzyme activity in *S. cerevisiae*.

**Rationale:** *S. cerevisiae* N-hyperglycosylates heterologous proteins at Asn-Xaa-Ser/Thr sequons (where Xaa ≠ Pro), which can destroy catalytic activity, particularly when sequons are near the active site.

**Method:**
1. Scan all final candidates for N-glycosylation sequons (Asn-Xaa-Ser/Thr where Xaa ≠ Pro) using simple sequence pattern matching (regex).
2. Map sequon positions onto Boltz-2 structures. Flag sequons within 10 Å of the active site as high-risk.
3. For high-risk sequons: generate alternative designs with Asn→Gln or Ser/Thr→Ala mutations. Re-score with ESMFold to confirm fold preservation.
4. Annotate all sequons in the output (even low-risk ones) for experimental awareness.

**Output:** Annotated candidate list with glycosylation risk flags. Modified sequences where needed.

**Compute:** Seconds (sequence-based pattern matching + structure mapping).

### 4.12 Stage 8 — Codon optimization and synthesis preparation (CPU-bound)

**Objective:** Prepare DNA sequences for synthesis and yeast expression.

**Specifications for *S. cerevisiae*:**
- Target GC content: 35–45%
- Codon Adaptation Index (CAI): > 0.7
- Avoid inhibitory codon pairs (CGA-CGA, CGA-CCG, etc.)
- Remove internal BsaI, BsmBI, NotI restriction sites (Golden Gate / Gibson compatibility)
- Remove cryptic splicing signals, poly-A signals, and homopolymer runs > 6 nt
- Add flanking sequences for cloning (promoter/terminator-compatible overhangs)

**Output:** Codon-optimized DNA sequences ready for synthesis ordering, plus a 96-well plate map per enzyme.

---

## 5. Workflow orchestration

The pipeline is implemented as a Snakemake workflow with the following DAG structure:

```
Stage 0: Bottleneck ID (CPU)
    ↓
Stage 1: MSA generation (GPU) ──────────────────────────────┐
    ↓                                                        │
Stage 2: WT holo structure prediction (GPU)                  │
    ↓  (substrate + product complexes)                       │
┌───────────────────────┬──────────────────┐                 │
│ Stage 3: RFdiffusion3 │ Stage 4b: Direct │                 │
│ partial diffusion     │ LigandMPNN on WT │                 │
│ (GPU)                 │ (GPU)            │                 │
└──────────┬────────────┴────────┬─────────┘                 │
           ↓                     ↓                           │
     Stage 4a: LigandMPNN       (merge)                      │
     on backbones (GPU)           ↓                           │
           ↓                                                  │
           └──────────→ Stage 4.5: Zero-shot fitness (CPU)    │
                              ↓                               │
                    Stage 5: ESMFold pre-screen (GPU)          │
                              ↓                               │
                    Stage 6: Boltz-2 screening (GPU) ←────────┘
                    (substrate + product docking)
                              ↓
                    Stage 6.5: MD simulations (GPU, OPTIONAL)
                              ↓
                    Stage 7: Protenix-v1 cross-val + Pareto ranking
                              ↓
                    Stage 7.5: N-glycosylation screening (CPU)
                              ↓
                    Stage 8: Codon optimization + plate maps
```

**Parallelization notes:**
- Stages 3 and 4b run in parallel (both GPU-bound; alternate or batch by enzyme).
- Stage 4.5 (zero-shot fitness) is CPU-bound and can run concurrently with GPU stages.
- Stage 6 is the longest single stage. Batch processing by enzyme allows intermediate inspection. Product docking can be run on only the top 50th percentile to halve compute.
- Stage 6.5 (MD) is optional for Round 1 fast-track.
- All GPU stages process one task at a time (single GPU). CPU stages can run concurrently.

---

## 6. Compute time estimate

| Stage | Task | Bound | Time estimate |
|-------|------|-------|---------------|
| 0 | Bottleneck ID | CPU | 1 day |
| 1 | MSA generation | GPU | 40 seconds |
| 2 | WT structure prediction (substrate + product) | GPU | 20 minutes |
| 3 | RFdiffusion3 partial diffusion | GPU | **~7 hours** |
| 4a | LigandMPNN on backbones | GPU | 13 hours |
| 4b | LigandMPNN direct on WT | GPU | 2.5 hours |
| 4.5 | Zero-shot fitness ensemble | CPU | 30 minutes |
| 5 | ESMFold pre-screen (softened) | GPU | 5.3 hours |
| 6 | Boltz-2 screening (substrate + product) | GPU | ~5 days |
| 6.5 | MD simulations (OPTIONAL) | GPU | 4–8 days |
| 7 | Protenix-v1 cross-validation + ranking | GPU | 7 hours |
| 7.5 | N-glycosylation screening | CPU | minutes |
| 8 | Codon optimization | CPU | 2 hours |
| **Total (without MD)** | | | **~7–8 days** |
| **Total (with MD)** | | | **~12–16 days** |

**Fast-track option:** Run Stage 4b only (direct LigandMPNN diversification on WT structures) → Stage 4.5 → Stage 5 → Stage 6 (substrate only) → Stage 7. Skips RFdiffusion3 and product docking. Produces 3,000 variants screened in ~3 days total. Use results to calibrate whether backbone-level diversity (Stage 3) and product inhibition screening are needed.

---

## 7. Data management

### 7.1 Directory structure

```
/data/enzyme_pipeline/
├── config/
│   ├── pipeline_config.yaml       # Global parameters
│   ├── enzyme_A.yaml              # Per-enzyme: fixed residues, substrate + product SMILES, contigs
│   ├── enzyme_B.yaml
│   ├── enzyme_C.yaml
│   └── enzyme_D.yaml
├── inputs/
│   ├── sequences/                 # WT FASTA files
│   ├── ligands/                   # Substrate AND product SDF or SMILES per enzyme
│   └── databases/                 # MMseqs2 indexed DBs
├── stage_01_msa/
├── stage_02_wt_structures/
│   ├── substrate_complexes/
│   └── product_complexes/
├── stage_03_rfdiffusion3/
│   ├── enzyme_A/
│   ├── enzyme_B/
│   ├── enzyme_C/
│   └── enzyme_D/
├── stage_04_ligandmpnn/
├── stage_04.5_fitness_scores/
├── stage_05_esmfold_prescreen/
├── stage_06_boltz2_screening/
│   ├── substrate_docking/
│   └── product_docking/
├── stage_06.5_md_simulations/     # Optional
├── stage_07_crossval_ranking/
├── stage_07.5_glycosylation/
├── stage_08_codon_opt/
│   ├── dna_sequences/
│   └── plate_maps/
├── logs/
└── results/
    ├── round_01/
    │   ├── final_rankings.csv
    │   ├── pareto_front.csv
    │   ├── synthesis_order.csv
    │   └── plate_maps/
    └── round_02/                  # After experimental data
```

### 7.2 Key file formats

| Stage | Input | Output |
|-------|-------|--------|
| MSA generation | FASTA | .a3m |
| Structure prediction | FASTA + SMILES | PDB + JSON (confidence) |
| RFdiffusion3 | PDB + contigs config | PDB backbones (all-atom) |
| LigandMPNN | PDB + ligand context | FASTA + scores CSV |
| Zero-shot fitness | FASTA + MSA (.a3m) | Scores CSV (per predictor + ensemble) |
| ESMFold | FASTA | PDB + pLDDT JSON |
| Boltz-2 screening | FASTA + SMILES | PDB + confidence + affinity JSON |
| MD simulations | PDB + ligand params | Trajectories (.xtc) + metrics CSV |
| Glycosylation screen | FASTA + PDB | Annotated CSV (sequons + risk flags) |
| Final output | — | Pareto-ranked CSV + codon-optimized DNA FASTA |

### 7.3 Logging and reproducibility

Every pipeline run logs: git commit hash of each tool, random seeds, all parameters, input file checksums (MD5), and wall-clock time per stage. Results are versioned by round number. Snakemake's `--archive` flag bundles the full reproducible workflow.

---

## 8. Success metrics and experimental integration

### 8.1 Computational metrics (per design round)

| Metric | Target |
|--------|--------|
| Designs passing structural filters (Stage 6) | ≥500 per enzyme |
| Boltz-2 / Protenix-v1 agreement rate | ≥70% |
| Sequence diversity (mean pairwise identity) | 60–85% to WT |
| Pareto-front coverage | ≥3 distinct clusters per enzyme |
| Designs selected for synthesis | 48–96 per enzyme |

### 8.2 Experimental metrics (post wet-lab)

| Metric | Target |
|--------|--------|
| Expression rate (detectable protein in yeast) | ≥50% of tested designs |
| Soluble/active fraction | ≥10% of tested designs |
| Activity improvement over WT | ≥2-fold in at least one variant per enzyme |
| Pathway output improvement | ≥3-fold after combining best variants |

### 8.3 DBTL feedback loop

**After Round 1:**
1. Correlate experimental activity with ALL computational scores (ESMFold pLDDT, RMSD, affinity, LigandMPNN confidence, CatPred, zero-shot fitness, product inhibition ratio, MD metrics if available).
2. Retrain a lightweight supervised model (random forest or GP) on experimental fitness data to learn optimal scoring weights.
3. Recalibrate composite scoring function using learned weights.
4. Identify positions where mutations consistently help/hurt — feed back as fixed or excluded residues in the next LigandMPNN round.
5. If initial designs show poor expression, incorporate ASR scaffolds or reduce sequence divergence (lower LigandMPNN temperature).

**Round 2 optimization with EVOLVEpro:**
6. Use **few-shot active learning** on Round 1 experimental data. The EVOLVEpro approach (Jiang et al., *Science* 2025) achieves 2-515× improvement with only 4 rounds, combining protein language model embeddings with regression on experimental measurements. EVOLVEpro itself has a non-commercial license; for commercial flexibility, implement the same approach using ESM-2 embeddings (MIT) + scikit-learn GP/RF regression.
7. Run EVOLVEpro in parallel with the generative pipeline (RFdiffusion3 → LigandMPNN → Boltz-2) on Round 2 to compare approaches and maximize coverage of fitness landscape.

**Aspirational (Round 3+):**
8. Move toward autonomous enzyme engineering, integrating ML predictions with biofoundry automation for closed-loop DBTL (cf. Zhao et al., *Nature Communications* 2025: 90× improvement without human intervention).

---

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Low expression in yeast | High | High | Include WT and conservative (T=0.1) designs as controls; ASR scaffolds as fallback; codon optimization with yeast-specific tables |
| Boltz-2 affinity scores don't correlate with experimental activity | Medium | High | Affinity is one signal among many in Pareto ranking, not a dominant weight; validate with orthogonal scoring (CatPred, RealKcat, zero-shot fitness) |
| RFdiffusion3 backbone diversity collapses (all designs converge) | Low | Medium | Monitor backbone RMSD distribution during generation; increase partial_T or reseed if clustering detected |
| GPU OOM during Boltz-2 complex prediction | Low | Medium | 96 GB handles 512 aa + ligand comfortably; batch larger complexes if needed |
| Rate-limiting enzyme misidentified by FBA | Medium | Medium | Run fast-track (Stage 4b only) on all four enzymes first; use experimental data to validate bottleneck prediction before full campaign |
| Designed enzymes fold but lack catalytic activity | High | High | Central unsolved problem; mitigate by fixing catalytic residues, using ligand-conditioned design, including WT-adjacent designs, and MD filtering (Stage 6.5) |
| Product inhibition limits pathway flux | Medium | High | NEW: Screen for product binding in Stage 6; prioritize variants with high substrate/product affinity ratio |
| N-glycosylation destroys activity in yeast | Medium | Medium | NEW: Screen for Asn-Xaa-Ser/Thr sequons in Stage 7.5; mutate high-risk sites |
| pLDDT filter rejects catalytically active designs | High | High | FIXED: Softened from hard gate (>80) to continuous score with floor at >60. ESMFold pLDDT preferred over Boltz pLDDT (better AUC) |
| Composite scoring weights are arbitrary | High | Medium | FIXED: Pareto-front ranking in Round 1 (no weights); learn weights from experimental data in Round 2 |
| Static structures miss dynamic requirements for catalysis | High | High | Optional MD simulations (Stage 6.5) filter for dynamic stability and substrate retention |
| ResiDPO-LigandMPNN code not yet released | Medium | Low | Base LigandMPNN remains effective; integrate ResiDPO when available for ~3× designability improvement |

---

## 10. Timeline

| Week | Activity |
|------|----------|
| 1 | Environment setup, database indexing, tool installation and validation |
| 2 | Stage 0 (bottleneck ID) + Stages 1–2 (MSA + WT structures with substrate + product) |
| 2–3 | Stages 3–4.5 (RFdiffusion3 + LigandMPNN + zero-shot fitness) |
| 3–4 | Stages 5–7.5 (screening + MD optional + ranking + glycosylation check) |
| 4 | Stage 8 (codon optimization) + synthesis order |
| 5–8 | DNA synthesis lead time |
| 8–10 | Cloning, transformation, expression screening |
| 10–12 | Activity assays, data analysis, DBTL feedback |
| 12–14 | Round 2 computational design (EVOLVEpro + generative pipeline, informed by Round 1 data) |

**Total time from project start to Round 1 experimental data: ~10–12 weeks.**
**Total time through Round 2 design completion: ~14–16 weeks.**
