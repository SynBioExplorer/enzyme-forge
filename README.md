# EnzymeForge

**GPU-accelerated computational enzyme engineering pipeline for metabolic pathway optimization in *S. cerevisiae*.**

EnzymeForge chains state-of-the-art protein design tools (Boltz-2, RFdiffusion3, LigandMPNN, Protenix-v1) into an end-to-end Design-Build-Test-Learn workflow that runs on a single GPU workstation. Given a multi-enzyme pathway and target metabolite, it generates optimized enzyme variants ready for DNA synthesis and experimental validation.

---

## Pipeline Overview

The pipeline takes 4 wild-type enzyme sequences through 12 stages of computational design, screening, and optimization, producing 48-96 synthesis-ready DNA sequences per enzyme in ~7-8 days on one NVIDIA RTX 6000 Pro (96 GB VRAM).

| Stage | Module | What it does | GPU/CPU | Time |
|-------|--------|-------------|---------|------|
| 0 | `stage0_bottleneck.py` | Identifies which enzymes limit pathway flux using flux balance analysis, so GPU time is spent on the right targets | CPU | ~1 day |
| 1 | `stage1_msa.py` | Generates multiple sequence alignments to capture evolutionary information needed by downstream predictors | GPU | ~40s |
| 2 | `stage2_wt_structures.py` | Predicts 3D structures of each enzyme bound to its substrate and product, providing the scaffolds for all design | GPU | ~20 min |
| 3 | `stage3_rfdiffusion3.py` | Creates diverse backbone conformations around the active site while preserving the global fold — the generative core of the pipeline | GPU | ~7 hrs |
| 4 | `stage4_ligandmpnn.py` | Designs amino acid sequences for each backbone that are compatible with substrate binding — converts structures to sequences | GPU | ~16 hrs |
| 4.5 | `stage4_5_fitness.py` | Scores all sequences for evolutionary plausibility using three complementary predictors — catches designs that "look wrong" to evolution | CPU/GPU | ~30 min |
| 5 | `stage5_esmfold.py` | Rapidly predicts structure from sequence alone to eliminate clearly misfolded designs before expensive screening | GPU | ~5.3 hrs |
| 6 | `stage6_boltz2_screen.py` | Full structural screening: docks each variant with substrate AND product to assess both binding and product inhibition risk | GPU | ~5 days |
| 6.5 | `stage6_5_md.py` | *Optional.* Runs molecular dynamics simulations to filter for conformational stability — catches designs that fold but can't catalyze | GPU | 4-8 days |
| 7 | `stage7_ranking.py` | Cross-validates top candidates with an independent predictor (Protenix-v1) and ranks by Pareto front across all metrics | GPU | ~7 hrs |
| 7.5 | `stage7_5_glycosylation.py` | Screens for N-glycosylation sites that yeast would modify, which can destroy enzyme activity | CPU | seconds |
| 8 | `stage8_codon_opt.py` | Converts protein sequences to yeast-optimized DNA, removes restriction sites, and generates 96-well plate maps for ordering | CPU | ~2 hrs |

**Total: ~7-8 days** (without MD) | **Fast-track: ~3 days** (skip backbone diversification) | **With MD: ~12-16 days**

### Pipeline DAG

```
Stage 0: Bottleneck ID ──► Which enzymes to prioritize
    │
Stage 1: MSA Generation ──────────────────────────────────┐
    │                                                       │
Stage 2: WT Structure Prediction (substrate + product)      │
    │                                                       │
    ├──────────────────────┬────────────────────┐           │
    │ Stage 3: RFdiffusion3│ Stage 4b: Direct   │           │
    │ backbone generation  │ LigandMPNN on WT   │           │
    │                      │                    │           │
    └──────────┬───────────┴─────────┬──────────┘           │
               │                     │                      │
         Stage 4a: LigandMPNN  ◄─── merge                  │
               │                                            │
               └──────► Stage 4.5: Zero-shot fitness        │
                              │                             │
                    Stage 5: ESMFold pre-screen              │
                              │                             │
                    Stage 6: Boltz-2 screening  ◄───────────┘
                              │
                    Stage 6.5: MD simulations (optional)
                              │
                    Stage 7: Protenix-v1 cross-val + Pareto ranking
                              │
                    Stage 7.5: N-glycosylation screening
                              │
                    Stage 8: Codon optimization + plate maps
```

---

## Why These Tools?

Every tool was selected for three criteria: **accuracy on relevant benchmarks**, **commercial usability** (MIT/BSD/Apache 2.0), and **single-GPU feasibility** (96 GB VRAM).

| Tool | Role | Why this one? | License |
|------|------|---------------|---------|
| **Boltz-2** | Structure prediction + binding affinity | Best open-source complex predictor; handles protein-ligand-cofactor systems in one pass; MIT license | MIT |
| **RFdiffusion3** | Backbone generation | 10x faster than RFdiffusion v1; all-atom representation; 90% win rate on 41 active-site benchmarks; 18% experimental success rate | BSD-3 |
| **LigandMPNN** | Sequence design | Only inverse folding model that conditions on bound ligands — critical for enzymes where the substrate shapes the active site | MIT |
| **ESMFold** | Fast pre-screening | ~1s/sequence structure prediction; no MSA needed; AUC 0.72 for discriminating design success (better than Boltz pLDDT's 0.60) | MIT |
| **Protenix-v1** | Cross-validation | First fully open model to outperform AlphaFold3; independent from Boltz-2 (different architecture = orthogonal validation) | Apache 2.0 |
| **TranceptEVE** | Fitness prediction | Best single zero-shot predictor on ProteinGym benchmark (Spearman ~0.46-0.50 with experimental fitness) | MIT |
| **ESM-2** | Fitness prediction | Complementary protein language model signal; log-likelihood correlates with fitness | MIT |
| **GEMME** | Fitness prediction | Fast MSA-based conservation scoring; strong on non-human proteins; adds evolutionary signal without GPU cost | MIT |
| **OpenMM** | MD simulations | GPU-accelerated; mature force fields (AMBER ff14SB); Python-native API; >1 ns/day on RTX 6000 | MIT |
| **CatPred** | Kinetics prediction | Predicts kcat, Km, Ki from sequence — provides catalytic activity estimates beyond structural metrics | MIT |
| **COBRApy** | Flux analysis | Standard tool for genome-scale metabolic modeling; integrates with yeast-GEM | GPL-2.0 |
| **MMseqs2-GPU** | MSA generation | GPU-accelerated homology search; eliminates API rate limiting by running locally | MIT |

### Why Not AlphaFold3 / ESM-3?

- **AlphaFold3**: Non-commercial license. Boltz-2 achieves comparable accuracy with MIT license.
- **ESM-3**: No ligand awareness (can't condition on substrates), scTM=0.486 vs RFdiffusion's 0.945 for designability, non-commercial license for useful model sizes.

---

## Design Decisions Explained

### Softened pLDDT Filtering (Stages 5-6)
Garcia et al. (*Protein Science* 2025) showed pLDDT has AUC of only 0.60 for predicting design success — barely better than random. Active sites are often in dynamic regions where pLDDT is inherently low. We use pLDDT > 60 as a floor (catch clearly misfolded) rather than a hard gate, and pass pLDDT as a continuous score to Pareto ranking.

### Product Inhibition Screening (Stage 6)
Pathway enzymes accumulate local product, which can inhibit catalysis. We dock each variant with both substrate and product SMILES, then rank by the substrate/product affinity ratio. Variants where product binds more tightly than substrate are flagged.

### Pareto Ranking Instead of Weighted Scoring (Stage 7)
The v0.1 composite score (0.2x pLDDT + 0.3x RMSD + 0.3x affinity + 0.2x LigandMPNN) had no evidence base. Boltz-2 affinity poorly correlates with catalytic activity (R~0.57). We rank by Pareto front across 7-8 metrics in Round 1 (no a priori weights), then learn weights from experimental data in Round 2.

### N-Glycosylation Screening (Stage 7.5)
*S. cerevisiae* hyperglycosylates heterologous proteins at Asn-Xaa-Ser/Thr sequons. Sequons near the active site (<10 A) can destroy activity. We screen and offer Asn->Gln substitutions.

### Zero-Shot Fitness Ensemble (Stage 4.5)
ProteinGym benchmarks show that ensembles across modalities outperform single large models, and scaling pLMs beyond 4B parameters doesn't help. We combine three orthogonal signals: autoregressive (TranceptEVE), language model (ESM-2), and conservation-based (GEMME).

### Local MSA Database
ColabFold's public API has rate limits. By downloading and indexing the full ColabFoldDB (~1.2 TB) + UniRef30 locally on NVMe, MSA generation runs in seconds without throttling.

---

## Installation

### Prerequisites
- NVIDIA GPU with >= 48 GB VRAM (96 GB recommended)
- CUDA >= 12.4
- Python 3.10-3.11
- Conda/Mamba

### Setup

```bash
# Clone
git clone https://github.com/your-username/enzyme-forge.git
cd enzyme-forge

# Install base dependencies
pip install pyyaml jsonschema pandas biopython numpy

# Download and index MSA databases (~1.4 TB, run once)
python -m pipeline.setup_databases --config config/pipeline_config.yaml --db-dir /data/databases

# Install each tool in its own conda environment (see pipeline-tech-reqs.md for versions)
# Boltz-2, RFdiffusion3, LigandMPNN, ESMFold, Protenix-v1, etc.
```

---

## Usage

Every stage is invoked the same way — all parameters come from the YAML config:

```bash
python -m pipeline.<stage_module> --config config/pipeline_config.yaml
```

### Running the Full Pipeline

```bash
# 1. Download databases (one-time, ~1.4 TB)
python -m pipeline.setup_databases --config config/pipeline_config.yaml

# 2. Identify bottleneck enzymes
python -m pipeline.stage0_bottleneck --config config/pipeline_config.yaml

# 3. Generate MSAs
python -m pipeline.stage1_msa --config config/pipeline_config.yaml

# 4. Predict wild-type structures (substrate + product complexes)
python -m pipeline.stage2_wt_structures --config config/pipeline_config.yaml

# 5. Generate diverse backbones
python -m pipeline.stage3_rfdiffusion3 --config config/pipeline_config.yaml

# 6. Design sequences
python -m pipeline.stage4_ligandmpnn --config config/pipeline_config.yaml

# 7. Zero-shot fitness scoring
python -m pipeline.stage4_5_fitness --config config/pipeline_config.yaml

# 8. Pre-screen with ESMFold
python -m pipeline.stage5_esmfold --config config/pipeline_config.yaml

# 9. Full structural screening + product inhibition
python -m pipeline.stage6_boltz2_screen --config config/pipeline_config.yaml

# 10. (Optional) MD simulations
python -m pipeline.stage6_5_md --config config/pipeline_config.yaml

# 11. Cross-validate and rank
python -m pipeline.stage7_ranking --config config/pipeline_config.yaml

# 12. N-glycosylation screening
python -m pipeline.stage7_5_glycosylation --config config/pipeline_config.yaml

# 13. Codon optimization + plate maps
python -m pipeline.stage8_codon_opt --config config/pipeline_config.yaml
```

### Fast-Track Mode (~3 days)

Skip backbone diversification — run LigandMPNN directly on wild-type structures:

```bash
python -m pipeline.stage1_msa --config config/pipeline_config.yaml
python -m pipeline.stage2_wt_structures --config config/pipeline_config.yaml
python -m pipeline.stage4_ligandmpnn --config config/pipeline_config.yaml  # uses WT branch only
python -m pipeline.stage4_5_fitness --config config/pipeline_config.yaml
python -m pipeline.stage5_esmfold --config config/pipeline_config.yaml
python -m pipeline.stage6_boltz2_screen --config config/pipeline_config.yaml
python -m pipeline.stage7_ranking --config config/pipeline_config.yaml
python -m pipeline.stage7_5_glycosylation --config config/pipeline_config.yaml
python -m pipeline.stage8_codon_opt --config config/pipeline_config.yaml
```

---

## Configuration

All parameters are defined in `config/pipeline_config.yaml`. Per-enzyme settings (substrate SMILES, fixed catalytic residues, RFdiffusion3 contigs) go in `config/enzyme_X.yaml`.

### Key Parameters

#### Stage 0 — Bottleneck ID
| Parameter | Default | Reasoning |
|-----------|---------|-----------|
| `fcc_threshold` | 0.3 | Enzymes with flux control coefficient > 0.3 are primary design targets; below 0.1 benefit more from expression optimization than protein engineering |

#### Stage 2 — Structure Prediction
| Parameter | Default | Reasoning |
|-----------|---------|-----------|
| `boltz2_seeds` | 5 | 5 independent seeds give sufficient conformational sampling for model selection by confidence score |
| `recycling_iterations` | 10 | Boltz-2 default; more recycling improves accuracy but with diminishing returns |

#### Stage 3 — RFdiffusion3
| Parameter | Default | Reasoning |
|-----------|---------|-----------|
| `partial_T_conservative` | 10 | Of 50 total timesteps — preserves most of the WT backbone, only remodels loops |
| `partial_T_standard` | 15 | Moderate diversification around the active site |
| `partial_T_aggressive` | 20 | Aggressive exploration for high-FCC enzymes where bigger changes are justified |
| `designs_primary` | 500 | High-FCC enzymes get 500 backbones to maximize diversity |
| `designs_secondary` | 200 | Low-FCC enzymes get 200 — enough for the library but conserving compute |

#### Stage 4 — LigandMPNN
| Parameter | Default | Reasoning |
|-----------|---------|-----------|
| `temperature_primary` | 0.1 | Low temperature = high confidence, conservative sequences that closely match the backbone |
| `temperature_exploratory` | 0.2 | Slightly higher diversity to explore more of sequence space |
| `temperature_wt_high` | 0.4 | Direct WT diversification at higher temperature to generate moderate-risk variants |
| `number_of_batches` | 16 | Standard sampling depth — 16 independent sequence samples per backbone |
| `omit_aa` | "C" | Suppress free cysteines to avoid misfolding in yeast cytoplasm (reducing environment) |
| `pack_side_chains` | true | Enable side-chain packing for more realistic models |
| `pack_with_ligand_context` | true | Critical: conditions sequence design on the bound substrate geometry |

#### Stage 5 — ESMFold Pre-screen
| Parameter | Default | Reasoning |
|-----------|---------|-----------|
| `plddt_hard_floor` | 60 | Catches only clearly misfolded designs; 70-80 cutoffs reject too many catalytically competent variants (Garcia et al. 2025) |
| `ca_rmsd_max` | 5.0 A | Eliminates sequences that fold to a completely different topology |

#### Stage 6 — Boltz-2 Screening
| Parameter | Default | Reasoning |
|-----------|---------|-----------|
| `ca_rmsd_max` | 2.0 A | Fold preservation — variants must maintain global structure within 2A of WT |
| `active_site_rmsd_max` | 1.5 A | Catalytic geometry — tighter threshold because active site positioning is critical for catalysis |
| `ligand_rmsd_max` | 2.0 A | Substrate must stay in roughly the same position as in the WT complex |

#### Stage 6.5 — MD Simulations (Optional)
| Parameter | Default | Reasoning |
|-----------|---------|-----------|
| `enabled` | false | Optional for Round 1 fast-track; recommended for Round 2+ when compute savings from learned filters offset the cost |
| `simulation_ns` | 20 | 20 ns is sufficient to observe active site stability and substrate retention without excessive compute |
| `top_n_candidates` | 200 | 50 per enzyme — balances thoroughness with compute budget |

#### Stage 7 — Ranking
| Parameter | Default | Reasoning |
|-----------|---------|-----------|
| `top_n_candidates` | 200 | 50 per enzyme for Protenix-v1 cross-validation |
| `protenix_ca_rmsd_agreement` | 2.0 A | Boltz-2 and Protenix-v1 must agree on fold within 2A — designs where two independent predictors disagree are unreliable |

#### Stage 8 — Codon Optimization
| Parameter | Default | Reasoning |
|-----------|---------|-----------|
| `target_gc_min/max` | 0.35-0.45 | Optimal GC range for *S. cerevisiae* expression; avoids mRNA secondary structure issues |
| `min_cai` | 0.7 | Codon Adaptation Index > 0.7 ensures efficient translation with yeast tRNAs |

### Per-Enzyme Configuration

Each enzyme gets its own YAML file (`config/enzyme_X.yaml`) with:

```yaml
name: "Enzyme A"
length: 428
sequence_file: "inputs/sequences/enzyme_A.fasta"
substrate_smiles: "CC(=O)SCoA"          # What it acts on
product_smiles: "CC(=O)CC(=O)SCoA"      # What it produces
fixed_residues: [145, 147, 201]          # Catalytic residues — never mutated
hotspot_residues: [140, 142, 145, 200]   # Active site — focus RFdiffusion3 here
contigs: "A1-139/0 5-15/A155-428"        # Which regions to redesign
cofactors: ["NAD+"]                      # Bound cofactors
```

---

## Output Structure

```
/data/enzyme_pipeline/
├── config/                           # Pipeline + per-enzyme YAML configs
├── inputs/sequences/                 # Wild-type FASTA files
├── stage_01_msa/                     # .a3m MSA files
├── stage_02_wt_structures/           # Boltz-2 holo PDBs (substrate + product)
├── stage_03_rfdiffusion3/enzyme_X/   # Diversified backbone PDBs
├── stage_04_ligandmpnn/              # Designed sequences (FASTA + scores)
├── stage_04.5_fitness_scores/        # Zero-shot fitness CSVs
├── stage_05_esmfold_prescreen/       # Filtered sequences + pLDDT scores
├── stage_06_boltz2_screening/        # Full screening results + product inhibition
├── stage_06.5_md_simulations/        # MD metrics (optional)
├── stage_07_crossval_ranking/        # Pareto-ranked final candidates
├── stage_07.5_glycosylation/         # Glycosylation risk annotations
├── stage_08_codon_opt/               # DNA sequences + 96-well plate maps
│   ├── dna_sequences/
│   └── plate_maps/
├── logs/                             # Structured JSON logs per stage
└── results/round_01/                 # Final deliverables
    ├── final_rankings.csv
    ├── pareto_front.csv
    └── synthesis_order.csv
```

---

## Testing

All modules are tested with mocked external tool calls (no GPU required):

```bash
# Run all 451 tests
python -m pytest tests/ -v

# Run tests for a specific stage
python -m pytest tests/test_stage3.py -v
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 48 GB VRAM | NVIDIA RTX 6000 Pro (96 GB GDDR7) |
| CPU | 8 cores | 24 cores (Intel i9 / AMD EPYC) |
| RAM | 64 GB | 128 GB DDR5 |
| Storage | 1.5 TB | 2+ TB NVMe SSD |
| OS | Ubuntu 22.04 | Ubuntu 24.04 LTS |
| CUDA | 12.4 | 13.0 (TensorRT optimized) |

---

## DBTL Integration

EnzymeForge is designed for iterative improvement:

- **Round 1**: Generative design (no experimental data) — Pareto ranking maximizes information gain
- **Round 2+**: Learn scoring weights from Round 1 data using random forest / GP regression; optionally add few-shot active learning (ESM-2 embeddings + GP regression)
- **Aspirational**: Closed-loop autonomous enzyme engineering with biofoundry automation

---

## References

- Watson et al., "De novo design of protein structure and function with RFdiffusion" (2023)
- Watson et al., "RFdiffusion3" (2025) — Baker Lab
- Dauparas et al., "Robust deep learning-based protein sequence design using ProteinMPNN" (2022)
- Wohlwend et al., "Boltz-2: Exploring the Limits of Structure Prediction" (2025)
- Abramson et al., "Accurate structure prediction of biomolecular interactions with AlphaFold 3" (2024)
- ByteDance, "Protenix v1" (2026)
- Notin et al., "ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction" (NeurIPS 2023)
- Garcia et al., "pLDDT is an imperfect discriminator of design success" (*Protein Science* 2025)
- Jiang et al., "EVOLVEpro: Few-shot protein engineering" (*Science* 2025)

---

## License

MIT

---

*Built at the [Australian Genome Foundry](https://www.mq.edu.au/research/research-centres-groups-and-facilities/healthy-people/facilities/australian-genome-foundry), Macquarie University.*
