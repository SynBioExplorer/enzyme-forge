# EnzymeForge

**GPU-accelerated computational enzyme engineering pipeline for metabolic pathway optimization in *S. cerevisiae*.**

EnzymeForge chains state-of-the-art protein design tools (Boltz-2, RFdiffusion3, LigandMPNN, Protenix-v1) into an end-to-end Design-Build-Test-Learn workflow that runs on a single GPU workstation. Given a multi-enzyme pathway and target metabolite, it generates optimized enzyme variants ready for DNA synthesis and experimental validation.

---

## Pipeline Overview

The pipeline takes wild-type enzyme sequences through 14 stages of computational design, screening, and optimization, producing 48-96 synthesis-ready DNA sequences per enzyme. With Boltz-2 NIM (TensorRT-compiled) and the optimized funnel, total runtime is **~2-3 days** on one NVIDIA RTX 6000 Pro (96 GB VRAM).

| Stage | Module | What it does | GPU/CPU | Time |
|-------|--------|-------------|---------|------|
| 0 | `stage0_bottleneck.py` | **Optional.** Identifies which enzymes limit pathway flux using thermodynamic FBA. When disabled (default), all enzymes are optimized equally | CPU | ~1 day |
| 1 | `stage1_msa.py` | Generates multiple sequence alignments via ColabFold + MMseqs2-GPU (~10s/enzyme on RTX 6000 Pro with Blackwell DPX kernels) | GPU | ~40s |
| 2 | `stage2_wt_structures.py` | Predicts 3D structures of each enzyme bound to substrate, product, and optional transition state analogue | GPU | ~20 min |
| 3 | `stage3_rfdiffusion3.py` | Creates ~5,000 diverse backbone conformations around the active site while preserving the global fold | GPU | ~20 hrs |
| 4 | `stage4_ligandmpnn.py` | Designs amino acid sequences for each backbone, optionally conditioning on transition state geometry. MMseqs2 clustering at 90% identity removes near-duplicates | GPU | ~16 hrs |
| 4.5 | `stage4_5_fitness.py` | Scores all sequences for evolutionary plausibility using TranceptEVE + ESM-2 + GEMME ensemble | CPU/GPU | ~30 min |
| 4.75 | `stage4_75_solubility.py` | **New.** Physics-based solubility pre-screen (CamSol-like intrinsic solubility, net charge, aggregation-prone regions). Flags likely-insoluble variants before expensive screening | CPU | ~5 min |
| 5 | `stage5_esmfold.py` | Batch ESMFold inference (bf16) with pre-ranking gate — passes only top N sequences to Stage 6 based on pLDDT + fitness composite | GPU | ~2 hrs |
| 6 | `stage6_boltz2_screen.py` | 3-way docking: substrate + product + transition state. Boltz-2 NIM (TRT) at 6.8s/inference. Incremental checkpointing for crash recovery | GPU | ~7-19 hrs |
| 6.5 | `stage6_5_md.py` | *Optional.* Gaussian accelerated MD (GaMD) for enhanced conformational sampling — captures dynamics that classical 20ns MD misses | GPU | 4-8 days |
| 7 | `stage7_ranking.py` | Cross-validates with Protenix-v1; Pareto-front ranking across 10 metrics including TS affinity and solubility | GPU | ~7 hrs |
| 7.5 | `stage7_5_glycosylation.py` | Screens for N-glycosylation sequons that yeast would modify, offers Asn->Gln substitutions | CPU | seconds |
| 8 | `stage8_codon_opt.py` | Codon optimization for *S. cerevisiae* with expression-hedging 96-well plate maps (WT controls, risk-stratified layout) | CPU | ~2 hrs |

**Total: ~2-3 days** (NIM + funnel) | **Without NIM: ~5-6 days** | **With MD: ~10-14 days**

### Pipeline DAG

```
Stage 0: Bottleneck ID (optional) ──► Enzyme priority
    │
Stage 1: MSA Generation ──────────────────────────────────┐
    │                                                       │
Stage 2: WT Structure Prediction (substrate+product+TS)     │
    │                                                       │
    ├──────────────────────┬────────────────────┐           │
    │ Stage 3: RFdiffusion3│ Stage 4b: Direct   │           │
    │ backbone generation  │ LigandMPNN on WT   │           │
    │ (~5,000 backbones)   │                    │           │
    └──────────┬───────────┴─────────┬──────────┘           │
               │                     │                      │
         Stage 4a: LigandMPNN ◄──── merge                   │
               │ + MMseqs2 clustering (90% identity)        │
               └──────► Stage 4.5: Zero-shot fitness        │
                              │                             │
                    Stage 4.75: Solubility pre-screen        │
                              │                             │
                    Stage 5: ESMFold + pre-ranking gate      │
                              │ (top 5,000 → Stage 6)       │
                    Stage 6: Boltz-2 3-way docking ◄────────┘
                              │ (substrate + product + TS)
                    Stage 6.5: GaMD simulations (optional)
                              │
                    Stage 7: Protenix-v1 + Pareto ranking
                              │
                    Stage 7.5: N-glycosylation screening
                              │
                    Stage 8: Codon opt + hedging plate maps
```

---

## v0.3 Improvements

Changes from v0.2 based on critical review and Didi et al. (2025) benchmarks:

| Improvement | What changed | Impact |
|-------------|-------------|--------|
| **Transition state docking** | 3-way docking (substrate + product + TS) instead of 2-way; TS affinity as Pareto metric | Better catalytic variant selection — enzymes stabilize TS, not substrate |
| **Solubility pre-screen** | New Stage 4.75 with CamSol-like physics-based scoring | Catches aggregation-prone variants before expensive screening |
| **Scaled funnel** | 2,000-5,000 backbones (up from 1,600); MMseqs2 clustering at 90% identity | More diversity in, less redundancy through |
| **Pre-ranking gate** | Stage 5 ranks by pLDDT + fitness, passes only top 5,000 to Stage 6 | Stage 6 compute cut by 60-80% |
| **Boltz-2 NIM** | TensorRT-compiled inference: 6.8s vs 18s per structure on RTX 6000 Pro | 2.6x faster Stage 6 (Didi et al. 2025) |
| **BF16 inference** | Autocast bf16 for ESMFold, validated by Didi et al. (no accuracy loss) | 2x throughput, halved VRAM |
| **Incremental checkpointing** | Stage 6 saves every 200 sequences; resume after crash | No more lost progress on multi-day runs |
| **GaMD enhanced sampling** | Gaussian accelerated MD replaces classical MD in Stage 6.5 | Captures conformational dynamics in shorter simulations |
| **Stage 0 optional** | FBA bottleneck ID is opt-in (default: optimize all enzymes equally) | Simpler startup; FBA available when kinetic data exists |
| **TFBA + ensemble FBA** | Thermodynamic FBA (eQuilibrator) and ensemble parameter sampling | More realistic flux estimates than basic FBA |
| **Expression-hedging plates** | WT controls in A1/H12; risk-stratified rows (conservative/moderate/aggressive) | Maximizes experimental information per plate |

---

## Why These Tools?

Every tool was selected for: **accuracy on relevant benchmarks**, **commercial usability** (MIT/BSD/Apache 2.0), and **single-GPU feasibility** (96 GB VRAM).

| Tool | Role | Why this one? | License |
|------|------|---------------|---------|
| **Boltz-2 (+ NIM)** | Structure prediction + binding affinity | Best open-source complex predictor; TRT-compiled NIM = 6.8s/inference on RTX 6000 Pro | MIT |
| **RFdiffusion3** | Backbone generation | 10x faster than v1; all-atom; 90% win rate on 41 active-site benchmarks; 18% experimental success rate | BSD-3 |
| **LigandMPNN** | Sequence design | Only inverse folding model that conditions on bound ligands — critical for enzymes | MIT |
| **ESMFold** | Fast pre-screening | ~1s/seq (bf16 batch); AUC 0.72 for design success discrimination | MIT |
| **Protenix-v1** | Cross-validation | First open model to outperform AlphaFold3; orthogonal to Boltz-2 | Apache 2.0 |
| **TranceptEVE** | Fitness prediction | Best single zero-shot predictor on ProteinGym (Spearman ~0.46-0.50) | MIT |
| **ESM-2** | Fitness prediction | Complementary PLM signal; log-likelihood correlates with fitness | MIT |
| **GEMME** | Fitness prediction | Fast MSA-based conservation; strong on non-human proteins | MIT |
| **OpenMM** | MD simulations (GaMD) | GPU-accelerated; AMBER ff14SB; Python API; GaMD for enhanced sampling | MIT |
| **CatPred** | Kinetics prediction | Predicts kcat, Km, Ki from sequence | MIT |
| **MMseqs2-GPU** | MSA generation | GPU-accelerated; Blackwell DPX kernels; 10s/query on RTX 6000 Pro | MIT |

### Why Not AlphaFold3 / ESM-3?

- **AlphaFold3**: Non-commercial license. Boltz-2 achieves comparable accuracy with MIT license.
- **ESM-3**: No ligand awareness (can't condition on substrates), scTM=0.486 vs RFdiffusion's 0.945, non-commercial license for useful model sizes.

---

## Design Decisions Explained

### Ground State vs. Transition State (Stage 4 + 6)
Enzymes catalyze by stabilizing the transition state (TS), not the substrate ground state. Over-optimizing for substrate affinity creates a thermodynamic sink. We optionally dock against TS analogues (via `ts_smiles` in config) alongside substrate/product, and include TS affinity in Pareto ranking.

### Softened pLDDT Filtering (Stage 5)
Garcia et al. (*Protein Science* 2025) showed pLDDT has AUC of only 0.60 for predicting design success. We use pLDDT > 60 as a floor and pass it as a continuous score to Pareto ranking.

### Solubility Pre-screening (Stage 4.75)
Expression failure is the #1 real-world failure mode for heterologous enzyme designs. CamSol-like physics-based scoring (no ML domain shift from E. coli to yeast) flags aggregation-prone variants before expensive Boltz-2 screening.

### Product Inhibition Screening (Stage 6)
Pathway enzymes accumulate local product. We dock each variant with substrate AND product SMILES, ranking by substrate/product affinity ratio.

### GaMD Enhanced Sampling (Stage 6.5)
Classical 20ns MD only captures side-chain relaxation. GaMD applies a harmonic boost potential to smooth the energy landscape, enabling observation of loop closures and domain motions within short simulations.

### Pareto Ranking (Stage 7)
No arbitrary weights. Round 1: rank by Pareto front across 10 metrics. Round 2+: learn weights from experimental data via GP regression.

### Expression-Hedging Plate Layout (Stage 8)
Wells A1/H12: WT controls. Rows A-B: conservative variants (fewest mutations). Rows G-H: aggressive variants. Middle rows: moderate risk. Maximizes information gain per plate.

### Funnel Optimization
The v0.2 pipeline spent 66% of compute (Stage 6) evaluating 12,000 sequences of which 98.3% were discarded. v0.3 fixes this with: (1) MMseqs2 clustering at 90% identity, (2) pre-ranking gate passing only top 5,000 to Stage 6, (3) Boltz-2 NIM at 2.6x speed. Result: Stage 6 drops from ~5 days to ~7-19 hours.

---

## Installation

### Prerequisites
- NVIDIA GPU with >= 48 GB VRAM (96 GB recommended, Blackwell architecture for DPX kernels)
- CUDA >= 12.4 (13.0 for TensorRT)
- Python 3.10-3.11
- Conda/Mamba

### Setup

```bash
git clone https://github.com/SynBioExplorer/enzyme-forge.git
cd enzyme-forge

pip install pyyaml jsonschema pandas biopython numpy

# Download MSA databases (~1.4 TB, run once)
python -m pipeline.setup_databases --config config/pipeline_config.yaml --db-dir /data/databases
```

---

## Usage

Every stage is invoked the same way:

```bash
python -m pipeline.<stage_module> --config config/pipeline_config.yaml
```

### Full Pipeline

```bash
# 0. (Optional) Bottleneck ID — set enabled: true in config first
python -m pipeline.stage0_bottleneck --config config/pipeline_config.yaml

# 1-2. MSA + WT structures
python -m pipeline.stage1_msa --config config/pipeline_config.yaml
python -m pipeline.stage2_wt_structures --config config/pipeline_config.yaml

# 3-4. Design
python -m pipeline.stage3_rfdiffusion3 --config config/pipeline_config.yaml
python -m pipeline.stage4_ligandmpnn --config config/pipeline_config.yaml

# 4.5-4.75. Fitness + solubility scoring
python -m pipeline.stage4_5_fitness --config config/pipeline_config.yaml
python -m pipeline.stage4_75_solubility --config config/pipeline_config.yaml

# 5-6. Screen (Stage 5 pre-ranks, Stage 6 does full Boltz-2 + TS docking)
python -m pipeline.stage5_esmfold --config config/pipeline_config.yaml
python -m pipeline.stage6_boltz2_screen --config config/pipeline_config.yaml

# 6.5. (Optional) GaMD simulations
python -m pipeline.stage6_5_md --config config/pipeline_config.yaml

# 7-8. Rank + prepare for synthesis
python -m pipeline.stage7_ranking --config config/pipeline_config.yaml
python -m pipeline.stage7_5_glycosylation --config config/pipeline_config.yaml
python -m pipeline.stage8_codon_opt --config config/pipeline_config.yaml
```

---

## Configuration

All parameters are in `config/pipeline_config.yaml`. Per-enzyme settings go in `config/enzyme_X.yaml`.

### Key Parameters

| Stage | Parameter | Default | Reasoning |
|-------|-----------|---------|-----------|
| 0 | `enabled` | `false` | Opt-in; when disabled, all enzymes get equal priority |
| 0 | `method` | `"tfba"` | Thermodynamic FBA more realistic than basic FBA |
| 3 | `designs_primary` | `2000` | Scaled up for larger funnel with downstream clustering |
| 3 | `designs_secondary` | `500` | Low-FCC enzymes get fewer but still substantial coverage |
| 4 | `clustering_identity` | `0.9` | MMseqs2 dedup at 90% identity removes near-duplicates |
| 4 | `omit_aa` | `"C"` | Suppress free cysteines for yeast cytoplasm |
| 4.75 | `risk_threshold` | `0.3` | Below this composite = high aggregation risk |
| 5 | `batch_size` | `8` | True batch ESMFold inference |
| 5 | `max_to_boltz2` | `5000` | Pre-ranking gate: only top 5K proceed to Stage 6 |
| 6 | `use_nim` | `false` | Boltz-2 NIM (TRT): 6.8s vs 18s per inference |
| 6 | `checkpoint_interval` | `200` | Save every 200 seqs for crash recovery |
| 6.5 | `md_method` | `"gamd"` | GaMD enhanced sampling (recommended over classical) |
| 6.5 | `gamd_sigma0` | `6.0` | Upper limit of boost potential std dev |
| 8 | `plate_layout` | `"hedging"` | WT controls + risk-stratified rows |

### Per-Enzyme Configuration

```yaml
name: "Enzyme A"
length: 428
sequence_file: "inputs/sequences/enzyme_A.fasta"
substrate_smiles: "CC(=O)SCoA"
product_smiles: "CC(=O)CC(=O)SCoA"
ts_smiles: ""                           # Optional: transition state analogue
fixed_residues: [145, 147, 201]         # Catalytic residues — never mutated
hotspot_residues: [140, 142, 145, 200]  # Focus RFdiffusion3 here
contigs: "A1-139/0 5-15/A155-428"
cofactors: ["NAD+"]
```

---

## Output Structure

```
/data/enzyme_pipeline/
├── config/                           # Pipeline + per-enzyme YAML configs
├── stage_01_msa/                     # .a3m MSA files
├── stage_02_wt_structures/           # Boltz-2 holo PDBs (substrate+product+TS)
├── stage_03_rfdiffusion3/            # ~5,000 backbone PDBs
├── stage_04_ligandmpnn/              # Designed sequences + MMseqs2 clusters
├── stage_04.5_fitness_scores/        # Zero-shot fitness CSVs
├── stage_04.75_solubility/           # Solubility scores + risk flags
├── stage_05_esmfold_prescreen/       # Filtered + pre-ranked sequences
├── stage_06_boltz2_screening/        # 3-way docking results + checkpoints
├── stage_06.5_md_simulations/        # GaMD metrics (optional)
├── stage_07_crossval_ranking/        # Pareto-ranked candidates
├── stage_07.5_glycosylation/         # Glycosylation risk annotations
├── stage_08_codon_opt/               # DNA sequences + hedging plate maps
├── logs/                             # Structured JSON logs
└── results/round_01/                 # Final deliverables
```

---

## Testing

```bash
# Run all 619 tests (no GPU required — all external tools are mocked)
python -m pytest tests/ -v

# Run tests for a specific stage
python -m pytest tests/test_stage6.py -v
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 48 GB VRAM | NVIDIA RTX 6000 Pro (96 GB GDDR7, Blackwell) |
| CPU | 8 cores | 24 cores (Intel i9 / AMD EPYC) |
| RAM | 64 GB | 128 GB DDR5 |
| Storage | 1.5 TB | 2+ TB NVMe SSD |
| OS | Ubuntu 22.04 | Ubuntu 24.04 LTS |
| CUDA | 12.4 | 13.0 (TensorRT) |

### RTX 6000 Pro Benchmarks (Didi et al. 2025)

| Task | Time | Notes |
|------|------|-------|
| MMseqs2-GPU MSA | 10.3s/query | Blackwell DPX kernels, 191x vs JackHMMER |
| OpenFold-TRT inference | 5.6s/structure | BF16, 2.54x vs vanilla PyTorch |
| Boltz-2 NIM inference | 6.83s/structure | TRT-compiled, no accuracy loss |
| 96 GB VRAM handles | up to 949 residues | L40S (48GB) and H100 (80GB) OOM'd on longest targets |

---

## DBTL Integration

- **Round 1**: Generative design (no experimental data) — Pareto ranking maximizes information gain
- **Round 2+**: Learn scoring weights from Round 1 data via GP regression; few-shot active learning (ESM-2 embeddings + scikit-learn)
- **Aspirational**: Closed-loop autonomous enzyme engineering with biofoundry automation

---

## References

- Watson et al., "De novo design of protein structure and function with RFdiffusion" (2023)
- Watson et al., "RFdiffusion3" (2025) — Baker Lab
- Dauparas et al., "Robust deep learning-based protein sequence design using ProteinMPNN" (2022)
- Wohlwend et al., "Boltz-2: Exploring the Limits of Structure Prediction" (2025)
- Didi et al., "Highly efficient protein structure prediction" (2025) — NVIDIA, Oxford, Steinegger
- ByteDance, "Protenix v1" (2026)
- Notin et al., "ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction" (NeurIPS 2023)
- Garcia et al., "pLDDT is an imperfect discriminator of design success" (*Protein Science* 2025)
- Jiang et al., "EVOLVEpro: Few-shot protein engineering" (*Science* 2025)

---

## License

MIT

---

*Built at the [Australian Genome Foundry](https://www.mq.edu.au/research/research-centres-groups-and-facilities/healthy-people/facilities/australian-genome-foundry), Macquarie University.*
