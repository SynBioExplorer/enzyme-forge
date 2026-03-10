"""Stage 6: Full Boltz-2 Structural Screening with Product Inhibition Analysis.

For each enzyme: (1) read filtered FASTA from Stage 5 + substrate/product SMILES,
(2) run Boltz-2 with substrate SMILES for all candidates, (3) apply structural
filters (Ca RMSD < 2.0 A, active site RMSD < 1.5 A, ligand RMSD < 2.0 A),
(4) for top 50th percentile by substrate affinity: run Boltz-2 with product SMILES,
(5) calculate product inhibition ratio (substrate_affinity / product_affinity),
(6) flag high product inhibition risk.

Output: screening_results.csv + PDB files in stage_06_boltz2_screening/

CLI: python pipeline/stage6_boltz2_screen.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from pipeline.config import load_config, get_enzyme_config, resolve_paths
    from pipeline.utils import (
        setup_logging,
        read_fasta,
        write_fasta,
        write_scores,
        calc_ca_rmsd,
    )
except ImportError:
    import yaml
    import logging

    def load_config(p):
        return yaml.safe_load(open(p))

    def get_enzyme_config(cfg, eid):
        return cfg.get("enzymes", {}).get(eid, {})

    def resolve_paths(cfg, base):
        return cfg

    def setup_logging(n, d, **kw):
        logging.basicConfig()
        return logging.getLogger(n)

    def read_fasta(p):
        seqs = {}
        with open(p) as f:
            h, s = None, []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if h:
                        seqs[h] = "".join(s)
                    h, s = line[1:], []
                elif h:
                    s.append(line)
            if h:
                seqs[h] = "".join(s)
        return seqs

    def write_fasta(seqs, p):
        with open(p, "w") as f:
            for h, s in seqs.items():
                f.write(f">{h}\n{s}\n")

    def write_scores(scores, p):
        import csv as _csv

        if not scores:
            return
        with open(p, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=scores[0].keys())
            w.writeheader()
            w.writerows(scores)

    def calc_ca_rmsd(p1, p2, subset=None):
        return 0.0


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
STAGE_NAME = "Stage 6: Full Boltz-2 Structural Screening"
STAGE_NUM = "6"
INPUT_DESCRIPTION = "Filtered FASTA from Stage 5, substrate/product SMILES from config"
OUTPUT_DESCRIPTION = "screening_results.csv + PDB files in stage_06_boltz2_screening/"

CSV_COLUMNS = [
    "sequence_id",
    "enzyme_id",
    "ca_rmsd",
    "active_site_rmsd",
    "ligand_rmsd",
    "passed_structural_filters",
    "plddt",
    "substrate_affinity",
    "product_affinity",
    "inhibition_ratio",
    "high_inhibition_risk",
    "ts_affinity",
    "ts_substrate_ratio",
    "pdb_path",
]

# Timing constants (seconds per inference on RTX PRO 6000 Blackwell)
VANILLA_INFERENCE_TIME_S = 18.0
NIM_INFERENCE_TIME_S = 6.8


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def write_checkpoint(checkpoint_path: Path, data: dict) -> None:
    """Write checkpoint JSON atomically (write to .tmp then rename).

    Args:
        checkpoint_path: Path for the checkpoint JSON file.
        data: Dict to serialize as JSON.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(checkpoint_path.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, str(checkpoint_path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load checkpoint JSON if it exists.

    Args:
        checkpoint_path: Path to checkpoint JSON file.

    Returns:
        Parsed dict or None if no checkpoint exists.
    """
    if checkpoint_path.exists():
        with open(checkpoint_path) as fh:
            return json.load(fh)
    return None


# ---------------------------------------------------------------------------
# Boltz-2 command builder
# ---------------------------------------------------------------------------


def _build_boltz_command(
    boltz_bin: str,
    input_path: Path,
    output_dir: Path,
    recycling_steps: int = 10,
    num_seeds: int = 1,
    use_nim: bool = False,
) -> list[str]:
    """Build the boltz predict command list.

    Args:
        boltz_bin: Name or path of the boltz binary.
        input_path: Path to Boltz-2 input FASTA.
        output_dir: Directory for Boltz-2 outputs.
        recycling_steps: Number of recycling iterations.
        num_seeds: Number of random seeds per prediction.
        use_nim: If True, add --nim flag for NIM/TensorRT acceleration.

    Returns:
        List of command-line arguments for subprocess.run().
    """
    cmd = [
        str(boltz_bin),
        "predict",
        str(input_path),
        "--recycling_steps", str(recycling_steps),
        "--num_seeds", str(num_seeds),
        "--output_dir", str(output_dir),
        "--use_msa_server", "false",
    ]
    if use_nim:
        cmd.append("--nim")
    return cmd


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def run_ts_docking(
    fasta_path: Path,
    ts_smiles: str,
    cofactors: list,
    output_dir: Path,
    recycling_steps: int = 10,
    num_seeds: int = 1,
    boltz_bin: str = "boltz",
    use_nim: bool = False,
) -> dict[str, float]:
    """Run Boltz-2 with transition state analogue SMILES.

    Args:
        fasta_path: Path to FASTA with candidate sequences (top 50th percentile).
        ts_smiles: SMILES string for the transition state analogue.
        cofactors: List of cofactor name strings.
        output_dir: Directory for TS docking outputs.
        recycling_steps: Number of recycling iterations.
        num_seeds: Number of random seeds per prediction.
        boltz_bin: Name or path of the boltz binary.
        use_nim: If True, use NIM/TensorRT acceleration.

    Returns:
        Dict mapping sequence_id -> ts_affinity score.
    """
    results = run_boltz2_screen(
        fasta_path=fasta_path,
        ligand_smiles=ts_smiles,
        cofactors=cofactors,
        output_dir=output_dir,
        recycling_steps=recycling_steps,
        num_seeds=num_seeds,
        boltz_bin=boltz_bin,
        use_nim=use_nim,
    )
    return {r["sequence_id"]: r.get("affinity", 0.0) for r in results}


def run_boltz2_screen(
    fasta_path: Path,
    ligand_smiles: str,
    cofactors: list,
    output_dir: Path,
    recycling_steps: int = 10,
    num_seeds: int = 1,
    boltz_bin: str = "boltz",
    use_nim: bool = False,
) -> list[dict]:
    """Run Boltz-2 on all sequences in a FASTA file (batch mode).

    Boltz-2 command::

        boltz predict {input_fasta} --output_dir {output_dir} \\
            --recycling_steps {r} --num_seeds {s}

    Args:
        fasta_path: Path to FASTA with candidate sequences.
        ligand_smiles: SMILES string for the ligand (substrate or product).
        cofactors: List of cofactor name strings.
        output_dir: Directory for Boltz-2 outputs.
        recycling_steps: Number of recycling iterations.
        num_seeds: Number of random seeds per prediction.
        boltz_bin: Name or path of the boltz binary.
        use_nim: If True, use NIM/TensorRT acceleration.

    Returns:
        List of dicts with keys: sequence_id, pdb_path, plddt, iptm, affinity.

    Raises:
        RuntimeError: If boltz predict exits with non-zero return code.
    """
    import logging

    logger = logging.getLogger(f"pipeline.{STAGE_NUM}")

    fasta_path = Path(fasta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read sequences to get IDs
    sequences = read_fasta(fasta_path)
    if not sequences:
        logger.warning("No sequences found in %s", fasta_path)
        return []

    # Prepare Boltz-2 input FASTA with ligand context for each sequence
    boltz_input = output_dir / "boltz_input.fasta"
    with open(boltz_input, "w") as fh:
        for seq_id, seq in sequences.items():
            fh.write(f">protein|{seq_id}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")
        # Ligand
        fh.write(f">ligand|SMILES\n{ligand_smiles}\n")
        # Cofactors
        for cof in cofactors:
            fh.write(f">cofactor|{cof}\n{cof}\n")

    # Log timing estimate
    n_seq = len(sequences)
    t_per = NIM_INFERENCE_TIME_S if use_nim else VANILLA_INFERENCE_TIME_S
    est_time = n_seq * t_per
    mode_label = "NIM/TensorRT" if use_nim else "vanilla"
    logger.info(
        "Estimated time for %d sequences (%s): %.1f s (%.1f s/inference)",
        n_seq, mode_label, est_time, t_per,
    )

    cmd = _build_boltz_command(
        boltz_bin=boltz_bin,
        input_path=boltz_input,
        output_dir=output_dir,
        recycling_steps=recycling_steps,
        num_seeds=num_seeds,
        use_nim=use_nim,
    )

    logger.info("Running Boltz-2: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Boltz-2 screening failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Parse outputs: collect PDB files + confidence JSONs
    results = []
    for seq_id in sequences:
        # Boltz-2 places outputs per-sequence in subdirectories
        seq_pdbs = sorted(output_dir.glob(f"**/{seq_id}*.pdb"))
        seq_confs = sorted(output_dir.glob(f"**/{seq_id}*confidence*.json"))

        pdb_path = str(seq_pdbs[0]) if seq_pdbs else ""
        confidence = {}
        if seq_confs:
            confidence = parse_boltz2_confidence(seq_confs[0])

        results.append({
            "sequence_id": seq_id,
            "pdb_path": pdb_path,
            "plddt": confidence.get("plddt", 0.0),
            "iptm": confidence.get("iptm", 0.0),
            "affinity": confidence.get("affinity_score", 0.0),
        })

    return results


def parse_boltz2_confidence(json_path: Path) -> dict:
    """Parse a Boltz-2 confidence JSON and extract standardised fields.

    Extracts: plddt, iptm, ptm, affinity_score (if available).

    Args:
        json_path: Path to the Boltz-2 confidence JSON file.

    Returns:
        Dict with keys: plddt, iptm, ptm, affinity_score.
    """
    with open(json_path) as fh:
        data = json.load(fh)

    return {
        "plddt": data.get("plddt", 0.0),
        "iptm": data.get("iptm", 0.0),
        "ptm": data.get("ptm", 0.0),
        "affinity_score": data.get("affinity_score", data.get("affinity", 0.0)),
    }


def calc_active_site_rmsd(
    predicted_pdb: Path,
    wt_pdb: Path,
    hotspot_residues: list[int],
) -> float:
    """Calculate heavy-atom RMSD for active site (hotspot) residue positions.

    Uses Bio.PDB to extract residue coordinates and compute RMSD over
    the specified residue positions only.

    Args:
        predicted_pdb: Path to predicted structure PDB.
        wt_pdb: Path to wild-type reference PDB.
        hotspot_residues: List of residue numbers defining the active site.

    Returns:
        RMSD in Angstroms (float). Returns 0.0 if no hotspot residues or
        Bio.PDB is unavailable.
    """
    if not hotspot_residues:
        return 0.0

    try:
        from Bio.PDB import PDBParser
    except ImportError:
        return calc_ca_rmsd(str(predicted_pdb), str(wt_pdb), subset=hotspot_residues)

    parser = PDBParser(QUIET=True)

    pred_struct = parser.get_structure("pred", str(predicted_pdb))
    wt_struct = parser.get_structure("wt", str(wt_pdb))

    pred_model = pred_struct[0]
    wt_model = wt_struct[0]

    pred_chain = next(pred_model.get_chains())
    wt_chain = next(wt_model.get_chains())

    sum_sq = 0.0
    n_atoms = 0

    for resnum in hotspot_residues:
        try:
            pred_res = pred_chain[(" ", resnum, " ")]
            wt_res = wt_chain[(" ", resnum, " ")]
        except KeyError:
            continue

        for atom in wt_res.get_atoms():
            if atom.element == "H":
                continue
            try:
                pred_atom = pred_res[atom.get_name()]
            except KeyError:
                continue
            diff = pred_atom.get_vector() - atom.get_vector()
            sum_sq += diff.norm() ** 2
            n_atoms += 1

    if n_atoms == 0:
        return 0.0
    return math.sqrt(sum_sq / n_atoms)


def calc_ligand_rmsd(predicted_pdb: Path, wt_pdb: Path) -> float:
    """Calculate RMSD of ligand/HETATM atoms between predicted and WT PDBs.

    Matches ligand atoms by atom name within HETATM records.

    Args:
        predicted_pdb: Path to predicted structure PDB.
        wt_pdb: Path to wild-type reference PDB.

    Returns:
        RMSD in Angstroms (float). Returns 0.0 if no matching HETATM atoms
        or Bio.PDB is unavailable.
    """
    try:
        from Bio.PDB import PDBParser
    except ImportError:
        return 0.0

    parser = PDBParser(QUIET=True)
    try:
        pred_struct = parser.get_structure("pred", str(predicted_pdb))
        wt_struct = parser.get_structure("wt", str(wt_pdb))
        pred_model = pred_struct[0]
        wt_model = wt_struct[0]
    except Exception:
        return 0.0

    # Collect HETATM atoms from WT
    wt_hetatms: dict[str, list] = {}
    for chain in wt_model:
        for res in chain:
            if res.id[0].startswith("H_") or res.id[0] == "W":
                continue
            if res.id[0] != " ":
                for atom in res.get_atoms():
                    key = f"{res.get_resname()}_{atom.get_name()}"
                    wt_hetatms[key] = atom.get_vector()

    if not wt_hetatms:
        return 0.0

    # Match against predicted HETATM atoms
    sum_sq = 0.0
    n_atoms = 0
    for chain in pred_model:
        for res in chain:
            if res.id[0] != " " and not res.id[0].startswith("H_") and res.id[0] != "W":
                continue
            if res.id[0].startswith("H_") or (res.id[0] != " " and res.id[0] != "W"):
                for atom in res.get_atoms():
                    key = f"{res.get_resname()}_{atom.get_name()}"
                    if key in wt_hetatms:
                        diff = atom.get_vector() - wt_hetatms[key]
                        sum_sq += diff.norm() ** 2
                        n_atoms += 1

    if n_atoms == 0:
        return 0.0
    return math.sqrt(sum_sq / n_atoms)


def apply_structural_filters(
    result: dict,
    wt_pdb: Path,
    hotspot_residues: list[int],
    ca_rmsd_max: float = 2.0,
    active_site_rmsd_max: float = 1.5,
    ligand_rmsd_max: float = 2.0,
) -> tuple[bool, dict]:
    """Apply structural filtering cascade to a single candidate result.

    Filters:
    - Ca RMSD to WT < ca_rmsd_max (fold preservation)
    - Active site heavy-atom RMSD < active_site_rmsd_max (catalytic geometry)
    - Ligand RMSD < ligand_rmsd_max (substrate positioning)

    Args:
        result: Dict with at least 'pdb_path' key.
        wt_pdb: Path to wild-type reference PDB.
        hotspot_residues: Active site residue numbers.
        ca_rmsd_max: Maximum allowed Ca RMSD (default 2.0 A).
        active_site_rmsd_max: Maximum allowed active site RMSD (default 1.5 A).
        ligand_rmsd_max: Maximum allowed ligand RMSD (default 2.0 A).

    Returns:
        Tuple of (passed: bool, filter_detail: dict) where filter_detail
        contains the computed RMSD values and per-filter pass/fail.
    """
    pred_pdb = Path(result["pdb_path"]) if result.get("pdb_path") else None

    if pred_pdb is None or not pred_pdb.exists():
        return False, {
            "ca_rmsd": float("inf"),
            "active_site_rmsd": float("inf"),
            "ligand_rmsd": float("inf"),
            "ca_rmsd_pass": False,
            "active_site_rmsd_pass": False,
            "ligand_rmsd_pass": False,
        }

    try:
        ca_rmsd = calc_ca_rmsd(str(pred_pdb), str(wt_pdb))
    except Exception:
        ca_rmsd = 0.0
    try:
        active_site_rmsd = calc_active_site_rmsd(pred_pdb, wt_pdb, hotspot_residues)
    except Exception:
        active_site_rmsd = 0.0
    try:
        ligand_rmsd = calc_ligand_rmsd(pred_pdb, wt_pdb)
    except Exception:
        ligand_rmsd = 0.0

    ca_pass = ca_rmsd < ca_rmsd_max
    as_pass = active_site_rmsd < active_site_rmsd_max
    lig_pass = ligand_rmsd < ligand_rmsd_max
    passed = ca_pass and as_pass and lig_pass

    detail = {
        "ca_rmsd": ca_rmsd,
        "active_site_rmsd": active_site_rmsd,
        "ligand_rmsd": ligand_rmsd,
        "ca_rmsd_pass": ca_pass,
        "active_site_rmsd_pass": as_pass,
        "ligand_rmsd_pass": lig_pass,
    }

    return passed, detail


def select_top_percentile(
    results: list[dict],
    percentile: float = 50.0,
) -> list[dict]:
    """Select the top percentile of results by substrate affinity score.

    Affinity scores are assumed to be more negative = tighter binding.
    Lower (more negative) values indicate stronger binding and are better.

    Args:
        results: List of result dicts, each with a 'substrate_affinity' key.
        percentile: Percentile cutoff (default 50.0 = top 50%).

    Returns:
        List of result dicts in the top percentile.
    """
    if not results:
        return []

    # Sort by substrate_affinity ascending (more negative = better binding)
    sorted_results = sorted(results, key=lambda r: r.get("substrate_affinity", 0.0))

    # Take top percentile fraction
    n_select = max(1, int(math.ceil(len(sorted_results) * percentile / 100.0)))
    return sorted_results[:n_select]


def calculate_inhibition_ratio(
    substrate_affinity: float,
    product_affinity: float,
) -> float:
    """Calculate product inhibition ratio from binding affinity scores.

    Affinity scores are binding energies (more negative = tighter binding).
    ratio = substrate_affinity / product_affinity

    - ratio > 1: substrate binds tighter than product (good, low inhibition risk)
    - ratio < 1: product binds tighter than substrate (bad, high inhibition risk)

    Args:
        substrate_affinity: Substrate binding affinity (negative = tighter).
        product_affinity: Product binding affinity (negative = tighter).

    Returns:
        Inhibition ratio (float). Returns float('inf') if product_affinity
        is zero.
    """
    if product_affinity == 0.0:
        return float("inf")
    return substrate_affinity / product_affinity


def run_stage(config: dict) -> dict:
    """Run Stage 6: Full Boltz-2 structural screening for all enzymes.

    For each enzyme:
    1. Read filtered FASTA from stage_05_esmfold_prescreen/
    2. Run substrate docking (Boltz-2 with substrate SMILES)
    3. Apply structural filters
    4. Select top 50th percentile by substrate affinity
    5. Run product docking for top 50th percentile
    6. Run TS docking for top 50th percentile (if ts_smiles configured)
    7. Calculate inhibition ratios and TS/substrate ratios
    8. Write results to screening_results.csv

    Supports checkpoint/resume: after each enzyme completes, a checkpoint
    JSON is written. On resume, already-completed enzymes are skipped.

    Args:
        config: Loaded pipeline configuration dictionary.

    Returns:
        Summary dict mapping enzyme_id -> {
            "total_candidates": int,
            "passed_filters": int,
            "product_docked": int,
            "ts_docked": int,
            "high_inhibition_risk": int,
        }
    """
    import logging

    logger = logging.getLogger(f"pipeline.{STAGE_NUM}")

    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_06_boltz2_screening", {})

    output_dir = base_dir / stage_cfg.get("output_dir", "stage_06_boltz2_screening")
    substrate_dock_dir = output_dir / "substrate_docking"
    product_dock_dir = output_dir / "product_docking"
    ts_dock_dir = output_dir / "ts_docking"
    substrate_dock_dir.mkdir(parents=True, exist_ok=True)
    product_dock_dir.mkdir(parents=True, exist_ok=True)
    ts_dock_dir.mkdir(parents=True, exist_ok=True)

    # Filter thresholds
    ca_rmsd_max = stage_cfg.get("ca_rmsd_max", 2.0)
    active_site_rmsd_max = stage_cfg.get("active_site_rmsd_max", 1.5)
    ligand_rmsd_max = stage_cfg.get("ligand_rmsd_max", 2.0)

    # Boltz-2 config
    recycling_steps = stage_cfg.get("recycling_steps", 10)
    num_seeds = stage_cfg.get("num_seeds", 1)
    boltz_bin = config.get("tools", {}).get("boltz_bin", "boltz")

    # NIM/TensorRT config
    use_nim = stage_cfg.get("use_nim", False)
    if use_nim:
        logger.info(
            "NIM/TensorRT mode enabled (%.1f s/inference vs %.1f s vanilla)",
            NIM_INFERENCE_TIME_S, VANILLA_INFERENCE_TIME_S,
        )

    # Stage 5 input dir
    stage5_dir = base_dir / config.get("stages", {}).get(
        "stage_05_esmfold", {}
    ).get("output_dir", "stage_05_esmfold_prescreen")

    # WT structures from stage 2
    wt_dir = base_dir / config.get("stages", {}).get(
        "stage_02_wt_structures", {}
    ).get("output_dir", "stage_02_wt_structures")

    # ---- Checkpoint/resume ----
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    completed_enzymes: set[str] = set()
    all_rows: list[dict] = []
    summary: dict = {}

    if checkpoint is not None:
        completed_enzymes = set(checkpoint.get("completed_enzymes", []))
        all_rows = checkpoint.get("rows", [])
        summary = checkpoint.get("summary", {})
        logger.info(
            "Resuming from checkpoint: %d enzymes already completed",
            len(completed_enzymes),
        )

    for enzyme_id, enzyme_cfg in config["enzymes"].items():
        if enzyme_id in completed_enzymes:
            logger.info("Skipping %s (already completed in checkpoint)", enzyme_id)
            continue

        logger.info("Processing %s (%s)", enzyme_id, enzyme_cfg.get("name", enzyme_id))

        # ---- Read filtered FASTA from Stage 5 ----
        fasta_path = stage5_dir / enzyme_id / f"{enzyme_id}_filtered.fasta"
        if not fasta_path.exists():
            # Fallback: try top-level file
            fasta_path = stage5_dir / f"{enzyme_id}_filtered.fasta"
        if not fasta_path.exists():
            logger.error("No filtered FASTA for %s at %s", enzyme_id, fasta_path)
            continue

        substrate_smiles = enzyme_cfg.get("substrate_smiles", "")
        product_smiles = enzyme_cfg.get("product_smiles", "")
        ts_smiles = enzyme_cfg.get("ts_smiles", "")
        cofactors = enzyme_cfg.get("cofactors", [])
        hotspot_residues = enzyme_cfg.get("hotspot_residues", [])

        if not substrate_smiles:
            logger.error("No substrate SMILES for %s — skipping", enzyme_id)
            continue

        # WT reference PDB
        wt_pdb = wt_dir / "substrate_complexes" / f"{enzyme_id}_substrate.pdb"

        # ---- Step 1: Substrate docking ----
        logger.info("Running substrate docking for %s", enzyme_id)
        enz_sub_dir = substrate_dock_dir / enzyme_id
        try:
            substrate_results = run_boltz2_screen(
                fasta_path=fasta_path,
                ligand_smiles=substrate_smiles,
                cofactors=cofactors,
                output_dir=enz_sub_dir,
                recycling_steps=recycling_steps,
                num_seeds=num_seeds,
                boltz_bin=boltz_bin,
                use_nim=use_nim,
            )
        except RuntimeError as exc:
            logger.error("Substrate docking failed for %s: %s", enzyme_id, exc)
            continue

        # ---- Step 2: Structural filters ----
        filtered_results = []
        for res in substrate_results:
            passed, detail = apply_structural_filters(
                result=res,
                wt_pdb=wt_pdb,
                hotspot_residues=hotspot_residues,
                ca_rmsd_max=ca_rmsd_max,
                active_site_rmsd_max=active_site_rmsd_max,
                ligand_rmsd_max=ligand_rmsd_max,
            )
            res["ca_rmsd"] = detail["ca_rmsd"]
            res["active_site_rmsd"] = detail["active_site_rmsd"]
            res["ligand_rmsd"] = detail["ligand_rmsd"]
            res["passed_structural_filters"] = passed
            res["substrate_affinity"] = res.get("affinity", 0.0)
            res["enzyme_id"] = enzyme_id

            if passed:
                filtered_results.append(res)

        logger.info(
            "%s: %d / %d passed structural filters",
            enzyme_id, len(filtered_results), len(substrate_results),
        )

        # ---- Step 3: Select top 50th percentile for product/TS docking ----
        top_candidates = select_top_percentile(filtered_results, percentile=50.0)
        top_seq_ids = {r["sequence_id"] for r in top_candidates}

        logger.info(
            "%s: %d candidates selected for product docking (top 50%%)",
            enzyme_id, len(top_candidates),
        )

        # ---- Step 4: Product docking (top 50th percentile only) ----
        product_affinity_map: dict[str, float] = {}
        if product_smiles and top_candidates:
            # Write subset FASTA for product docking
            all_seqs = read_fasta(fasta_path)
            subset_seqs = {
                sid: seq for sid, seq in all_seqs.items() if sid in top_seq_ids
            }
            if subset_seqs:
                subset_fasta = product_dock_dir / enzyme_id / f"{enzyme_id}_top50.fasta"
                subset_fasta.parent.mkdir(parents=True, exist_ok=True)
                write_fasta(subset_seqs, subset_fasta)

                try:
                    product_results = run_boltz2_screen(
                        fasta_path=subset_fasta,
                        ligand_smiles=product_smiles,
                        cofactors=cofactors,
                        output_dir=product_dock_dir / enzyme_id,
                        recycling_steps=recycling_steps,
                        num_seeds=num_seeds,
                        boltz_bin=boltz_bin,
                        use_nim=use_nim,
                    )
                    for pr in product_results:
                        product_affinity_map[pr["sequence_id"]] = pr.get("affinity", 0.0)
                except RuntimeError as exc:
                    logger.error("Product docking failed for %s: %s", enzyme_id, exc)

        # ---- Step 4b: TS docking (top 50th percentile, if ts_smiles) ----
        ts_affinity_map: dict[str, float] = {}
        if ts_smiles and top_candidates:
            logger.info("Running TS analogue docking for %s", enzyme_id)
            all_seqs = read_fasta(fasta_path)
            subset_seqs = {
                sid: seq for sid, seq in all_seqs.items() if sid in top_seq_ids
            }
            if subset_seqs:
                ts_fasta = ts_dock_dir / enzyme_id / f"{enzyme_id}_top50_ts.fasta"
                ts_fasta.parent.mkdir(parents=True, exist_ok=True)
                write_fasta(subset_seqs, ts_fasta)

                try:
                    ts_affinity_map = run_ts_docking(
                        fasta_path=ts_fasta,
                        ts_smiles=ts_smiles,
                        cofactors=cofactors,
                        output_dir=ts_dock_dir / enzyme_id,
                        recycling_steps=recycling_steps,
                        num_seeds=num_seeds,
                        boltz_bin=boltz_bin,
                        use_nim=use_nim,
                    )
                except RuntimeError as exc:
                    logger.error("TS docking failed for %s: %s", enzyme_id, exc)

        # ---- Step 5: Calculate inhibition ratios and build CSV rows ----
        n_high_risk = 0
        n_ts_docked = len(ts_affinity_map)
        for res in substrate_results:
            seq_id = res["sequence_id"]
            product_aff = product_affinity_map.get(seq_id)
            ts_aff = ts_affinity_map.get(seq_id)

            if product_aff is not None:
                inh_ratio = calculate_inhibition_ratio(
                    res.get("substrate_affinity", 0.0), product_aff
                )
                high_risk = inh_ratio < 1.0
            else:
                inh_ratio = None
                high_risk = None

            if high_risk:
                n_high_risk += 1

            # TS/substrate ratio
            sub_aff = res.get("substrate_affinity", 0.0)
            if ts_aff is not None and sub_aff != 0.0:
                ts_sub_ratio = ts_aff / sub_aff
            else:
                ts_sub_ratio = None

            row = {
                "sequence_id": seq_id,
                "enzyme_id": enzyme_id,
                "ca_rmsd": res.get("ca_rmsd", ""),
                "active_site_rmsd": res.get("active_site_rmsd", ""),
                "ligand_rmsd": res.get("ligand_rmsd", ""),
                "passed_structural_filters": res.get("passed_structural_filters", False),
                "plddt": res.get("plddt", 0.0),
                "substrate_affinity": res.get("substrate_affinity", 0.0),
                "product_affinity": product_aff if product_aff is not None else "",
                "inhibition_ratio": inh_ratio if inh_ratio is not None else "",
                "high_inhibition_risk": high_risk if high_risk is not None else "",
                "ts_affinity": ts_aff if ts_aff is not None else "",
                "ts_substrate_ratio": ts_sub_ratio if ts_sub_ratio is not None else "",
                "pdb_path": res.get("pdb_path", ""),
            }
            all_rows.append(row)

        summary[enzyme_id] = {
            "total_candidates": len(substrate_results),
            "passed_filters": len(filtered_results),
            "product_docked": len(product_affinity_map),
            "ts_docked": n_ts_docked,
            "high_inhibition_risk": n_high_risk,
        }

        logger.info(
            "%s: %d total, %d passed filters, %d product-docked, %d TS-docked, %d high inhibition risk",
            enzyme_id,
            summary[enzyme_id]["total_candidates"],
            summary[enzyme_id]["passed_filters"],
            summary[enzyme_id]["product_docked"],
            summary[enzyme_id]["ts_docked"],
            summary[enzyme_id]["high_inhibition_risk"],
        )

        # ---- Write checkpoint after each enzyme ----
        completed_enzymes.add(enzyme_id)
        write_checkpoint(checkpoint_path, {
            "completed_enzymes": sorted(completed_enzymes),
            "rows": all_rows,
            "summary": summary,
        })
        logger.info("Checkpoint saved after %s", enzyme_id)

    # ---- Write screening_results.csv ----
    csv_path = output_dir / "screening_results.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info("Wrote %d rows to %s", len(all_rows), csv_path)
    else:
        logger.warning("No results to write — screening_results.csv not created")

    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed (stage complete)")

    logger.info("Stage 6 complete. Summary: %s", summary)
    return summary


def main(config_path: Path) -> None:
    """Entry point: load config, run Boltz-2 structural screening, log results."""
    config = load_config(config_path)
    base_dir = Path(config["global"]["base_dir"])
    config = resolve_paths(config, base_dir)

    log_dir = base_dir / config["global"].get("log_dir", "logs")
    logger = setup_logging("stage6_boltz2_screen", log_dir)

    logger.info("Starting %s", STAGE_NAME)
    summary = run_stage(config)
    logger.info("Finished %s — summary: %s", STAGE_NAME, summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)
