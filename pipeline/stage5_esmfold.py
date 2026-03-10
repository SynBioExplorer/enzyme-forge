"""Stage 5: ESMFold Pre-screening with softened pLDDT filtering.

Rapidly eliminates clearly misfolded designs using ESMFold single-sequence
structure prediction (~1 s per sequence, no MSA required).

Filtering (softened from v0.1 per Garcia et al. 2025):
- Hard floor: discard sequences with mean pLDDT < 60
- Hard filter: discard sequences with Ca RMSD > 5 A to WT
- Soft signal: record pLDDT as continuous score for Stage 7 ranking

Expected retention: ~60-70% of input (~19,000 sequences).

CLI: python pipeline/stage5_esmfold.py --config config.yaml
"""
from __future__ import annotations

import argparse
import io
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pipeline.config import load_config, get_enzyme_config, resolve_paths
    from pipeline.utils import (
        setup_logging,
        read_fasta,
        write_fasta,
        write_scores,
    )
except ImportError:
    import csv as _csv

    import yaml

    def load_config(p):  # type: ignore[misc]
        return yaml.safe_load(open(p))

    def get_enzyme_config(cfg, eid):  # type: ignore[misc]
        return cfg.get("enzymes", {}).get(eid, {})

    def resolve_paths(cfg, base):  # type: ignore[misc]
        return cfg

    def setup_logging(n, d, **kw):  # type: ignore[misc]
        logging.basicConfig()
        return logging.getLogger(n)

    def read_fasta(p):  # type: ignore[misc]
        seqs: dict[str, str] = {}
        with open(p) as f:
            h, s = None, []  # type: ignore[assignment]
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if h:
                        seqs[h] = "".join(s)
                    h, s = line[1:].split()[0], []
                elif h:
                    s.append(line)
            if h:
                seqs[h] = "".join(s)
        return seqs

    def write_fasta(seqs, p):  # type: ignore[misc]
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for h, s in seqs.items():
                f.write(f">{h}\n{s}\n")

    def write_scores(scores, p):  # type: ignore[misc]
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        if not scores:
            return
        import csv
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=scores[0].keys())
            w.writeheader()
            w.writerows(scores)


# ---------------------------------------------------------------------------
# Module constants (StageModuleProtocol)
# ---------------------------------------------------------------------------

STAGE_NAME = "Stage 5: ESMFold Pre-screening"
STAGE_NUM = "5"
INPUT_DESCRIPTION = "Designed FASTA sequences from Stage 4"
OUTPUT_DESCRIPTION = (
    "Filtered FASTA + plddt_scores.csv + PDB files in stage_05_esmfold_prescreen/"
)

logger = logging.getLogger(__name__)

# Module-level model cache
_cached_model = None
_cached_tokenizer = None


# ---------------------------------------------------------------------------
# ESMFold model loading
# ---------------------------------------------------------------------------

def load_esmfold_model(
    device: str = "cuda",
) -> tuple:
    """Load ESMFold model via HuggingFace transformers.

    The model is cached at module level so subsequent calls are free.

    Args:
        device: Device string ("cuda" or "cpu").

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        ImportError: If transformers is not installed.
    """
    global _cached_model, _cached_tokenizer

    if _cached_model is not None and _cached_tokenizer is not None:
        return _cached_model, _cached_tokenizer

    from transformers import AutoTokenizer, EsmForProteinFolding

    logger.info("Loading ESMFold model (facebook/esmfold_v1) ...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.to(device)
    model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    logger.info("ESMFold model loaded on %s", device)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Single-sequence prediction
# ---------------------------------------------------------------------------

def predict_structure_esmfold(
    sequence: str,
    model: Any,
    tokenizer: Any,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run ESMFold prediction for a single sequence.

    Args:
        sequence: Amino-acid sequence string.
        model: ESMFold model instance.
        tokenizer: ESMFold tokenizer instance.
        device: Device string.

    Returns:
        Dictionary with keys:
        - plddt: float (mean pLDDT over all residues)
        - plddt_per_residue: list[float]
        - pdb_string: str (PDB-format string of the predicted structure)
    """
    import torch
    from transformers.models.esm.openfold_utils.protein import to_pdb, Protein

    with torch.no_grad():
        inputs = tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

    # pLDDT is stored in outputs.plddt — shape (1, L)
    plddt_tensor = outputs.plddt[0]  # (L,)
    plddt_per_residue = plddt_tensor.cpu().numpy().tolist()
    mean_plddt = float(np.mean(plddt_per_residue))

    # Convert to PDB string
    pdb_string = _output_to_pdb(outputs, sequence)

    return {
        "plddt": mean_plddt,
        "plddt_per_residue": plddt_per_residue,
        "pdb_string": pdb_string,
    }


def _output_to_pdb(outputs: Any, sequence: str) -> str:
    """Convert ESMFold model output to a PDB string.

    Attempts the standard transformers conversion. Falls back to manual
    coordinate extraction if the helper is unavailable.
    """
    import torch

    try:
        # transformers >= 4.x provides a conversion utility
        from transformers.models.esm.openfold_utils.protein import (
            Protein,
            to_pdb,
        )

        # positions shape: (1, L, 37, 3) or (1, L, 14, 3)
        positions = outputs.positions[-1]  # last recycling step
        atom_mask = outputs.atom37_atom_exists[0]
        aatype = outputs.esmfold.trunk.structure_module.aatype
        residue_index = torch.arange(len(sequence), device=positions.device)
        b_factors = outputs.plddt[0].unsqueeze(-1).expand(-1, atom_mask.shape[-1])

        protein = Protein(
            atom_positions=positions[0].cpu().numpy(),
            atom_mask=atom_mask.cpu().numpy(),
            aatype=aatype[0].cpu().numpy(),
            residue_index=residue_index.cpu().numpy(),
            b_factors=b_factors.cpu().numpy(),
            chain_index=np.zeros(len(sequence), dtype=np.int32),
        )
        return to_pdb(protein)
    except Exception:
        # Fallback: write coordinates manually from output positions
        positions = outputs.positions[-1][0].cpu().numpy()  # (L, 37, 3)
        plddt = outputs.plddt[0].cpu().numpy()

        lines: list[str] = []
        atom_idx = 1
        for i in range(len(sequence)):
            # CA is atom index 1 in atom37 representation
            x, y, z = positions[i, 1]
            bfac = float(plddt[i])
            lines.append(
                f"ATOM  {atom_idx:5d}  CA  ALA A{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}"
            )
            atom_idx += 1
        lines.append("END")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def batch_predict(
    sequences: dict[str, str],
    model: Any,
    tokenizer: Any,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, dict]:
    """Predict structures for multiple sequences, handling OOM gracefully.

    Sequences are processed individually (ESMFold doesn't natively batch
    variable-length sequences well), but grouped for progress logging.
    On OOM, the current batch is retried with batch_size // 2. If
    batch_size reaches 1 and OOM persists, the offending sequence is
    skipped.

    Args:
        sequences: {sequence_id: amino_acid_sequence}.
        model: ESMFold model.
        tokenizer: ESMFold tokenizer.
        batch_size: Logical batch size for progress logging / OOM recovery.
        device: Device string.

    Returns:
        {sequence_id: {"plddt": float, "pdb_string": str}}.
    """
    seq_ids = list(sequences.keys())
    results: dict[str, dict] = {}
    total = len(seq_ids)

    idx = 0
    current_batch_size = batch_size

    while idx < total:
        batch_ids = seq_ids[idx : idx + current_batch_size]
        try:
            for sid in batch_ids:
                seq = sequences[sid]
                pred = predict_structure_esmfold(seq, model, tokenizer, device)
                results[sid] = {
                    "plddt": pred["plddt"],
                    "pdb_string": pred["pdb_string"],
                }
            logger.info(
                "Predicted %d/%d sequences", min(idx + len(batch_ids), total), total
            )
            idx += current_batch_size
            current_batch_size = batch_size  # reset after success
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "CUDA" in str(exc):
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(
                        "OOM at index %d — reducing batch size to %d",
                        idx, current_batch_size,
                    )
                else:
                    # Single sequence caused OOM — skip it
                    skipped_id = seq_ids[idx]
                    logger.error(
                        "OOM on single sequence %s (len=%d) — skipping",
                        skipped_id, len(sequences[skipped_id]),
                    )
                    idx += 1
            else:
                raise

    return results


# ---------------------------------------------------------------------------
# Ca RMSD from PDB strings
# ---------------------------------------------------------------------------

def calc_ca_rmsd_from_strings(pdb_str1: str, pdb_str2: str) -> float:
    """Calculate Ca RMSD between two structures given as PDB strings.

    Uses Bio.PDB.Superimposer for optimal superposition. When the two
    structures have different numbers of residues the comparison is
    truncated to the shorter length.

    Args:
        pdb_str1: PDB-format string of structure 1.
        pdb_str2: PDB-format string of structure 2.

    Returns:
        RMSD in Angstroms after optimal superposition.
    """
    from Bio.PDB import PDBParser, Superimposer

    parser = PDBParser(QUIET=True)

    struct1 = parser.get_structure("s1", io.StringIO(pdb_str1))
    struct2 = parser.get_structure("s2", io.StringIO(pdb_str2))

    def _get_ca(struct):
        cas = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    het = residue.get_id()[0]
                    if het.strip() and het != " ":
                        continue
                    for atom in residue:
                        if atom.get_name() == "CA":
                            cas.append(atom)
            break  # first model only
        return cas

    ca1 = _get_ca(struct1)
    ca2 = _get_ca(struct2)

    n = min(len(ca1), len(ca2))
    if n == 0:
        return 0.0

    ca1 = ca1[:n]
    ca2 = ca2[:n]

    sup = Superimposer()
    sup.set_atoms(ca1, ca2)
    return float(sup.rms)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_filters(
    sequence_id: str,
    pdb_string: str,
    plddt: float,
    wt_pdb_path: Path,
    plddt_floor: float = 60.0,
    rmsd_max: float = 5.0,
) -> tuple[bool, dict[str, Any]]:
    """Apply hard-floor and hard-filter criteria to a predicted structure.

    1. Hard floor: reject if mean pLDDT < *plddt_floor*.
    2. Hard filter: reject if Ca RMSD to WT > *rmsd_max*.

    Args:
        sequence_id: Identifier for logging.
        pdb_string: Predicted PDB string.
        plddt: Mean pLDDT score.
        wt_pdb_path: Path to the WT PDB file.
        plddt_floor: Minimum acceptable mean pLDDT (default 60).
        rmsd_max: Maximum acceptable Ca RMSD to WT in Angstroms (default 5).

    Returns:
        Tuple of (passed: bool, info: dict) where *info* contains:
        - passed_plddt_filter: bool
        - passed_rmsd_filter: bool
        - passed_all_filters: bool
        - rmsd_to_wt: float | None
        - reason: str (empty if passed)
    """
    info: dict[str, Any] = {
        "sequence_id": sequence_id,
        "mean_plddt": plddt,
        "passed_plddt_filter": True,
        "passed_rmsd_filter": True,
        "passed_all_filters": True,
        "rmsd_to_wt": None,
        "reason": "",
    }

    # 1. pLDDT hard floor
    if plddt < plddt_floor:
        info["passed_plddt_filter"] = False
        info["passed_all_filters"] = False
        info["reason"] = f"pLDDT {plddt:.1f} < {plddt_floor}"
        return False, info

    # 2. Ca RMSD hard filter
    wt_pdb_str = wt_pdb_path.read_text()
    rmsd = calc_ca_rmsd_from_strings(pdb_string, wt_pdb_str)
    info["rmsd_to_wt"] = rmsd

    if rmsd > rmsd_max:
        info["passed_rmsd_filter"] = False
        info["passed_all_filters"] = False
        info["reason"] = f"RMSD {rmsd:.2f} > {rmsd_max}"
        return False, info

    return True, info


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(config: dict[str, Any]) -> dict[str, Any]:
    """Execute Stage 5: ESMFold pre-screening for all enzymes.

    Steps per enzyme:
    1. Read designed sequences from stage_04_ligandmpnn/
    2. Batch-predict structures with ESMFold
    3. Apply hard floor (pLDDT) and hard filter (RMSD) vs WT
    4. Write passing sequences to stage_05_esmfold_prescreen/{enzyme_id}_filtered.fasta
    5. Write all PDB files to stage_05_esmfold_prescreen/structures/
    6. Write plddt_scores.csv with all scores (including filtered-out)

    Args:
        config: Loaded pipeline configuration dict.

    Returns:
        {"retained": dict, "filtered": dict, "retention_rates": dict}
    """
    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_05_esmfold", {})
    output_dir_name = stage_cfg.get("output_dir", "stage_05_esmfold_prescreen")
    output_dir = base_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    structures_dir = output_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    plddt_floor = float(stage_cfg.get("plddt_hard_floor", 60))
    rmsd_max = float(stage_cfg.get("ca_rmsd_max", 5.0))
    batch_size = int(stage_cfg.get("batch_size", 8))
    device = str(stage_cfg.get("device", "cuda"))

    # Input directories
    input_dir = base_dir / config.get("stages", {}).get(
        "stage_04_ligandmpnn", {}
    ).get("output_dir", "stage_04_ligandmpnn")

    wt_dir = base_dir / config.get("stages", {}).get(
        "stage_02_wt_structures", {}
    ).get("output_dir", "stage_02_wt_structures")

    # Load ESMFold model once
    model, tokenizer = load_esmfold_model(device)

    enzyme_ids = sorted(config.get("enzymes", {}).keys())
    all_scores: list[dict] = []
    retained: dict[str, int] = {}
    filtered: dict[str, int] = {}
    retention_rates: dict[str, float] = {}

    for enzyme_id in enzyme_ids:
        logger.info("Processing %s ...", enzyme_id)

        # Find input FASTA
        input_fasta = input_dir / f"{enzyme_id}_designed.fasta"
        if not input_fasta.exists():
            input_fasta = input_dir / f"{enzyme_id}.fasta"
        if not input_fasta.exists():
            logger.warning(
                "No input FASTA found for %s in %s — skipping", enzyme_id, input_dir
            )
            continue

        sequences = read_fasta(input_fasta)
        if not sequences:
            logger.warning("Empty FASTA for %s — skipping", enzyme_id)
            continue

        logger.info("%s: %d input sequences", enzyme_id, len(sequences))

        # Batch predict
        predictions = batch_predict(
            sequences, model, tokenizer, batch_size=batch_size, device=device
        )

        # Find WT PDB for RMSD comparison
        wt_pdb = wt_dir / "substrate_complexes" / f"{enzyme_id}.pdb"
        if not wt_pdb.exists():
            wt_pdb = wt_dir / f"{enzyme_id}.pdb"
        if not wt_pdb.exists():
            logger.error(
                "WT PDB not found for %s — skipping RMSD filter", enzyme_id
            )
            wt_pdb = None

        # Apply filters and collect results
        passing_seqs: dict[str, str] = {}
        enzyme_retained = 0
        enzyme_filtered = 0

        for seq_id, pred in predictions.items():
            # Save PDB file for every predicted structure
            pdb_path = structures_dir / f"{seq_id}.pdb"
            pdb_path.write_text(pred["pdb_string"])

            if wt_pdb is not None:
                passed, info = apply_filters(
                    sequence_id=seq_id,
                    pdb_string=pred["pdb_string"],
                    plddt=pred["plddt"],
                    wt_pdb_path=wt_pdb,
                    plddt_floor=plddt_floor,
                    rmsd_max=rmsd_max,
                )
            else:
                # Without WT, only apply pLDDT floor
                passed = pred["plddt"] >= plddt_floor
                info = {
                    "sequence_id": seq_id,
                    "mean_plddt": pred["plddt"],
                    "passed_plddt_filter": pred["plddt"] >= plddt_floor,
                    "passed_rmsd_filter": True,
                    "passed_all_filters": passed,
                    "rmsd_to_wt": None,
                    "reason": "" if passed else f"pLDDT {pred['plddt']:.1f} < {plddt_floor}",
                }

            info["enzyme_id"] = enzyme_id
            all_scores.append({
                "sequence_id": info["sequence_id"],
                "enzyme_id": enzyme_id,
                "mean_plddt": info["mean_plddt"],
                "rmsd_to_wt": info.get("rmsd_to_wt"),
                "passed_plddt_filter": info["passed_plddt_filter"],
                "passed_rmsd_filter": info["passed_rmsd_filter"],
                "passed_all_filters": info["passed_all_filters"],
            })

            if passed:
                passing_seqs[seq_id] = sequences[seq_id]
                enzyme_retained += 1
            else:
                enzyme_filtered += 1

        # Write filtered FASTA
        output_fasta = output_dir / f"{enzyme_id}_filtered.fasta"
        write_fasta(passing_seqs, output_fasta)

        retained[enzyme_id] = enzyme_retained
        filtered[enzyme_id] = enzyme_filtered
        total = enzyme_retained + enzyme_filtered
        rate = (enzyme_retained / total * 100) if total > 0 else 0.0
        retention_rates[enzyme_id] = rate

        logger.info(
            "%s: %d/%d retained (%.1f%%), %d filtered out",
            enzyme_id, enzyme_retained, total, rate, enzyme_filtered,
        )

    # Write scores CSV
    scores_csv = output_dir / "plddt_scores.csv"
    write_scores(all_scores, scores_csv)
    logger.info("Wrote scores to %s (%d rows)", scores_csv, len(all_scores))

    return {
        "retained": retained,
        "filtered": filtered,
        "retention_rates": retention_rates,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(config_path: Path) -> None:
    """Load config, set up logging, run Stage 5."""
    config_path = Path(config_path)
    config = load_config(config_path)

    base_dir = Path(config["global"]["base_dir"])
    config = resolve_paths(config, base_dir)

    log_dir = base_dir / config["global"].get("log_dir", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "stage5_esmfold.log"),
        ],
    )

    logger.info("=" * 60)
    logger.info("%s — starting", STAGE_NAME)
    logger.info("Config: %s", config_path)
    logger.info("=" * 60)

    results = run_stage(config)

    logger.info("=" * 60)
    logger.info("%s — complete", STAGE_NAME)
    for eid, rate in results["retention_rates"].items():
        logger.info(
            "  %s: %d retained (%.1f%%)",
            eid, results["retained"][eid], rate,
        )
    logger.info("=" * 60)


def _cli() -> None:
    """Argument parser wrapper for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Stage 5: ESMFold Pre-screening",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    _cli()
