"""Stage 4: Sequence Design with LigandMPNN.

Two design branches:
  A) RFdiffusion3 backbones → LigandMPNN at T=0.1, 0.2 (8–16 batches each)
  B) Direct WT diversification → LigandMPNN on WT Boltz-2 structures at T=0.1, 0.2, 0.4
     (250 seqs/enzyme/temperature)

Fixed catalytic residues, --pack_side_chains 1, --pack_with_ligand_context 1,
--omit_aa C. Outputs merged FASTA + ligandmpnn_scores.csv in stage_04_ligandmpnn/.

CLI: python pipeline/stage4_ligandmpnn.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

try:
    from pipeline.config import load_config, get_enzyme_config, resolve_paths
    from pipeline.utils import setup_logging, read_fasta, write_fasta, write_scores
except ImportError:
    import yaml
    import csv

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
                    h, s = line[1:], []
                else:
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
        import csv as _csv

        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=scores[0].keys())
            w.writeheader()
            w.writerows(scores)


# ---------------------------------------------------------------------------
# Module constants (StageModuleProtocol)
# ---------------------------------------------------------------------------

STAGE_NAME = "Stage 4: Sequence Design (LigandMPNN)"
STAGE_NUM = "4"
INPUT_DESCRIPTION = "Backbone PDBs from Stage 3, WT PDBs from Stage 2, substrate SMILES"
OUTPUT_DESCRIPTION = "FASTA files + ligandmpnn_scores.csv in stage_04_ligandmpnn/"

logger = logging.getLogger(__name__)

# Default temperatures for each branch
BACKBONE_TEMPERATURES = [0.1, 0.2]
WT_TEMPERATURES = [0.1, 0.2, 0.4]
WT_SEQS_PER_TEMP = 250


# ---------------------------------------------------------------------------
# LigandMPNN execution
# ---------------------------------------------------------------------------

def run_ligandmpnn(
    pdb_path: Path,
    output_dir: Path,
    temperature: float,
    n_batches: int,
    fixed_residues: list[int],
    ligandmpnn_dir: Path,
    omit_aa: str = "C",
    seed: int = 42,
) -> tuple[Path, Path]:
    """Run LigandMPNN on a single PDB.

    Args:
        pdb_path: Path to input PDB file.
        output_dir: Directory for outputs.
        temperature: Sampling temperature.
        n_batches: Number of sequence batches.
        fixed_residues: Residue numbers to keep fixed.
        ligandmpnn_dir: LigandMPNN installation directory.
        omit_aa: Amino acids to omit (default "C").
        seed: Random seed.

    Returns:
        Tuple of (fasta_output_path, scores_json_path).

    Raises:
        RuntimeError: If the LigandMPNN command fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_residues_str = " ".join(str(r) for r in fixed_residues)

    cmd = [
        "python", str(Path(ligandmpnn_dir) / "run.py"),
        "--model_type", "ligand_mpnn",
        "--pdb_path", str(pdb_path),
        "--out_folder", str(output_dir),
        "--temperature", str(temperature),
        "--number_of_batches", str(n_batches),
        "--pack_side_chains", "1",
        "--pack_with_ligand_context", "1",
        "--omit_aa", omit_aa,
        "--seed", str(seed),
    ]

    if fixed_residues:
        cmd.extend(["--fixed_residues", fixed_residues_str])

    logger.info("Running LigandMPNN: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"LigandMPNN failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # LigandMPNN outputs: seqs/ directory with .fa files and scores
    seqs_dir = output_dir / "seqs"
    fasta_files = sorted(seqs_dir.glob("*.fa")) if seqs_dir.exists() else []

    if not fasta_files:
        # Fallback: check for .fasta files directly in output_dir
        fasta_files = sorted(output_dir.glob("*.fa"))

    fasta_path = fasta_files[0] if fasta_files else output_dir / "designed.fa"
    scores_path = output_dir / "scores.json"

    return fasta_path, scores_path


# ---------------------------------------------------------------------------
# Backbone design branch (Branch A)
# ---------------------------------------------------------------------------

def _parse_ligandmpnn_output(
    fasta_path: Path,
    scores_path: Path,
    temperature: float,
    backbone_name: str,
) -> list[dict[str, Any]]:
    """Parse LigandMPNN FASTA + scores into a list of sequence records.

    Returns:
        List of dicts with keys: sequence, sequence_id, score, temperature, backbone.
    """
    records: list[dict[str, Any]] = []

    # Parse sequences
    sequences: dict[str, str] = {}
    if fasta_path.exists():
        sequences = read_fasta(fasta_path)

    # Parse scores
    score_map: dict[str, float] = {}
    if scores_path.exists():
        try:
            with open(scores_path) as fh:
                score_data = json.load(fh)
            if isinstance(score_data, dict):
                score_map = {
                    k: float(v) if isinstance(v, (int, float)) else 0.0
                    for k, v in score_data.items()
                }
        except (json.JSONDecodeError, ValueError):
            pass

    for seq_id, sequence in sequences.items():
        records.append({
            "sequence": sequence,
            "sequence_id": seq_id,
            "score": score_map.get(seq_id, 0.0),
            "temperature": temperature,
            "backbone": backbone_name,
        })

    return records


def design_backbone(
    backbone_pdb: Path,
    enzyme_config: dict[str, Any],
    output_dir: Path,
    temperatures: list[float],
    n_batches_per_temp: int,
    ligandmpnn_dir: Path,
) -> list[dict[str, Any]]:
    """Run LigandMPNN at each temperature for one backbone.

    Args:
        backbone_pdb: Path to backbone PDB.
        enzyme_config: Enzyme configuration dict (must include fixed_residues).
        output_dir: Base output directory.
        temperatures: List of sampling temperatures.
        n_batches_per_temp: Number of batches per temperature.
        ligandmpnn_dir: LigandMPNN installation directory.

    Returns:
        List of sequence record dicts.
    """
    fixed_residues = enzyme_config.get("fixed_residues", [])
    backbone_name = backbone_pdb.stem
    all_records: list[dict[str, Any]] = []

    for temp in temperatures:
        temp_dir = output_dir / f"{backbone_name}_T{temp}"
        try:
            fasta_path, scores_path = run_ligandmpnn(
                pdb_path=backbone_pdb,
                output_dir=temp_dir,
                temperature=temp,
                n_batches=n_batches_per_temp,
                fixed_residues=fixed_residues,
                ligandmpnn_dir=ligandmpnn_dir,
            )
            records = _parse_ligandmpnn_output(
                fasta_path, scores_path, temp, backbone_name,
            )
            all_records.extend(records)
        except RuntimeError as exc:
            logger.error(
                "LigandMPNN failed for %s at T=%s: %s",
                backbone_name, temp, exc,
            )

    return all_records


# ---------------------------------------------------------------------------
# WT direct diversification branch (Branch B)
# ---------------------------------------------------------------------------

def design_wt_branch(
    wt_pdb: Path,
    enzyme_id: str,
    enzyme_config: dict[str, Any],
    output_dir: Path,
    n_sequences_per_temp: int,
    ligandmpnn_dir: Path,
) -> list[dict[str, Any]]:
    """Run LigandMPNN directly on WT structure at T=0.1, 0.2, 0.4.

    Args:
        wt_pdb: Path to wild-type PDB from Stage 2.
        enzyme_id: Enzyme identifier.
        enzyme_config: Enzyme configuration dict.
        output_dir: Base output directory for WT branch.
        n_sequences_per_temp: Number of sequences per temperature.
        ligandmpnn_dir: LigandMPNN installation directory.

    Returns:
        List of designed sequence dicts.
    """
    fixed_residues = enzyme_config.get("fixed_residues", [])
    all_records: list[dict[str, Any]] = []

    for temp in WT_TEMPERATURES:
        # Calculate batches needed: each batch produces 1 seq, so n_batches = n_seqs
        n_batches = n_sequences_per_temp
        temp_dir = output_dir / f"{enzyme_id}_wt_T{temp}"

        try:
            fasta_path, scores_path = run_ligandmpnn(
                pdb_path=wt_pdb,
                output_dir=temp_dir,
                temperature=temp,
                n_batches=n_batches,
                fixed_residues=fixed_residues,
                ligandmpnn_dir=ligandmpnn_dir,
                seed=int(temp * 1000),
            )
            records = _parse_ligandmpnn_output(
                fasta_path, scores_path, temp, f"{enzyme_id}_wt",
            )
            all_records.extend(records)
        except RuntimeError as exc:
            logger.error(
                "LigandMPNN WT branch failed for %s at T=%s: %s",
                enzyme_id, temp, exc,
            )

    return all_records


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_sequences(sequences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove exact duplicate amino acid sequences, keeping best score.

    Args:
        sequences: List of sequence record dicts (must have "sequence" and "score").

    Returns:
        Deduplicated list, keeping the record with the best (lowest) score
        for each unique sequence.
    """
    best: dict[str, dict[str, Any]] = {}

    for record in sequences:
        seq = record["sequence"]
        if seq not in best or record["score"] < best[seq]["score"]:
            best[seq] = record

    deduped = list(best.values())
    n_removed = len(sequences) - len(deduped)
    if n_removed > 0:
        logger.info(
            "Deduplication: %d → %d sequences (%d duplicates removed)",
            len(sequences), len(deduped), n_removed,
        )
    return deduped


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(config: dict[str, Any]) -> dict[str, Any]:
    """Execute Stage 4: sequence design for all enzymes.

    Branch A: For each enzyme, iterate over backbone PDBs from stage_03_rfdiffusion3/
    Branch B: For each enzyme, run WT diversification from stage_02_wt_structures/
    Merge and deduplicate all sequences, write merged FASTA + scores CSV.

    Args:
        config: Loaded pipeline configuration dictionary.

    Returns:
        Dict with keys: total_sequences, per_enzyme, fasta_paths.
    """
    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_04_ligandmpnn", {})
    output_dir_name = stage_cfg.get("output_dir", "stage_04_ligandmpnn")
    output_dir = base_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tool path
    ligandmpnn_dir = Path(
        config.get("tools", {}).get("ligandmpnn_dir", "/opt/tools/LigandMPNN")
    )

    # Stage parameters
    n_batches = stage_cfg.get("number_of_batches", 16)
    omit_aa = stage_cfg.get("omit_aa", "C")
    wt_seqs_per_temp = stage_cfg.get("wt_sequences_per_temp", WT_SEQS_PER_TEMP)

    # Input directories
    backbone_dir = base_dir / config.get("stages", {}).get(
        "stage_03_rfdiffusion3", {}
    ).get("output_dir", "stage_03_rfdiffusion3")

    wt_dir = base_dir / config.get("stages", {}).get(
        "stage_02_wt_structures", {}
    ).get("output_dir", "stage_02_wt_structures")

    enzyme_ids = sorted(config.get("enzymes", {}).keys())
    per_enzyme: dict[str, int] = {}
    fasta_paths: dict[str, str] = {}
    all_scores: list[dict[str, Any]] = []

    for enzyme_id in enzyme_ids:
        enzyme_cfg = config["enzymes"][enzyme_id]
        enzyme_out_dir = output_dir / enzyme_id
        enzyme_out_dir.mkdir(parents=True, exist_ok=True)

        enzyme_records: list[dict[str, Any]] = []

        # ----- Branch A: RFdiffusion3 backbones -----
        enzyme_backbone_dir = backbone_dir / enzyme_id
        if enzyme_backbone_dir.exists():
            backbone_pdbs = sorted(enzyme_backbone_dir.rglob("*.pdb"))
            logger.info(
                "%s Branch A: found %d backbone PDBs in %s",
                enzyme_id, len(backbone_pdbs), enzyme_backbone_dir,
            )
            for pdb in backbone_pdbs:
                records = design_backbone(
                    backbone_pdb=pdb,
                    enzyme_config=enzyme_cfg,
                    output_dir=enzyme_out_dir / "branch_a",
                    temperatures=BACKBONE_TEMPERATURES,
                    n_batches_per_temp=n_batches,
                    ligandmpnn_dir=ligandmpnn_dir,
                )
                for r in records:
                    r["enzyme_id"] = enzyme_id
                    r["design_branch"] = "rfd3_backbone"
                enzyme_records.extend(records)
        else:
            logger.warning(
                "%s Branch A: backbone dir %s not found — skipping",
                enzyme_id, enzyme_backbone_dir,
            )

        # ----- Branch B: WT direct diversification -----
        wt_pdb = wt_dir / "substrate_complexes" / f"{enzyme_id}_substrate.pdb"
        if not wt_pdb.exists():
            wt_pdb = wt_dir / "substrate_complexes" / f"{enzyme_id}.pdb"
        if not wt_pdb.exists():
            wt_pdb = wt_dir / f"{enzyme_id}.pdb"

        if wt_pdb.exists():
            logger.info("%s Branch B: WT PDB = %s", enzyme_id, wt_pdb)
            wt_records = design_wt_branch(
                wt_pdb=wt_pdb,
                enzyme_id=enzyme_id,
                enzyme_config=enzyme_cfg,
                output_dir=enzyme_out_dir / "branch_b",
                n_sequences_per_temp=wt_seqs_per_temp,
                ligandmpnn_dir=ligandmpnn_dir,
            )
            for r in wt_records:
                r["enzyme_id"] = enzyme_id
                r["design_branch"] = "wt_direct"
            enzyme_records.extend(wt_records)
        else:
            logger.warning(
                "%s Branch B: WT PDB not found — skipping", enzyme_id,
            )

        # ----- Merge and deduplicate -----
        enzyme_records = deduplicate_sequences(enzyme_records)
        per_enzyme[enzyme_id] = len(enzyme_records)

        # Write enzyme FASTA
        fasta_seqs = {
            r["sequence_id"]: r["sequence"] for r in enzyme_records
        }
        fasta_path = output_dir / f"{enzyme_id}_designed.fasta"
        write_fasta(fasta_seqs, fasta_path)
        fasta_paths[enzyme_id] = str(fasta_path)

        # Collect scores
        for r in enzyme_records:
            all_scores.append({
                "sequence_id": r["sequence_id"],
                "enzyme_id": r["enzyme_id"],
                "backbone_pdb": r["backbone"],
                "temperature": r["temperature"],
                "ligandmpnn_score": r["score"],
                "design_branch": r["design_branch"],
            })

        logger.info(
            "%s: %d sequences after deduplication", enzyme_id, len(enzyme_records),
        )

    # Write combined scores CSV
    if all_scores:
        scores_path = output_dir / "ligandmpnn_scores.csv"
        write_scores(all_scores, scores_path)
        logger.info("Wrote scores to %s", scores_path)

    total = sum(per_enzyme.values())
    logger.info("Stage 4 complete. Total sequences: %d", total)

    return {
        "total_sequences": total,
        "per_enzyme": per_enzyme,
        "fasta_paths": fasta_paths,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(config_path: Path) -> None:
    """Load config, set up logging, run Stage 4."""
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
            logging.FileHandler(log_dir / "stage4_ligandmpnn.log"),
        ],
    )

    logger.info("=" * 60)
    logger.info("%s — starting", STAGE_NAME)
    logger.info("Config: %s", config_path)
    logger.info("=" * 60)

    results = run_stage(config)

    logger.info("=" * 60)
    logger.info("%s — complete", STAGE_NAME)
    logger.info("  Total sequences: %d", results["total_sequences"])
    for eid, count in results["per_enzyme"].items():
        logger.info("  %s: %d sequences", eid, count)
    logger.info("=" * 60)


def _cli() -> None:
    """Argument parser wrapper for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Sequence Design with LigandMPNN",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    _cli()
