"""Stage 1: MSA Generation using ColabFold-search with MMseqs2-GPU backend.

Reads WT FASTA per enzyme, runs colabfold_search against local pre-indexed
UniRef30 + ColabFoldDB, validates output .a3m files (depth, coverage),
and reports MSA statistics (Neff, coverage per position).

CLI: python pipeline/stage1_msa.py --config config.yaml
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from pipeline.config import load_config, get_enzyme_config, resolve_paths
    from pipeline.utils import setup_logging, read_fasta
except ImportError:
    import yaml
    import logging

    def load_config(p):
        return yaml.safe_load(open(p))

    def get_enzyme_config(cfg, eid):
        return cfg["enzymes"][eid]

    def resolve_paths(cfg, base):
        return cfg

    def setup_logging(n, d, **kw):
        logging.basicConfig()
        return logging.getLogger(n)

    def read_fasta(p):
        seqs = {}
        with open(p) as f:
            header, seq = None, []
            for line in f:
                if line.startswith(">"):
                    if header:
                        seqs[header] = "".join(seq)
                    header, seq = line[1:].strip(), []
                else:
                    seq.append(line.strip())
            if header:
                seqs[header] = "".join(seq)
        return seqs


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
STAGE_NAME = "Stage 1: MSA Generation"
STAGE_NUM = "1"
INPUT_DESCRIPTION = "WT FASTA sequences per enzyme"
OUTPUT_DESCRIPTION = ".a3m MSA files per enzyme in stage_01_msa/"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def run_colabfold_search(
    query_fasta: Path,
    db_dir: Path,
    output_dir: Path,
    colabfold_bin: str = "colabfold_search",
    mmseqs_bin: str = "mmseqs",
    db_load_mode: int = 2,
) -> Path:
    """Run colabfold_search to generate an MSA for the given query.

    Args:
        query_fasta: Path to input FASTA file (single or multi-sequence).
        db_dir: Directory containing the pre-indexed ColabFold databases.
            The command joins uniref30 and colabfolddb paths via ':'.
        output_dir: Directory for colabfold_search output.
        colabfold_bin: Name or path of the colabfold_search binary.
        mmseqs_bin: Name or path of the mmseqs binary.
        db_load_mode: Database load mode (2 = pre-loaded on GPU).

    Returns:
        Path to the output .a3m file.

    Raises:
        RuntimeError: If colabfold_search exits with non-zero return code.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(colabfold_bin),
        "--mmseqs", str(mmseqs_bin),
        "--db-load-mode", str(db_load_mode),
        str(query_fasta),
        str(db_dir),
        str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"colabfold_search failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Find the output .a3m file — colabfold_search names it after the query
    a3m_files = list(output_dir.glob("*.a3m"))
    if not a3m_files:
        raise RuntimeError(
            f"colabfold_search produced no .a3m files in {output_dir}"
        )

    return a3m_files[0]


def parse_a3m(a3m_path: Path) -> tuple[str, list[str]]:
    """Parse an A3M file into the query sequence and list of aligned sequences.

    Returns:
        Tuple of (query_sequence, list_of_all_sequences_including_query).
    """
    headers: list[str] = []
    sequences: list[str] = []
    current_seq: list[str] = []

    with open(a3m_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                headers.append(line[1:])
                current_seq = []
            elif line.strip():
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))

    query_seq = sequences[0] if sequences else ""
    return query_seq, sequences


def validate_a3m(a3m_path: Path, min_depth: int = 10) -> dict:
    """Validate an A3M MSA file and return summary statistics.

    Args:
        a3m_path: Path to the .a3m file.
        min_depth: Minimum acceptable MSA depth. Logs a warning if below.

    Returns:
        Dictionary with keys: depth, coverage, neff, path.
    """
    import logging

    logger = logging.getLogger(f"pipeline.{STAGE_NUM}")
    query_seq, sequences = parse_a3m(a3m_path)
    depth = len(sequences)

    if depth == 0:
        logger.warning("Empty MSA file: %s", a3m_path)
        return {"depth": 0, "coverage": 0.0, "neff": 0.0, "path": str(a3m_path)}

    # Coverage: fraction of query positions covered (non-gap) across the MSA.
    # Only consider match-state columns (uppercase + gap in query).
    query_len = len(query_seq)
    if query_len == 0:
        return {"depth": depth, "coverage": 0.0, "neff": 0.0, "path": str(a3m_path)}

    # Calculate per-position coverage
    coverage_per_pos = _per_position_coverage(query_seq, sequences)
    mean_coverage = sum(coverage_per_pos) / len(coverage_per_pos) if coverage_per_pos else 0.0

    # Neff: simple approximation — number of unique sequences / total * depth
    unique_seqs = len(set(sequences))
    neff = unique_seqs / depth * depth if depth > 0 else 0.0
    # Simplifies to just unique_seqs, but follow the spec formula
    neff = float(unique_seqs)

    if depth < min_depth:
        logger.warning(
            "Low MSA depth for %s: %d (minimum recommended: %d)",
            a3m_path.name, depth, min_depth,
        )

    return {
        "depth": depth,
        "coverage": round(mean_coverage, 4),
        "neff": neff,
        "path": str(a3m_path),
    }


def parse_a3m_stats(a3m_path: Path) -> dict:
    """Detailed MSA statistics including per-position coverage.

    Returns:
        Dictionary with keys: depth, coverage, neff, path, per_position_coverage.
    """
    stats = validate_a3m(a3m_path)
    query_seq, sequences = parse_a3m(a3m_path)
    stats["per_position_coverage"] = _per_position_coverage(query_seq, sequences)
    return stats


def _per_position_coverage(query_seq: str, sequences: list[str]) -> list[float]:
    """Compute per-position coverage for match-state columns.

    Match-state columns are uppercase letters or gaps in the query.
    For each column, coverage = fraction of sequences with a non-gap character.
    """
    if not sequences or not query_seq:
        return []

    query_len = len(query_seq)
    # Match state columns are those corresponding to the query
    # In A3M format, lowercase = insertions relative to query
    # We need to extract match-state columns from each sequence.
    n_seqs = len(sequences)
    coverage = [0.0] * query_len

    for seq in sequences:
        col = 0
        for char in seq:
            if char.isupper() or char == "-":
                # Match state column
                if col < query_len:
                    if char != "-":
                        coverage[col] += 1.0
                    col += 1
            # lowercase = insertion, skip

    # Normalize
    for i in range(query_len):
        coverage[i] = coverage[i] / n_seqs if n_seqs > 0 else 0.0

    return coverage


def run_stage(config: dict) -> dict:
    """Run Stage 1: MSA generation for all enzymes.

    Args:
        config: Loaded pipeline configuration dictionary.

    Returns:
        Dictionary mapping enzyme IDs to their output .a3m file paths.
    """
    import logging

    logger = logging.getLogger(f"pipeline.{STAGE_NUM}")

    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_01_msa", {})
    output_dir = base_dir / stage_cfg.get("output_dir", "stage_01_msa")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Database paths
    db_dir = config["databases"]["colabfold_db_dir"]

    # Tool config
    tools = config.get("tools", {})
    colabfold_bin = "colabfold_search"

    results: dict[str, str] = {}

    for enzyme_id, enzyme_cfg in config["enzymes"].items():
        logger.info("Processing %s (%s)", enzyme_id, enzyme_cfg.get("name", enzyme_id))

        # Read WT FASTA
        seq_file = enzyme_cfg["sequence_file"]
        if not isinstance(seq_file, Path):
            seq_file = Path(seq_file)
            if not seq_file.is_absolute():
                seq_file = base_dir / seq_file

        sequences = read_fasta(seq_file)
        if not sequences:
            logger.error("No sequences found in %s for %s", seq_file, enzyme_id)
            continue

        # Write to temp FASTA for colabfold_search input
        enzyme_out_dir = output_dir / enzyme_id
        enzyme_out_dir.mkdir(parents=True, exist_ok=True)
        query_fasta = enzyme_out_dir / f"{enzyme_id}.fasta"
        with open(query_fasta, "w") as fh:
            for header, seq in sequences.items():
                fh.write(f">{header}\n{seq}\n")

        # Run colabfold_search
        try:
            a3m_path = run_colabfold_search(
                query_fasta=query_fasta,
                db_dir=Path(str(db_dir)),
                output_dir=enzyme_out_dir,
                colabfold_bin=colabfold_bin,
            )
        except RuntimeError as exc:
            logger.error("colabfold_search failed for %s: %s", enzyme_id, exc)
            continue

        # Validate and log statistics
        stats = validate_a3m(a3m_path)
        logger.info(
            "%s MSA stats — depth: %d, coverage: %.4f, Neff: %.1f, path: %s",
            enzyme_id, stats["depth"], stats["coverage"], stats["neff"], stats["path"],
        )

        results[enzyme_id] = str(a3m_path)

    logger.info("Stage 1 complete. MSAs generated for: %s", list(results.keys()))
    return results


def main(config_path: Path) -> None:
    """Entry point: load config, run MSA generation, log results."""
    config = load_config(config_path)
    base_dir = Path(config["global"]["base_dir"])
    config = resolve_paths(config, base_dir)

    log_dir = base_dir / config["global"].get("log_dir", "logs")
    logger = setup_logging("stage1_msa", log_dir)

    logger.info("Starting %s", STAGE_NAME)
    results = run_stage(config)
    logger.info("Finished %s — outputs: %s", STAGE_NAME, results)


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
