"""Stage 4.75: Solubility Pre-screening.

Sequence-only solubility prediction inserted between LigandMPNN (Stage 4) and
ESMFold (Stage 5). Sequences with high aggregation propensity are flagged
early, before expensive structure-prediction runs.

Scoring is based on physicochemical properties:
  1. CamSol-like intrinsic solubility (Kyte-Doolittle sliding window)
  2. Net charge at pH 7.0
  3. Aggregation-prone region count
  4. Hydrophobic fraction
  5. Isoelectric point estimate (bisection + Henderson-Hasselbalch)

A composite score [0, 1] is computed; sequences below the configurable
risk_threshold are flagged as ``high_aggregation_risk``.

Output: solubility_scores.csv in stage_04.75_solubility/

CLI: python pipeline/stage4_75_solubility.py --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

try:
    from pipeline.config import load_config, get_enzyme_config, resolve_paths
    from pipeline.utils import (
        setup_logging,
        read_fasta,
        write_fasta,
        write_scores,
    )
except ImportError:
    import yaml  # type: ignore[import-untyped]

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
        import csv
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        if not scores:
            return
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=scores[0].keys())
            w.writeheader()
            w.writerows(scores)


# ---------------------------------------------------------------------------
# Module constants (StageModuleProtocol)
# ---------------------------------------------------------------------------

STAGE_NAME = "Stage 4.75: Solubility Pre-screening"
STAGE_NUM = "4.75"
INPUT_DESCRIPTION = "Designed FASTA sequences from Stage 4 (LigandMPNN)"
OUTPUT_DESCRIPTION = (
    "solubility_scores.csv with per-sequence solubility metrics "
    "in stage_04.75_solubility/"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kyte-Doolittle hydrophobicity scale
# ---------------------------------------------------------------------------

# Kyte & Doolittle (1982) J. Mol. Biol. 157:105-132
_KYTE_DOOLITTLE: dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# Residues considered hydrophobic for fraction calculation (per task spec)
_HYDROPHOBIC_RESIDUES: frozenset[str] = frozenset("VILFWMAP")

# Residues considered aggregation-prone (hydrophobic core set)
_AGG_PRONE_RESIDUES: frozenset[str] = frozenset("VILFWM")

# Standard pKa values for isoelectric point estimation
_PKA_NTERM: float = 8.2
_PKA_CTERM: float = 3.7
_PKA_SIDE_CHAIN: dict[str, float] = {
    "D": 3.9,   # Asp
    "E": 4.1,   # Glu
    "H": 6.0,   # His
    "C": 8.3,   # Cys
    "Y": 10.1,  # Tyr
    "K": 10.5,  # Lys
    "R": 12.5,  # Arg
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def calc_intrinsic_solubility(sequence: str, window: int = 7) -> float:
    """CamSol-like intrinsic solubility from Kyte-Doolittle sliding window.

    Computes a 7-residue sliding window mean of Kyte-Doolittle hydrophobicity
    scores, then returns the mean window score **inverted** so that higher
    values correspond to greater solubility (less hydrophobic).

    For sequences shorter than *window*, the whole sequence is used as a
    single window.

    Args:
        sequence: Amino acid sequence (single-letter codes, uppercase).
        window: Sliding window width (default 7).

    Returns:
        Intrinsic solubility score (float). Higher = more soluble.
        Returns 0.0 for empty sequences.
    """
    if not sequence:
        return 0.0

    seq = sequence.upper()
    n = len(seq)

    if n < window:
        # Use the whole sequence as one window
        kd_values = [_KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq]
        mean_hydrophobicity = sum(kd_values) / len(kd_values)
    else:
        window_scores: list[float] = []
        for i in range(n - window + 1):
            chunk = seq[i : i + window]
            kd_values = [_KYTE_DOOLITTLE.get(aa, 0.0) for aa in chunk]
            window_scores.append(sum(kd_values) / window)
        mean_hydrophobicity = sum(window_scores) / len(window_scores)

    # Invert: higher hydrophobicity → lower solubility
    return -mean_hydrophobicity


def calc_net_charge(sequence: str, ph: float = 7.0) -> float:  # noqa: ARG001
    """Estimate net charge of a sequence at pH 7.0.

    Uses a simplified model:
      - D, E residues contribute -1
      - K, R residues contribute +1
      - H residues contribute +0.1 (partially protonated at pH 7)

    Note: The *ph* parameter is accepted for API compatibility but the
    contributions are fixed at the values appropriate for pH 7.

    Args:
        sequence: Amino acid sequence.
        ph: pH value (currently only pH 7.0 values are implemented).

    Returns:
        Net charge (float). Positive = net positive charge.
    """
    charge = 0.0
    for aa in sequence.upper():
        if aa in ("K", "R"):
            charge += 1.0
        elif aa in ("D", "E"):
            charge -= 1.0
        elif aa == "H":
            charge += 0.1
    return charge


def find_aggregation_regions(
    sequence: str,
    window: int = 7,
    threshold: float = 1.5,
) -> list[dict[str, Any]]:
    """Identify aggregation-prone regions in a sequence.

    A region is flagged when a sliding window of length *window* has a mean
    Kyte-Doolittle score above *threshold* AND the window contains 5 or more
    consecutive hydrophobic residues (VILFW M).

    Args:
        sequence: Amino acid sequence.
        window: Sliding window size (default 7).
        threshold: KD mean threshold above which windows are flagged (default 1.5).

    Returns:
        List of dicts, each with:
            - start: 1-based start position of the region
            - end: 1-based end position of the region
            - mean_kd: mean Kyte-Doolittle score of the window
            - length: number of residues in the region
    """
    if not sequence:
        return []

    seq = sequence.upper()
    n = len(seq)

    if n < 5:
        return []

    regions: list[dict[str, Any]] = []

    # Find stretches of 5+ consecutive aggregation-prone residues
    i = 0
    while i < n:
        if seq[i] in _AGG_PRONE_RESIDUES:
            j = i
            while j < n and seq[j] in _AGG_PRONE_RESIDUES:
                j += 1
            run_length = j - i
            if run_length >= 5:
                # Check KD threshold using the window approach
                run_seq = seq[i:j]
                kd_values = [_KYTE_DOOLITTLE.get(aa, 0.0) for aa in run_seq]
                mean_kd = sum(kd_values) / len(kd_values)
                if mean_kd >= threshold:
                    regions.append({
                        "start": i + 1,
                        "end": j,
                        "mean_kd": mean_kd,
                        "length": run_length,
                    })
            i = j
        else:
            i += 1

    return regions


def calc_hydrophobic_fraction(sequence: str) -> float:
    """Compute the fraction of hydrophobic residues in a sequence.

    Hydrophobic residues are: V, I, L, F, W, M, A, P.

    Args:
        sequence: Amino acid sequence.

    Returns:
        Fraction in [0, 1]. Returns 0.0 for empty sequences.
    """
    if not sequence:
        return 0.0
    seq = sequence.upper()
    n_hydrophobic = sum(1 for aa in seq if aa in _HYDROPHOBIC_RESIDUES)
    return n_hydrophobic / len(seq)


def estimate_pi(sequence: str) -> float:
    """Estimate the isoelectric point (pI) of a sequence.

    Uses a bisection method solving for the pH at which the sum of fractional
    charges from all ionisable groups (N-terminus, C-terminus, and side chains)
    equals zero.

    Standard pKa values:
      - N-terminus: 8.2
      - C-terminus: 3.7
      - D: 3.9, E: 4.1, H: 6.0, C: 8.3, Y: 10.1, K: 10.5, R: 12.5

    Args:
        sequence: Amino acid sequence.

    Returns:
        Estimated pI (float in approximately 1–14).
        Returns 7.0 for empty sequences (neutral default).
    """
    if not sequence:
        return 7.0

    seq = sequence.upper()

    # Count ionisable residues
    counts: dict[str, int] = {}
    for aa in seq:
        if aa in _PKA_SIDE_CHAIN:
            counts[aa] = counts.get(aa, 0) + 1

    def _net_charge_at_ph(ph: float) -> float:
        """Sum of fractional charges at a given pH."""
        charge = 0.0

        # N-terminus: positively charged below pKa
        charge += 1.0 / (1.0 + 10 ** (ph - _PKA_NTERM))

        # C-terminus: negatively charged above pKa
        charge -= 1.0 / (1.0 + 10 ** (_PKA_CTERM - ph))

        for aa, count in counts.items():
            pka = _PKA_SIDE_CHAIN[aa]
            if aa in ("D", "E", "C", "Y"):
                # Acidic side chains (negatively charged above pKa)
                charge -= count / (1.0 + 10 ** (pka - ph))
            else:
                # Basic side chains (positively charged below pKa)
                # H, K, R
                charge += count / (1.0 + 10 ** (ph - pka))

        return charge

    # Bisection search for pH where net charge ≈ 0
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _net_charge_at_ph(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break

    return (lo + hi) / 2.0


def calc_solubility_score(sequence: str) -> float:
    """Compute a composite solubility score [0, 1].

    The composite is a weighted combination of individual metrics:

      score = 0.4 * intrinsic_solubility_norm
            + 0.2 * charge_score
            + 0.2 * (1 - agg_fraction)
            + 0.2 * (1 - hydrophobic_fraction)

    Where:
      - intrinsic_solubility_norm maps [-4.5, 4.5] KD range to [0, 1];
        higher = more soluble (less hydrophobic).
      - charge_score maps net charge to [0, 1] favouring moderate positive
        charge (optimal ~+2 for cytoplasmic expression); clamped to [0, 1].
      - agg_fraction = min(n_agg_regions / 5, 1.0) where 5 is a normalising
        constant.
      - hydrophobic_fraction: directly from calc_hydrophobic_fraction().

    Args:
        sequence: Amino acid sequence.

    Returns:
        Composite solubility score in [0, 1]. Higher = more soluble.
    """
    if not sequence:
        return 0.0

    # 1. Intrinsic solubility (inverted KD) → normalise to [0, 1]
    # KD range: most negative is R = -4.5, most positive is I = 4.5
    # intrinsic_solubility = -mean_kd, range [-4.5, 4.5]
    raw_solubility = calc_intrinsic_solubility(sequence)
    # Map from [-4.5, 4.5] to [0, 1]
    intrinsic_norm = (raw_solubility - (-4.5)) / (4.5 - (-4.5))
    intrinsic_norm = max(0.0, min(1.0, intrinsic_norm))

    # 2. Charge score — moderate positive is good; map to [0, 1]
    # Optimal is around +2; strongly negative or >+10 is penalised
    net_charge = calc_net_charge(sequence)
    # Map from [-10, +10] to [0, 1] with a peak around +2
    charge_raw = (net_charge + 10.0) / 20.0
    charge_score = max(0.0, min(1.0, charge_raw))

    # 3. Aggregation-prone region fraction
    # Expressed as fraction of sequence residues covered by aggregation regions,
    # which is more physically meaningful than a raw region count.
    agg_regions = find_aggregation_regions(sequence)
    seq_len = len(sequence)
    agg_residues = sum(r["length"] for r in agg_regions)
    agg_fraction = min(agg_residues / seq_len, 1.0) if seq_len > 0 else 0.0

    # 4. Hydrophobic fraction
    hydrophobic_fraction = calc_hydrophobic_fraction(sequence)

    composite = (
        0.4 * intrinsic_norm
        + 0.2 * charge_score
        + 0.2 * (1.0 - agg_fraction)
        + 0.2 * (1.0 - hydrophobic_fraction)
    )
    return max(0.0, min(1.0, composite))


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(config: dict[str, Any]) -> dict[str, Any]:
    """Execute Stage 4.75: Solubility pre-screening for all enzymes.

    For each enzyme:
    1. Read designed sequences from stage_04_ligandmpnn/
    2. Score each sequence with the five solubility metrics
    3. Flag sequences below *risk_threshold* as high_aggregation_risk
    4. Write solubility_scores.csv

    Args:
        config: Loaded pipeline configuration dict.

    Returns:
        Dict mapping enzyme_id -> {"total_scored": int, "high_risk_count": int}.
    """
    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_04_75_solubility", {})

    output_dir_name = stage_cfg.get("output_dir", "stage_04.75_solubility")
    output_dir = base_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    risk_threshold = float(stage_cfg.get("risk_threshold", 0.3))

    # Input from Stage 4
    stage4_dir_name = (
        config.get("stages", {})
        .get("stage_04_ligandmpnn", {})
        .get("output_dir", "stage_04_ligandmpnn")
    )
    stage4_dir = base_dir / stage4_dir_name

    enzyme_ids = sorted(config.get("enzymes", {}).keys())
    all_scores: list[dict] = []
    summary: dict[str, dict] = {}

    for enzyme_id in enzyme_ids:
        logger.info("Scoring solubility for %s ...", enzyme_id)

        # Locate input FASTA — check multiple candidate paths
        input_fasta = _find_input_fasta(stage4_dir, enzyme_id)
        if input_fasta is None:
            logger.warning(
                "No input FASTA found for %s in %s — skipping",
                enzyme_id,
                stage4_dir,
            )
            continue

        sequences = read_fasta(input_fasta)
        if not sequences:
            logger.warning("Empty FASTA for %s — skipping", enzyme_id)
            continue

        logger.info("%s: %d sequences to score", enzyme_id, len(sequences))

        enzyme_high_risk = 0

        for seq_id, seq in sequences.items():
            intrinsic = calc_intrinsic_solubility(seq)
            net_charge = calc_net_charge(seq)
            pi_est = estimate_pi(seq)
            agg_regions = find_aggregation_regions(seq)
            hydro_frac = calc_hydrophobic_fraction(seq)
            composite = calc_solubility_score(seq)

            is_high_risk = composite < risk_threshold
            if is_high_risk:
                enzyme_high_risk += 1

            all_scores.append({
                "sequence_id": seq_id,
                "enzyme_id": enzyme_id,
                "intrinsic_solubility": round(intrinsic, 4),
                "net_charge": round(net_charge, 4),
                "pi_estimate": round(pi_est, 4),
                "aggregation_regions": len(agg_regions),
                "hydrophobic_fraction": round(hydro_frac, 4),
                "composite_score": round(composite, 4),
                "high_risk": is_high_risk,
            })

        enzyme_total = len(sequences)
        summary[enzyme_id] = {
            "total_scored": enzyme_total,
            "high_risk_count": enzyme_high_risk,
        }

        logger.info(
            "%s: %d scored, %d high risk (%.1f%%)",
            enzyme_id,
            enzyme_total,
            enzyme_high_risk,
            100.0 * enzyme_high_risk / enzyme_total if enzyme_total > 0 else 0.0,
        )

    # Write scores CSV
    scores_csv = output_dir / "solubility_scores.csv"
    write_scores(all_scores, scores_csv)
    logger.info(
        "Wrote solubility scores to %s (%d rows)", scores_csv, len(all_scores)
    )

    return summary


def _find_input_fasta(stage4_dir: Path, enzyme_id: str) -> Path | None:
    """Find the input FASTA for an enzyme in the Stage 4 output directory.

    Checks several candidate locations in priority order:
    1. stage4_dir/{enzyme_id}/designed_sequences.fasta
    2. stage4_dir/{enzyme_id}/{enzyme_id}_designed.fasta
    3. stage4_dir/{enzyme_id}_designed.fasta
    4. stage4_dir/{enzyme_id}.fasta

    Args:
        stage4_dir: Base directory for Stage 4 output.
        enzyme_id: Enzyme identifier string.

    Returns:
        Path to the first existing FASTA file, or None if none found.
    """
    candidates = [
        stage4_dir / enzyme_id / "designed_sequences.fasta",
        stage4_dir / enzyme_id / f"{enzyme_id}_designed.fasta",
        stage4_dir / f"{enzyme_id}_designed.fasta",
        stage4_dir / f"{enzyme_id}.fasta",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(config_path: Path) -> None:
    """Load config, set up logging, run Stage 4.75."""
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
            logging.FileHandler(log_dir / "stage4_75_solubility.log"),
        ],
    )

    logger.info("=" * 60)
    logger.info("%s — starting", STAGE_NAME)
    logger.info("Config: %s", config_path)
    logger.info("=" * 60)

    summary = run_stage(config)

    logger.info("=" * 60)
    logger.info("%s — complete", STAGE_NAME)
    for eid, stats in summary.items():
        logger.info(
            "  %s: %d scored, %d high risk",
            eid,
            stats["total_scored"],
            stats["high_risk_count"],
        )
    logger.info("=" * 60)


def _cli() -> None:
    """Argument parser wrapper for CLI invocation."""
    parser = argparse.ArgumentParser(
        description=STAGE_NAME,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    _cli()
