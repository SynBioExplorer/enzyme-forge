"""Stage 7.5: N-glycosylation Sequon Screening.

S. cerevisiae N-hyperglycosylates heterologous proteins at Asn-Xaa-Ser/Thr
sequons (Xaa != Pro), which can destroy catalytic activity when sequons are
near the active site.

Steps:
1. Read final ranked FASTA + PDB structures from Stage 7
2. Scan for Asn-Xaa-Ser/Thr sequons using regex
3. Map sequon positions onto structure
4. Flag sequons within 10 A of active site as high-risk
5. For high-risk: generate Asn->Gln and Ser/Thr->Ala alternatives
6. Re-score alternatives with ESMFold for fold preservation
7. Annotate all sequons in output

Output: glycosylation_report.csv + modified FASTA in stage_07.5_glycosylation/

CLI: python pipeline/stage7_5_glycosylation.py --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
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
# Module constants
# ---------------------------------------------------------------------------

STAGE_NAME = "Stage 7.5: N-glycosylation Screening"
STAGE_NUM = "7.5"
INPUT_DESCRIPTION = "Final ranked FASTA + PDB structures from Stage 7"
OUTPUT_DESCRIPTION = "glycosylation_report.csv + modified FASTA in stage_07.5_glycosylation/"

SEQUON_PATTERN = re.compile(r"N[^P][ST]")

CSV_COLUMNS = [
    "sequence_id",
    "enzyme_id",
    "sequon_position",
    "sequon_str",
    "asn_pos",
    "min_dist_to_active_site",
    "risk_level",
    "alternative_sequence_id",
    "alternative_mutation",
    "alternative_plddt",
    "fold_preserved",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def find_glycosylation_sequons(sequence: str) -> list[dict]:
    """Scan *sequence* for N-glycosylation sequons (N-X-S/T, X != P).

    Returns list of dicts with keys:
        position: 1-based position of the sequon start (the Asn)
        sequon: the 3-character sequon string
        asn_pos: 1-based position of Asn
        xaa_pos: 1-based position of the middle residue
        st_pos:  1-based position of the Ser/Thr
    """
    results: list[dict] = []
    for match in SEQUON_PATTERN.finditer(sequence):
        i = match.start()  # 0-based
        results.append({
            "position": i + 1,
            "sequon": match.group(),
            "asn_pos": i + 1,
            "xaa_pos": i + 2,
            "st_pos": i + 3,
        })
    return results


def get_residue_coords(
    pdb_path: Path, residue_positions: list[int],
) -> dict[int, tuple]:
    """Extract CA coordinates for given 1-based residue positions from a PDB.

    Returns {residue_pos: (x, y, z)} for each position found.
    Uses Bio.PDB if available, otherwise falls back to manual ATOM line parsing.
    """
    pos_set = set(residue_positions)
    coords: dict[int, tuple] = {}

    try:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", str(pdb_path))
        model = structure[0]

        for chain in model:
            for residue in chain:
                het = residue.get_id()[0]
                if het.strip() and het != " ":
                    continue
                resseq = residue.get_id()[1]
                if resseq in pos_set:
                    for atom in residue:
                        if atom.get_name() == "CA":
                            c = atom.get_vector().get_array()
                            coords[resseq] = (float(c[0]), float(c[1]), float(c[2]))
                            break
        return coords
    except ImportError:
        pass

    # Fallback: manual ATOM line parsing for CA atoms
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                try:
                    resseq = int(line[22:26].strip())
                except ValueError:
                    continue
                if resseq not in pos_set:
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords[resseq] = (x, y, z)
                except ValueError:
                    continue
    except OSError:
        logger.warning("Cannot read PDB file: %s", pdb_path)

    return coords


def _euclidean_distance(a: tuple, b: tuple) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def calc_min_distance_to_active_site(
    sequon_positions: list[int],
    hotspot_residues: list[int],
    pdb_path: Path,
) -> dict[int, float]:
    """For each sequon Asn position, find minimum CA distance to any hotspot.

    Args:
        sequon_positions: 1-based Asn positions of sequons.
        hotspot_residues: 1-based residue numbers of active site.
        pdb_path: Path to PDB file.

    Returns:
        {asn_position: min_distance_angstroms}.
        Positions not found in the PDB get distance ``float('inf')``.
    """
    if not hotspot_residues or not sequon_positions:
        return {pos: float("inf") for pos in sequon_positions}

    all_positions = list(set(sequon_positions) | set(hotspot_residues))
    coords = get_residue_coords(pdb_path, all_positions)

    result: dict[int, float] = {}
    for seq_pos in sequon_positions:
        if seq_pos not in coords:
            result[seq_pos] = float("inf")
            continue
        min_dist = float("inf")
        for hp in hotspot_residues:
            if hp in coords:
                d = _euclidean_distance(coords[seq_pos], coords[hp])
                if d < min_dist:
                    min_dist = d
        result[seq_pos] = min_dist
    return result


def classify_risk(
    sequon_pos: int,
    min_dist_to_active_site: float,
    distance_threshold: float = 10.0,
) -> str:
    """Classify a sequon as high_risk or low_risk based on distance.

    Returns ``"high_risk"`` if *min_dist_to_active_site* < *distance_threshold*,
    otherwise ``"low_risk"``.
    """
    if min_dist_to_active_site < distance_threshold:
        return "high_risk"
    return "low_risk"


def generate_alternatives(
    sequence: str, sequon: dict, strategy: str = "asn_to_gln",
) -> list[dict]:
    """Generate alternative sequences that abolish a glycosylation sequon.

    Args:
        sequence: Original amino acid sequence.
        sequon: Sequon dict as returned by :func:`find_glycosylation_sequons`.
        strategy: ``"asn_to_gln"`` replaces Asn with Gln at sequon start;
                  ``"st_to_ala"`` replaces Ser/Thr at +2 with Ala.

    Returns:
        List with a single dict containing:
        - modified_sequence: the mutated sequence
        - mutation: human-readable mutation string (e.g. "N42Q")
        - strategy: the strategy name
    """
    seq_list = list(sequence)
    results: list[dict] = []

    if strategy == "asn_to_gln":
        idx = sequon["asn_pos"] - 1  # 0-based
        original = seq_list[idx]
        seq_list[idx] = "Q"
        results.append({
            "modified_sequence": "".join(seq_list),
            "mutation": f"{original}{sequon['asn_pos']}Q",
            "strategy": strategy,
        })
    elif strategy == "st_to_ala":
        idx = sequon["st_pos"] - 1  # 0-based
        original = seq_list[idx]
        seq_list[idx] = "A"
        results.append({
            "modified_sequence": "".join(seq_list),
            "mutation": f"{original}{sequon['st_pos']}A",
            "strategy": strategy,
        })
    return results


def score_with_esmfold(
    sequences: dict[str, str],
    reference_pdb: Path | None = None,
) -> dict[str, dict]:
    """Score alternative sequences with ESMFold for fold preservation.

    Args:
        sequences: {seq_id: amino_acid_sequence}.
        reference_pdb: Optional reference PDB for RMSD (unused for now).

    Returns:
        {seq_id: {"plddt": float|None, "fold_preserved": bool|None}}.
        Returns None scores with a warning if ESMFold is unavailable.
    """
    if not sequences:
        return {}

    try:
        from pipeline.stage5_esmfold import load_esmfold_model, predict_structure_esmfold
    except ImportError:
        pass
    else:
        try:
            model, tokenizer = load_esmfold_model()
            results: dict[str, dict] = {}
            for sid, seq in sequences.items():
                pred = predict_structure_esmfold(seq, model, tokenizer)
                plddt = pred["plddt"]
                results[sid] = {
                    "plddt": plddt,
                    "fold_preserved": plddt >= 60.0,
                }
            return results
        except Exception as exc:
            logger.warning("ESMFold prediction failed: %s", exc)

    # Fallback: ESMFold not available
    logger.warning(
        "ESMFold not available — returning None scores for %d sequences",
        len(sequences),
    )
    return {
        sid: {"plddt": None, "fold_preserved": None}
        for sid in sequences
    }


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(config: dict) -> dict:
    """Run Stage 7.5: N-glycosylation screening for all enzymes.

    For each enzyme:
    1. Read final rankings from stage_07_crossval_ranking/
    2. For each candidate: find sequons, map to structure, classify risk
    3. For high-risk sequons: generate alternatives and score with ESMFold
    4. Write glycosylation_report.csv and modified FASTA

    Args:
        config: Loaded pipeline configuration dict.

    Returns:
        Summary dict mapping enzyme_id -> {
            "total_candidates": int,
            "total_sequons": int,
            "high_risk_sequons": int,
            "alternatives_generated": int,
        }
    """
    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_07_5_glycosylation", {})

    output_dir = base_dir / stage_cfg.get("output_dir", "stage_07.5_glycosylation")
    output_dir.mkdir(parents=True, exist_ok=True)

    distance_threshold = float(stage_cfg.get("distance_threshold", 10.0))
    plddt_threshold = float(stage_cfg.get("plddt_threshold", 60.0))

    # Input: Stage 7 rankings
    stage7_cfg = config.get("stages", {}).get("stage_07_crossval", {})
    stage7_dir = base_dir / stage7_cfg.get("output_dir", "stage_07_crossval_ranking")

    # WT structures from Stage 2
    wt_dir = base_dir / config.get("stages", {}).get(
        "stage_02_wt_structures", {}
    ).get("output_dir", "stage_02_wt_structures")

    all_rows: list[dict] = []
    all_alternative_seqs: dict[str, str] = {}
    summary: dict[str, dict] = {}

    for enzyme_id, enzyme_cfg in config.get("enzymes", {}).items():
        logger.info("Processing %s", enzyme_id)

        hotspot_residues = enzyme_cfg.get("hotspot_residues", [])

        # Find ranked FASTA from Stage 7
        fasta_path = stage7_dir / enzyme_id / f"{enzyme_id}_ranked.fasta"
        if not fasta_path.exists():
            fasta_path = stage7_dir / f"{enzyme_id}_ranked.fasta"
        if not fasta_path.exists():
            logger.warning("No ranked FASTA for %s — skipping", enzyme_id)
            continue

        sequences = read_fasta(fasta_path)
        if not sequences:
            logger.warning("Empty FASTA for %s — skipping", enzyme_id)
            continue

        # PDB directory for this enzyme from Stage 7
        pdb_dir = stage7_dir / enzyme_id / "structures"
        if not pdb_dir.exists():
            pdb_dir = stage7_dir / "structures"

        # WT PDB as fallback for structure mapping
        wt_pdb = wt_dir / "substrate_complexes" / f"{enzyme_id}_substrate.pdb"
        if not wt_pdb.exists():
            wt_pdb = wt_dir / f"{enzyme_id}.pdb"

        enz_total_sequons = 0
        enz_high_risk = 0
        enz_alternatives = 0

        for seq_id, sequence in sequences.items():
            sequons = find_glycosylation_sequons(sequence)

            if not sequons:
                # No sequons — add a single row noting no sequons found
                all_rows.append({
                    "sequence_id": seq_id,
                    "enzyme_id": enzyme_id,
                    "sequon_position": "",
                    "sequon_str": "",
                    "asn_pos": "",
                    "min_dist_to_active_site": "",
                    "risk_level": "none",
                    "alternative_sequence_id": "",
                    "alternative_mutation": "",
                    "alternative_plddt": "",
                    "fold_preserved": "",
                })
                continue

            # Find PDB for this sequence
            pdb_path = pdb_dir / f"{seq_id}.pdb"
            if not pdb_path.exists():
                pdb_path = wt_pdb

            # Calculate distances to active site
            asn_positions = [s["asn_pos"] for s in sequons]
            if pdb_path.exists() and hotspot_residues:
                distances = calc_min_distance_to_active_site(
                    asn_positions, hotspot_residues, pdb_path,
                )
            else:
                distances = {pos: float("inf") for pos in asn_positions}

            for sequon in sequons:
                enz_total_sequons += 1
                asn_pos = sequon["asn_pos"]
                min_dist = distances.get(asn_pos, float("inf"))
                risk = classify_risk(asn_pos, min_dist, distance_threshold)

                alt_seq_id = ""
                alt_mutation = ""
                alt_plddt: Any = ""
                alt_fold_preserved: Any = ""

                if risk == "high_risk":
                    enz_high_risk += 1

                    # Generate both alternative strategies
                    alternatives_to_score: dict[str, str] = {}
                    alt_metadata: list[dict] = []

                    for strategy in ("asn_to_gln", "st_to_ala"):
                        alts = generate_alternatives(sequence, sequon, strategy)
                        for alt in alts:
                            aid = f"{seq_id}_glyco_{sequon['asn_pos']}_{strategy}"
                            alternatives_to_score[aid] = alt["modified_sequence"]
                            alt_metadata.append({
                                "alt_id": aid,
                                "mutation": alt["mutation"],
                                "strategy": alt["strategy"],
                                "modified_sequence": alt["modified_sequence"],
                            })

                    # Score alternatives with ESMFold
                    esmfold_scores = score_with_esmfold(alternatives_to_score)

                    for meta in alt_metadata:
                        aid = meta["alt_id"]
                        score_info = esmfold_scores.get(aid, {})
                        plddt_val = score_info.get("plddt")
                        preserved = score_info.get("fold_preserved")

                        # Only add to output FASTA if fold is preserved
                        if preserved is True:
                            all_alternative_seqs[aid] = meta["modified_sequence"]

                        enz_alternatives += 1

                        all_rows.append({
                            "sequence_id": seq_id,
                            "enzyme_id": enzyme_id,
                            "sequon_position": sequon["position"],
                            "sequon_str": sequon["sequon"],
                            "asn_pos": asn_pos,
                            "min_dist_to_active_site": round(min_dist, 2) if min_dist != float("inf") else "inf",
                            "risk_level": risk,
                            "alternative_sequence_id": aid,
                            "alternative_mutation": meta["mutation"],
                            "alternative_plddt": plddt_val if plddt_val is not None else "",
                            "fold_preserved": preserved if preserved is not None else "",
                        })
                else:
                    # Low-risk: annotate but no alternatives
                    all_rows.append({
                        "sequence_id": seq_id,
                        "enzyme_id": enzyme_id,
                        "sequon_position": sequon["position"],
                        "sequon_str": sequon["sequon"],
                        "asn_pos": asn_pos,
                        "min_dist_to_active_site": round(min_dist, 2) if min_dist != float("inf") else "inf",
                        "risk_level": risk,
                        "alternative_sequence_id": "",
                        "alternative_mutation": "",
                        "alternative_plddt": "",
                        "fold_preserved": "",
                    })

        summary[enzyme_id] = {
            "total_candidates": len(sequences),
            "total_sequons": enz_total_sequons,
            "high_risk_sequons": enz_high_risk,
            "alternatives_generated": enz_alternatives,
        }

        logger.info(
            "%s: %d candidates, %d sequons, %d high-risk, %d alternatives",
            enzyme_id,
            len(sequences),
            enz_total_sequons,
            enz_high_risk,
            enz_alternatives,
        )

    # Write glycosylation_report.csv
    report_path = output_dir / "glycosylation_report.csv"
    write_scores(all_rows, report_path)
    logger.info("Wrote %d rows to %s", len(all_rows), report_path)

    # Write modified FASTA with alternatives that preserve fold
    if all_alternative_seqs:
        alt_fasta_path = output_dir / "alternative_sequences.fasta"
        write_fasta(all_alternative_seqs, alt_fasta_path)
        logger.info(
            "Wrote %d alternative sequences to %s",
            len(all_alternative_seqs), alt_fasta_path,
        )

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(config_path: Path) -> None:
    """Load config, set up logging, run Stage 7.5."""
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
            logging.FileHandler(log_dir / "stage7_5_glycosylation.log"),
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
        logger.info("  %s: %s", eid, stats)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)
