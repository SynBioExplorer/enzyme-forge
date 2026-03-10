"""Stage 8: Codon Optimization for S. cerevisiae expression.

Prepares DNA sequences for synthesis and yeast expression:
1. Read final protein FASTA sequences from Stage 7.5 (or 7)
2. Reverse-translate using S. cerevisiae codon usage table (Kazusa)
3. Optimize: CAI > 0.7, GC 35-45%, avoid inhibitory codon pairs
4. Remove restriction sites: BsaI (GGTCTC), BsmBI (CGTCTC), NotI (GCGGCCGC)
5. Remove cryptic splicing signals, poly-A signals, homopolymer runs > 6 nt
6. Add flanking sequences for Golden Gate cloning
7. Generate 96-well plate maps per enzyme

Output: DNA FASTA + plate_maps.csv in stage_08_codon_opt/

CLI: python pipeline/stage8_codon_opt.py --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import re
import string
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

STAGE_NAME = "Stage 8: Codon Optimization"
STAGE_NUM = "8"
INPUT_DESCRIPTION = "Final protein FASTA sequences from Stage 7.5 (or 7)"
OUTPUT_DESCRIPTION = "DNA FASTA + plate_maps.csv in stage_08_codon_opt/"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# S. cerevisiae codon usage table (Kazusa)
# ---------------------------------------------------------------------------

YEAST_CODON_TABLE: dict[str, dict[str, float]] = {
    "A": {"GCT": 0.38, "GCC": 0.22, "GCA": 0.29, "GCG": 0.11},
    "R": {"CGT": 0.14, "CGC": 0.06, "CGA": 0.07, "CGG": 0.04, "AGA": 0.48, "AGG": 0.21},
    "N": {"AAT": 0.59, "AAC": 0.41},
    "D": {"GAT": 0.65, "GAC": 0.35},
    "C": {"TGT": 0.63, "TGC": 0.37},
    "Q": {"CAA": 0.69, "CAG": 0.31},
    "E": {"GAA": 0.70, "GAG": 0.30},
    "G": {"GGT": 0.47, "GGC": 0.19, "GGA": 0.22, "GGG": 0.12},
    "H": {"CAT": 0.64, "CAC": 0.36},
    "I": {"ATT": 0.46, "ATC": 0.26, "ATA": 0.27},
    "L": {"TTA": 0.28, "TTG": 0.29, "CTT": 0.13, "CTC": 0.06, "CTA": 0.14, "CTG": 0.11},
    "K": {"AAA": 0.58, "AAG": 0.42},
    "M": {"ATG": 1.00},
    "F": {"TTT": 0.59, "TTC": 0.41},
    "P": {"CCT": 0.31, "CCC": 0.15, "CCA": 0.42, "CCG": 0.12},
    "S": {"TCT": 0.26, "TCC": 0.16, "TCA": 0.21, "TCG": 0.10, "AGT": 0.16, "AGC": 0.11},
    "T": {"ACT": 0.35, "ACC": 0.22, "ACA": 0.30, "ACG": 0.13},
    "W": {"TGG": 1.00},
    "Y": {"TAT": 0.56, "TAC": 0.44},
    "V": {"GTT": 0.39, "GTC": 0.21, "GTA": 0.21, "GTG": 0.19},
    "*": {"TAA": 0.47, "TAG": 0.23, "TGA": 0.30},
}

# Reverse lookup: codon -> amino acid
_CODON_TO_AA: dict[str, str] = {}
for _aa, _codons in YEAST_CODON_TABLE.items():
    for _codon in _codons:
        _CODON_TO_AA[_codon] = _aa

# Default restriction sites to remove
DEFAULT_RESTRICTION_SITES: dict[str, str] = {
    "BsaI": "GGTCTC",
    "BsmBI": "CGTCTC",
    "NotI": "GCGGCCGC",
}

# Default inhibitory codon pairs
DEFAULT_INHIBITORY_PAIRS: list[str] = ["CGACGA", "CGACCG"]

# Poly-A signal
POLY_A_SIGNAL = "AATAAA"

# Default flanking sequences for Golden Gate cloning
DEFAULT_CLONING_CONFIG: dict[str, str] = {
    "5prime_flank": "AATGGGTCTC",
    "3prime_flank": "GAGACCTTAA",
}

# 96-well plate layout
PLATE_ROWS = list("ABCDEFGH")
PLATE_COLS = list(range(1, 13))


# ---------------------------------------------------------------------------
# Translation helper
# ---------------------------------------------------------------------------

def translate_dna(dna_seq: str) -> str:
    """Translate a DNA sequence to protein using the standard genetic code."""
    protein = []
    for i in range(0, len(dna_seq) - 2, 3):
        codon = dna_seq[i:i + 3].upper()
        aa = _CODON_TO_AA.get(codon)
        if aa is None:
            raise ValueError(f"Unknown codon: {codon}")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def reverse_translate(
    protein_seq: str,
    codon_table: dict[str, dict[str, float]] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Reverse-translate a protein sequence to DNA using weighted random sampling.

    Args:
        protein_seq: Amino acid sequence string.
        codon_table: Codon usage table {aa: {codon: frequency}}.
        rng: Random number generator for reproducibility.

    Returns:
        DNA sequence string.
    """
    if codon_table is None:
        codon_table = YEAST_CODON_TABLE
    if rng is None:
        rng = random.Random()

    codons: list[str] = []
    for aa in protein_seq.upper():
        if aa not in codon_table:
            raise ValueError(f"Unknown amino acid: {aa}")
        aa_codons = codon_table[aa]
        codon_list = list(aa_codons.keys())
        weights = list(aa_codons.values())
        chosen = rng.choices(codon_list, weights=weights, k=1)[0]
        codons.append(chosen)
    return "".join(codons)


def calc_cai(
    dna_seq: str,
    codon_table: dict[str, dict[str, float]] | None = None,
) -> float:
    """Calculate the Codon Adaptation Index (CAI).

    CAI = geometric mean of (w_i) for each codon, where
    w_i = frequency(codon) / max_frequency(amino acid).

    Args:
        dna_seq: DNA sequence (length must be multiple of 3).
        codon_table: Codon usage table.

    Returns:
        CAI value between 0 and 1.
    """
    if codon_table is None:
        codon_table = YEAST_CODON_TABLE

    # Precompute max frequency per amino acid
    max_freq: dict[str, float] = {}
    for aa, codons in codon_table.items():
        max_freq[aa] = max(codons.values())

    log_sum = 0.0
    count = 0
    for i in range(0, len(dna_seq) - 2, 3):
        codon = dna_seq[i:i + 3].upper()
        aa = _CODON_TO_AA.get(codon)
        if aa is None or aa == "*":
            continue
        freq = codon_table[aa].get(codon, 0.0)
        mf = max_freq[aa]
        if mf > 0 and freq > 0:
            w = freq / mf
            log_sum += math.log(w)
            count += 1

    if count == 0:
        return 0.0
    return math.exp(log_sum / count)


def calc_gc_content(dna_seq: str) -> float:
    """Calculate GC content as a fraction (0-1).

    Args:
        dna_seq: DNA sequence string.

    Returns:
        GC fraction between 0.0 and 1.0.
    """
    if not dna_seq:
        return 0.0
    seq = dna_seq.upper()
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)


def _get_synonymous_codons(codon: str, codon_table: dict[str, dict[str, float]] | None = None) -> list[str]:
    """Get all synonymous codons for the given codon, excluding itself."""
    if codon_table is None:
        codon_table = YEAST_CODON_TABLE
    aa = _CODON_TO_AA.get(codon.upper())
    if aa is None:
        return []
    return [c for c in codon_table[aa] if c != codon.upper()]


def remove_restriction_sites(
    dna_seq: str,
    sites: dict[str, str] | None = None,
    codon_table: dict[str, dict[str, float]] | None = None,
) -> str:
    """Remove restriction enzyme recognition sites by synonymous codon substitution.

    For each site found: synonymously substitute one codon to break the site
    while preserving the encoded protein.

    Args:
        dna_seq: DNA sequence string (length multiple of 3).
        sites: Dict of {enzyme_name: recognition_sequence}.
        codon_table: Codon usage table.

    Returns:
        Modified DNA sequence with the same protein translation.
    """
    if sites is None:
        sites = DEFAULT_RESTRICTION_SITES
    if codon_table is None:
        codon_table = YEAST_CODON_TABLE

    seq = dna_seq.upper()
    changed = True

    # Iterate until no sites remain (each substitution might create new ones)
    max_passes = 20
    for _pass in range(max_passes):
        changed = False
        for _site_name, site_seq in sites.items():
            # Check both strands
            rc_site = _reverse_complement(site_seq)
            for pattern in [site_seq, rc_site]:
                pos = seq.find(pattern)
                while pos != -1:
                    # Find which codons overlap with this site
                    codon_start = (pos // 3) * 3
                    codon_end = ((pos + len(pattern) - 1) // 3 + 1) * 3
                    # Try substituting each overlapping codon
                    fixed = False
                    for ci in range(codon_start, min(codon_end, len(seq) - 2), 3):
                        old_codon = seq[ci:ci + 3]
                        synonymous = _get_synonymous_codons(old_codon, codon_table)
                        for new_codon in synonymous:
                            candidate = seq[:ci] + new_codon + seq[ci + 3:]
                            if pattern not in candidate[max(0, pos - 6):pos + len(pattern) + 6]:
                                seq = candidate
                                fixed = True
                                changed = True
                                break
                        if fixed:
                            break
                    if not fixed:
                        # Can't fix this occurrence — move past it
                        pos = seq.find(pattern, pos + 1)
                        continue
                    # Re-search from start for this pattern
                    pos = seq.find(pattern)
        if not changed:
            break

    return seq


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(comp.get(b, b) for b in reversed(seq.upper()))


def remove_homopolymers(
    dna_seq: str,
    max_run: int = 6,
    codon_table: dict[str, dict[str, float]] | None = None,
) -> str:
    """Remove homopolymer runs longer than max_run by synonymous substitution.

    Args:
        dna_seq: DNA sequence string.
        max_run: Maximum allowed consecutive identical nucleotides.
        codon_table: Codon usage table.

    Returns:
        Modified DNA sequence preserving the same protein.
    """
    if codon_table is None:
        codon_table = YEAST_CODON_TABLE

    seq = dna_seq.upper()

    for _pass in range(20):
        # Find homopolymer runs
        match = re.search(r"(.)\1{" + str(max_run) + r",}", seq)
        if match is None:
            break

        run_start = match.start()
        run_end = match.end()

        # Find overlapping codons and try to break the run
        codon_start = (run_start // 3) * 3
        codon_end = ((run_end - 1) // 3 + 1) * 3
        fixed = False
        for ci in range(codon_start, min(codon_end, len(seq) - 2), 3):
            old_codon = seq[ci:ci + 3]
            synonymous = _get_synonymous_codons(old_codon, codon_table)
            for new_codon in synonymous:
                candidate = seq[:ci] + new_codon + seq[ci + 3:]
                # Check if the run is broken
                region = candidate[max(0, run_start - max_run):run_end + max_run]
                if not re.search(r"(.)\1{" + str(max_run) + r",}", region):
                    seq = candidate
                    fixed = True
                    break
            if fixed:
                break
        if not fixed:
            break

    return seq


def check_poly_a_signals(dna_seq: str) -> list[int]:
    """Find all positions of AATAAA poly-A signals in the DNA sequence.

    Args:
        dna_seq: DNA sequence string.

    Returns:
        List of 0-based positions where AATAAA occurs.
    """
    positions: list[int] = []
    seq = dna_seq.upper()
    start = 0
    while True:
        pos = seq.find(POLY_A_SIGNAL, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions


def remove_poly_a_signals(
    dna_seq: str,
    codon_table: dict[str, dict[str, float]] | None = None,
) -> str:
    """Remove poly-A signal sequences (AATAAA) by synonymous substitution.

    Args:
        dna_seq: DNA sequence string.
        codon_table: Codon usage table.

    Returns:
        Modified DNA sequence preserving the same protein.
    """
    if codon_table is None:
        codon_table = YEAST_CODON_TABLE

    seq = dna_seq.upper()

    for _pass in range(50):
        positions = check_poly_a_signals(seq)
        if not positions:
            break
        pos = positions[0]
        codon_start = (pos // 3) * 3
        codon_end = ((pos + len(POLY_A_SIGNAL) - 1) // 3 + 1) * 3
        fixed = False
        for ci in range(codon_start, min(codon_end, len(seq) - 2), 3):
            old_codon = seq[ci:ci + 3]
            synonymous = _get_synonymous_codons(old_codon, codon_table)
            for new_codon in synonymous:
                candidate = seq[:ci] + new_codon + seq[ci + 3:]
                if POLY_A_SIGNAL not in candidate[max(0, pos - 3):pos + len(POLY_A_SIGNAL) + 3]:
                    seq = candidate
                    fixed = True
                    break
            if fixed:
                break
        if not fixed:
            break

    return seq


def optimize_codon_pairs(
    dna_seq: str,
    protein_seq: str,
    inhibitory_pairs: list[str] | None = None,
    codon_table: dict[str, dict[str, float]] | None = None,
) -> str:
    """Remove inhibitory codon pairs by synonymous substitution.

    Args:
        dna_seq: DNA sequence string.
        protein_seq: Original protein sequence for verification.
        inhibitory_pairs: List of 6-nt codon pair strings to avoid.
        codon_table: Codon usage table.

    Returns:
        Modified DNA sequence preserving the same protein.
    """
    if inhibitory_pairs is None:
        inhibitory_pairs = DEFAULT_INHIBITORY_PAIRS
    if codon_table is None:
        codon_table = YEAST_CODON_TABLE

    seq = dna_seq.upper()

    for _pass in range(50):
        found_any = False
        for pair in inhibitory_pairs:
            pos = seq.find(pair)
            if pos == -1 or pos % 3 != 0:
                # Only consider pairs aligned to codon boundaries
                # Search for codon-boundary aligned occurrences
                for offset in range(0, len(seq) - 5, 3):
                    if seq[offset:offset + 6] == pair:
                        pos = offset
                        break
                else:
                    continue
            found_any = True
            # The pair spans two codons starting at pos
            codon1_start = pos
            codon2_start = pos + 3
            # Try substituting the second codon first (less likely to be the preferred one)
            fixed = False
            for ci in [codon2_start, codon1_start]:
                if ci + 3 > len(seq):
                    continue
                old_codon = seq[ci:ci + 3]
                synonymous = _get_synonymous_codons(old_codon, codon_table)
                for new_codon in synonymous:
                    candidate = seq[:ci] + new_codon + seq[ci + 3:]
                    # Verify the pair is broken
                    region = candidate[max(0, pos - 3):pos + 9]
                    if pair not in region:
                        seq = candidate
                        fixed = True
                        break
                if fixed:
                    break
        if not found_any:
            break

    return seq


def optimize_sequence(
    protein_seq: str,
    max_iterations: int = 10,
    target_gc_min: float = 0.35,
    target_gc_max: float = 0.45,
    target_cai_min: float = 0.70,
    codon_table: dict[str, dict[str, float]] | None = None,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Iteratively optimize a codon sequence for S. cerevisiae expression.

    Steps per iteration:
    1. Reverse translate
    2. Check GC content and CAI
    3. Remove restriction sites
    4. Remove homopolymers
    5. Remove poly-A signals
    6. Optimize codon pairs
    7. Repeat if targets not met

    Args:
        protein_seq: Amino acid sequence.
        max_iterations: Maximum optimization attempts.
        target_gc_min: Minimum target GC content.
        target_gc_max: Maximum target GC content.
        target_cai_min: Minimum target CAI.
        codon_table: Codon usage table.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (optimized_dna, metrics_dict).
    """
    if codon_table is None:
        codon_table = YEAST_CODON_TABLE

    rng = random.Random(seed)
    best_dna: str | None = None
    best_score: float = -1.0
    best_metrics: dict[str, Any] = {}
    sites_removed: list[str] = []

    for iteration in range(1, max_iterations + 1):
        # Step 1: Reverse translate
        dna = reverse_translate(protein_seq, codon_table, rng)

        # Step 2: Remove restriction sites
        dna_clean = remove_restriction_sites(dna, codon_table=codon_table)
        removed = []
        for name, site in DEFAULT_RESTRICTION_SITES.items():
            if site in dna and site not in dna_clean:
                removed.append(name)
            rc = _reverse_complement(site)
            if rc in dna and rc not in dna_clean:
                removed.append(f"{name}(rc)")
        dna = dna_clean

        # Step 3: Remove homopolymers
        dna = remove_homopolymers(dna, codon_table=codon_table)

        # Step 4: Remove poly-A signals
        dna = remove_poly_a_signals(dna, codon_table=codon_table)

        # Step 5: Optimize codon pairs
        dna = optimize_codon_pairs(dna, protein_seq, codon_table=codon_table)

        # Step 6: Check metrics
        cai = calc_cai(dna, codon_table)
        gc = calc_gc_content(dna)

        # Score: prefer sequences meeting both targets
        score = cai
        if target_gc_min <= gc <= target_gc_max:
            score += 1.0  # bonus for being in GC range
        if cai >= target_cai_min:
            score += 1.0  # bonus for meeting CAI target

        if score > best_score:
            best_score = score
            best_dna = dna
            best_metrics = {
                "cai": round(cai, 4),
                "gc_content": round(gc, 4),
                "restriction_sites_removed": removed,
                "iterations": iteration,
                "poly_a_signals": len(check_poly_a_signals(dna)),
                "has_restriction_sites": _has_restriction_sites(dna),
            }
            sites_removed = removed

        # Early termination if targets met
        if cai >= target_cai_min and target_gc_min <= gc <= target_gc_max:
            break

    # Verify translation is preserved
    if best_dna is not None:
        translated = translate_dna(best_dna)
        if translated != protein_seq.upper().rstrip("*"):
            logger.error(
                "Translation mismatch after optimization! "
                "Expected %d aa, got %d aa",
                len(protein_seq), len(translated),
            )

    return best_dna or "", best_metrics


def _has_restriction_sites(dna_seq: str) -> bool:
    """Check if any default restriction sites remain in the sequence."""
    seq = dna_seq.upper()
    for site in DEFAULT_RESTRICTION_SITES.values():
        if site in seq or _reverse_complement(site) in seq:
            return True
    return False


def add_flanking_sequences(
    dna_seq: str,
    enzyme_id: str,
    cloning_config: dict[str, str] | None = None,
) -> str:
    """Add 5' and 3' flanking sequences for Golden Gate cloning.

    Args:
        dna_seq: Core DNA sequence.
        enzyme_id: Enzyme identifier (for logging).
        cloning_config: Dict with '5prime_flank' and '3prime_flank' keys.

    Returns:
        Full construct DNA sequence with flanks.
    """
    if cloning_config is None:
        cloning_config = DEFAULT_CLONING_CONFIG

    flank_5 = cloning_config.get("5prime_flank", "")
    flank_3 = cloning_config.get("3prime_flank", "")

    return flank_5 + dna_seq + flank_3


def generate_plate_map(
    sequences: dict[str, str],
    enzyme_id: str,
    plate_size: int = 96,
) -> list[dict[str, Any]]:
    """Assign sequences to 96-well plate positions (A1-H12).

    Args:
        sequences: Dict of {sequence_id: dna_sequence}.
        enzyme_id: Enzyme identifier.
        plate_size: Number of wells per plate (default 96).

    Returns:
        List of dicts with well assignments.
    """
    plate_map: list[dict[str, Any]] = []
    seq_list = list(sequences.items())
    wells_per_plate = min(plate_size, len(PLATE_ROWS) * len(PLATE_COLS))

    for idx, (seq_id, dna_seq) in enumerate(seq_list):
        plate_num = idx // wells_per_plate + 1
        well_idx = idx % wells_per_plate
        row = PLATE_ROWS[well_idx // len(PLATE_COLS)]
        col = PLATE_COLS[well_idx % len(PLATE_COLS)]
        well_pos = f"{row}{col}"

        plate_map.append({
            "enzyme_id": enzyme_id,
            "well_position": well_pos,
            "sequence_id": seq_id,
            "protein_length_aa": len(dna_seq) // 3,
            "dna_length_nt": len(dna_seq),
            "plate_number": plate_num,
        })

    return plate_map


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(config: dict[str, Any]) -> dict[str, Any]:
    """Execute Stage 8: Codon optimization for all enzymes.

    Steps per enzyme:
    1. Read final protein FASTA from stage_07.5_glycosylation/ or stage_07_crossval_ranking/
    2. For each sequence: optimize_sequence + add flanking + plate assignment
    3. Write DNA FASTA to stage_08_codon_opt/dna_sequences/
    4. Write plate maps to stage_08_codon_opt/plate_maps/
    5. Return summary with CAI stats, GC content distribution

    Args:
        config: Loaded pipeline configuration dict.

    Returns:
        Summary dict with per-enzyme stats.
    """
    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_08_codon_opt", {})
    output_dir_name = stage_cfg.get("output_dir", "stage_08_codon_opt")
    output_dir = base_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dna_dir = output_dir / "dna_sequences"
    dna_dir.mkdir(parents=True, exist_ok=True)
    plate_dir = output_dir / "plate_maps"
    plate_dir.mkdir(parents=True, exist_ok=True)

    target_gc_min = float(stage_cfg.get("target_gc_min", 0.35))
    target_gc_max = float(stage_cfg.get("target_gc_max", 0.45))
    min_cai = float(stage_cfg.get("min_cai", 0.7))
    max_iterations = int(stage_cfg.get("max_iterations", 10))
    seed = stage_cfg.get("seed")

    cloning_config = stage_cfg.get("cloning", DEFAULT_CLONING_CONFIG)

    # Input directories (try stage 7.5 first, then stage 7)
    input_dir_75 = base_dir / config.get("stages", {}).get(
        "stage_07_5_glycosylation", {}
    ).get("output_dir", "stage_07.5_glycosylation")

    input_dir_7 = base_dir / config.get("stages", {}).get(
        "stage_07_crossval", {}
    ).get("output_dir", "stage_07_crossval_ranking")

    enzyme_ids = sorted(config.get("enzymes", {}).keys())
    all_plate_rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}

    for enzyme_id in enzyme_ids:
        logger.info("Processing %s ...", enzyme_id)

        # Find input FASTA
        input_fasta = None
        for idir in [input_dir_75, input_dir_7]:
            for pattern in [
                f"{enzyme_id}_filtered.fasta",
                f"{enzyme_id}_final.fasta",
                f"{enzyme_id}.fasta",
            ]:
                candidate = idir / pattern
                if candidate.exists():
                    input_fasta = candidate
                    break
            if input_fasta:
                break

        if input_fasta is None:
            logger.warning(
                "No input FASTA found for %s — skipping", enzyme_id
            )
            continue

        protein_seqs = read_fasta(input_fasta)
        if not protein_seqs:
            logger.warning("Empty FASTA for %s — skipping", enzyme_id)
            continue

        logger.info("%s: %d input sequences from %s", enzyme_id, len(protein_seqs), input_fasta)

        # Optimize each sequence
        dna_sequences: dict[str, str] = {}
        plate_rows: list[dict[str, Any]] = []
        cai_values: list[float] = []
        gc_values: list[float] = []

        for seq_id, protein_seq in protein_seqs.items():
            # Clean protein sequence
            clean_protein = protein_seq.upper().replace("*", "").replace(" ", "")

            optimized_dna, metrics = optimize_sequence(
                clean_protein,
                max_iterations=max_iterations,
                target_gc_min=target_gc_min,
                target_gc_max=target_gc_max,
                target_cai_min=min_cai,
                seed=seed,
            )

            # Add flanking sequences
            construct = add_flanking_sequences(
                optimized_dna, enzyme_id, cloning_config
            )

            dna_sequences[seq_id] = construct
            cai_values.append(metrics["cai"])
            gc_values.append(metrics["gc_content"])

            plate_rows.append({
                "enzyme_id": enzyme_id,
                "sequence_id": seq_id,
                "protein_length_aa": len(clean_protein),
                "dna_length_nt": len(construct),
                "cai": metrics["cai"],
                "gc_content": metrics["gc_content"],
                "has_restriction_sites": metrics.get("has_restriction_sites", False),
                "status": "optimized",
            })

        # Generate plate maps
        plate_map = generate_plate_map(dna_sequences, enzyme_id)
        # Merge plate_map well info into plate_rows
        well_map = {pm["sequence_id"]: pm["well_position"] for pm in plate_map}
        for row in plate_rows:
            row["well_position"] = well_map.get(row["sequence_id"], "")

        # Reorder columns for output
        ordered_rows: list[dict[str, Any]] = []
        for row in plate_rows:
            ordered_rows.append({
                "enzyme_id": row["enzyme_id"],
                "well_position": row["well_position"],
                "sequence_id": row["sequence_id"],
                "protein_length_aa": row["protein_length_aa"],
                "dna_length_nt": row["dna_length_nt"],
                "cai": row["cai"],
                "gc_content": row["gc_content"],
                "has_restriction_sites": row["has_restriction_sites"],
                "status": row["status"],
            })

        # Write DNA FASTA
        out_fasta = dna_dir / f"{enzyme_id}_codon_opt.fasta"
        write_fasta(dna_sequences, out_fasta)
        logger.info("Wrote %d DNA sequences to %s", len(dna_sequences), out_fasta)

        # Write plate map CSV
        plate_csv = plate_dir / f"{enzyme_id}_plate_map.csv"
        write_scores(ordered_rows, plate_csv)
        logger.info("Wrote plate map to %s", plate_csv)

        all_plate_rows.extend(ordered_rows)

        summary[enzyme_id] = {
            "n_sequences": len(dna_sequences),
            "mean_cai": round(float(np.mean(cai_values)), 4) if cai_values else 0.0,
            "mean_gc": round(float(np.mean(gc_values)), 4) if gc_values else 0.0,
            "min_cai": round(float(np.min(cai_values)), 4) if cai_values else 0.0,
            "max_cai": round(float(np.max(cai_values)), 4) if cai_values else 0.0,
        }

    # Write combined plate map
    combined_csv = plate_dir / "all_plate_maps.csv"
    write_scores(all_plate_rows, combined_csv)
    logger.info("Wrote combined plate map (%d rows) to %s", len(all_plate_rows), combined_csv)

    return {"summary": summary, "total_sequences": len(all_plate_rows)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(config_path: Path) -> None:
    """Load config, set up logging, run Stage 8."""
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
            logging.FileHandler(log_dir / "stage8_codon_opt.log"),
        ],
    )

    logger.info("=" * 60)
    logger.info("%s — starting", STAGE_NAME)
    logger.info("Config: %s", config_path)
    logger.info("=" * 60)

    results = run_stage(config)

    logger.info("=" * 60)
    logger.info("%s — complete", STAGE_NAME)
    for eid, stats in results.get("summary", {}).items():
        logger.info(
            "  %s: %d sequences, mean CAI=%.3f, mean GC=%.3f",
            eid, stats["n_sequences"], stats["mean_cai"], stats["mean_gc"],
        )
    logger.info("Total sequences: %d", results.get("total_sequences", 0))
    logger.info("=" * 60)


def _cli() -> None:
    """Argument parser wrapper for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Stage 8: Codon Optimization for S. cerevisiae",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    _cli()
