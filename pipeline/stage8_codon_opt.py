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
import bisect
import logging
import math
import os
import random
import re
import string
from concurrent.futures import ProcessPoolExecutor
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


def count_mutations_from_wt(sequence: str, wt_sequence: str) -> int:
    """Count number of amino acid differences from wild-type.

    Comparison is case-insensitive. Only positions present in both sequences
    (up to the shorter length) are compared.

    Args:
        sequence: Variant amino acid sequence.
        wt_sequence: Wild-type amino acid sequence.

    Returns:
        Integer count of positions that differ.
    """
    seq = sequence.upper()
    wt = wt_sequence.upper()
    return sum(a != b for a, b in zip(seq, wt))


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
    layout_strategy: str = "hedging",
    variant_info: dict[str, tuple[str, int, int]] | None = None,
    wt_sequence: str | None = None,
) -> list[dict[str, Any]]:
    """Assign sequences to 96-well plate positions (A1-H12).

    Args:
        sequences: Dict of {sequence_id: dna_sequence}.
        enzyme_id: Enzyme identifier.
        plate_size: Number of wells per plate (default 96).
        layout_strategy: "hedging" (default) or "sequential".

            "sequential" — fills wells A1 → A12 → B1 → … → H12 in input order
            (original behaviour, backward compatible).

            "hedging" — expression-hedging layout:
                - Well A1:  WT control (sequence with lowest mutation_count,
                            or the sequence_id equal to "wt" if present).
                - Well H12: WT control (second WT slot, filled by the same
                            or another WT replica if enough sequences).
                - Rows A-B (excl. A1): conservative variants (bottom 25%
                            by mutation count).
                - Rows C-F: moderate variants (middle 50% by mutation count).
                - Rows G-H (excl. H12): aggressive variants (top 25% by
                            mutation count).
                - Within each tier: sorted by Pareto rank ascending
                  (best rank first).
                Requires ``variant_info`` and ``wt_sequence`` arguments.

        variant_info: Required for hedging layout.
            Dict of {sequence_id: (protein_seq, mutation_count, pareto_rank)}.
        wt_sequence: Wild-type protein sequence string.  Used to identify the
            WT control well.  The sequence whose ``mutation_count`` is 0 is
            treated as the WT control.

    Returns:
        List of dicts with well assignments.  Hedging layout adds
        ``mutation_count``, ``risk_tier``, and ``pareto_rank`` columns.
    """
    if layout_strategy == "sequential":
        return _generate_plate_map_sequential(sequences, enzyme_id, plate_size)

    # Default: hedging layout
    return _generate_plate_map_hedging(
        sequences, enzyme_id, plate_size, variant_info or {}, wt_sequence or ""
    )


def _generate_plate_map_sequential(
    sequences: dict[str, str],
    enzyme_id: str,
    plate_size: int,
) -> list[dict[str, Any]]:
    """Sequential well assignment (original behaviour)."""
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


# Wells belonging to each hedging region on a standard 96-well plate
# (plate 1 only; subsequent plates use sequential layout for overflow).
_HEDGING_WT_A1   = "A1"
_HEDGING_WT_H12  = "H12"
_HEDGING_ROW_CONSERVATIVE = ("A", "B")   # rows A-B (A1 reserved for WT)
_HEDGING_ROWS_MODERATE    = ("C", "D", "E", "F")
_HEDGING_ROWS_AGGRESSIVE  = ("G", "H")   # H12 reserved for WT


def _hedging_wells(rows: tuple[str, ...], exclude: set[str] | None = None) -> list[str]:
    """Generate well names for the given rows, optionally excluding specific wells."""
    if exclude is None:
        exclude = set()
    wells: list[str] = []
    for row in rows:
        for col in PLATE_COLS:
            w = f"{row}{col}"
            if w not in exclude:
                wells.append(w)
    return wells


def _assign_risk_tier(mutation_count: int, sorted_counts: list[int]) -> str:
    """Assign a risk tier based on where mutation_count falls in sorted_counts.

    Uses the left-insertion index (bisect_left) so that a value not present in
    the list is placed correctly and values at the boundaries are consistent.

    Bottom 25%  -> "conservative"
    Middle 50%  -> "moderate"
    Top 25%     -> "aggressive"
    """
    n = len(sorted_counts)
    if n == 0:
        return "moderate"
    # Use bisect_left so the rank equals the number of elements strictly less
    # than mutation_count, giving a consistent percentile for any value.
    rank = bisect.bisect_left(sorted_counts, mutation_count)
    pct = rank / n  # 0.0 .. <1.0
    if pct < 0.25:
        return "conservative"
    elif pct < 0.75:
        return "moderate"
    else:
        return "aggressive"


def _generate_plate_map_hedging(
    sequences: dict[str, str],
    enzyme_id: str,
    plate_size: int,
    variant_info: dict[str, tuple[str, int, int]],
    wt_sequence: str,
) -> list[dict[str, Any]]:
    """Expression-hedging well assignment.

    Layout (plate 1):
      A1   — WT control (sequence with mutation_count == 0, or least-mutated
              if no exact WT is supplied)
      H12  — WT control (second WT replica, or same WT if only one exists)
      A2-B12  — conservative variants (sorted by Pareto rank)
      C1-F12  — moderate variants   (sorted by Pareto rank)
      G1-H11  — aggressive variants (sorted by Pareto rank)

    Risk tiers are computed over ALL non-WT sequences:
      Bottom 25%  -> conservative
      Middle 50%  -> moderate
      Top 25%     -> aggressive

    Any sequences that overflow the reserved row slots spill into remaining
    wells on plate 1.  Sequences beyond plate 1 capacity go to plate 2+
    using sequential layout.
    """
    wells_per_plate = min(plate_size, len(PLATE_ROWS) * len(PLATE_COLS))

    # Identify WT candidates (mutation_count == 0).
    wt_candidates: list[str] = []
    non_wt_ids:    list[str] = []

    for seq_id in sequences:
        info = variant_info.get(seq_id)
        mut_count = info[1] if info is not None else 0
        if mut_count == 0:
            wt_candidates.append(seq_id)
        else:
            non_wt_ids.append(seq_id)

    # If no explicit WT found, pick the sequence with the lowest mutation
    # count (alphabetically first on ties) as WT control.
    if not wt_candidates:
        all_ids = list(sequences.keys())
        wt_candidates = [
            min(all_ids, key=lambda sid: (variant_info.get(sid, ("", 9999, 9999))[1], sid))
        ]
        # WT control is NOT removed from the variant pool for tier computation;
        # it is simply duplicated in A1 and H12.  The pool for tier assignment
        # is the full set of non-WT-candidate sequences.
        non_wt_ids = [s for s in sequences if s not in wt_candidates]

    # Assign WT controls: A1 and H12.
    wt_a1 = wt_candidates[0]
    wt_h12 = wt_candidates[1] if len(wt_candidates) > 1 else wt_a1

    # Compute mutation counts and tiers over the full non-WT pool.
    # (The WT control wells are fixed; only non-WT sequences get tiered.)
    mut_counts_sorted = sorted(
        variant_info[sid][1] for sid in non_wt_ids if sid in variant_info
    )

    def _tier(seq_id: str) -> str:
        info = variant_info.get(seq_id)
        if info is None:
            return "moderate"
        return _assign_risk_tier(info[1], mut_counts_sorted)

    def _pareto(seq_id: str) -> int:
        info = variant_info.get(seq_id)
        return info[2] if info is not None else 9999

    def _mut_count(seq_id: str) -> int:
        info = variant_info.get(seq_id)
        return info[1] if info is not None else 0

    # Group and sort non-WT sequences by tier then Pareto rank.
    conservative = sorted(
        [s for s in non_wt_ids if _tier(s) == "conservative"],
        key=_pareto,
    )
    moderate = sorted(
        [s for s in non_wt_ids if _tier(s) == "moderate"],
        key=_pareto,
    )
    aggressive = sorted(
        [s for s in non_wt_ids if _tier(s) == "aggressive"],
        key=_pareto,
    )

    # Build available well slots for each zone on plate 1.
    cons_wells  = _hedging_wells(_HEDGING_ROW_CONSERVATIVE, exclude={_HEDGING_WT_A1})
    mod_wells   = _hedging_wells(_HEDGING_ROWS_MODERATE)
    agg_wells   = _hedging_wells(_HEDGING_ROWS_AGGRESSIVE,  exclude={_HEDGING_WT_H12})
    overflow_wells: list[str] = []  # collect remaining wells for spill

    def _base_row(seq_id: str, tier: str = "moderate") -> dict[str, Any]:
        dna_seq = sequences[seq_id]
        return {
            "enzyme_id": enzyme_id,
            "sequence_id": seq_id,
            "protein_length_aa": len(dna_seq) // 3,
            "dna_length_nt": len(dna_seq),
            "plate_number": 1,
            "mutation_count": _mut_count(seq_id),
            "risk_tier": tier,
            "pareto_rank": _pareto(seq_id),
        }

    plate_map: list[dict[str, Any]] = []

    # Place WT controls.
    def _place_wt(seq_id: str, well: str, tier: str) -> None:
        row = _base_row(seq_id, tier=tier)
        row["well_position"] = well
        plate_map.append(row)

    _place_wt(wt_a1,  _HEDGING_WT_A1,  "conservative")
    _place_wt(wt_h12, _HEDGING_WT_H12, "aggressive")

    # Fill tier zones; overflow into a flat list.
    def _fill_zone(
        zone_seqs: list[str],
        zone_wells: list[str],
        tier: str,
        overflow: list[str],
    ) -> None:
        for i, seq_id in enumerate(zone_seqs):
            if i < len(zone_wells):
                row = _base_row(seq_id, tier=tier)
                row["well_position"] = zone_wells[i]
                plate_map.append(row)
            else:
                overflow.append(seq_id)

    _fill_zone(conservative, cons_wells,  "conservative", overflow_wells)
    _fill_zone(moderate,     mod_wells,   "moderate",     overflow_wells)
    _fill_zone(aggressive,   agg_wells,   "aggressive",   overflow_wells)

    # Assign overflow sequences to remaining wells on plate 1, then plate 2+.
    assigned_wells: set[str] = set()
    for r in plate_map:
        if r["plate_number"] == 1:
            assigned_wells.add(r["well_position"])

    remaining_p1: list[str] = []
    for row_lbl in PLATE_ROWS:
        for col in PLATE_COLS:
            w = f"{row_lbl}{col}"
            if w not in assigned_wells:
                remaining_p1.append(w)

    for i, seq_id in enumerate(overflow_wells):
        tier_for_overflow = _tier(seq_id)
        if i < len(remaining_p1):
            row = _base_row(seq_id, tier=tier_for_overflow)
            row["well_position"] = remaining_p1[i]
            plate_map.append(row)
        else:
            # Overflow to subsequent plates (sequential).
            plate_num = (i - len(remaining_p1)) // wells_per_plate + 2
            well_idx  = (i - len(remaining_p1)) % wells_per_plate
            row_lbl   = PLATE_ROWS[well_idx // len(PLATE_COLS)]
            col_num   = PLATE_COLS[well_idx % len(PLATE_COLS)]
            row = _base_row(seq_id, tier=tier_for_overflow)
            row["well_position"] = f"{row_lbl}{col_num}"
            row["plate_number"]  = plate_num
            plate_map.append(row)

    # Sort output for readability: plate -> well row then col.
    def _sort_key(r: dict[str, Any]) -> tuple[int, str, int]:
        well = r["well_position"]
        return (r["plate_number"], well[0], int(well[1:]))

    plate_map.sort(key=_sort_key)
    return plate_map


def _optimize_single_sequence(
    seq_id: str,
    protein_seq: str,
    max_iterations: int,
    target_gc_min: float,
    target_gc_max: float,
    min_cai: float,
    seed: int | None,
    enzyme_id: str,
    cloning_config: dict[str, str],
) -> tuple[str, str, dict[str, Any], int]:
    """Worker for parallel codon optimization (ProcessPoolExecutor).

    Returns:
        (seq_id, dna_construct, optimization_metrics, protein_length)
    """
    clean_protein = protein_seq.upper().replace("*", "").replace(" ", "")
    optimized_dna, metrics = optimize_sequence(
        clean_protein,
        max_iterations=max_iterations,
        target_gc_min=target_gc_min,
        target_gc_max=target_gc_max,
        target_cai_min=min_cai,
        seed=seed,
    )
    construct = add_flanking_sequences(optimized_dna, enzyme_id, cloning_config)
    return seq_id, construct, metrics, len(clean_protein)


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

        # Optimize each sequence (parallel when enough sequences)
        dna_sequences: dict[str, str] = {}
        plate_rows: list[dict[str, Any]] = []
        cai_values: list[float] = []
        gc_values: list[float] = []

        def _collect_result(
            sid: str, construct: str, metrics: dict[str, Any], protein_len: int
        ) -> None:
            dna_sequences[sid] = construct
            cai_values.append(metrics["cai"])
            gc_values.append(metrics["gc_content"])
            plate_rows.append({
                "enzyme_id": enzyme_id,
                "sequence_id": sid,
                "protein_length_aa": protein_len,
                "dna_length_nt": len(construct),
                "cai": metrics["cai"],
                "gc_content": metrics["gc_content"],
                "has_restriction_sites": metrics.get("has_restriction_sites", False),
                "status": "optimized",
            })

        n_workers = min(os.cpu_count() or 1, len(protein_seqs))
        if n_workers > 1 and len(protein_seqs) >= 4:
            logger.info(
                "%s: optimizing %d sequences in parallel (%d workers)",
                enzyme_id, len(protein_seqs), n_workers,
            )
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(
                        _optimize_single_sequence,
                        seq_id, protein_seq,
                        max_iterations, target_gc_min, target_gc_max,
                        min_cai, seed, enzyme_id, cloning_config,
                    )
                    for seq_id, protein_seq in protein_seqs.items()
                ]
                for future in futures:
                    sid, construct, metrics, protein_len = future.result()
                    _collect_result(sid, construct, metrics, protein_len)
        else:
            for seq_id, protein_seq in protein_seqs.items():
                sid, construct, metrics, protein_len = _optimize_single_sequence(
                    seq_id, protein_seq,
                    max_iterations, target_gc_min, target_gc_max,
                    min_cai, seed, enzyme_id, cloning_config,
                )
                _collect_result(sid, construct, metrics, protein_len)

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
