"""Tests for Stage 8: Codon Optimization (pipeline/stage8_codon_opt.py).

All tests are pure Python (no GPU needed).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import pytest

from pipeline.stage8_codon_opt import (
    STAGE_NAME,
    STAGE_NUM,
    INPUT_DESCRIPTION,
    OUTPUT_DESCRIPTION,
    YEAST_CODON_TABLE,
    DEFAULT_RESTRICTION_SITES,
    DEFAULT_INHIBITORY_PAIRS,
    reverse_translate,
    translate_dna,
    calc_cai,
    calc_gc_content,
    remove_restriction_sites,
    remove_homopolymers,
    check_poly_a_signals,
    remove_poly_a_signals,
    optimize_codon_pairs,
    optimize_sequence,
    add_flanking_sequences,
    generate_plate_map,
    run_stage,
    _reverse_complement,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _write_fasta(path: Path, sequences: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for header, seq in sequences.items():
            fh.write(f">{header}\n{seq}\n")
    return path


def _make_stage8_config(tmp_path: Path, enzyme_ids: list[str] | None = None) -> dict:
    if enzyme_ids is None:
        enzyme_ids = ["enzyme_A"]

    base_dir = tmp_path / "pipeline"
    base_dir.mkdir(exist_ok=True)

    # Create input dir (stage 7.5)
    input_dir = base_dir / "stage_07.5_glycosylation"
    input_dir.mkdir(parents=True)
    for eid in enzyme_ids:
        _write_fasta(
            input_dir / f"{eid}_filtered.fasta",
            {
                f"{eid}_var1": "MKTLYKE",
                f"{eid}_var2": "AVILWFG",
            },
        )

    config = {
        "global": {
            "project_name": "test_codon_opt",
            "base_dir": str(base_dir),
            "log_dir": "logs",
            "results_dir": "results",
        },
        "databases": {
            "colabfold_db_dir": str(base_dir / "db"),
            "mmseqs2_db_dir": str(base_dir / "db"),
        },
        "tools": {},
        "stages": {
            "stage_07_5_glycosylation": {
                "output_dir": "stage_07.5_glycosylation",
            },
            "stage_07_crossval": {
                "output_dir": "stage_07_crossval_ranking",
            },
            "stage_08_codon_opt": {
                "output_dir": "stage_08_codon_opt",
                "target_gc_min": 0.35,
                "target_gc_max": 0.45,
                "min_cai": 0.7,
                "max_iterations": 5,
            },
        },
        "enzymes": {
            eid: {
                "name": eid.replace("_", " ").title(),
                "length": 100,
                "sequence_file": "dummy.fasta",
                "substrate_smiles": "C",
                "product_smiles": "C",
                "fixed_residues": [],
                "hotspot_residues": [],
                "contigs": "A1-100",
                "cofactors": [],
            }
            for eid in enzyme_ids
        },
    }
    return config


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_stage_name(self):
        assert "Codon" in STAGE_NAME

    def test_stage_num(self):
        assert STAGE_NUM == "8"

    def test_descriptions(self):
        assert INPUT_DESCRIPTION
        assert OUTPUT_DESCRIPTION


# ---------------------------------------------------------------------------
# Codon table validation
# ---------------------------------------------------------------------------

class TestCodonTable:
    def test_all_20_amino_acids_covered(self):
        for aa in ALL_AMINO_ACIDS:
            assert aa in YEAST_CODON_TABLE, f"Missing amino acid: {aa}"

    def test_stop_codon_present(self):
        assert "*" in YEAST_CODON_TABLE

    def test_frequencies_sum_approximately_one(self):
        for aa, codons in YEAST_CODON_TABLE.items():
            total = sum(codons.values())
            assert abs(total - 1.0) < 0.02, f"{aa}: frequencies sum to {total}"

    def test_codons_are_three_nucleotides(self):
        for aa, codons in YEAST_CODON_TABLE.items():
            for codon in codons:
                assert len(codon) == 3
                assert all(nt in "ACGT" for nt in codon)


# ---------------------------------------------------------------------------
# reverse_translate
# ---------------------------------------------------------------------------

class TestReverseTranslate:
    def test_produces_valid_dna(self):
        protein = "MKTAL"
        dna = reverse_translate(protein, rng=random.Random(42))
        assert len(dna) == len(protein) * 3
        assert all(nt in "ACGT" for nt in dna)

    def test_all_20_amino_acids(self):
        dna = reverse_translate(ALL_AMINO_ACIDS, rng=random.Random(42))
        assert len(dna) == 20 * 3
        # Verify translation round-trips
        translated = translate_dna(dna)
        assert translated == ALL_AMINO_ACIDS

    def test_reproducible_with_seed(self):
        protein = "MKTLYKE"
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        dna1 = reverse_translate(protein, rng=rng1)
        dna2 = reverse_translate(protein, rng=rng2)
        assert dna1 == dna2

    def test_unknown_amino_acid_raises(self):
        with pytest.raises(ValueError, match="Unknown amino acid"):
            reverse_translate("MX", rng=random.Random(42))

    def test_translation_preserves_protein(self):
        protein = "MKTLYKERAVILWFGSTCMPNDHQ"
        dna = reverse_translate(protein, rng=random.Random(42))
        assert translate_dna(dna) == protein


# ---------------------------------------------------------------------------
# calc_cai
# ---------------------------------------------------------------------------

class TestCalcCAI:
    def test_returns_between_0_and_1(self):
        dna = reverse_translate("MKTLYKERAVILWFG", rng=random.Random(42))
        cai = calc_cai(dna)
        assert 0.0 < cai <= 1.0

    def test_optimal_codons_give_high_cai(self):
        # Build sequence using only the most frequent codons
        optimal_codons = []
        protein = "MKTLYKE"
        for aa in protein:
            codons = YEAST_CODON_TABLE[aa]
            best = max(codons, key=codons.get)
            optimal_codons.append(best)
        dna = "".join(optimal_codons)
        cai = calc_cai(dna)
        assert cai == pytest.approx(1.0, abs=0.001)

    def test_rare_codons_give_low_cai(self):
        # Build sequence using only the rarest codons
        rare_codons = []
        protein = "ARGLKE"
        for aa in protein:
            codons = YEAST_CODON_TABLE[aa]
            worst = min(codons, key=codons.get)
            rare_codons.append(worst)
        dna = "".join(rare_codons)
        cai = calc_cai(dna)
        assert cai < 0.5

    def test_empty_sequence(self):
        assert calc_cai("") == 0.0


# ---------------------------------------------------------------------------
# calc_gc_content
# ---------------------------------------------------------------------------

class TestCalcGCContent:
    def test_all_gc(self):
        assert calc_gc_content("GCGCGC") == pytest.approx(1.0)

    def test_all_at(self):
        assert calc_gc_content("ATATAT") == pytest.approx(0.0)

    def test_mixed(self):
        assert calc_gc_content("ATGC") == pytest.approx(0.5)

    def test_empty(self):
        assert calc_gc_content("") == 0.0

    def test_known_sequence(self):
        # 4 G/C out of 6: A T G C G C
        assert calc_gc_content("ATGCGC") == pytest.approx(4 / 6)


# ---------------------------------------------------------------------------
# remove_restriction_sites
# ---------------------------------------------------------------------------

class TestRemoveRestrictionSites:
    def test_removes_bsai_site(self):
        # Build a sequence that contains a BsaI site (GGTCTC)
        # "GY" encodes to GGT + TAT, but we need GGTCTC
        # Manually construct: GGT CTC -> G(Gly) L(Leu)? CTC = Leu
        # Actually, let's just insert the site into a longer sequence
        protein = "MGLSAK"  # 6 aa = 18 nt
        # Keep trying seeds until we can inject a site
        # Instead, manually build a DNA with a BsaI site
        dna = "ATGGGTCTCGCTAAAAAG"  # contains GGTCTC at pos 3
        assert "GGTCTC" in dna
        result = remove_restriction_sites(dna)
        assert "GGTCTC" not in result
        # Verify same protein
        assert translate_dna(result) == translate_dna(dna)

    def test_removes_notI_site(self):
        # NotI = GCGGCCGC (8 nt)
        # Build DNA with NotI inside
        dna = "ATGGCGGCCGCAAAAAAG"
        assert "GCGGCCGC" in dna
        result = remove_restriction_sites(dna)
        assert "GCGGCCGC" not in result
        assert translate_dna(result) == translate_dna(dna)

    def test_no_change_when_no_sites(self):
        dna = "ATGAAAGCT"  # M K A
        result = remove_restriction_sites(dna)
        assert result == dna

    def test_preserves_protein_translation(self):
        protein = "MKTLYKERAVILWFG"
        rng = random.Random(1)
        dna = reverse_translate(protein, rng=rng)
        result = remove_restriction_sites(dna)
        assert translate_dna(result) == protein


# ---------------------------------------------------------------------------
# remove_homopolymers
# ---------------------------------------------------------------------------

class TestRemoveHomopolymers:
    def test_breaks_long_runs(self):
        import re
        # AAA AAA AAA = K K K (9 consecutive A's)
        dna = "AAAAAAAAA"
        result = remove_homopolymers(dna, max_run=6)
        # AAA->AAG substitution should break the run (K is AAA or AAG)
        assert not re.search(r"(.)\1{6,}", result)
        # Protein should still be KKK
        assert translate_dna(result) == "KKK"

    def test_preserves_protein(self):
        # K K K = AAA AAA AAA = 9 A's
        dna = "AAAAAAAAA"  # KKK
        result = remove_homopolymers(dna, max_run=6)
        assert translate_dna(result) == translate_dna(dna)

    def test_no_change_when_no_homopolymers(self):
        dna = "ATGAAAGCT"
        result = remove_homopolymers(dna, max_run=6)
        assert result == dna

    def test_six_is_allowed(self):
        # 6 consecutive A's should be fine with max_run=6
        dna = "ATGAAAAAAGCT"  # 6 A's
        result = remove_homopolymers(dna, max_run=6)
        assert result == dna


# ---------------------------------------------------------------------------
# check_poly_a_signals
# ---------------------------------------------------------------------------

class TestCheckPolyASignals:
    def test_finds_aataaa(self):
        dna = "ATGAATAAAGCT"
        positions = check_poly_a_signals(dna)
        assert 3 in positions

    def test_no_signal(self):
        dna = "ATGGCTGCTGCT"
        assert check_poly_a_signals(dna) == []

    def test_multiple_signals(self):
        dna = "AATAAAGCTAATAAA"
        positions = check_poly_a_signals(dna)
        assert len(positions) == 2


# ---------------------------------------------------------------------------
# optimize_codon_pairs
# ---------------------------------------------------------------------------

class TestOptimizeCodonPairs:
    def test_removes_cgacga_pair(self):
        # CGA CGA = Arg Arg with inhibitory pair
        dna = "ATGCGACGAGCT"  # M R R A
        protein = translate_dna(dna)
        result = optimize_codon_pairs(dna, protein)
        # Should not contain CGACGA at codon boundaries
        has_pair = False
        for i in range(0, len(result) - 5, 3):
            if result[i:i + 6] == "CGACGA":
                has_pair = True
        assert not has_pair
        assert translate_dna(result) == protein

    def test_preserves_protein(self):
        dna = "ATGCGACCGGCT"  # Contains CGACCG
        protein = translate_dna(dna)
        result = optimize_codon_pairs(dna, protein)
        assert translate_dna(result) == protein


# ---------------------------------------------------------------------------
# optimize_sequence
# ---------------------------------------------------------------------------

class TestOptimizeSequence:
    def test_achieves_target_cai(self):
        protein = "MKTLYKERAVILWFGSTCMPNDHQ"
        dna, metrics = optimize_sequence(protein, max_iterations=10, seed=42)
        assert metrics["cai"] >= 0.5  # Should be reasonably high

    def test_achieves_target_gc(self):
        protein = "MKTLYKERAVILWFGSTCMPNDHQ"
        dna, metrics = optimize_sequence(
            protein, max_iterations=10, seed=42,
            target_gc_min=0.30, target_gc_max=0.50,
        )
        assert 0.25 <= metrics["gc_content"] <= 0.55

    def test_returns_valid_dna(self):
        protein = "MKTLYKE"
        dna, metrics = optimize_sequence(protein, seed=42)
        assert len(dna) == len(protein) * 3
        assert all(nt in "ACGT" for nt in dna)

    def test_translation_preserved(self):
        protein = "MKTLYKERAVILWFG"
        dna, _ = optimize_sequence(protein, seed=42)
        assert translate_dna(dna) == protein

    def test_metrics_keys(self):
        protein = "MKTLYKE"
        _, metrics = optimize_sequence(protein, seed=42)
        assert "cai" in metrics
        assert "gc_content" in metrics
        assert "iterations" in metrics
        assert "restriction_sites_removed" in metrics


# ---------------------------------------------------------------------------
# add_flanking_sequences
# ---------------------------------------------------------------------------

class TestAddFlankingSequences:
    def test_adds_default_flanks(self):
        dna = "ATGAAAGCT"
        result = add_flanking_sequences(dna, "enzyme_A")
        assert result.startswith("AATGGGTCTC")
        assert result.endswith("GAGACCTTAA")
        assert dna in result

    def test_custom_flanks(self):
        dna = "ATGAAAGCT"
        config = {"5prime_flank": "AAAA", "3prime_flank": "TTTT"}
        result = add_flanking_sequences(dna, "enzyme_A", config)
        assert result == "AAAA" + dna + "TTTT"

    def test_empty_flanks(self):
        dna = "ATGAAAGCT"
        config = {"5prime_flank": "", "3prime_flank": ""}
        result = add_flanking_sequences(dna, "enzyme_A", config)
        assert result == dna


# ---------------------------------------------------------------------------
# generate_plate_map
# ---------------------------------------------------------------------------

class TestGeneratePlateMap:
    def test_assigns_96_wells(self):
        sequences = {f"seq_{i}": "ATG" * 10 for i in range(96)}
        plate_map = generate_plate_map(sequences, "enzyme_A")
        assert len(plate_map) == 96
        wells = [row["well_position"] for row in plate_map]
        # First well should be A1
        assert wells[0] == "A1"
        # Last well should be H12
        assert wells[-1] == "H12"

    def test_well_positions_unique(self):
        sequences = {f"seq_{i}": "ATG" * 10 for i in range(96)}
        plate_map = generate_plate_map(sequences, "enzyme_A")
        wells = [row["well_position"] for row in plate_map]
        assert len(set(wells)) == 96

    def test_fewer_than_96(self):
        sequences = {f"seq_{i}": "ATG" * 10 for i in range(10)}
        plate_map = generate_plate_map(sequences, "enzyme_A")
        assert len(plate_map) == 10
        assert plate_map[0]["well_position"] == "A1"

    def test_more_than_96(self):
        sequences = {f"seq_{i}": "ATG" * 10 for i in range(100)}
        plate_map = generate_plate_map(sequences, "enzyme_A")
        assert len(plate_map) == 100
        # 97th sequence should be on plate 2, well A1
        assert plate_map[96]["plate_number"] == 2
        assert plate_map[96]["well_position"] == "A1"

    def test_plate_map_fields(self):
        sequences = {"seq_1": "ATGAAAGCT"}
        plate_map = generate_plate_map(sequences, "enzyme_A")
        row = plate_map[0]
        assert row["enzyme_id"] == "enzyme_A"
        assert row["sequence_id"] == "seq_1"
        assert row["well_position"] == "A1"
        assert row["dna_length_nt"] == 9
        assert row["protein_length_aa"] == 3

    def test_well_ordering(self):
        """Wells go A1, A2, ..., A12, B1, B2, ..., H12."""
        sequences = {f"seq_{i}": "ATG" for i in range(24)}
        plate_map = generate_plate_map(sequences, "enzyme_A")
        wells = [row["well_position"] for row in plate_map]
        # First 12 should be A1-A12
        assert wells[:12] == [f"A{c}" for c in range(1, 13)]
        # Next 12 should be B1-B12
        assert wells[12:24] == [f"B{c}" for c in range(1, 13)]


# ---------------------------------------------------------------------------
# run_stage (integration test)
# ---------------------------------------------------------------------------

class TestRunStage:
    def test_run_stage_produces_outputs(self, tmp_path):
        config = _make_stage8_config(tmp_path, ["enzyme_A"])
        results = run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        dna_dir = base_dir / "stage_08_codon_opt" / "dna_sequences"
        plate_dir = base_dir / "stage_08_codon_opt" / "plate_maps"

        # Check DNA FASTA exists
        fasta_file = dna_dir / "enzyme_A_codon_opt.fasta"
        assert fasta_file.exists()

        # Check plate map exists
        plate_file = plate_dir / "enzyme_A_plate_map.csv"
        assert plate_file.exists()

        # Check combined plate map
        combined = plate_dir / "all_plate_maps.csv"
        assert combined.exists()

        # Check summary
        assert "enzyme_A" in results["summary"]
        assert results["summary"]["enzyme_A"]["n_sequences"] == 2
        assert results["total_sequences"] == 2

    def test_run_stage_multiple_enzymes(self, tmp_path):
        config = _make_stage8_config(tmp_path, ["enzyme_A", "enzyme_B"])
        results = run_stage(config)
        assert "enzyme_A" in results["summary"]
        assert "enzyme_B" in results["summary"]
        assert results["total_sequences"] == 4

    def test_run_stage_missing_input_skips(self, tmp_path):
        config = _make_stage8_config(tmp_path, ["enzyme_A"])
        # Add an enzyme with no input FASTA
        config["enzymes"]["enzyme_Z"] = {
            "name": "Enzyme Z",
            "length": 50,
            "sequence_file": "dummy.fasta",
            "substrate_smiles": "C",
            "product_smiles": "C",
            "fixed_residues": [],
            "hotspot_residues": [],
            "contigs": "A1-50",
            "cofactors": [],
        }
        results = run_stage(config)
        assert "enzyme_Z" not in results["summary"]


# ---------------------------------------------------------------------------
# Edge cases / helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_reverse_complement(self):
        assert _reverse_complement("ATGC") == "GCAT"
        assert _reverse_complement("GGTCTC") == "GAGACC"

    def test_translate_dna(self):
        assert translate_dna("ATGAAAGCT") == "MKA"
        assert translate_dna("ATG") == "M"

    def test_translate_stops_at_stop_codon(self):
        assert translate_dna("ATGTAAATG") == "M"
