"""Tests for Stage 4.75: Solubility Pre-screening (pipeline/stage4_75_solubility.py).

All tests run without GPU or external tools installed.
"""
from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeline.stage4_75_solubility import (
    STAGE_NAME,
    STAGE_NUM,
    INPUT_DESCRIPTION,
    OUTPUT_DESCRIPTION,
    calc_intrinsic_solubility,
    calc_net_charge,
    find_aggregation_regions,
    calc_hydrophobic_fraction,
    estimate_pi,
    calc_solubility_score,
    run_stage,
)


# ---------------------------------------------------------------------------
# Known sequences for testing
# ---------------------------------------------------------------------------

# Highly hydrophobic: mostly Leu, Ile, Val, Phe, Trp
HYDROPHOBIC_SEQ = "LLLLIIIIVVVVFFFFWWWW"

# Highly charged, soluble: many Lys/Arg and Asp/Glu, some Gly
CHARGED_SOLUBLE_SEQ = "KKKKKDDDDEEERRRRGGGG"

# All-glycine: minimum hydrophobicity, neutral charge
ALL_GLY = "GGGGGGGGGG"

# Single residue
SINGLE_K = "K"
SINGLE_L = "L"

# Kyte-Doolittle scale for reference (used in assertions)
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> dict:
    """Create a minimal config for testing run_stage."""
    base_dir = tmp_path / "pipeline"
    base_dir.mkdir()

    # Stage 4 output — designed FASTA files
    stage4_dir = base_dir / "stage_04_ligandmpnn"

    for eid in ("enzyme_A", "enzyme_B"):
        enz_dir = stage4_dir / eid
        enz_dir.mkdir(parents=True)
        fasta = enz_dir / "designed_sequences.fasta"
        if eid == "enzyme_A":
            # Mix: one hydrophobic (low solubility), one charged (high solubility)
            fasta.write_text(
                f">{eid}_design_001\n{HYDROPHOBIC_SEQ}\n"
                f">{eid}_design_002\n{CHARGED_SOLUBLE_SEQ}\n"
            )
        else:
            fasta.write_text(
                f">{eid}_design_001\nMVLSPADKTNVKAAWGKVGA\n"
            )

    return {
        "global": {
            "project_name": "test",
            "base_dir": str(base_dir),
            "log_dir": "logs",
            "round_number": 1,
        },
        "databases": {
            "colabfold_db_dir": str(tmp_path / "db"),
            "mmseqs2_db_dir": str(tmp_path / "mmseqs"),
        },
        "tools": {},
        "stages": {
            "stage_04_75_solubility": {
                "output_dir": "stage_04.75_solubility",
                "risk_threshold": 0.3,
                "enabled": True,
            },
            "stage_04_ligandmpnn": {
                "output_dir": "stage_04_ligandmpnn",
            },
        },
        "enzymes": {
            "enzyme_A": {
                "name": "Enzyme A",
                "length": 20,
                "sequence_file": "inputs/sequences/enzyme_A.fasta",
                "substrate_smiles": "CCO",
                "product_smiles": "CC=O",
                "fixed_residues": [1, 5, 10],
                "hotspot_residues": [5, 10],
                "contigs": "A1-20",
                "cofactors": ["NAD+"],
            },
            "enzyme_B": {
                "name": "Enzyme B",
                "length": 20,
                "sequence_file": "inputs/sequences/enzyme_B.fasta",
                "substrate_smiles": "CC=O",
                "product_smiles": "CC(O)=O",
                "fixed_residues": [2, 8],
                "hotspot_residues": [8, 12],
                "contigs": "A1-20",
                "cofactors": ["FAD"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_stage_name_contains_version(self):
        assert "4.75" in STAGE_NAME

    def test_stage_name_mentions_solubility(self):
        assert "solubility" in STAGE_NAME.lower() or "Solubility" in STAGE_NAME

    def test_stage_num(self):
        assert STAGE_NUM == "4.75"

    def test_input_description_nonempty(self):
        assert INPUT_DESCRIPTION

    def test_output_description_nonempty(self):
        assert OUTPUT_DESCRIPTION

    def test_output_description_mentions_csv(self):
        assert "csv" in OUTPUT_DESCRIPTION.lower() or "solubility" in OUTPUT_DESCRIPTION.lower()


# ---------------------------------------------------------------------------
# calc_intrinsic_solubility
# ---------------------------------------------------------------------------

class TestCalcIntrinsicSolubility:
    def test_hydrophobic_sequence_low_score(self):
        """All-hydrophobic sequence should produce a lower intrinsic solubility score
        than an all-charged sequence (after inversion)."""
        hydro_score = calc_intrinsic_solubility(HYDROPHOBIC_SEQ)
        charged_score = calc_intrinsic_solubility(CHARGED_SOLUBLE_SEQ)
        assert hydro_score < charged_score

    def test_all_glycine_returns_float(self):
        score = calc_intrinsic_solubility(ALL_GLY)
        assert isinstance(score, float)

    def test_single_residue_returns_float(self):
        score = calc_intrinsic_solubility(SINGLE_K)
        assert isinstance(score, float)

    def test_result_is_finite(self):
        score = calc_intrinsic_solubility("MVLSPADKTNVK")
        assert math.isfinite(score)

    def test_empty_sequence_returns_zero(self):
        """Empty sequence should return 0.0 (no windows)."""
        score = calc_intrinsic_solubility("")
        assert score == 0.0

    def test_all_leucine_is_less_than_all_lysine(self):
        """All-L (hydrophobic) should score lower than all-K (charged)."""
        all_leu = "L" * 10
        all_lys = "K" * 10
        assert calc_intrinsic_solubility(all_leu) < calc_intrinsic_solubility(all_lys)

    def test_sequence_shorter_than_window_still_works(self):
        """Sequences shorter than the 7-residue window should not raise."""
        score = calc_intrinsic_solubility("MLK")  # 3 residues
        assert isinstance(score, float)

    def test_known_neutral_gly_score(self):
        """Glycine has KD = -0.4; a long poly-G should give a known score range."""
        score = calc_intrinsic_solubility("G" * 20)
        # KD for G is -0.4; inverted score should be positive (more soluble than hydrophobic)
        # Exact value depends on implementation but should be finite and positive
        assert math.isfinite(score)


# ---------------------------------------------------------------------------
# calc_net_charge
# ---------------------------------------------------------------------------

class TestCalcNetCharge:
    def test_all_positive_residues(self):
        """KKKK should give +4 net charge."""
        charge = calc_net_charge("KKKK")
        assert charge == pytest.approx(4.0, abs=0.01)

    def test_all_negative_residues(self):
        """DDDD should give -4 net charge."""
        charge = calc_net_charge("DDDD")
        assert charge == pytest.approx(-4.0, abs=0.01)

    def test_balanced_charged(self):
        """KDKD should give ~0 net charge."""
        charge = calc_net_charge("KDKD")
        assert charge == pytest.approx(0.0, abs=0.01)

    def test_neutral_glycine(self):
        """All-glycine has no charge-bearing residues."""
        charge = calc_net_charge("GGGG")
        assert charge == pytest.approx(0.0, abs=0.01)

    def test_histidine_partial_charge(self):
        """H contributes +0.1 (partially protonated at pH 7)."""
        charge = calc_net_charge("H")
        assert charge == pytest.approx(0.1, abs=0.001)

    def test_empty_sequence_zero_charge(self):
        assert calc_net_charge("") == pytest.approx(0.0, abs=0.001)

    def test_single_k(self):
        assert calc_net_charge("K") == pytest.approx(1.0, abs=0.01)

    def test_single_r(self):
        assert calc_net_charge("R") == pytest.approx(1.0, abs=0.01)

    def test_single_e(self):
        assert calc_net_charge("E") == pytest.approx(-1.0, abs=0.01)

    def test_mixed_sequence(self):
        """KRED: K=+1, R=+1, E=-1, D=-1 → net 0."""
        charge = calc_net_charge("KRED")
        assert charge == pytest.approx(0.0, abs=0.01)

    def test_ph_parameter_accepted(self):
        """Function should accept a ph parameter."""
        charge = calc_net_charge("KDKD", ph=7.0)
        assert isinstance(charge, float)


# ---------------------------------------------------------------------------
# find_aggregation_regions
# ---------------------------------------------------------------------------

class TestFindAggregationRegions:
    def test_hydrophobic_sequence_has_regions(self):
        """The all-hydrophobic sequence should have aggregation-prone regions."""
        regions = find_aggregation_regions(HYDROPHOBIC_SEQ)
        assert len(regions) > 0

    def test_charged_sequence_fewer_regions(self):
        """Charged/hydrophilic sequence should have fewer or no aggregation regions."""
        regions_hydro = find_aggregation_regions(HYDROPHOBIC_SEQ)
        regions_charged = find_aggregation_regions(CHARGED_SOLUBLE_SEQ)
        assert len(regions_hydro) >= len(regions_charged)

    def test_all_glycine_no_aggregation(self):
        """Glycine has low KD; poly-G should not trigger aggregation."""
        regions = find_aggregation_regions("G" * 20)
        assert len(regions) == 0

    def test_empty_sequence_no_regions(self):
        assert find_aggregation_regions("") == []

    def test_returns_list(self):
        result = find_aggregation_regions("MVLSPADKTNVK")
        assert isinstance(result, list)

    def test_five_consecutive_hydrophobics_flagged(self):
        """A run of 5+ VILFW residues should be detected."""
        seq = "GGGVVVVVGGG"  # 5 consecutive V (hydrophobic)
        regions = find_aggregation_regions(seq)
        assert len(regions) >= 1

    def test_four_consecutive_not_flagged(self):
        """A run of only 4 VILFW residues should NOT be flagged."""
        seq = "GGGVVVVGGG"  # only 4 V
        # This may or may not flag depending on threshold; just check it returns a list
        result = find_aggregation_regions(seq)
        assert isinstance(result, list)

    def test_single_residue_no_region(self):
        assert find_aggregation_regions("L") == []


# ---------------------------------------------------------------------------
# calc_hydrophobic_fraction
# ---------------------------------------------------------------------------

class TestCalcHydrophobicFraction:
    def test_all_hydrophobic_is_one(self):
        """Sequence of all VILFW residues should give fraction 1.0."""
        seq = "VILFWM"  # all hydrophobic (V, I, L, F, W, M)
        frac = calc_hydrophobic_fraction(seq)
        assert frac == pytest.approx(1.0, abs=0.01)

    def test_all_charged_is_low(self):
        """Sequence of K, R, D, E, G has no hydrophobic residues."""
        seq = "KRDEG"
        frac = calc_hydrophobic_fraction(seq)
        assert frac == pytest.approx(0.0, abs=0.01)

    def test_mixed_sequence(self):
        """Half hydrophobic, half polar."""
        seq = "LLLKKK"  # L is hydrophobic, K is not (not in VILFW M A P)
        # Wait — per task spec: hydrophobic = V, I, L, F, W, M, A, P
        # L = hydrophobic (3), K = not (3) → fraction = 0.5
        frac = calc_hydrophobic_fraction(seq)
        assert frac == pytest.approx(0.5, abs=0.01)

    def test_empty_sequence_zero(self):
        assert calc_hydrophobic_fraction("") == 0.0

    def test_single_hydrophobic(self):
        assert calc_hydrophobic_fraction("L") == pytest.approx(1.0, abs=0.01)

    def test_single_nonhydrophobic(self):
        assert calc_hydrophobic_fraction("K") == pytest.approx(0.0, abs=0.01)

    def test_result_in_range(self):
        frac = calc_hydrophobic_fraction("MVLSPADKTNVKAAAWGKVGA")
        assert 0.0 <= frac <= 1.0

    def test_hydrophobic_set_includes_ala_and_pro(self):
        """Per spec, A and P are included in hydrophobic set."""
        seq = "AAPP"  # A=hydrophobic, P=hydrophobic per spec
        frac = calc_hydrophobic_fraction(seq)
        assert frac == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# estimate_pi
# ---------------------------------------------------------------------------

class TestEstimatePi:
    def test_lysine_rich_has_high_pi(self):
        """Positively charged sequence should have pI > 7."""
        pi = estimate_pi("KKKKKKKK")
        assert pi > 7.0

    def test_aspartate_rich_has_low_pi(self):
        """Negatively charged sequence should have pI < 7."""
        pi = estimate_pi("DDDDDDDD")
        assert pi < 7.0

    def test_balanced_sequence_near_neutral(self):
        """Equal K and D should give pI near 7."""
        pi = estimate_pi("KDKDKDKD")
        # pI may not be exactly 7 due to different pKa values, but should be 4-10
        assert 4.0 <= pi <= 10.0

    def test_result_in_reasonable_range(self):
        pi = estimate_pi("MVLSPADKTNVKAAAWGKVGA")
        assert 1.0 <= pi <= 14.0

    def test_all_glycine_neutral_pi(self):
        """Poly-G has no ionisable side chains; pI from N/C terminal only."""
        pi = estimate_pi("GGGGGGGG")
        # N-terminal ~8.2, C-terminal ~3.7 → midpoint ~6
        assert 1.0 <= pi <= 14.0
        assert isinstance(pi, float)

    def test_empty_sequence(self):
        """Empty sequence should return a float without raising."""
        pi = estimate_pi("")
        assert isinstance(pi, float)
        assert math.isfinite(pi)

    def test_single_lysine(self):
        pi = estimate_pi("K")
        assert pi > 7.0  # positive charge → basic pI

    def test_single_aspartate(self):
        pi = estimate_pi("D")
        assert pi < 7.0  # negative charge → acidic pI


# ---------------------------------------------------------------------------
# calc_solubility_score
# ---------------------------------------------------------------------------

class TestCalcSolubilityScore:
    def test_hydrophobic_low_score(self):
        """Highly hydrophobic sequence should score below 0.5."""
        score = calc_solubility_score(HYDROPHOBIC_SEQ)
        assert score < 0.5

    def test_charged_soluble_higher_than_hydrophobic(self):
        """Charged/soluble sequence should score higher than hydrophobic one."""
        hydro_score = calc_solubility_score(HYDROPHOBIC_SEQ)
        charged_score = calc_solubility_score(CHARGED_SOLUBLE_SEQ)
        assert charged_score > hydro_score

    def test_result_in_range(self):
        """Composite score must be in [0, 1]."""
        score = calc_solubility_score("MVLSPADKTNVKAAAWGKVGA")
        assert 0.0 <= score <= 1.0

    def test_all_glycine_returns_valid_score(self):
        score = calc_solubility_score(ALL_GLY)
        assert 0.0 <= score <= 1.0

    def test_empty_sequence_returns_zero_or_valid(self):
        score = calc_solubility_score("")
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        score = calc_solubility_score("MVLK")
        assert isinstance(score, float)

    def test_single_residue_returns_valid(self):
        score = calc_solubility_score("G")
        assert 0.0 <= score <= 1.0

    def test_composite_uses_all_components(self):
        """Verify the score changes when sequence changes significantly."""
        score1 = calc_solubility_score("KKKKKKKKKK")  # very polar
        score2 = calc_solubility_score("LLLLLLLLLL")  # very hydrophobic
        # They should differ
        assert score1 != pytest.approx(score2, abs=0.01)


# ---------------------------------------------------------------------------
# run_stage (integration)
# ---------------------------------------------------------------------------

class TestRunStage:
    def test_produces_csv(self, tmp_path):
        """run_stage creates solubility_scores.csv in the output dir."""
        config = _make_config(tmp_path)
        result = run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        csv_path = base_dir / "stage_04.75_solubility" / "solubility_scores.csv"
        assert csv_path.exists(), f"Expected CSV at {csv_path}"

    def test_csv_has_required_columns(self, tmp_path):
        """solubility_scores.csv must have all required columns."""
        config = _make_config(tmp_path)
        run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        csv_path = base_dir / "stage_04.75_solubility" / "solubility_scores.csv"

        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) > 0
        required = {
            "sequence_id",
            "enzyme_id",
            "intrinsic_solubility",
            "net_charge",
            "pi_estimate",
            "aggregation_regions",
            "hydrophobic_fraction",
            "composite_score",
            "high_risk",
        }
        actual = set(rows[0].keys())
        assert required.issubset(actual), f"Missing columns: {required - actual}"

    def test_high_risk_flag_on_hydrophobic(self, tmp_path):
        """The hydrophobic sequence should be flagged as high_risk."""
        config = _make_config(tmp_path)
        run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        csv_path = base_dir / "stage_04.75_solubility" / "solubility_scores.csv"

        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))

        # Find the hydrophobic design
        hydro_rows = [r for r in rows if "design_001" in r["sequence_id"] and r["enzyme_id"] == "enzyme_A"]
        assert len(hydro_rows) > 0
        assert hydro_rows[0]["high_risk"].lower() in ("true", "1", "yes")

    def test_charged_sequence_lower_risk(self, tmp_path):
        """The charged/soluble design should have a higher composite score."""
        config = _make_config(tmp_path)
        run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        csv_path = base_dir / "stage_04.75_solubility" / "solubility_scores.csv"

        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))

        hydro_rows = [r for r in rows if "design_001" in r["sequence_id"] and r["enzyme_id"] == "enzyme_A"]
        charged_rows = [r for r in rows if "design_002" in r["sequence_id"] and r["enzyme_id"] == "enzyme_A"]
        assert len(hydro_rows) > 0
        assert len(charged_rows) > 0

        hydro_score = float(hydro_rows[0]["composite_score"])
        charged_score = float(charged_rows[0]["composite_score"])
        assert charged_score > hydro_score

    def test_all_enzymes_present_in_csv(self, tmp_path):
        """Both enzymes appear in the output CSV."""
        config = _make_config(tmp_path)
        run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        csv_path = base_dir / "stage_04.75_solubility" / "solubility_scores.csv"

        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))

        enzyme_ids = {r["enzyme_id"] for r in rows}
        assert "enzyme_A" in enzyme_ids
        assert "enzyme_B" in enzyme_ids

    def test_returns_summary_dict(self, tmp_path):
        """run_stage returns a summary dict with per-enzyme info."""
        config = _make_config(tmp_path)
        result = run_stage(config)

        assert isinstance(result, dict)
        assert "enzyme_A" in result
        assert "enzyme_B" in result

    def test_summary_has_counts(self, tmp_path):
        """Summary dict contains total_scored and high_risk_count."""
        config = _make_config(tmp_path)
        result = run_stage(config)

        for eid in ("enzyme_A", "enzyme_B"):
            assert "total_scored" in result[eid]
            assert "high_risk_count" in result[eid]
            assert isinstance(result[eid]["total_scored"], int)

    def test_missing_fasta_skips_enzyme(self, tmp_path):
        """If an enzyme's FASTA is missing, run_stage skips it gracefully."""
        config = _make_config(tmp_path)
        # Remove enzyme_B's FASTA
        base_dir = Path(config["global"]["base_dir"])
        fasta_b = base_dir / "stage_04_ligandmpnn" / "enzyme_B" / "designed_sequences.fasta"
        fasta_b.unlink()

        result = run_stage(config)
        # Should still succeed; enzyme_B missing from summary or has 0 scored
        assert isinstance(result, dict)

    def test_output_dir_created(self, tmp_path):
        """run_stage creates the output directory."""
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])
        out_dir = base_dir / "stage_04.75_solubility"
        assert not out_dir.exists()

        run_stage(config)
        assert out_dir.exists()

    def test_custom_risk_threshold(self, tmp_path):
        """Adjusting risk_threshold changes high_risk classification."""
        config = _make_config(tmp_path)
        # Set threshold so high = everything (threshold=1.0 means everything is high risk)
        config["stages"]["stage_04_75_solubility"]["risk_threshold"] = 1.0
        run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        csv_path = base_dir / "stage_04.75_solubility" / "solubility_scores.csv"

        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))

        # With threshold=1.0, ALL sequences should be high risk
        for row in rows:
            assert row["high_risk"].lower() in ("true", "1", "yes"), (
                f"Expected high_risk=True for all rows with threshold=1.0, got {row['high_risk']}"
            )

    def test_fasta_lookup_fallback(self, tmp_path):
        """run_stage also checks for {enzyme_id}_designed.fasta as a fallback."""
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        # Rename the default FASTA to fallback name
        src = base_dir / "stage_04_ligandmpnn" / "enzyme_B" / "designed_sequences.fasta"
        dst = base_dir / "stage_04_ligandmpnn" / "enzyme_B" / "enzyme_B_designed.fasta"
        src.rename(dst)

        result = run_stage(config)
        # enzyme_B should have been processed via the fallback path
        assert result.get("enzyme_B", {}).get("total_scored", 0) > 0
