"""Tests for Stage 7.5: N-glycosylation Screening (pipeline/stage7_5_glycosylation.py).

All tests run without ESMFold, GPU, or BioPython's PDB parser installed.
"""
from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeline.stage7_5_glycosylation import (
    STAGE_NAME,
    STAGE_NUM,
    INPUT_DESCRIPTION,
    OUTPUT_DESCRIPTION,
    find_glycosylation_sequons,
    get_residue_coords,
    calc_min_distance_to_active_site,
    classify_risk,
    generate_alternatives,
    score_with_esmfold,
    run_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_minimal_pdb(path: Path, residues: dict[int, tuple]) -> None:
    """Write a minimal PDB with CA atoms at specified positions.

    Args:
        path: Output PDB path.
        residues: {residue_number: (x, y, z)} for CA atoms.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for i, (resnum, (x, y, z)) in enumerate(sorted(residues.items()), start=1):
            fh.write(
                f"ATOM  {i:5d}  CA  ALA A{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
            )
        fh.write("END\n")


def _make_config(tmp_path: Path) -> dict:
    """Create a minimal config for testing run_stage."""
    base_dir = tmp_path / "pipeline"
    base_dir.mkdir()

    # Create Stage 7 output with ranked FASTA
    stage7_dir = base_dir / "stage_07_crossval_ranking"

    for eid in ("enzyme_A", "enzyme_B"):
        enz_dir = stage7_dir / eid
        enz_dir.mkdir(parents=True)

        # Ranked FASTA — enzyme_A has a sequon, enzyme_B does not
        if eid == "enzyme_A":
            # NAS is a valid sequon (N-A-S); NPS is NOT (P at middle)
            fasta_content = (
                f">{eid}_design_001\nMVLSNASKTNVKNPSGKVGA\n"
                f">{eid}_design_002\nMVLSPADKTNVKAAWGKVGA\n"
            )
        else:
            fasta_content = (
                f">{eid}_design_001\nMVLSPADKTNVKAAWGKVGA\n"
            )
        fasta = enz_dir / f"{eid}_ranked.fasta"
        fasta.write_text(fasta_content)

        # PDB structures
        struct_dir = enz_dir / "structures"
        struct_dir.mkdir()

        # For enzyme_A design_001: put sequon Asn at pos 5 near hotspot at pos 10
        if eid == "enzyme_A":
            residues = {}
            for r in range(1, 21):
                residues[r] = (float(r), 0.0, 0.0)
            _write_minimal_pdb(
                struct_dir / f"{eid}_design_001.pdb", residues,
            )
            _write_minimal_pdb(
                struct_dir / f"{eid}_design_002.pdb", residues,
            )
        else:
            residues = {r: (float(r), 0.0, 0.0) for r in range(1, 21)}
            _write_minimal_pdb(
                struct_dir / f"{eid}_design_001.pdb", residues,
            )

    # WT structures
    wt_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
    wt_dir.mkdir(parents=True)
    for eid in ("enzyme_A", "enzyme_B"):
        residues = {r: (float(r), 0.0, 0.0) for r in range(1, 21)}
        _write_minimal_pdb(wt_dir / f"{eid}_substrate.pdb", residues)

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
            "stage_07_crossval": {
                "output_dir": "stage_07_crossval_ranking",
            },
            "stage_07_5_glycosylation": {
                "output_dir": "stage_07.5_glycosylation",
                "distance_threshold": 10.0,
            },
            "stage_02_wt_structures": {
                "output_dir": "stage_02_wt_structures",
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
                "hotspot_residues": [10, 15],
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
    def test_stage_name(self):
        assert "7.5" in STAGE_NAME
        assert "glycosylation" in STAGE_NAME.lower()

    def test_stage_num(self):
        assert STAGE_NUM == "7.5"

    def test_input_description(self):
        assert INPUT_DESCRIPTION

    def test_output_description(self):
        assert "glycosylation" in OUTPUT_DESCRIPTION.lower()


# ---------------------------------------------------------------------------
# find_glycosylation_sequons
# ---------------------------------------------------------------------------

class TestFindGlycosylationSequons:
    def test_single_sequon_nas(self):
        """NAS is a valid N-glycosylation sequon."""
        result = find_glycosylation_sequons("MVLNASKTG")
        assert len(result) == 1
        assert result[0]["position"] == 4  # 1-based
        assert result[0]["sequon"] == "NAS"
        assert result[0]["asn_pos"] == 4
        assert result[0]["xaa_pos"] == 5
        assert result[0]["st_pos"] == 6

    def test_sequon_nxt(self):
        """NXT (X=any non-Pro, T at end) is valid."""
        result = find_glycosylation_sequons("ANGTM")
        assert len(result) == 1
        assert result[0]["sequon"] == "NGT"

    def test_excludes_nps(self):
        """NPS should NOT be matched (Pro at middle position)."""
        result = find_glycosylation_sequons("MVLNPSKTG")
        assert len(result) == 0

    def test_excludes_npt(self):
        """NPT should NOT be matched (Pro at middle position)."""
        result = find_glycosylation_sequons("ANPTM")
        assert len(result) == 0

    def test_multiple_sequons(self):
        """Multiple non-overlapping sequons are found."""
        # NAS at pos 1, NGT at pos 7
        result = find_glycosylation_sequons("NASXXXXNGT")
        assert len(result) == 2
        assert result[0]["position"] == 1
        assert result[0]["sequon"] == "NAS"
        assert result[1]["position"] == 8
        assert result[1]["sequon"] == "NGT"

    def test_no_sequons(self):
        """Sequence with no sequons returns empty list."""
        result = find_glycosylation_sequons("MVLSPADKTNVKAAWGKVGA")
        assert result == []

    def test_empty_sequence(self):
        result = find_glycosylation_sequons("")
        assert result == []

    def test_sequon_at_start(self):
        result = find_glycosylation_sequons("NASTM")
        assert len(result) == 1
        assert result[0]["position"] == 1

    def test_sequon_at_end(self):
        result = find_glycosylation_sequons("MVNAS")
        assert len(result) == 1
        assert result[0]["position"] == 3

    def test_positions_are_1_based(self):
        """Verify all positions are 1-based, not 0-based."""
        result = find_glycosylation_sequons("NAS")
        assert result[0]["asn_pos"] == 1
        assert result[0]["xaa_pos"] == 2
        assert result[0]["st_pos"] == 3


# ---------------------------------------------------------------------------
# classify_risk
# ---------------------------------------------------------------------------

class TestClassifyRisk:
    def test_high_risk_close_to_active_site(self):
        assert classify_risk(5, 8.0) == "high_risk"

    def test_low_risk_far_from_active_site(self):
        assert classify_risk(5, 15.0) == "low_risk"

    def test_exactly_at_threshold_is_low_risk(self):
        """Distance == threshold should be low_risk (strict less-than)."""
        assert classify_risk(5, 10.0) == "low_risk"

    def test_just_below_threshold_is_high_risk(self):
        assert classify_risk(5, 9.99) == "high_risk"

    def test_custom_threshold(self):
        assert classify_risk(5, 4.0, distance_threshold=5.0) == "high_risk"
        assert classify_risk(5, 6.0, distance_threshold=5.0) == "low_risk"

    def test_infinity_is_low_risk(self):
        assert classify_risk(5, float("inf")) == "low_risk"

    def test_zero_distance_is_high_risk(self):
        assert classify_risk(5, 0.0) == "high_risk"


# ---------------------------------------------------------------------------
# generate_alternatives
# ---------------------------------------------------------------------------

class TestGenerateAlternatives:
    def test_asn_to_gln_mutation(self):
        """Asn->Gln correctly replaces N with Q."""
        sequence = "MVLNASKTG"
        sequon = find_glycosylation_sequons(sequence)[0]
        alts = generate_alternatives(sequence, sequon, strategy="asn_to_gln")
        assert len(alts) == 1
        assert alts[0]["modified_sequence"] == "MVLQASKTG"
        assert alts[0]["mutation"] == "N4Q"
        assert alts[0]["strategy"] == "asn_to_gln"

    def test_st_to_ala_mutation_ser(self):
        """Ser->Ala correctly replaces S with A at +2 position."""
        sequence = "MVLNASKTG"
        sequon = find_glycosylation_sequons(sequence)[0]
        alts = generate_alternatives(sequence, sequon, strategy="st_to_ala")
        assert len(alts) == 1
        assert alts[0]["modified_sequence"] == "MVLNAAKTG"
        assert alts[0]["mutation"] == "S6A"
        assert alts[0]["strategy"] == "st_to_ala"

    def test_st_to_ala_mutation_thr(self):
        """Thr->Ala correctly replaces T with A."""
        sequence = "ANGTM"
        sequon = find_glycosylation_sequons(sequence)[0]
        alts = generate_alternatives(sequence, sequon, strategy="st_to_ala")
        assert len(alts) == 1
        assert alts[0]["modified_sequence"] == "ANGAM"
        assert alts[0]["mutation"] == "T4A"

    def test_alternative_abolishes_sequon(self):
        """The alternative sequence should no longer contain the sequon."""
        sequence = "MVLNASKTG"
        sequon = find_glycosylation_sequons(sequence)[0]

        for strategy in ("asn_to_gln", "st_to_ala"):
            alts = generate_alternatives(sequence, sequon, strategy=strategy)
            modified = alts[0]["modified_sequence"]
            remaining_sequons = find_glycosylation_sequons(modified)
            assert len(remaining_sequons) == 0, (
                f"Strategy {strategy} did not abolish sequon: {modified}"
            )

    def test_sequence_length_preserved(self):
        """Alternative sequence has same length as original."""
        sequence = "MVLNASKTG"
        sequon = find_glycosylation_sequons(sequence)[0]
        for strategy in ("asn_to_gln", "st_to_ala"):
            alts = generate_alternatives(sequence, sequon, strategy=strategy)
            assert len(alts[0]["modified_sequence"]) == len(sequence)


# ---------------------------------------------------------------------------
# calc_min_distance_to_active_site
# ---------------------------------------------------------------------------

class TestCalcMinDistanceToActiveSite:
    def test_known_distances(self, tmp_path):
        """Test with known coordinates — residues at (5,0,0) and (10,0,0)."""
        pdb_path = tmp_path / "test.pdb"
        residues = {
            5: (5.0, 0.0, 0.0),   # sequon Asn position
            10: (10.0, 0.0, 0.0),  # hotspot
            15: (15.0, 0.0, 0.0),  # hotspot
        }
        _write_minimal_pdb(pdb_path, residues)

        result = calc_min_distance_to_active_site(
            sequon_positions=[5],
            hotspot_residues=[10, 15],
            pdb_path=pdb_path,
        )
        assert result[5] == pytest.approx(5.0, abs=0.1)

    def test_multiple_hotspots_uses_minimum(self, tmp_path):
        """Should return the minimum distance to ANY hotspot residue."""
        pdb_path = tmp_path / "test.pdb"
        residues = {
            5: (0.0, 0.0, 0.0),
            8: (3.0, 0.0, 0.0),   # closer hotspot
            20: (50.0, 0.0, 0.0),  # far hotspot
        }
        _write_minimal_pdb(pdb_path, residues)

        result = calc_min_distance_to_active_site(
            sequon_positions=[5],
            hotspot_residues=[8, 20],
            pdb_path=pdb_path,
        )
        assert result[5] == pytest.approx(3.0, abs=0.1)

    def test_empty_hotspots_returns_inf(self, tmp_path):
        pdb_path = tmp_path / "test.pdb"
        _write_minimal_pdb(pdb_path, {5: (0.0, 0.0, 0.0)})

        result = calc_min_distance_to_active_site(
            sequon_positions=[5],
            hotspot_residues=[],
            pdb_path=pdb_path,
        )
        assert result[5] == float("inf")

    def test_missing_residue_returns_inf(self, tmp_path):
        """If residue is not in PDB, distance should be infinity."""
        pdb_path = tmp_path / "test.pdb"
        _write_minimal_pdb(pdb_path, {10: (0.0, 0.0, 0.0)})

        result = calc_min_distance_to_active_site(
            sequon_positions=[99],  # not in PDB
            hotspot_residues=[10],
            pdb_path=pdb_path,
        )
        assert result[99] == float("inf")


# ---------------------------------------------------------------------------
# score_with_esmfold (mocked)
# ---------------------------------------------------------------------------

class TestScoreWithEsmfold:
    def test_unavailable_returns_none_scores(self):
        """When ESMFold is not available, returns None pLDDT and fold_preserved."""
        with patch(
            "pipeline.stage7_5_glycosylation.score_with_esmfold",
            wraps=score_with_esmfold,
        ):
            # Force import to fail
            with patch.dict("sys.modules", {"pipeline.stage5_esmfold": None}):
                result = score_with_esmfold({"s1": "MVLSPADKTG"})

        assert "s1" in result
        assert result["s1"]["plddt"] is None
        assert result["s1"]["fold_preserved"] is None

    def test_empty_sequences(self):
        assert score_with_esmfold({}) == {}


# ---------------------------------------------------------------------------
# run_stage (integration with mocked ESMFold)
# ---------------------------------------------------------------------------

class TestRunStage:
    def test_produces_csv_with_required_columns(self, tmp_path):
        """run_stage writes glycosylation_report.csv with correct columns."""
        config = _make_config(tmp_path)

        with patch(
            "pipeline.stage7_5_glycosylation.score_with_esmfold"
        ) as mock_esmfold:
            mock_esmfold.return_value = {}
            summary = run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        report = base_dir / "stage_07.5_glycosylation" / "glycosylation_report.csv"
        assert report.exists()

        with open(report) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) > 0

        required_columns = {
            "sequence_id",
            "enzyme_id",
            "sequon_position",
            "sequon_str",
            "risk_level",
        }
        actual_columns = set(rows[0].keys())
        assert required_columns.issubset(actual_columns), (
            f"Missing columns: {required_columns - actual_columns}"
        )

    def test_high_risk_sequon_generates_alternatives(self, tmp_path):
        """High-risk sequons trigger alternative generation."""
        config = _make_config(tmp_path)

        mock_scores = {}

        def esmfold_side_effect(sequences, **kw):
            result = {}
            for sid in sequences:
                result[sid] = {"plddt": 75.0, "fold_preserved": True}
                mock_scores[sid] = 75.0
            return result

        with patch(
            "pipeline.stage7_5_glycosylation.score_with_esmfold",
            side_effect=esmfold_side_effect,
        ):
            summary = run_stage(config)

        # enzyme_A design_001 has NAS at pos 5 — with hotspot at 10,
        # distance is 5.0 A which is < 10 A threshold → high risk
        assert summary["enzyme_A"]["high_risk_sequons"] > 0
        assert summary["enzyme_A"]["alternatives_generated"] > 0

    def test_low_risk_sequon_no_alternatives(self, tmp_path):
        """Low-risk sequons are annotated but no alternatives generated."""
        config = _make_config(tmp_path)
        # Set threshold very low so everything is low-risk
        config["stages"]["stage_07_5_glycosylation"]["distance_threshold"] = 1.0

        with patch(
            "pipeline.stage7_5_glycosylation.score_with_esmfold"
        ) as mock_esmfold:
            mock_esmfold.return_value = {}
            summary = run_stage(config)

        assert summary["enzyme_A"]["high_risk_sequons"] == 0
        assert summary["enzyme_A"]["alternatives_generated"] == 0

    def test_enzyme_without_sequons(self, tmp_path):
        """Enzymes without sequons get a row with risk_level='none'."""
        config = _make_config(tmp_path)

        with patch(
            "pipeline.stage7_5_glycosylation.score_with_esmfold"
        ) as mock_esmfold:
            mock_esmfold.return_value = {}
            summary = run_stage(config)

        # enzyme_B has no sequons
        assert summary["enzyme_B"]["total_sequons"] == 0

        base_dir = Path(config["global"]["base_dir"])
        report = base_dir / "stage_07.5_glycosylation" / "glycosylation_report.csv"
        with open(report) as fh:
            rows = list(csv.DictReader(fh))

        enzyme_b_rows = [r for r in rows if r["enzyme_id"] == "enzyme_B"]
        assert len(enzyme_b_rows) > 0
        assert enzyme_b_rows[0]["risk_level"] == "none"

    def test_summary_has_all_enzymes(self, tmp_path):
        """Summary dict includes both enzymes."""
        config = _make_config(tmp_path)

        with patch(
            "pipeline.stage7_5_glycosylation.score_with_esmfold"
        ) as mock_esmfold:
            mock_esmfold.return_value = {}
            summary = run_stage(config)

        assert "enzyme_A" in summary
        assert "enzyme_B" in summary
        for eid in summary:
            assert "total_candidates" in summary[eid]
            assert "total_sequons" in summary[eid]
            assert "high_risk_sequons" in summary[eid]
            assert "alternatives_generated" in summary[eid]

    def test_alternative_fasta_written_when_fold_preserved(self, tmp_path):
        """Modified FASTA is written when alternatives pass fold check."""
        config = _make_config(tmp_path)

        def esmfold_side_effect(sequences, **kw):
            return {
                sid: {"plddt": 75.0, "fold_preserved": True}
                for sid in sequences
            }

        with patch(
            "pipeline.stage7_5_glycosylation.score_with_esmfold",
            side_effect=esmfold_side_effect,
        ):
            run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        alt_fasta = base_dir / "stage_07.5_glycosylation" / "alternative_sequences.fasta"
        assert alt_fasta.exists()
        content = alt_fasta.read_text()
        assert ">" in content  # has at least one sequence

    def test_nps_not_flagged(self, tmp_path):
        """NPS (Pro in middle) should not be detected as a sequon."""
        config = _make_config(tmp_path)

        with patch(
            "pipeline.stage7_5_glycosylation.score_with_esmfold"
        ) as mock_esmfold:
            mock_esmfold.return_value = {}
            summary = run_stage(config)

        # enzyme_A design_001 has NPS at pos 13 — this should NOT be a sequon
        # Total sequons should only count NAS at pos 5
        base_dir = Path(config["global"]["base_dir"])
        report = base_dir / "stage_07.5_glycosylation" / "glycosylation_report.csv"
        with open(report) as fh:
            rows = list(csv.DictReader(fh))

        # Check no row has NPS as a sequon
        for row in rows:
            assert row.get("sequon_str") != "NPS"
