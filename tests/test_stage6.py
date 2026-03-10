"""Tests for Stage 6: Full Boltz-2 Structural Screening (pipeline/stage6_boltz2_screen.py).

All tests run without Boltz-2 or GPU.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.stage6_boltz2_screen import (
    STAGE_NAME,
    STAGE_NUM,
    INPUT_DESCRIPTION,
    OUTPUT_DESCRIPTION,
    CSV_COLUMNS,
    run_boltz2_screen,
    parse_boltz2_confidence,
    calc_active_site_rmsd,
    calc_ligand_rmsd,
    apply_structural_filters,
    select_top_percentile,
    calculate_inhibition_ratio,
    run_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> dict:
    """Create a minimal config for testing."""
    base_dir = tmp_path / "pipeline"
    base_dir.mkdir()

    # Stage 5 output dirs with filtered FASTAs
    for eid in ("enzyme_A", "enzyme_B"):
        s5_dir = base_dir / "stage_05_esmfold_prescreen"
        s5_dir.mkdir(parents=True, exist_ok=True)
        fasta = s5_dir / f"{eid}_filtered.fasta"
        fasta.write_text(
            f">{eid}_var1\nMVLSPADKTNVKAAWGKVGA\n"
            f">{eid}_var2\nMVLSPADKTNVKAAWGKVGG\n"
            f">{eid}_var3\nMVLSPADKTNVKAAWGKVGV\n"
            f">{eid}_var4\nMVLSPADKTNVKAAWGKVGI\n"
        )

    # WT PDB stubs
    wt_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
    wt_dir.mkdir(parents=True, exist_ok=True)
    for eid in ("enzyme_A", "enzyme_B"):
        (wt_dir / f"{eid}_substrate.pdb").write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00\nEND\n"
        )

    return {
        "global": {
            "project_name": "test",
            "base_dir": str(base_dir),
            "log_dir": "logs",
            "round_number": 1,
        },
        "tools": {},
        "stages": {
            "stage_02_wt_structures": {
                "output_dir": "stage_02_wt_structures",
            },
            "stage_05_esmfold": {
                "output_dir": "stage_05_esmfold_prescreen",
            },
            "stage_06_boltz2_screening": {
                "output_dir": "stage_06_boltz2_screening",
                "ca_rmsd_max": 2.0,
                "active_site_rmsd_max": 1.5,
                "ligand_rmsd_max": 2.0,
                "recycling_steps": 10,
                "num_seeds": 1,
            },
        },
        "enzymes": {
            "enzyme_A": {
                "name": "Enzyme A",
                "length": 428,
                "sequence_file": "inputs/sequences/enzyme_A.fasta",
                "substrate_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "product_smiles": "OC1=CC=CC=C1C(=O)O",
                "fixed_residues": [100, 200, 300],
                "hotspot_residues": [98, 99, 100, 101],
                "contigs": "A1-428",
                "cofactors": ["NAD+"],
            },
            "enzyme_B": {
                "name": "Enzyme B",
                "length": 512,
                "sequence_file": "inputs/sequences/enzyme_B.fasta",
                "substrate_smiles": "C(=O)O",
                "product_smiles": "CO",
                "fixed_residues": [],
                "hotspot_residues": [],
                "contigs": "A1-512",
                "cofactors": [],
            },
        },
    }


def _create_mock_boltz_output(output_dir: Path, seq_ids: list[str]) -> None:
    """Create mock Boltz-2 output files for given sequence IDs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, sid in enumerate(seq_ids):
        pdb_path = output_dir / f"{sid}_model_0.pdb"
        pdb_path.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00\nEND\n"
        )
        conf_path = output_dir / f"{sid}_confidence.json"
        conf_data = {
            "plddt": 0.80 + i * 0.02,
            "iptm": 0.70 + i * 0.03,
            "ptm": 0.75 + i * 0.01,
            "affinity_score": -8.0 - i * 0.5,
        }
        conf_path.write_text(json.dumps(conf_data))


# ---------------------------------------------------------------------------
# Tests: module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_stage_name(self):
        assert STAGE_NAME == "Stage 6: Full Boltz-2 Structural Screening"

    def test_stage_num(self):
        assert STAGE_NUM == "6"

    def test_input_description(self):
        assert "FASTA" in INPUT_DESCRIPTION
        assert "SMILES" in INPUT_DESCRIPTION

    def test_output_description(self):
        assert "screening_results.csv" in OUTPUT_DESCRIPTION
        assert "stage_06" in OUTPUT_DESCRIPTION

    def test_csv_columns(self):
        assert "sequence_id" in CSV_COLUMNS
        assert "enzyme_id" in CSV_COLUMNS
        assert "substrate_affinity" in CSV_COLUMNS
        assert "product_affinity" in CSV_COLUMNS
        assert "inhibition_ratio" in CSV_COLUMNS
        assert "high_inhibition_risk" in CSV_COLUMNS
        assert "pdb_path" in CSV_COLUMNS


# ---------------------------------------------------------------------------
# Tests: parse_boltz2_confidence
# ---------------------------------------------------------------------------


class TestParseBoltz2Confidence:
    def test_extracts_all_fields(self, tmp_path: Path):
        conf = {
            "plddt": 0.85,
            "iptm": 0.72,
            "ptm": 0.78,
            "affinity_score": -9.2,
        }
        json_path = tmp_path / "confidence.json"
        json_path.write_text(json.dumps(conf))

        result = parse_boltz2_confidence(json_path)

        assert result["plddt"] == 0.85
        assert result["iptm"] == 0.72
        assert result["ptm"] == 0.78
        assert result["affinity_score"] == -9.2

    def test_defaults_for_missing_fields(self, tmp_path: Path):
        conf = {"plddt": 0.80}
        json_path = tmp_path / "confidence.json"
        json_path.write_text(json.dumps(conf))

        result = parse_boltz2_confidence(json_path)

        assert result["plddt"] == 0.80
        assert result["iptm"] == 0.0
        assert result["ptm"] == 0.0
        assert result["affinity_score"] == 0.0

    def test_uses_affinity_fallback_key(self, tmp_path: Path):
        """If 'affinity_score' is missing, fall back to 'affinity' key."""
        conf = {"plddt": 0.80, "affinity": -7.5}
        json_path = tmp_path / "confidence.json"
        json_path.write_text(json.dumps(conf))

        result = parse_boltz2_confidence(json_path)

        assert result["affinity_score"] == -7.5

    def test_empty_json(self, tmp_path: Path):
        json_path = tmp_path / "confidence.json"
        json_path.write_text("{}")

        result = parse_boltz2_confidence(json_path)

        assert result["plddt"] == 0.0
        assert result["affinity_score"] == 0.0


# ---------------------------------------------------------------------------
# Tests: apply_structural_filters
# ---------------------------------------------------------------------------


class TestApplyStructuralFilters:
    def test_passes_when_all_below_thresholds(self, tmp_path: Path):
        """All RMSDs below thresholds => should pass."""
        pdb_path = tmp_path / "pred.pdb"
        pdb_path.write_text("ATOM      1  CA  ALA A   1\nEND\n")
        wt_pdb = tmp_path / "wt.pdb"
        wt_pdb.write_text("ATOM      1  CA  ALA A   1\nEND\n")

        result = {"pdb_path": str(pdb_path)}

        # calc_ca_rmsd (fallback) returns 0.0 which is < 2.0
        # calc_active_site_rmsd and calc_ligand_rmsd also return 0.0 without Bio.PDB
        passed, detail = apply_structural_filters(
            result, wt_pdb, hotspot_residues=[1, 2, 3],
            ca_rmsd_max=2.0, active_site_rmsd_max=1.5, ligand_rmsd_max=2.0,
        )

        assert passed is True
        assert detail["ca_rmsd_pass"] is True
        assert detail["active_site_rmsd_pass"] is True
        assert detail["ligand_rmsd_pass"] is True

    def test_fails_when_ca_rmsd_above_threshold(self, tmp_path: Path):
        """Ca RMSD above threshold => should fail."""
        pdb_path = tmp_path / "pred.pdb"
        pdb_path.write_text("ATOM\nEND\n")
        wt_pdb = tmp_path / "wt.pdb"
        wt_pdb.write_text("ATOM\nEND\n")

        result = {"pdb_path": str(pdb_path)}

        with patch("pipeline.stage6_boltz2_screen.calc_ca_rmsd", return_value=3.5):
            passed, detail = apply_structural_filters(
                result, wt_pdb, hotspot_residues=[],
                ca_rmsd_max=2.0,
            )

        assert passed is False
        assert detail["ca_rmsd"] == 3.5
        assert detail["ca_rmsd_pass"] is False

    def test_fails_when_active_site_rmsd_above_threshold(self, tmp_path: Path):
        pdb_path = tmp_path / "pred.pdb"
        pdb_path.write_text("ATOM\nEND\n")
        wt_pdb = tmp_path / "wt.pdb"
        wt_pdb.write_text("ATOM\nEND\n")

        result = {"pdb_path": str(pdb_path)}

        with patch("pipeline.stage6_boltz2_screen.calc_ca_rmsd", return_value=1.0), \
             patch("pipeline.stage6_boltz2_screen.calc_active_site_rmsd", return_value=2.0), \
             patch("pipeline.stage6_boltz2_screen.calc_ligand_rmsd", return_value=0.5):
            passed, detail = apply_structural_filters(
                result, wt_pdb, hotspot_residues=[10, 20],
                active_site_rmsd_max=1.5,
            )

        assert passed is False
        assert detail["active_site_rmsd"] == 2.0
        assert detail["active_site_rmsd_pass"] is False

    def test_fails_when_ligand_rmsd_above_threshold(self, tmp_path: Path):
        pdb_path = tmp_path / "pred.pdb"
        pdb_path.write_text("ATOM\nEND\n")
        wt_pdb = tmp_path / "wt.pdb"
        wt_pdb.write_text("ATOM\nEND\n")

        result = {"pdb_path": str(pdb_path)}

        with patch("pipeline.stage6_boltz2_screen.calc_ca_rmsd", return_value=1.0), \
             patch("pipeline.stage6_boltz2_screen.calc_active_site_rmsd", return_value=0.5), \
             patch("pipeline.stage6_boltz2_screen.calc_ligand_rmsd", return_value=2.5):
            passed, detail = apply_structural_filters(
                result, wt_pdb, hotspot_residues=[],
                ligand_rmsd_max=2.0,
            )

        assert passed is False
        assert detail["ligand_rmsd"] == 2.5
        assert detail["ligand_rmsd_pass"] is False

    def test_fails_when_pdb_missing(self, tmp_path: Path):
        wt_pdb = tmp_path / "wt.pdb"
        wt_pdb.write_text("ATOM\nEND\n")

        result = {"pdb_path": str(tmp_path / "nonexistent.pdb")}

        passed, detail = apply_structural_filters(result, wt_pdb, hotspot_residues=[])

        assert passed is False
        assert detail["ca_rmsd"] == float("inf")

    def test_exact_threshold_fails(self, tmp_path: Path):
        """Exact threshold value should fail (strict less-than)."""
        pdb_path = tmp_path / "pred.pdb"
        pdb_path.write_text("ATOM\nEND\n")
        wt_pdb = tmp_path / "wt.pdb"
        wt_pdb.write_text("ATOM\nEND\n")

        result = {"pdb_path": str(pdb_path)}

        with patch("pipeline.stage6_boltz2_screen.calc_ca_rmsd", return_value=2.0):
            passed, detail = apply_structural_filters(
                result, wt_pdb, hotspot_residues=[],
                ca_rmsd_max=2.0,
            )

        assert passed is False
        assert detail["ca_rmsd_pass"] is False


# ---------------------------------------------------------------------------
# Tests: calculate_inhibition_ratio
# ---------------------------------------------------------------------------


class TestCalculateInhibitionRatio:
    def test_substrate_tighter_than_product(self):
        """Substrate binds tighter (more negative) => ratio > 1 (good)."""
        ratio = calculate_inhibition_ratio(-10.0, -5.0)
        assert ratio == pytest.approx(2.0)

    def test_product_tighter_than_substrate(self):
        """Product binds tighter => ratio < 1 (bad, inhibition risk)."""
        ratio = calculate_inhibition_ratio(-5.0, -10.0)
        assert ratio == pytest.approx(0.5)

    def test_equal_affinities(self):
        """Equal affinities => ratio = 1."""
        ratio = calculate_inhibition_ratio(-8.0, -8.0)
        assert ratio == pytest.approx(1.0)

    def test_zero_product_affinity(self):
        """Zero product affinity => returns inf."""
        ratio = calculate_inhibition_ratio(-5.0, 0.0)
        assert ratio == float("inf")

    def test_positive_affinities(self):
        """Positive affinities (repulsion) still computed correctly."""
        ratio = calculate_inhibition_ratio(2.0, 4.0)
        assert ratio == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: select_top_percentile
# ---------------------------------------------------------------------------


class TestSelectTopPercentile:
    def test_returns_top_50_percent(self):
        results = [
            {"sequence_id": "a", "substrate_affinity": -5.0},
            {"sequence_id": "b", "substrate_affinity": -10.0},
            {"sequence_id": "c", "substrate_affinity": -3.0},
            {"sequence_id": "d", "substrate_affinity": -8.0},
        ]
        top = select_top_percentile(results, percentile=50.0)

        assert len(top) == 2
        # Most negative (tightest binding) first
        ids = [r["sequence_id"] for r in top]
        assert "b" in ids
        assert "d" in ids

    def test_returns_all_at_100_percent(self):
        results = [
            {"sequence_id": "a", "substrate_affinity": -5.0},
            {"sequence_id": "b", "substrate_affinity": -10.0},
        ]
        top = select_top_percentile(results, percentile=100.0)
        assert len(top) == 2

    def test_returns_at_least_one(self):
        results = [{"sequence_id": "a", "substrate_affinity": -5.0}]
        top = select_top_percentile(results, percentile=10.0)
        assert len(top) >= 1

    def test_empty_input(self):
        top = select_top_percentile([], percentile=50.0)
        assert top == []

    def test_sorted_by_affinity(self):
        results = [
            {"sequence_id": "x", "substrate_affinity": -2.0},
            {"sequence_id": "y", "substrate_affinity": -9.0},
            {"sequence_id": "z", "substrate_affinity": -6.0},
        ]
        top = select_top_percentile(results, percentile=70.0)

        # Should be sorted: most negative first
        affinities = [r["substrate_affinity"] for r in top]
        assert affinities == sorted(affinities)


# ---------------------------------------------------------------------------
# Tests: run_boltz2_screen
# ---------------------------------------------------------------------------


class TestRunBoltz2Screen:
    @patch("pipeline.stage6_boltz2_screen.subprocess.run")
    def test_constructs_correct_command(self, mock_run: MagicMock, tmp_path: Path):
        seq_ids = ["var1", "var2"]
        out_dir = tmp_path / "output"
        _create_mock_boltz_output(out_dir, seq_ids)

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        fasta = tmp_path / "input.fasta"
        fasta.write_text(">var1\nACDEF\n>var2\nGHIKL\n")

        run_boltz2_screen(
            fasta_path=fasta,
            ligand_smiles="CCO",
            cofactors=["NAD+"],
            output_dir=out_dir,
            recycling_steps=10,
            num_seeds=1,
            boltz_bin="boltz",
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "boltz"
        assert cmd[1] == "predict"
        assert "--recycling_steps" in cmd
        assert "10" in cmd
        assert "--num_seeds" in cmd
        assert "1" in cmd
        assert "--output_dir" in cmd
        assert "--use_msa_server" in cmd

    @patch("pipeline.stage6_boltz2_screen.subprocess.run")
    def test_raises_on_nonzero_exit(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="CUDA error"
        )
        fasta = tmp_path / "input.fasta"
        fasta.write_text(">var1\nACDEF\n")

        with pytest.raises(RuntimeError, match="Boltz-2 screening failed"):
            run_boltz2_screen(
                fasta_path=fasta,
                ligand_smiles="CCO",
                cofactors=[],
                output_dir=tmp_path / "out",
            )

    @patch("pipeline.stage6_boltz2_screen.subprocess.run")
    def test_returns_results_per_sequence(self, mock_run: MagicMock, tmp_path: Path):
        seq_ids = ["var1", "var2", "var3"]
        out_dir = tmp_path / "output"
        _create_mock_boltz_output(out_dir, seq_ids)

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        fasta = tmp_path / "input.fasta"
        fasta.write_text(">var1\nACDEF\n>var2\nGHIKL\n>var3\nMNPQR\n")

        results = run_boltz2_screen(
            fasta_path=fasta,
            ligand_smiles="CCO",
            cofactors=[],
            output_dir=out_dir,
        )

        assert len(results) == 3
        for res in results:
            assert "sequence_id" in res
            assert "pdb_path" in res
            assert "plddt" in res
            assert "iptm" in res
            assert "affinity" in res

    @patch("pipeline.stage6_boltz2_screen.subprocess.run")
    def test_custom_boltz_bin(self, mock_run: MagicMock, tmp_path: Path):
        out_dir = tmp_path / "output"
        _create_mock_boltz_output(out_dir, ["var1"])
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        fasta = tmp_path / "input.fasta"
        fasta.write_text(">var1\nACDEF\n")

        run_boltz2_screen(
            fasta_path=fasta,
            ligand_smiles="C",
            cofactors=[],
            output_dir=out_dir,
            boltz_bin="/opt/boltz/bin/boltz",
        )

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/opt/boltz/bin/boltz"

    @patch("pipeline.stage6_boltz2_screen.subprocess.run")
    def test_writes_boltz_input_with_ligand(self, mock_run: MagicMock, tmp_path: Path):
        out_dir = tmp_path / "output"
        _create_mock_boltz_output(out_dir, ["var1"])
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        fasta = tmp_path / "input.fasta"
        fasta.write_text(">var1\nACDEF\n")

        run_boltz2_screen(
            fasta_path=fasta,
            ligand_smiles="CCO",
            cofactors=["NAD+"],
            output_dir=out_dir,
        )

        boltz_input = out_dir / "boltz_input.fasta"
        assert boltz_input.exists()
        content = boltz_input.read_text()
        assert ">protein|var1" in content
        assert ">ligand|SMILES" in content
        assert "CCO" in content
        assert ">cofactor|NAD+" in content


# ---------------------------------------------------------------------------
# Tests: run_stage
# ---------------------------------------------------------------------------


class TestRunStage:
    @patch("pipeline.stage6_boltz2_screen.run_boltz2_screen")
    def test_creates_output_directories(self, mock_screen: MagicMock, tmp_path: Path):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        mock_screen.return_value = [
            {
                "sequence_id": "enzyme_A_var1",
                "pdb_path": str(tmp_path / "dummy.pdb"),
                "plddt": 0.85,
                "iptm": 0.72,
                "affinity": -8.0,
            },
        ]

        run_stage(config)

        stage_dir = base_dir / "stage_06_boltz2_screening"
        assert (stage_dir / "substrate_docking").exists()
        assert (stage_dir / "product_docking").exists()

    @patch("pipeline.stage6_boltz2_screen.run_boltz2_screen")
    @patch("pipeline.stage6_boltz2_screen.apply_structural_filters")
    def test_produces_csv_with_correct_columns(
        self, mock_filters: MagicMock, mock_screen: MagicMock, tmp_path: Path
    ):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        mock_screen.return_value = [
            {
                "sequence_id": "enzyme_A_var1",
                "pdb_path": "",
                "plddt": 0.85,
                "iptm": 0.72,
                "affinity": -8.0,
            },
        ]
        mock_filters.return_value = (
            True,
            {
                "ca_rmsd": 1.0,
                "active_site_rmsd": 0.8,
                "ligand_rmsd": 1.2,
                "ca_rmsd_pass": True,
                "active_site_rmsd_pass": True,
                "ligand_rmsd_pass": True,
            },
        )

        run_stage(config)

        csv_path = base_dir / "stage_06_boltz2_screening" / "screening_results.csv"
        assert csv_path.exists()

        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            columns = reader.fieldnames
            rows = list(reader)

        for col in CSV_COLUMNS:
            assert col in columns, f"Missing column: {col}"
        assert len(rows) > 0

    @patch("pipeline.stage6_boltz2_screen.run_boltz2_screen")
    @patch("pipeline.stage6_boltz2_screen.apply_structural_filters")
    def test_processes_all_enzymes(
        self, mock_filters: MagicMock, mock_screen: MagicMock, tmp_path: Path
    ):
        config = _make_config(tmp_path)

        mock_screen.return_value = [
            {
                "sequence_id": "var1",
                "pdb_path": "",
                "plddt": 0.85,
                "iptm": 0.72,
                "affinity": -8.0,
            },
        ]
        mock_filters.return_value = (
            True,
            {
                "ca_rmsd": 1.0,
                "active_site_rmsd": 0.8,
                "ligand_rmsd": 1.2,
                "ca_rmsd_pass": True,
                "active_site_rmsd_pass": True,
                "ligand_rmsd_pass": True,
            },
        )

        summary = run_stage(config)

        assert "enzyme_A" in summary
        assert "enzyme_B" in summary

    @patch("pipeline.stage6_boltz2_screen.run_boltz2_screen")
    @patch("pipeline.stage6_boltz2_screen.apply_structural_filters")
    def test_summary_counts(
        self, mock_filters: MagicMock, mock_screen: MagicMock, tmp_path: Path
    ):
        config = _make_config(tmp_path)
        # Only test one enzyme
        del config["enzymes"]["enzyme_B"]

        mock_screen.return_value = [
            {"sequence_id": "v1", "pdb_path": "", "plddt": 0.8, "iptm": 0.7, "affinity": -8.0},
            {"sequence_id": "v2", "pdb_path": "", "plddt": 0.7, "iptm": 0.6, "affinity": -6.0},
            {"sequence_id": "v3", "pdb_path": "", "plddt": 0.6, "iptm": 0.5, "affinity": -4.0},
            {"sequence_id": "v4", "pdb_path": "", "plddt": 0.5, "iptm": 0.4, "affinity": -3.0},
        ]

        # First two pass, last two fail
        pass_results = [
            (True, {"ca_rmsd": 1.0, "active_site_rmsd": 0.8, "ligand_rmsd": 1.0,
                     "ca_rmsd_pass": True, "active_site_rmsd_pass": True, "ligand_rmsd_pass": True}),
            (True, {"ca_rmsd": 1.5, "active_site_rmsd": 1.0, "ligand_rmsd": 1.5,
                     "ca_rmsd_pass": True, "active_site_rmsd_pass": True, "ligand_rmsd_pass": True}),
            (False, {"ca_rmsd": 3.0, "active_site_rmsd": 2.0, "ligand_rmsd": 2.5,
                      "ca_rmsd_pass": False, "active_site_rmsd_pass": False, "ligand_rmsd_pass": False}),
            (False, {"ca_rmsd": 4.0, "active_site_rmsd": 3.0, "ligand_rmsd": 3.0,
                      "ca_rmsd_pass": False, "active_site_rmsd_pass": False, "ligand_rmsd_pass": False}),
        ]
        mock_filters.side_effect = pass_results

        summary = run_stage(config)

        assert summary["enzyme_A"]["total_candidates"] == 4
        assert summary["enzyme_A"]["passed_filters"] == 2

    @patch("pipeline.stage6_boltz2_screen.run_boltz2_screen")
    def test_handles_boltz_failure_gracefully(
        self, mock_screen: MagicMock, tmp_path: Path
    ):
        config = _make_config(tmp_path)
        mock_screen.side_effect = RuntimeError("Boltz-2 crashed")

        # Should not raise
        summary = run_stage(config)

        # Enzymes with failures are skipped
        assert "enzyme_A" not in summary
        assert "enzyme_B" not in summary

    @patch("pipeline.stage6_boltz2_screen.run_boltz2_screen")
    @patch("pipeline.stage6_boltz2_screen.apply_structural_filters")
    def test_product_docking_only_for_top_percentile(
        self, mock_filters: MagicMock, mock_screen: MagicMock, tmp_path: Path
    ):
        config = _make_config(tmp_path)
        del config["enzymes"]["enzyme_B"]

        # IDs must match the FASTA file created by _make_config
        substrate_results = [
            {"sequence_id": "enzyme_A_var1", "pdb_path": "", "plddt": 0.9, "iptm": 0.8, "affinity": -10.0},
            {"sequence_id": "enzyme_A_var2", "pdb_path": "", "plddt": 0.8, "iptm": 0.7, "affinity": -5.0},
        ]

        product_results = [
            {"sequence_id": "enzyme_A_var1", "pdb_path": "", "plddt": 0.85, "iptm": 0.75, "affinity": -6.0},
        ]

        # First call: substrate docking, second call: product docking
        mock_screen.side_effect = [substrate_results, product_results]

        mock_filters.return_value = (
            True,
            {"ca_rmsd": 1.0, "active_site_rmsd": 0.8, "ligand_rmsd": 1.0,
             "ca_rmsd_pass": True, "active_site_rmsd_pass": True, "ligand_rmsd_pass": True},
        )

        summary = run_stage(config)

        # run_boltz2_screen should be called twice: substrate + product
        assert mock_screen.call_count == 2
        # Product docking should only include top 50% (1 of 2 candidates)
        assert summary["enzyme_A"]["product_docked"] == 1
