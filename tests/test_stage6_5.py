"""Tests for Stage 6.5: MD Simulations (pipeline/stage6_5_md.py).

All tests run without OpenMM, mdtraj, or GPU installed.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.stage6_5_md import (
    STAGE_NAME,
    STAGE_NUM,
    INPUT_DESCRIPTION,
    OUTPUT_DESCRIPTION,
    apply_md_filters,
    run_md_simulation,
    analyze_trajectory,
    run_stage,
    _select_top_candidates,
    DEFAULT_ACTIVE_SITE_RMSD_MAX,
    DEFAULT_BACKBONE_RMSD_MAX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_minimal_pdb(path: Path, n_residues: int = 5) -> Path:
    """Write a minimal PDB with n CA atoms."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for i in range(1, n_residues + 1):
            x, y, z = float(i), 0.0, 0.0
            fh.write(
                f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
            )
        fh.write("END\n")
    return path


def _write_screening_csv(path: Path, candidates: list[dict]) -> Path:
    """Write a screening_results.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not candidates:
        path.write_text("")
        return path
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=candidates[0].keys())
        writer.writeheader()
        writer.writerows(candidates)
    return path


def _make_config(tmp_path: Path, optional_md: bool = True) -> dict:
    """Build a minimal pipeline config for testing run_stage."""
    base_dir = tmp_path / "pipeline"
    base_dir.mkdir(exist_ok=True)

    # Create WT PDB files
    wt_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
    wt_dir.mkdir(parents=True)
    for eid in ("enzyme_A", "enzyme_B"):
        _write_minimal_pdb(wt_dir / f"{eid}.pdb")

    # Create Stage 6 output dir and screening results
    stage6_dir = base_dir / "stage_06_boltz2_screening"
    stage6_dir.mkdir(parents=True)

    candidates = []
    for eid in ("enzyme_A", "enzyme_B"):
        enzyme_dir = stage6_dir / eid
        enzyme_dir.mkdir(parents=True)
        for i in range(60):
            seq_id = f"{eid}_design_{i}"
            pdb_path = enzyme_dir / f"{seq_id}.pdb"
            _write_minimal_pdb(pdb_path)
            candidates.append({
                "sequence_id": seq_id,
                "enzyme_id": eid,
                "pdb_path": str(pdb_path),
                "boltz2_plddt": 0.85 - i * 0.005,
            })

    _write_screening_csv(stage6_dir / "screening_results.csv", candidates)

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
            "stage_02_wt_structures": {"output_dir": "stage_02_wt_structures"},
            "stage_06_boltz2_screening": {"output_dir": "stage_06_boltz2_screening"},
            "stage_06_5_md": {
                "enabled": optional_md,
                "output_dir": "stage_06.5_md_simulations",
                "production_ns": 10.0,
            },
        },
        "enzymes": {
            "enzyme_A": {
                "name": "Enzyme A",
                "length": 428,
                "sequence_file": "inputs/sequences/enzyme_A.fasta",
                "substrate_smiles": "C",
                "product_smiles": "C",
                "fixed_residues": [],
                "hotspot_residues": [100, 150, 200],
                "contigs": "A1-428",
                "cofactors": [],
            },
            "enzyme_B": {
                "name": "Enzyme B",
                "length": 512,
                "sequence_file": "inputs/sequences/enzyme_B.fasta",
                "substrate_smiles": "C",
                "product_smiles": "C",
                "fixed_residues": [],
                "hotspot_residues": [80, 120],
                "contigs": "A1-512",
                "cofactors": [],
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests: module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_stage_name(self):
        assert STAGE_NAME == "Stage 6.5: MD Simulations (Optional)"

    def test_stage_num(self):
        assert STAGE_NUM == "6.5"

    def test_input_description(self):
        assert "PDB" in INPUT_DESCRIPTION
        assert "Stage 6" in INPUT_DESCRIPTION

    def test_output_description(self):
        assert "md_metrics.csv" in OUTPUT_DESCRIPTION
        assert "stage_06.5" in OUTPUT_DESCRIPTION


# ---------------------------------------------------------------------------
# Tests: apply_md_filters
# ---------------------------------------------------------------------------

class TestApplyMdFilters:
    def test_all_pass(self):
        metrics = {
            "backbone_rmsd_plateau": 1.5,
            "active_site_rmsd": 1.0,
        }
        passed, reason = apply_md_filters(metrics)
        assert passed is True
        assert reason == "passed"

    def test_backbone_rmsd_too_high(self):
        metrics = {
            "backbone_rmsd_plateau": 4.0,
            "active_site_rmsd": 1.0,
        }
        passed, reason = apply_md_filters(metrics)
        assert passed is False
        assert "backbone RMSD" in reason

    def test_active_site_rmsd_too_high(self):
        metrics = {
            "backbone_rmsd_plateau": 1.5,
            "active_site_rmsd": 3.0,
        }
        passed, reason = apply_md_filters(metrics)
        assert passed is False
        assert "active site RMSD" in reason

    def test_both_fail(self):
        metrics = {
            "backbone_rmsd_plateau": 5.0,
            "active_site_rmsd": 4.0,
        }
        passed, reason = apply_md_filters(metrics)
        assert passed is False
        assert "backbone" in reason
        assert "active site" in reason

    def test_at_threshold_passes(self):
        """Exactly at threshold should pass (not strictly greater)."""
        metrics = {
            "backbone_rmsd_plateau": DEFAULT_BACKBONE_RMSD_MAX,
            "active_site_rmsd": DEFAULT_ACTIVE_SITE_RMSD_MAX,
        }
        passed, reason = apply_md_filters(metrics)
        assert passed is True

    def test_just_above_threshold_fails(self):
        metrics = {
            "backbone_rmsd_plateau": DEFAULT_BACKBONE_RMSD_MAX + 0.01,
            "active_site_rmsd": 0.5,
        }
        passed, reason = apply_md_filters(metrics)
        assert passed is False

    def test_custom_thresholds(self):
        metrics = {
            "backbone_rmsd_plateau": 4.0,
            "active_site_rmsd": 3.5,
        }
        # With relaxed thresholds, should pass
        passed, reason = apply_md_filters(
            metrics, active_site_rmsd_max=5.0, backbone_rmsd_max=5.0
        )
        assert passed is True

    def test_zero_values_pass(self):
        metrics = {
            "backbone_rmsd_plateau": 0.0,
            "active_site_rmsd": 0.0,
        }
        passed, reason = apply_md_filters(metrics)
        assert passed is True

    def test_missing_keys_default_to_zero(self):
        """Missing keys should default to 0.0 (passes)."""
        passed, reason = apply_md_filters({})
        assert passed is True


# ---------------------------------------------------------------------------
# Tests: run_md_simulation — graceful skip when OpenMM unavailable
# ---------------------------------------------------------------------------

class TestRunMdSimulationNoOpenMM:
    @patch("pipeline.stage6_5_md._OPENMM_AVAILABLE", False)
    def test_skips_when_openmm_unavailable(self, tmp_path: Path):
        pdb = _write_minimal_pdb(tmp_path / "test.pdb")
        wt_pdb = _write_minimal_pdb(tmp_path / "wt.pdb")

        result = run_md_simulation(
            pdb_path=pdb,
            output_dir=tmp_path / "output",
            hotspot_residues=[1, 2, 3],
            wt_pdb_path=wt_pdb,
        )

        assert result["skipped"] is True
        assert "OpenMM" in result["skip_reason"]
        assert result["simulation_ns"] == 0.0
        assert math.isnan(result["backbone_rmsd_mean"])
        assert math.isnan(result["active_site_rmsd"])

    @patch("pipeline.stage6_5_md._OPENMM_AVAILABLE", False)
    def test_creates_output_dir(self, tmp_path: Path):
        pdb = _write_minimal_pdb(tmp_path / "test.pdb")
        wt_pdb = _write_minimal_pdb(tmp_path / "wt.pdb")
        out_dir = tmp_path / "new_output"

        run_md_simulation(
            pdb_path=pdb,
            output_dir=out_dir,
            hotspot_residues=[],
            wt_pdb_path=wt_pdb,
        )

        assert out_dir.exists()


# ---------------------------------------------------------------------------
# Tests: analyze_trajectory (mocked mdtraj)
# ---------------------------------------------------------------------------

class TestAnalyzeTrajectory:
    @patch("pipeline.stage6_5_md._MDTRAJ_AVAILABLE", True)
    @patch("pipeline.stage6_5_md.mdtraj", create=True)
    def test_returns_correct_structure(self, mock_mdtraj: MagicMock, tmp_path: Path):
        """analyze_trajectory should return dict with expected keys."""
        # Setup mock trajectory
        n_frames = 100
        mock_traj = MagicMock()
        mock_traj.__len__ = lambda self: n_frames
        mock_traj.__getitem__ = lambda self, key: mock_traj

        mock_topology = MagicMock()
        mock_topology.select.return_value = np.array([0, 1, 2, 3])
        mock_traj.topology = mock_topology

        mock_mdtraj.load.return_value = mock_traj
        mock_mdtraj.rmsd.return_value = np.random.uniform(0.05, 0.15, n_frames)
        mock_mdtraj.rmsf.return_value = np.array([0.05, 0.08, 0.06, 0.07])

        traj_path = tmp_path / "traj.dcd"
        traj_path.touch()
        pdb_path = _write_minimal_pdb(tmp_path / "design.pdb")
        wt_pdb = _write_minimal_pdb(tmp_path / "wt.pdb")

        result = analyze_trajectory(
            traj_path=traj_path,
            pdb_path=pdb_path,
            hotspot_residues=[1, 2, 3],
            wt_pdb_path=wt_pdb,
        )

        assert "backbone_rmsd_mean" in result
        assert "backbone_rmsd_plateau" in result
        assert "active_site_rmsd" in result
        assert "active_site_rmsf" in result
        assert isinstance(result["backbone_rmsd_mean"], float)
        assert isinstance(result["active_site_rmsd"], float)

    @patch("pipeline.stage6_5_md._MDTRAJ_AVAILABLE", False)
    def test_falls_back_to_simplified(self, tmp_path: Path):
        """Without mdtraj, should use simplified static analysis."""
        traj_path = tmp_path / "traj.dcd"
        traj_path.touch()
        pdb_path = _write_minimal_pdb(tmp_path / "design.pdb")
        wt_pdb = _write_minimal_pdb(tmp_path / "wt.pdb")

        result = analyze_trajectory(
            traj_path=traj_path,
            pdb_path=pdb_path,
            hotspot_residues=[1, 2],
            wt_pdb_path=wt_pdb,
        )

        assert "backbone_rmsd_mean" in result
        assert "backbone_rmsd_plateau" in result
        assert "active_site_rmsd" in result
        assert "active_site_rmsf" in result
        # RMSF not available in simplified mode
        assert result["active_site_rmsf"] == 0.0

    @patch("pipeline.stage6_5_md._MDTRAJ_AVAILABLE", True)
    @patch("pipeline.stage6_5_md.mdtraj", create=True)
    def test_empty_hotspot_residues(self, mock_mdtraj: MagicMock, tmp_path: Path):
        """Empty hotspot list should return zero active site metrics."""
        n_frames = 50
        mock_traj = MagicMock()
        mock_traj.__len__ = lambda self: n_frames
        mock_traj.__getitem__ = lambda self, key: mock_traj

        mock_topology = MagicMock()
        mock_topology.select.return_value = np.array([0, 1, 2])
        mock_traj.topology = mock_topology

        mock_mdtraj.load.return_value = mock_traj
        mock_mdtraj.rmsd.return_value = np.random.uniform(0.05, 0.15, n_frames)

        traj_path = tmp_path / "traj.dcd"
        traj_path.touch()
        pdb_path = _write_minimal_pdb(tmp_path / "design.pdb")
        wt_pdb = _write_minimal_pdb(tmp_path / "wt.pdb")

        result = analyze_trajectory(
            traj_path=traj_path,
            pdb_path=pdb_path,
            hotspot_residues=[],
            wt_pdb_path=wt_pdb,
        )

        assert result["active_site_rmsd"] == 0.0
        assert result["active_site_rmsf"] == 0.0


# ---------------------------------------------------------------------------
# Tests: _select_top_candidates
# ---------------------------------------------------------------------------

class TestSelectTopCandidates:
    def test_selects_top_n_per_enzyme(self, tmp_path: Path):
        candidates = []
        for eid in ("enzyme_A", "enzyme_B"):
            for i in range(60):
                candidates.append({
                    "sequence_id": f"{eid}_d{i}",
                    "enzyme_id": eid,
                })

        csv_path = _write_screening_csv(tmp_path / "results.csv", candidates)
        selected = _select_top_candidates(csv_path, n_per_enzyme=50, max_total=200)

        # 50 per enzyme × 2 enzymes = 100
        assert len(selected) == 100

        enzyme_a = [c for c in selected if c["enzyme_id"] == "enzyme_A"]
        enzyme_b = [c for c in selected if c["enzyme_id"] == "enzyme_B"]
        assert len(enzyme_a) == 50
        assert len(enzyme_b) == 50

    def test_respects_max_total(self, tmp_path: Path):
        candidates = []
        for eid in ("enzyme_A", "enzyme_B", "enzyme_C", "enzyme_D"):
            for i in range(60):
                candidates.append({
                    "sequence_id": f"{eid}_d{i}",
                    "enzyme_id": eid,
                })

        csv_path = _write_screening_csv(tmp_path / "results.csv", candidates)
        selected = _select_top_candidates(csv_path, n_per_enzyme=60, max_total=200)

        assert len(selected) == 200

    def test_handles_empty_csv(self, tmp_path: Path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("sequence_id,enzyme_id\n")
        selected = _select_top_candidates(csv_path)
        assert selected == []

    def test_handles_missing_csv(self, tmp_path: Path):
        csv_path = tmp_path / "nonexistent.csv"
        selected = _select_top_candidates(csv_path)
        assert selected == []


# ---------------------------------------------------------------------------
# Tests: run_stage
# ---------------------------------------------------------------------------

class TestRunStage:
    @patch("pipeline.stage6_5_md._OPENMM_AVAILABLE", False)
    def test_skips_when_openmm_unavailable(self, tmp_path: Path):
        """Stage should complete gracefully when OpenMM is not installed."""
        config = _make_config(tmp_path, optional_md=True)
        result = run_stage(config)

        assert result["skipped"] is True
        assert "OpenMM" in result.get("reason", "")
        assert result["total_simulated"] == 0

        # Should write empty metrics file
        base_dir = Path(config["global"]["base_dir"])
        metrics_path = base_dir / "stage_06.5_md_simulations" / "md_metrics.csv"
        assert metrics_path.exists()

    def test_skips_when_disabled(self, tmp_path: Path):
        """Stage should skip when enabled=false."""
        config = _make_config(tmp_path, optional_md=False)
        result = run_stage(config)

        assert result["skipped"] is True
        assert "disabled" in result.get("reason", "").lower()

    @patch("pipeline.stage6_5_md._OPENMM_AVAILABLE", True)
    @patch("pipeline.stage6_5_md.run_md_simulation")
    def test_selects_top_200_candidates(self, mock_run_md: MagicMock, tmp_path: Path):
        """Should select top 200 candidates from screening results."""
        config = _make_config(tmp_path, optional_md=True)

        # Mock successful MD runs
        mock_run_md.return_value = {
            "backbone_rmsd_mean": 1.5,
            "backbone_rmsd_plateau": 1.8,
            "active_site_rmsd": 1.2,
            "active_site_rmsf": 0.5,
            "simulation_ns": 10.0,
            "trajectory_path": "/fake/traj.dcd",
            "skipped": False,
        }

        result = run_stage(config)

        # 50 per enzyme × 2 enzymes = 100 candidates (we only have 2 enzymes)
        assert result["total_simulated"] == 100
        assert result["passed"] == 100
        assert result["failed"] == 0
        assert mock_run_md.call_count == 100

    @patch("pipeline.stage6_5_md._OPENMM_AVAILABLE", True)
    @patch("pipeline.stage6_5_md.run_md_simulation")
    def test_writes_md_metrics_csv(self, mock_run_md: MagicMock, tmp_path: Path):
        """Should write md_metrics.csv with correct columns."""
        config = _make_config(tmp_path, optional_md=True)

        mock_run_md.return_value = {
            "backbone_rmsd_mean": 1.5,
            "backbone_rmsd_plateau": 1.8,
            "active_site_rmsd": 1.2,
            "active_site_rmsf": 0.5,
            "simulation_ns": 10.0,
            "trajectory_path": "/fake/traj.dcd",
            "skipped": False,
        }

        run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        metrics_path = base_dir / "stage_06.5_md_simulations" / "md_metrics.csv"
        assert metrics_path.exists()

        with open(metrics_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 100
        expected_cols = {
            "sequence_id", "enzyme_id", "backbone_rmsd_mean",
            "backbone_rmsd_plateau", "active_site_rmsd", "active_site_rmsf",
            "passed_md_filters", "simulation_ns", "trajectory_path",
        }
        assert expected_cols.issubset(set(rows[0].keys()))

    @patch("pipeline.stage6_5_md._OPENMM_AVAILABLE", True)
    @patch("pipeline.stage6_5_md.run_md_simulation")
    def test_filters_applied_correctly(self, mock_run_md: MagicMock, tmp_path: Path):
        """Should correctly filter candidates based on RMSD thresholds."""
        config = _make_config(tmp_path, optional_md=True)

        call_count = [0]

        def alternating_metrics(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                # Fails backbone RMSD
                return {
                    "backbone_rmsd_mean": 4.0,
                    "backbone_rmsd_plateau": 4.0,
                    "active_site_rmsd": 1.0,
                    "active_site_rmsf": 0.5,
                    "simulation_ns": 10.0,
                    "trajectory_path": "",
                    "skipped": False,
                }
            else:
                # Passes
                return {
                    "backbone_rmsd_mean": 1.5,
                    "backbone_rmsd_plateau": 1.5,
                    "active_site_rmsd": 1.0,
                    "active_site_rmsf": 0.5,
                    "simulation_ns": 10.0,
                    "trajectory_path": "",
                    "skipped": False,
                }

        mock_run_md.side_effect = alternating_metrics

        result = run_stage(config)

        assert result["passed"] == 50
        assert result["failed"] == 50

    @patch("pipeline.stage6_5_md._OPENMM_AVAILABLE", True)
    def test_handles_missing_screening_results(self, tmp_path: Path):
        """Should handle missing screening results gracefully."""
        config = _make_config(tmp_path, optional_md=True)
        base_dir = Path(config["global"]["base_dir"])

        # Remove screening results
        screening_csv = base_dir / "stage_06_boltz2_screening" / "screening_results.csv"
        screening_csv.unlink()

        result = run_stage(config)

        assert result["skipped"] is True
        assert "not found" in result.get("reason", "").lower()
