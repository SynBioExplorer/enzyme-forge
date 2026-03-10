"""Tests for Stage 3: Backbone Diversification (pipeline/stage3_rfdiffusion3.py).

All tests run without actual RFdiffusion3, Foundry CLI, or GPU.
"""
from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.stage3_rfdiffusion3 import (
    STAGE_NAME,
    STAGE_NUM,
    INPUT_DESCRIPTION,
    OUTPUT_DESCRIPTION,
    determine_design_strategy,
    run_rfdiffusion3,
    calc_radius_of_gyration,
    calc_pocket_volume,
    prefilter_backbones,
    run_stage,
    _load_fcc_from_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_minimal_pdb(path: Path, coords: list[tuple[float, float, float]]) -> Path:
    """Write a minimal PDB file with CA atoms at the given coordinates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for i, (x, y, z) in enumerate(coords, start=1):
            # PDB ATOM format: columns are fixed-width
            fh.write(
                f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
            )
        fh.write("END\n")
    return path


def _write_full_pdb(path: Path, residues: dict[int, list[tuple[str, float, float, float]]]) -> Path:
    """Write a PDB with multiple atoms per residue.

    residues: {resseq: [(atom_name, x, y, z), ...]}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    atom_idx = 1
    with open(path, "w") as fh:
        for resseq in sorted(residues.keys()):
            for atom_name, x, y, z in residues[resseq]:
                fh.write(
                    f"ATOM  {atom_idx:5d}  {atom_name:<3s} ALA A{resseq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
                )
                atom_idx += 1
        fh.write("END\n")
    return path


@pytest.fixture
def simple_pdb(tmp_path: Path) -> Path:
    """PDB with 4 CA atoms arranged in a known geometry."""
    # Tetrahedron-like arrangement centered near origin
    coords = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (5.0, 10.0, 0.0),
        (5.0, 5.0, 10.0),
    ]
    return _write_minimal_pdb(tmp_path / "simple.pdb", coords)


@pytest.fixture
def compact_pdb(tmp_path: Path) -> Path:
    """Compact protein — small Rg."""
    coords = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (1.5, 1.5, 1.5)]
    return _write_minimal_pdb(tmp_path / "compact.pdb", coords)


@pytest.fixture
def extended_pdb(tmp_path: Path) -> Path:
    """Extended protein — large Rg."""
    coords = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0)]
    return _write_minimal_pdb(tmp_path / "extended.pdb", coords)


@pytest.fixture
def bottleneck_csv(tmp_path: Path) -> Path:
    """Write a bottleneck_results.csv with known FCC values."""
    csv_path = tmp_path / "stage_00_bottleneck" / "bottleneck_results.csv"
    csv_path.parent.mkdir(parents=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["enzyme_id", "fcc", "kcat_pred", "km_pred", "priority", "notes"]
        )
        writer.writeheader()
        writer.writerow({"enzyme_id": "enzyme_A", "fcc": "0.4", "kcat_pred": "12.5",
                          "km_pred": "0.45", "priority": "primary", "notes": ""})
        writer.writerow({"enzyme_id": "enzyme_B", "fcc": "0.3", "kcat_pred": "8.3",
                          "km_pred": "0.82", "priority": "secondary", "notes": ""})
        writer.writerow({"enzyme_id": "enzyme_C", "fcc": "0.2", "kcat_pred": "25.1",
                          "km_pred": "0.31", "priority": "secondary", "notes": ""})
        writer.writerow({"enzyme_id": "enzyme_D", "fcc": "0.1", "kcat_pred": "15.7",
                          "km_pred": "0.55", "priority": "secondary", "notes": ""})
    return csv_path


def _make_config(tmp_path: Path) -> dict:
    """Build a minimal pipeline config for testing run_stage."""
    base_dir = tmp_path / "pipeline"
    base_dir.mkdir(exist_ok=True)

    # Create WT PDB files
    wt_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
    wt_dir.mkdir(parents=True)
    for eid in ("enzyme_A", "enzyme_B"):
        coords = [(float(i), 0.0, 0.0) for i in range(10)]
        _write_minimal_pdb(wt_dir / f"{eid}.pdb", coords)

    # Create bottleneck CSV
    bn_dir = base_dir / "stage_00_bottleneck"
    bn_dir.mkdir(parents=True)
    with open(bn_dir / "bottleneck_results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["enzyme_id", "fcc", "kcat_pred", "km_pred", "priority", "notes"]
        )
        writer.writeheader()
        writer.writerow({"enzyme_id": "enzyme_A", "fcc": "0.4", "kcat_pred": "12.5",
                          "km_pred": "0.45", "priority": "primary", "notes": ""})
        writer.writerow({"enzyme_id": "enzyme_B", "fcc": "0.2", "kcat_pred": "8.3",
                          "km_pred": "0.82", "priority": "secondary", "notes": ""})

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
        "tools": {
            "rfdiffusion3_foundry_bin": "foundry",
        },
        "stages": {
            "stage_00_bottleneck": {"output_dir": "stage_00_bottleneck"},
            "stage_02_wt_structures": {"output_dir": "stage_02_wt_structures"},
            "stage_03_rfdiffusion3": {"output_dir": "stage_03_rfdiffusion3"},
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
                "contigs": "A1-50/0 10-20/A51-428",
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
        assert STAGE_NAME == "Stage 3: Backbone Diversification (RFdiffusion3)"

    def test_stage_num(self):
        assert STAGE_NUM == "3"

    def test_input_description(self):
        assert "PDB" in INPUT_DESCRIPTION
        assert "FCC" in INPUT_DESCRIPTION

    def test_output_description(self):
        assert "stage_03_rfdiffusion3" in OUTPUT_DESCRIPTION


# ---------------------------------------------------------------------------
# Tests: determine_design_strategy
# ---------------------------------------------------------------------------

class TestDetermineDesignStrategy:
    def test_high_fcc_returns_aggressive(self):
        """FCC > 0.3 → aggressive strategy with two rounds."""
        strategies = determine_design_strategy("enzyme_A", 0.5)
        assert len(strategies) == 2
        assert strategies[0]["partial_T"] == 15
        assert strategies[0]["n_designs"] == 500
        assert strategies[0]["strategy"] == "aggressive"
        assert strategies[1]["partial_T"] == 20
        assert strategies[1]["n_designs"] == 100
        assert strategies[1]["strategy"] == "aggressive_exploration"

    def test_low_fcc_returns_conservative(self):
        """FCC <= 0.3 → conservative strategy with one round."""
        strategies = determine_design_strategy("enzyme_D", 0.1)
        assert len(strategies) == 1
        assert strategies[0]["partial_T"] == 10
        assert strategies[0]["n_designs"] == 200
        assert strategies[0]["strategy"] == "conservative"

    def test_boundary_fcc_exactly_0_3(self):
        """FCC == 0.3 → conservative (boundary is <=)."""
        strategies = determine_design_strategy("enzyme_B", 0.3)
        assert len(strategies) == 1
        assert strategies[0]["strategy"] == "conservative"

    def test_fcc_just_above_threshold(self):
        """FCC = 0.31 → aggressive."""
        strategies = determine_design_strategy("enzyme_X", 0.31)
        assert len(strategies) == 2
        assert strategies[0]["strategy"] == "aggressive"

    def test_high_fcc_total_designs(self):
        """High-FCC strategy should total 600 designs (500 + 100)."""
        strategies = determine_design_strategy("enzyme_A", 0.4)
        total = sum(s["n_designs"] for s in strategies)
        assert total == 600


# ---------------------------------------------------------------------------
# Tests: run_rfdiffusion3
# ---------------------------------------------------------------------------

class TestRunRfdiffusion3:
    @patch("pipeline.stage3_rfdiffusion3.subprocess.run")
    def test_constructs_correct_command(self, mock_run: MagicMock, tmp_path: Path):
        """Verify the subprocess command contains all expected arguments."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        (out_dir / "design_0.pdb").write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_pdb = tmp_path / "input.pdb"
        input_pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n")

        run_rfdiffusion3(
            input_pdb=input_pdb,
            contigs="A1-50/0 10-20/A51-428",
            hotspot_residues="100,150,200",
            partial_T=15,
            n_designs=500,
            output_dir=out_dir,
            foundry_bin="/opt/foundry/bin/foundry",
            seed=42,
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/opt/foundry/bin/foundry"
        assert cmd[1] == "run"
        assert cmd[2] == "rfd3"
        assert "--pdb" in cmd
        assert str(input_pdb) in cmd
        assert "--contigs" in cmd
        assert "A1-50/0 10-20/A51-428" in cmd
        assert "--hotspot_residues" in cmd
        assert "100,150,200" in cmd
        assert "--partial_T" in cmd
        assert "15" in cmd
        assert "--num_designs" in cmd
        assert "500" in cmd
        assert "--output_dir" in cmd
        assert str(out_dir) in cmd
        assert "--seed" in cmd
        assert "42" in cmd

    @patch("pipeline.stage3_rfdiffusion3.subprocess.run")
    def test_raises_on_nonzero_exit(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="CUDA OOM"
        )

        input_pdb = tmp_path / "input.pdb"
        input_pdb.write_text("END\n")

        with pytest.raises(RuntimeError, match="RFdiffusion3 failed"):
            run_rfdiffusion3(
                input_pdb=input_pdb,
                contigs="A1-100",
                hotspot_residues="50",
                partial_T=10,
                n_designs=10,
                output_dir=tmp_path / "out",
            )

    @patch("pipeline.stage3_rfdiffusion3.subprocess.run")
    def test_returns_generated_pdbs(self, mock_run: MagicMock, tmp_path: Path):
        """Should return list of PDB files found in output dir."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        for i in range(3):
            (out_dir / f"design_{i}.pdb").write_text("END\n")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_pdb = tmp_path / "input.pdb"
        input_pdb.write_text("END\n")

        result = run_rfdiffusion3(
            input_pdb=input_pdb,
            contigs="A1-100",
            hotspot_residues="50",
            partial_T=10,
            n_designs=3,
            output_dir=out_dir,
        )

        assert len(result) == 3
        assert all(p.suffix == ".pdb" for p in result)

    @patch("pipeline.stage3_rfdiffusion3.subprocess.run")
    def test_creates_output_directory(self, mock_run: MagicMock, tmp_path: Path):
        """Output directory should be created if it doesn't exist."""
        out_dir = tmp_path / "new_output"
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_pdb = tmp_path / "input.pdb"
        input_pdb.write_text("END\n")

        run_rfdiffusion3(
            input_pdb=input_pdb,
            contigs="A1-100",
            hotspot_residues="50",
            partial_T=10,
            n_designs=1,
            output_dir=out_dir,
        )

        assert out_dir.exists()


# ---------------------------------------------------------------------------
# Tests: calc_radius_of_gyration
# ---------------------------------------------------------------------------

class TestCalcRadiusOfGyration:
    def test_known_geometry(self, simple_pdb: Path):
        """4 CA atoms at known coords → calculable Rg."""
        rg = calc_radius_of_gyration(simple_pdb)
        assert rg > 0
        # Centroid = (5, 3.75, 2.5)
        # Compute expected Rg manually
        coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 10.0, 0.0],
            [5.0, 5.0, 10.0],
        ])
        centroid = coords.mean(axis=0)
        sq_dists = np.sum((coords - centroid) ** 2, axis=1)
        expected_rg = float(np.sqrt(sq_dists.mean()))
        assert abs(rg - expected_rg) < 1e-6

    def test_compact_smaller_than_extended(self, compact_pdb: Path, extended_pdb: Path):
        """Compact structure should have smaller Rg than extended."""
        rg_compact = calc_radius_of_gyration(compact_pdb)
        rg_extended = calc_radius_of_gyration(extended_pdb)
        assert rg_compact < rg_extended

    def test_single_atom_rg_zero(self, tmp_path: Path):
        """Single atom → Rg = 0."""
        pdb = _write_minimal_pdb(tmp_path / "single.pdb", [(5.0, 5.0, 5.0)])
        rg = calc_radius_of_gyration(pdb)
        assert rg == 0.0

    def test_empty_pdb_rg_zero(self, tmp_path: Path):
        """Empty PDB → Rg = 0."""
        pdb = tmp_path / "empty.pdb"
        pdb.write_text("END\n")
        rg = calc_radius_of_gyration(pdb)
        assert rg == 0.0

    def test_two_atoms_symmetric(self, tmp_path: Path):
        """Two atoms at (-5,0,0) and (5,0,0) → Rg = 5.0."""
        pdb = _write_minimal_pdb(tmp_path / "two.pdb", [(-5.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
        rg = calc_radius_of_gyration(pdb)
        assert abs(rg - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# Tests: calc_pocket_volume
# ---------------------------------------------------------------------------

class TestCalcPocketVolume:
    def test_empty_hotspots_returns_zero(self, simple_pdb: Path):
        vol = calc_pocket_volume(simple_pdb, hotspot_residues=[])
        assert vol == 0.0

    def test_nonzero_for_valid_pocket(self, tmp_path: Path):
        """Pocket around hotspot residues should have nonzero volume."""
        # Create PDB with atoms at known positions, leaving space for a pocket
        residues = {
            1: [("CA", 0.0, 0.0, 0.0), ("CB", 1.0, 0.0, 0.0)],
            2: [("CA", 5.0, 0.0, 0.0), ("CB", 5.0, 1.0, 0.0)],
            3: [("CA", 0.0, 5.0, 0.0), ("CB", 1.0, 5.0, 0.0)],
        }
        pdb = _write_full_pdb(tmp_path / "pocket.pdb", residues)
        vol = calc_pocket_volume(pdb, hotspot_residues=[1, 2, 3], pocket_radius=5.0)
        assert vol > 0

    def test_missing_hotspot_residues(self, tmp_path: Path):
        """Hotspot residues not in PDB → volume 0."""
        residues = {1: [("CA", 0.0, 0.0, 0.0)]}
        pdb = _write_full_pdb(tmp_path / "miss.pdb", residues)
        vol = calc_pocket_volume(pdb, hotspot_residues=[99, 100])
        assert vol == 0.0


# ---------------------------------------------------------------------------
# Tests: prefilter_backbones
# ---------------------------------------------------------------------------

class TestPrefilterBackbones:
    def test_empty_input_returns_empty(self):
        assert prefilter_backbones([], hotspot_residues=[]) == []

    def test_no_filters_passes_all(self, compact_pdb: Path, extended_pdb: Path):
        """With no thresholds set, all PDBs pass."""
        result = prefilter_backbones(
            [compact_pdb, extended_pdb],
            hotspot_residues=[],
        )
        assert len(result) == 2

    def test_rg_max_filters_extended(self, compact_pdb: Path, extended_pdb: Path):
        """Extended PDB should fail rg_max filter."""
        result = prefilter_backbones(
            [compact_pdb, extended_pdb],
            hotspot_residues=[],
            rg_max=10.0,
        )
        assert compact_pdb in result
        assert extended_pdb not in result

    def test_rg_min_filters_compact(self, compact_pdb: Path, extended_pdb: Path):
        """Compact PDB should fail rg_min filter."""
        result = prefilter_backbones(
            [compact_pdb, extended_pdb],
            hotspot_residues=[],
            rg_min=10.0,
        )
        assert compact_pdb not in result
        assert extended_pdb in result

    def test_all_pass_with_generous_thresholds(self, compact_pdb: Path, extended_pdb: Path):
        result = prefilter_backbones(
            [compact_pdb, extended_pdb],
            hotspot_residues=[],
            rg_max=1000.0,
            rg_min=0.0,
        )
        assert len(result) == 2

    def test_all_fail_strict_thresholds(self, compact_pdb: Path, extended_pdb: Path):
        """Very strict thresholds should reject everything."""
        result = prefilter_backbones(
            [compact_pdb, extended_pdb],
            hotspot_residues=[],
            rg_max=0.001,
        )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: _load_fcc_from_csv
# ---------------------------------------------------------------------------

class TestLoadFccFromCsv:
    def test_loads_values(self, bottleneck_csv: Path):
        fcc_map = _load_fcc_from_csv(bottleneck_csv)
        assert fcc_map["enzyme_A"] == pytest.approx(0.4)
        assert fcc_map["enzyme_B"] == pytest.approx(0.3)
        assert fcc_map["enzyme_C"] == pytest.approx(0.2)
        assert fcc_map["enzyme_D"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Tests: run_stage
# ---------------------------------------------------------------------------

class TestRunStage:
    @patch("pipeline.stage3_rfdiffusion3.run_rfdiffusion3")
    def test_reads_fcc_and_applies_strategy(self, mock_rfd3: MagicMock, tmp_path: Path):
        """High-FCC enzyme_A should get aggressive strategy (2 calls),
        low-FCC enzyme_B should get conservative (1 call)."""
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        def fake_rfd3(input_pdb, contigs, hotspot_residues, partial_T,
                       n_designs, output_dir, foundry_bin="foundry", seed=42):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pdbs = []
            for i in range(min(n_designs, 3)):
                p = output_dir / f"design_{i}.pdb"
                _write_minimal_pdb(p, [(float(i), 0.0, 0.0)])
                pdbs.append(p)
            return pdbs

        mock_rfd3.side_effect = fake_rfd3

        results = run_stage(config)

        # enzyme_A (FCC=0.4 > 0.3) → 2 calls (pT15 + pT20)
        # enzyme_B (FCC=0.2 ≤ 0.3) → 1 call (pT10)
        a_calls = [c for c in mock_rfd3.call_args_list
                    if "enzyme_A" in str(c)]
        b_calls = [c for c in mock_rfd3.call_args_list
                    if "enzyme_B" in str(c)]
        assert len(a_calls) == 2
        assert len(b_calls) == 1

        assert "enzyme_A" in results
        assert "enzyme_B" in results

    @patch("pipeline.stage3_rfdiffusion3.run_rfdiffusion3")
    def test_creates_output_directories(self, mock_rfd3: MagicMock, tmp_path: Path):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        mock_rfd3.return_value = []

        run_stage(config)

        output_dir = base_dir / "stage_03_rfdiffusion3"
        assert output_dir.exists()
        assert (output_dir / "enzyme_A").exists()
        assert (output_dir / "enzyme_B").exists()

    @patch("pipeline.stage3_rfdiffusion3.run_rfdiffusion3")
    def test_handles_rfdiffusion3_failure(self, mock_rfd3: MagicMock, tmp_path: Path):
        """If RFdiffusion3 fails for one enzyme, others should still proceed."""
        config = _make_config(tmp_path)

        call_count = [0]

        def sometimes_fail(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("GPU OOM")
            output_dir = Path(kwargs.get("output_dir", args[5] if len(args) > 5 else "."))
            output_dir.mkdir(parents=True, exist_ok=True)
            p = output_dir / "design_0.pdb"
            _write_minimal_pdb(p, [(0.0, 0.0, 0.0)])
            return [p]

        mock_rfd3.side_effect = sometimes_fail

        results = run_stage(config)
        # enzyme_A fails (2 calls), enzyme_B succeeds (1 call)
        assert results["enzyme_A"] == []
        assert len(results["enzyme_B"]) >= 1

    @patch("pipeline.stage3_rfdiffusion3.run_rfdiffusion3")
    def test_uses_default_fcc_when_csv_missing(self, mock_rfd3: MagicMock, tmp_path: Path):
        """When bottleneck CSV is missing, use default FCC=0.25 (conservative)."""
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        # Remove the bottleneck CSV
        bn_csv = base_dir / "stage_00_bottleneck" / "bottleneck_results.csv"
        bn_csv.unlink()

        mock_rfd3.return_value = []

        run_stage(config)

        # Default FCC=0.25 ≤ 0.3 → conservative for all → 1 call per enzyme
        assert mock_rfd3.call_count == 2  # one per enzyme

    @patch("pipeline.stage3_rfdiffusion3.run_rfdiffusion3")
    def test_partial_T_values_in_calls(self, mock_rfd3: MagicMock, tmp_path: Path):
        """Verify correct partial_T values are passed to RFdiffusion3."""
        config = _make_config(tmp_path)

        mock_rfd3.return_value = []

        run_stage(config)

        partial_ts = []
        for call in mock_rfd3.call_args_list:
            # partial_T is the 4th positional arg or keyword
            if call.kwargs.get("partial_T"):
                partial_ts.append(call.kwargs["partial_T"])
            elif len(call.args) > 3:
                partial_ts.append(call.args[3])

        # enzyme_A (FCC=0.4): partial_T=15, partial_T=20
        # enzyme_B (FCC=0.2): partial_T=10
        assert 15 in partial_ts
        assert 20 in partial_ts
        assert 10 in partial_ts

    @patch("pipeline.stage3_rfdiffusion3.run_rfdiffusion3")
    def test_skips_enzyme_without_wt_pdb(self, mock_rfd3: MagicMock, tmp_path: Path):
        """Enzyme with missing WT PDB should be skipped."""
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        # Remove enzyme_B WT PDB
        wt_pdb = base_dir / "stage_02_wt_structures" / "substrate_complexes" / "enzyme_B.pdb"
        wt_pdb.unlink()

        mock_rfd3.return_value = []
        results = run_stage(config)

        # enzyme_B should be skipped (not in results or empty)
        assert "enzyme_B" not in results
