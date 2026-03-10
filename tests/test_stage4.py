"""Tests for Stage 4: Sequence Design with LigandMPNN (pipeline/stage4_ligandmpnn.py).

All tests run without LigandMPNN or GPU.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from pipeline.stage4_ligandmpnn import (
    STAGE_NAME,
    STAGE_NUM,
    INPUT_DESCRIPTION,
    OUTPUT_DESCRIPTION,
    BACKBONE_TEMPERATURES,
    WT_TEMPERATURES,
    run_ligandmpnn,
    design_backbone,
    design_wt_branch,
    deduplicate_sequences,
    run_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> dict:
    """Create a minimal config for testing."""
    base_dir = tmp_path / "pipeline"
    base_dir.mkdir()

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
            "ligandmpnn_dir": "/opt/tools/LigandMPNN",
        },
        "stages": {
            "stage_02_wt_structures": {
                "output_dir": "stage_02_wt_structures",
            },
            "stage_03_rfdiffusion3": {
                "output_dir": "stage_03_rfdiffusion3",
            },
            "stage_04_ligandmpnn": {
                "output_dir": "stage_04_ligandmpnn",
                "number_of_batches": 16,
                "omit_aa": "C",
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


_mock_output_counter = 0


def _create_mock_ligandmpnn_output(output_dir: Path, n_seqs: int = 3) -> None:
    """Create mock LigandMPNN output files with unique sequences per call."""
    global _mock_output_counter
    _mock_output_counter += 1
    tag = _mock_output_counter

    seqs_dir = output_dir / "seqs"
    seqs_dir.mkdir(parents=True, exist_ok=True)

    # Write FASTA file — use tag to make sequences unique across calls
    fasta_path = seqs_dir / "designed.fa"
    with open(fasta_path, "w") as fh:
        for i in range(n_seqs):
            fh.write(f">seq_{tag}_{i}\n")
            fh.write(f"MVLSPADKTNVKAAWGKVGA{'A' * i}{'G' * tag}\n")

    # Write scores JSON
    scores_path = output_dir / "scores.json"
    scores = {f"seq_{tag}_{i}": 0.5 + i * 0.1 for i in range(n_seqs)}
    with open(scores_path, "w") as fh:
        json.dump(scores, fh)


def _create_dummy_pdb(path: Path) -> Path:
    """Create a minimal dummy PDB file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00\nEND\n"
    )
    return path


# ---------------------------------------------------------------------------
# Tests: module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_stage_name(self):
        assert STAGE_NAME == "Stage 4: Sequence Design (LigandMPNN)"

    def test_stage_num(self):
        assert STAGE_NUM == "4"

    def test_input_description(self):
        assert "Backbone PDBs" in INPUT_DESCRIPTION
        assert "WT PDBs" in INPUT_DESCRIPTION

    def test_output_description(self):
        assert "FASTA" in OUTPUT_DESCRIPTION
        assert "ligandmpnn_scores.csv" in OUTPUT_DESCRIPTION


# ---------------------------------------------------------------------------
# Tests: run_ligandmpnn
# ---------------------------------------------------------------------------

class TestRunLigandmpnn:
    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_constructs_correct_command(self, mock_run: MagicMock, tmp_path: Path):
        out_dir = tmp_path / "output"
        _create_mock_ligandmpnn_output(out_dir)

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        pdb_path = _create_dummy_pdb(tmp_path / "input.pdb")

        run_ligandmpnn(
            pdb_path=pdb_path,
            output_dir=out_dir,
            temperature=0.1,
            n_batches=16,
            fixed_residues=[100, 200, 300],
            ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
            omit_aa="C",
            seed=42,
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        # Verify command components
        assert cmd[0] == "python"
        assert "/opt/tools/LigandMPNN/run.py" in cmd[1]
        assert "--model_type" in cmd
        assert "ligand_mpnn" in cmd
        assert "--pdb_path" in cmd
        assert str(pdb_path) in cmd
        assert "--temperature" in cmd
        assert "0.1" in cmd
        assert "--number_of_batches" in cmd
        assert "16" in cmd
        assert "--pack_side_chains" in cmd
        assert "1" in cmd
        assert "--pack_with_ligand_context" in cmd
        assert "--omit_aa" in cmd
        assert "C" in cmd
        assert "--seed" in cmd
        assert "42" in cmd
        assert "--fixed_residues" in cmd
        assert "100 200 300" in cmd

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_no_fixed_residues_flag_when_empty(self, mock_run: MagicMock, tmp_path: Path):
        out_dir = tmp_path / "output"
        _create_mock_ligandmpnn_output(out_dir)

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        pdb_path = _create_dummy_pdb(tmp_path / "input.pdb")

        run_ligandmpnn(
            pdb_path=pdb_path,
            output_dir=out_dir,
            temperature=0.2,
            n_batches=8,
            fixed_residues=[],
            ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
        )

        cmd = mock_run.call_args[0][0]
        assert "--fixed_residues" not in cmd

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_raises_on_nonzero_exit(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="CUDA error"
        )

        pdb_path = _create_dummy_pdb(tmp_path / "input.pdb")

        with pytest.raises(RuntimeError, match="LigandMPNN failed"):
            run_ligandmpnn(
                pdb_path=pdb_path,
                output_dir=tmp_path / "output",
                temperature=0.1,
                n_batches=16,
                fixed_residues=[],
                ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
            )

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_returns_fasta_and_scores_paths(self, mock_run: MagicMock, tmp_path: Path):
        out_dir = tmp_path / "output"
        _create_mock_ligandmpnn_output(out_dir)

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        pdb_path = _create_dummy_pdb(tmp_path / "input.pdb")

        fasta_path, scores_path = run_ligandmpnn(
            pdb_path=pdb_path,
            output_dir=out_dir,
            temperature=0.1,
            n_batches=16,
            fixed_residues=[],
            ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
        )

        assert fasta_path.suffix == ".fa"
        assert scores_path.name == "scores.json"


# ---------------------------------------------------------------------------
# Tests: deduplicate_sequences
# ---------------------------------------------------------------------------

class TestDeduplicateSequences:
    def test_removes_exact_duplicates(self):
        sequences = [
            {"sequence": "ACDEF", "sequence_id": "s1", "score": 0.5, "temperature": 0.1, "backbone": "b1"},
            {"sequence": "ACDEF", "sequence_id": "s2", "score": 0.3, "temperature": 0.2, "backbone": "b2"},
            {"sequence": "GHIJK", "sequence_id": "s3", "score": 0.7, "temperature": 0.1, "backbone": "b1"},
        ]

        result = deduplicate_sequences(sequences)

        assert len(result) == 2
        unique_seqs = {r["sequence"] for r in result}
        assert unique_seqs == {"ACDEF", "GHIJK"}

    def test_keeps_best_score(self):
        sequences = [
            {"sequence": "ACDEF", "sequence_id": "s1", "score": 0.5, "temperature": 0.1, "backbone": "b1"},
            {"sequence": "ACDEF", "sequence_id": "s2", "score": 0.3, "temperature": 0.2, "backbone": "b2"},
            {"sequence": "ACDEF", "sequence_id": "s3", "score": 0.8, "temperature": 0.1, "backbone": "b3"},
        ]

        result = deduplicate_sequences(sequences)

        assert len(result) == 1
        assert result[0]["score"] == 0.3
        assert result[0]["sequence_id"] == "s2"

    def test_no_duplicates(self):
        sequences = [
            {"sequence": "ACDEF", "sequence_id": "s1", "score": 0.5, "temperature": 0.1, "backbone": "b1"},
            {"sequence": "GHIJK", "sequence_id": "s2", "score": 0.3, "temperature": 0.2, "backbone": "b2"},
        ]

        result = deduplicate_sequences(sequences)

        assert len(result) == 2

    def test_empty_list(self):
        result = deduplicate_sequences([])
        assert result == []

    def test_single_sequence(self):
        sequences = [
            {"sequence": "ACDEF", "sequence_id": "s1", "score": 0.5, "temperature": 0.1, "backbone": "b1"},
        ]

        result = deduplicate_sequences(sequences)

        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: design_backbone
# ---------------------------------------------------------------------------

class TestDesignBackbone:
    @patch("pipeline.stage4_ligandmpnn.run_ligandmpnn")
    def test_runs_multiple_temperatures(self, mock_lmpnn: MagicMock, tmp_path: Path):
        pdb_path = _create_dummy_pdb(tmp_path / "backbone.pdb")

        # Setup mock outputs for each temperature call
        def create_output(pdb_path, output_dir, temperature, n_batches,
                         fixed_residues, ligandmpnn_dir, **kwargs):
            _create_mock_ligandmpnn_output(output_dir, n_seqs=2)
            fasta_path = output_dir / "seqs" / "designed.fa"
            scores_path = output_dir / "scores.json"
            return fasta_path, scores_path

        mock_lmpnn.side_effect = create_output

        enzyme_config = {"fixed_residues": [100, 200]}

        records = design_backbone(
            backbone_pdb=pdb_path,
            enzyme_config=enzyme_config,
            output_dir=tmp_path / "output",
            temperatures=[0.1, 0.2],
            n_batches_per_temp=16,
            ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
        )

        # Called once per temperature
        assert mock_lmpnn.call_count == 2

    @patch("pipeline.stage4_ligandmpnn.run_ligandmpnn")
    def test_returns_records_with_required_keys(self, mock_lmpnn: MagicMock, tmp_path: Path):
        pdb_path = _create_dummy_pdb(tmp_path / "backbone.pdb")

        def create_output(pdb_path, output_dir, **kwargs):
            _create_mock_ligandmpnn_output(output_dir, n_seqs=2)
            fasta_path = output_dir / "seqs" / "designed.fa"
            scores_path = output_dir / "scores.json"
            return fasta_path, scores_path

        mock_lmpnn.side_effect = create_output

        records = design_backbone(
            backbone_pdb=pdb_path,
            enzyme_config={"fixed_residues": []},
            output_dir=tmp_path / "output",
            temperatures=[0.1],
            n_batches_per_temp=8,
            ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
        )

        assert len(records) > 0
        for r in records:
            assert "sequence" in r
            assert "sequence_id" in r
            assert "score" in r
            assert "temperature" in r
            assert "backbone" in r

    @patch("pipeline.stage4_ligandmpnn.run_ligandmpnn")
    def test_passes_fixed_residues(self, mock_lmpnn: MagicMock, tmp_path: Path):
        pdb_path = _create_dummy_pdb(tmp_path / "backbone.pdb")

        def create_output(pdb_path, output_dir, **kwargs):
            _create_mock_ligandmpnn_output(output_dir, n_seqs=1)
            return output_dir / "seqs" / "designed.fa", output_dir / "scores.json"

        mock_lmpnn.side_effect = create_output

        design_backbone(
            backbone_pdb=pdb_path,
            enzyme_config={"fixed_residues": [10, 20, 30]},
            output_dir=tmp_path / "output",
            temperatures=[0.1],
            n_batches_per_temp=8,
            ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
        )

        call_kwargs = mock_lmpnn.call_args_list[0]
        assert call_kwargs.kwargs.get("fixed_residues") == [10, 20, 30] or \
               call_kwargs[1].get("fixed_residues") == [10, 20, 30]


# ---------------------------------------------------------------------------
# Tests: design_wt_branch
# ---------------------------------------------------------------------------

class TestDesignWtBranch:
    @patch("pipeline.stage4_ligandmpnn.run_ligandmpnn")
    def test_runs_three_temperatures(self, mock_lmpnn: MagicMock, tmp_path: Path):
        wt_pdb = _create_dummy_pdb(tmp_path / "wt.pdb")

        def create_output(pdb_path, output_dir, **kwargs):
            _create_mock_ligandmpnn_output(output_dir, n_seqs=3)
            return output_dir / "seqs" / "designed.fa", output_dir / "scores.json"

        mock_lmpnn.side_effect = create_output

        records = design_wt_branch(
            wt_pdb=wt_pdb,
            enzyme_id="enzyme_A",
            enzyme_config={"fixed_residues": [100]},
            output_dir=tmp_path / "wt_output",
            n_sequences_per_temp=250,
            ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
        )

        # Should run at T=0.1, 0.2, 0.4
        assert mock_lmpnn.call_count == 3

    @patch("pipeline.stage4_ligandmpnn.run_ligandmpnn")
    def test_wt_temperatures_are_correct(self, mock_lmpnn: MagicMock, tmp_path: Path):
        wt_pdb = _create_dummy_pdb(tmp_path / "wt.pdb")

        def create_output(pdb_path, output_dir, **kwargs):
            _create_mock_ligandmpnn_output(output_dir, n_seqs=1)
            return output_dir / "seqs" / "designed.fa", output_dir / "scores.json"

        mock_lmpnn.side_effect = create_output

        design_wt_branch(
            wt_pdb=wt_pdb,
            enzyme_id="enzyme_A",
            enzyme_config={"fixed_residues": []},
            output_dir=tmp_path / "wt_output",
            n_sequences_per_temp=250,
            ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
        )

        temps_used = [
            c.kwargs.get("temperature") for c in mock_lmpnn.call_args_list
        ]
        assert temps_used == [0.1, 0.2, 0.4]

    @patch("pipeline.stage4_ligandmpnn.run_ligandmpnn")
    def test_returns_records_with_wt_backbone_name(self, mock_lmpnn: MagicMock, tmp_path: Path):
        wt_pdb = _create_dummy_pdb(tmp_path / "wt.pdb")

        def create_output(pdb_path, output_dir, **kwargs):
            _create_mock_ligandmpnn_output(output_dir, n_seqs=2)
            return output_dir / "seqs" / "designed.fa", output_dir / "scores.json"

        mock_lmpnn.side_effect = create_output

        records = design_wt_branch(
            wt_pdb=wt_pdb,
            enzyme_id="enzyme_A",
            enzyme_config={"fixed_residues": []},
            output_dir=tmp_path / "wt_output",
            n_sequences_per_temp=250,
            ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
        )

        assert len(records) > 0
        for r in records:
            assert "enzyme_A_wt" in r["backbone"]


# ---------------------------------------------------------------------------
# Tests: run_stage
# ---------------------------------------------------------------------------

class TestRunStage:
    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_processes_both_branches(self, mock_run: MagicMock, tmp_path: Path):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        # Create backbone PDBs for Branch A (enzyme_A only)
        bb_dir = base_dir / "stage_03_rfdiffusion3" / "enzyme_A"
        bb_dir.mkdir(parents=True)
        _create_dummy_pdb(bb_dir / "backbone_001.pdb")

        # Create WT PDBs for Branch B
        wt_sub_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
        wt_sub_dir.mkdir(parents=True)
        _create_dummy_pdb(wt_sub_dir / "enzyme_A_substrate.pdb")
        _create_dummy_pdb(wt_sub_dir / "enzyme_B_substrate.pdb")

        def fake_run(cmd, **kwargs):
            # Parse output dir from command
            out_idx = cmd.index("--out_folder") + 1
            out_dir = Path(cmd[out_idx])
            _create_mock_ligandmpnn_output(out_dir, n_seqs=2)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        results = run_stage(config)

        assert results["total_sequences"] > 0
        assert "enzyme_A" in results["per_enzyme"]
        assert "enzyme_B" in results["per_enzyme"]
        assert results["per_enzyme"]["enzyme_A"] > 0

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_writes_fasta_files(self, mock_run: MagicMock, tmp_path: Path):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        # Create WT PDBs only (Branch B)
        wt_sub_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
        wt_sub_dir.mkdir(parents=True)
        _create_dummy_pdb(wt_sub_dir / "enzyme_A_substrate.pdb")

        def fake_run(cmd, **kwargs):
            out_idx = cmd.index("--out_folder") + 1
            out_dir = Path(cmd[out_idx])
            _create_mock_ligandmpnn_output(out_dir, n_seqs=2)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        results = run_stage(config)

        fasta_path = Path(results["fasta_paths"]["enzyme_A"])
        assert fasta_path.exists()
        assert fasta_path.suffix == ".fasta"

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_output_csv_has_required_columns(self, mock_run: MagicMock, tmp_path: Path):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        # Create WT PDBs
        wt_sub_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
        wt_sub_dir.mkdir(parents=True)
        _create_dummy_pdb(wt_sub_dir / "enzyme_A_substrate.pdb")

        def fake_run(cmd, **kwargs):
            out_idx = cmd.index("--out_folder") + 1
            out_dir = Path(cmd[out_idx])
            _create_mock_ligandmpnn_output(out_dir, n_seqs=2)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        run_stage(config)

        csv_path = base_dir / "stage_04_ligandmpnn" / "ligandmpnn_scores.csv"
        assert csv_path.exists()

        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            assert len(rows) > 0

            required_cols = {
                "sequence_id", "enzyme_id", "backbone_pdb",
                "temperature", "ligandmpnn_score", "design_branch",
            }
            assert required_cols.issubset(set(reader.fieldnames))

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_csv_contains_design_branch_values(self, mock_run: MagicMock, tmp_path: Path):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        # Create both backbone and WT PDBs for enzyme_A
        bb_dir = base_dir / "stage_03_rfdiffusion3" / "enzyme_A"
        bb_dir.mkdir(parents=True)
        _create_dummy_pdb(bb_dir / "backbone_001.pdb")

        wt_sub_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
        wt_sub_dir.mkdir(parents=True)
        _create_dummy_pdb(wt_sub_dir / "enzyme_A_substrate.pdb")

        def fake_run(cmd, **kwargs):
            out_idx = cmd.index("--out_folder") + 1
            out_dir = Path(cmd[out_idx])
            _create_mock_ligandmpnn_output(out_dir, n_seqs=2)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        run_stage(config)

        csv_path = base_dir / "stage_04_ligandmpnn" / "ligandmpnn_scores.csv"
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            branches = {row["design_branch"] for row in reader}

        assert "rfd3_backbone" in branches
        assert "wt_direct" in branches

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_creates_output_directory(self, mock_run: MagicMock, tmp_path: Path):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        # No inputs → no sequences but directory should exist
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        run_stage(config)

        output_dir = base_dir / "stage_04_ligandmpnn"
        assert output_dir.exists()

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_handles_missing_backbones_gracefully(self, mock_run: MagicMock, tmp_path: Path):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        # Only WT exists, no backbone dir
        wt_sub_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
        wt_sub_dir.mkdir(parents=True)
        _create_dummy_pdb(wt_sub_dir / "enzyme_A_substrate.pdb")

        def fake_run(cmd, **kwargs):
            out_idx = cmd.index("--out_folder") + 1
            out_dir = Path(cmd[out_idx])
            _create_mock_ligandmpnn_output(out_dir, n_seqs=1)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        # Should not raise
        results = run_stage(config)
        assert results["total_sequences"] >= 0

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_returns_expected_keys(self, mock_run: MagicMock, tmp_path: Path):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        wt_sub_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
        wt_sub_dir.mkdir(parents=True)
        _create_dummy_pdb(wt_sub_dir / "enzyme_A_substrate.pdb")

        def fake_run(cmd, **kwargs):
            out_idx = cmd.index("--out_folder") + 1
            out_dir = Path(cmd[out_idx])
            _create_mock_ligandmpnn_output(out_dir, n_seqs=2)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        results = run_stage(config)

        assert "total_sequences" in results
        assert "per_enzyme" in results
        assert "fasta_paths" in results
        assert isinstance(results["total_sequences"], int)
        assert isinstance(results["per_enzyme"], dict)
        assert isinstance(results["fasta_paths"], dict)
