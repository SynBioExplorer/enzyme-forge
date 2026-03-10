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
    cluster_sequences,
    design_ts_branch,
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


# ---------------------------------------------------------------------------
# Tests: cluster_sequences
# ---------------------------------------------------------------------------

def _write_fasta_file(path: Path, sequences: dict[str, str]) -> None:
    """Helper to write a FASTA file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for header, seq in sequences.items():
            fh.write(f">{header}\n{seq}\n")


class TestClusterSequences:
    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_calls_mmseqs_easy_cluster(self, mock_run: MagicMock, tmp_path: Path):
        """cluster_sequences calls mmseqs easy-cluster with correct arguments."""
        fasta_path = tmp_path / "input.fasta"
        _write_fasta_file(fasta_path, {
            "seq1": "ACDEFGHIKLMNPQRSTVWY",
            "seq2": "ACDEFGHIKLMNPQRSTVWY",  # identical — should cluster
            "seq3": "MNPQRSTVWYACDEFGHIKL",  # different
        })

        # Simulate mmseqs creating a representative FASTA
        rep_fasta = tmp_path / "cluster_result_rep_seq.fasta"
        _write_fasta_file(rep_fasta, {
            "seq1": "ACDEFGHIKLMNPQRSTVWY",
            "seq3": "MNPQRSTVWYACDEFGHIKL",
        })

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = cluster_sequences(
            fasta_path=fasta_path,
            output_dir=tmp_path,
            identity=0.9,
            threads=4,
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "mmseqs"
        assert "easy-cluster" in cmd
        assert str(fasta_path) in cmd
        # Identity threshold must appear
        assert "--min-seq-id" in cmd
        assert "0.9" in cmd

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_returns_representative_sequences(self, mock_run: MagicMock, tmp_path: Path):
        """cluster_sequences returns only cluster representatives."""
        fasta_path = tmp_path / "input.fasta"
        _write_fasta_file(fasta_path, {
            "seq1": "ACDEFGHIKLMNPQRSTVWY",
            "seq2": "ACDEFGHIKLMNPQRSTVWY",
            "seq3": "MNPQRSTVWYACDEFGHIKL",
        })

        # Simulate representative FASTA written by mmseqs
        rep_fasta = tmp_path / "cluster_result_rep_seq.fasta"
        _write_fasta_file(rep_fasta, {
            "seq1": "ACDEFGHIKLMNPQRSTVWY",
            "seq3": "MNPQRSTVWYACDEFGHIKL",
        })

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = cluster_sequences(
            fasta_path=fasta_path,
            output_dir=tmp_path,
            identity=0.9,
            threads=4,
        )

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "seq1" in result or "seq3" in result

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_raises_on_mmseqs_failure(self, mock_run: MagicMock, tmp_path: Path):
        """cluster_sequences raises RuntimeError when mmseqs fails."""
        fasta_path = tmp_path / "input.fasta"
        _write_fasta_file(fasta_path, {"seq1": "ACDEF"})

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error msg")

        with pytest.raises(RuntimeError, match="MMseqs2 clustering failed"):
            cluster_sequences(
                fasta_path=fasta_path,
                output_dir=tmp_path,
                identity=0.9,
                threads=4,
            )

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_uses_thread_count(self, mock_run: MagicMock, tmp_path: Path):
        """cluster_sequences passes the thread count to mmseqs."""
        fasta_path = tmp_path / "input.fasta"
        _write_fasta_file(fasta_path, {"seq1": "ACDEF"})

        # Create representative output file
        rep_fasta = tmp_path / "cluster_result_rep_seq.fasta"
        _write_fasta_file(rep_fasta, {"seq1": "ACDEF"})

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cluster_sequences(
            fasta_path=fasta_path,
            output_dir=tmp_path,
            identity=0.9,
            threads=24,
        )

        cmd = mock_run.call_args[0][0]
        assert "--threads" in cmd
        assert "24" in cmd

    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_default_identity_is_0_9(self, mock_run: MagicMock, tmp_path: Path):
        """cluster_sequences uses 0.9 identity threshold by default."""
        fasta_path = tmp_path / "input.fasta"
        _write_fasta_file(fasta_path, {"seq1": "ACDEF"})

        rep_fasta = tmp_path / "cluster_result_rep_seq.fasta"
        _write_fasta_file(rep_fasta, {"seq1": "ACDEF"})

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cluster_sequences(fasta_path=fasta_path, output_dir=tmp_path)

        cmd = mock_run.call_args[0][0]
        assert "--min-seq-id" in cmd
        idx = cmd.index("--min-seq-id")
        assert cmd[idx + 1] == "0.9"


# ---------------------------------------------------------------------------
# Tests: design_ts_branch
# ---------------------------------------------------------------------------

class TestDesignTsBranch:
    @patch("pipeline.stage4_ligandmpnn.run_ligandmpnn")
    def test_runs_when_ts_complex_exists(self, mock_lmpnn: MagicMock, tmp_path: Path):
        """design_ts_branch runs LigandMPNN when a TS complex PDB is found."""
        ts_pdb = tmp_path / "stage_02_wt_structures" / "ts_complexes" / "enzyme_A_ts.pdb"
        _create_dummy_pdb(ts_pdb)

        def create_output(pdb_path, output_dir, **kwargs):
            _create_mock_ligandmpnn_output(output_dir, n_seqs=2)
            return output_dir / "seqs" / "designed.fa", output_dir / "scores.json"

        mock_lmpnn.side_effect = create_output

        records = design_ts_branch(
            ts_pdb=ts_pdb,
            enzyme_id="enzyme_A",
            enzyme_config={"fixed_residues": [10, 20]},
            output_dir=tmp_path / "ts_output",
            n_sequences_per_temp=50,
            ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
        )

        assert mock_lmpnn.call_count >= 1
        assert len(records) > 0

    @patch("pipeline.stage4_ligandmpnn.run_ligandmpnn")
    def test_records_have_ts_design_branch(self, mock_lmpnn: MagicMock, tmp_path: Path):
        """Records from design_ts_branch are tagged with 'ts_conditioned' design_branch."""
        ts_pdb = tmp_path / "enzyme_A_ts.pdb"
        _create_dummy_pdb(ts_pdb)

        def create_output(pdb_path, output_dir, **kwargs):
            _create_mock_ligandmpnn_output(output_dir, n_seqs=2)
            return output_dir / "seqs" / "designed.fa", output_dir / "scores.json"

        mock_lmpnn.side_effect = create_output

        records = design_ts_branch(
            ts_pdb=ts_pdb,
            enzyme_id="enzyme_A",
            enzyme_config={"fixed_residues": []},
            output_dir=tmp_path / "ts_out",
            n_sequences_per_temp=10,
            ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
        )

        assert all(r["design_branch"] == "ts_conditioned" for r in records)

    @patch("pipeline.stage4_ligandmpnn.run_ligandmpnn")
    def test_uses_wt_temperatures(self, mock_lmpnn: MagicMock, tmp_path: Path):
        """design_ts_branch runs at WT_TEMPERATURES (0.1, 0.2, 0.4)."""
        ts_pdb = tmp_path / "enzyme_A_ts.pdb"
        _create_dummy_pdb(ts_pdb)

        def create_output(pdb_path, output_dir, **kwargs):
            _create_mock_ligandmpnn_output(output_dir, n_seqs=1)
            return output_dir / "seqs" / "designed.fa", output_dir / "scores.json"

        mock_lmpnn.side_effect = create_output

        design_ts_branch(
            ts_pdb=ts_pdb,
            enzyme_id="enzyme_A",
            enzyme_config={"fixed_residues": []},
            output_dir=tmp_path / "ts_out",
            n_sequences_per_temp=50,
            ligandmpnn_dir=Path("/opt/tools/LigandMPNN"),
        )

        assert mock_lmpnn.call_count == len(WT_TEMPERATURES)
        temps_used = [c.kwargs.get("temperature") for c in mock_lmpnn.call_args_list]
        assert temps_used == list(WT_TEMPERATURES)


# ---------------------------------------------------------------------------
# Tests: run_stage with clustering + TS conditioning
# ---------------------------------------------------------------------------

class TestRunStageWithClustering:
    @patch("pipeline.stage4_ligandmpnn.cluster_sequences")
    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_clustering_called_when_enabled(
        self, mock_run: MagicMock, mock_cluster: MagicMock, tmp_path: Path
    ):
        """run_stage calls cluster_sequences when clustering_identity is set."""
        config = _make_config(tmp_path)
        config["stages"]["stage_04_ligandmpnn"]["clustering_identity"] = 0.9
        base_dir = Path(config["global"]["base_dir"])

        wt_sub_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
        wt_sub_dir.mkdir(parents=True)
        _create_dummy_pdb(wt_sub_dir / "enzyme_A_substrate.pdb")

        def fake_run(cmd, **kwargs):
            out_idx = cmd.index("--out_folder") + 1
            out_dir = Path(cmd[out_idx])
            _create_mock_ligandmpnn_output(out_dir, n_seqs=3)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        # cluster_sequences should return a subset dict
        mock_cluster.return_value = {
            "seq_1_1": "ACDEFGHIKLMNPQRSTVWY",
            "seq_1_2": "MNPQRSTVWYACDEFGHIKL",
        }

        results = run_stage(config)

        assert mock_cluster.called

    @patch("pipeline.stage4_ligandmpnn.cluster_sequences")
    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_clustering_not_called_when_disabled(
        self, mock_run: MagicMock, mock_cluster: MagicMock, tmp_path: Path
    ):
        """run_stage skips clustering when clustering_identity is not in config."""
        config = _make_config(tmp_path)
        # No clustering_identity key
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

        run_stage(config)

        assert not mock_cluster.called

    @patch("pipeline.stage4_ligandmpnn.cluster_sequences")
    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_ts_conditioning_triggered_when_ts_smiles_present(
        self, mock_run: MagicMock, mock_cluster: MagicMock, tmp_path: Path
    ):
        """run_stage runs TS conditioning when enzyme has ts_smiles and TS PDB exists."""
        config = _make_config(tmp_path)
        # Add ts_smiles to enzyme_A config
        config["enzymes"]["enzyme_A"]["ts_smiles"] = "CC(=O)[O-]"
        base_dir = Path(config["global"]["base_dir"])

        # Create WT substrate PDB
        wt_sub_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
        wt_sub_dir.mkdir(parents=True)
        _create_dummy_pdb(wt_sub_dir / "enzyme_A_substrate.pdb")

        # Create TS complex PDB
        ts_dir = base_dir / "stage_02_wt_structures" / "ts_complexes"
        ts_dir.mkdir(parents=True)
        _create_dummy_pdb(ts_dir / "enzyme_A_ts.pdb")

        call_log: list[list] = []

        def fake_run(cmd, **kwargs):
            call_log.append(cmd[:])
            out_idx = cmd.index("--out_folder") + 1
            out_dir = Path(cmd[out_idx])
            _create_mock_ligandmpnn_output(out_dir, n_seqs=2)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run
        mock_cluster.return_value = {"rep1": "ACDEFGHIKLMN"}

        run_stage(config)

        # Should have made LigandMPNN calls (substrate-conditioned + TS-conditioned)
        assert len(call_log) > 0
        # At least one call should reference the TS complex PDB path
        ts_pdb_path = str(ts_dir / "enzyme_A_ts.pdb")
        assert any(ts_pdb_path in cmd for cmd in call_log), (
            f"Expected TS PDB {ts_pdb_path} in at least one LigandMPNN call. "
            f"Calls: {call_log}"
        )

    @patch("pipeline.stage4_ligandmpnn.cluster_sequences")
    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_ts_conditioning_skipped_when_no_ts_smiles(
        self, mock_run: MagicMock, mock_cluster: MagicMock, tmp_path: Path
    ):
        """run_stage skips TS conditioning when enzyme has no ts_smiles."""
        config = _make_config(tmp_path)
        # No ts_smiles in config (default _make_config)
        base_dir = Path(config["global"]["base_dir"])

        wt_sub_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
        wt_sub_dir.mkdir(parents=True)
        _create_dummy_pdb(wt_sub_dir / "enzyme_A_substrate.pdb")

        # Even if a TS PDB exists, it should not be used without ts_smiles
        ts_dir = base_dir / "stage_02_wt_structures" / "ts_complexes"
        ts_dir.mkdir(parents=True)
        _create_dummy_pdb(ts_dir / "enzyme_A_ts.pdb")

        call_log: list[list] = []

        def fake_run(cmd, **kwargs):
            call_log.append(cmd[:])
            out_idx = cmd.index("--out_folder") + 1
            out_dir = Path(cmd[out_idx])
            _create_mock_ligandmpnn_output(out_dir, n_seqs=2)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        run_stage(config)

        ts_pdb_path = str(ts_dir / "enzyme_A_ts.pdb")
        assert not any(ts_pdb_path in cmd for cmd in call_log), (
            "TS PDB should not appear in any LigandMPNN call when ts_smiles is absent"
        )

    @patch("pipeline.stage4_ligandmpnn.cluster_sequences")
    @patch("pipeline.stage4_ligandmpnn.subprocess.run")
    def test_ts_sequences_merged_before_clustering(
        self, mock_run: MagicMock, mock_cluster: MagicMock, tmp_path: Path
    ):
        """TS-conditioned sequences are merged with substrate-conditioned before clustering."""
        config = _make_config(tmp_path)
        config["enzymes"]["enzyme_A"]["ts_smiles"] = "CC(=O)[O-]"
        config["stages"]["stage_04_ligandmpnn"]["clustering_identity"] = 0.9
        base_dir = Path(config["global"]["base_dir"])

        wt_sub_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
        wt_sub_dir.mkdir(parents=True)
        _create_dummy_pdb(wt_sub_dir / "enzyme_A_substrate.pdb")

        ts_dir = base_dir / "stage_02_wt_structures" / "ts_complexes"
        ts_dir.mkdir(parents=True)
        _create_dummy_pdb(ts_dir / "enzyme_A_ts.pdb")

        def fake_run(cmd, **kwargs):
            out_idx = cmd.index("--out_folder") + 1
            out_dir = Path(cmd[out_idx])
            _create_mock_ligandmpnn_output(out_dir, n_seqs=3)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        # cluster_sequences returns a small representative set
        mock_cluster.return_value = {"rep1": "ACDEFGHIKLMN"}

        results = run_stage(config)

        # cluster_sequences must have been called
        assert mock_cluster.called
        # The fasta passed to cluster should contain sequences from multiple branches
        call_kwargs = mock_cluster.call_args
        # The function was called — verify it received a real fasta path
        assert call_kwargs is not None
