"""Tests for Stage 2: Wild-type Structure Prediction (pipeline/stage2_wt_structures.py).

All tests run without Boltz-2 or GPU.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.stage2_wt_structures import (
    STAGE_NAME,
    STAGE_NUM,
    INPUT_DESCRIPTION,
    OUTPUT_DESCRIPTION,
    prepare_boltz_input,
    run_boltz2,
    select_best_model,
    run_boltz2_for_complex,
    run_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> dict:
    """Create a minimal config for testing."""
    base_dir = tmp_path / "pipeline"
    base_dir.mkdir()
    seq_dir = base_dir / "inputs" / "sequences"
    seq_dir.mkdir(parents=True)

    for eid in ("enzyme_A", "enzyme_B"):
        fasta = seq_dir / f"{eid}.fasta"
        fasta.write_text(f">{eid}\nMVLSPADKTNVKAAWGKVGA\n")

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
            "stage_02_wt_structures": {
                "output_dir": "stage_02_wt_structures",
                "recycling_steps": 10,
                "num_seeds": 5,
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


def _create_mock_boltz_output(output_dir: Path, num_seeds: int = 3) -> None:
    """Create mock Boltz-2 output files (PDBs + confidence JSONs)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_seeds):
        pdb_path = output_dir / f"model_{i}.pdb"
        pdb_path.write_text(
            f"ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00\nEND\n"
        )

        conf_path = output_dir / f"model_{i}_confidence.json"
        conf_data = {
            "plddt": 0.80 + i * 0.02,
            "iptm": 0.60 + i * 0.05,
            "ptm": 0.70 + i * 0.03,
            "complex_plddt": 0.78 + i * 0.01,
        }
        conf_path.write_text(json.dumps(conf_data))


# ---------------------------------------------------------------------------
# Tests: module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_stage_name(self):
        assert STAGE_NAME == "Stage 2: Wild-type Structure Prediction"

    def test_stage_num(self):
        assert STAGE_NUM == "2"

    def test_input_description(self):
        assert "FASTA" in INPUT_DESCRIPTION
        assert "SMILES" in INPUT_DESCRIPTION

    def test_output_description(self):
        assert "PDB" in OUTPUT_DESCRIPTION
        assert "substrate" in OUTPUT_DESCRIPTION
        assert "product" in OUTPUT_DESCRIPTION


# ---------------------------------------------------------------------------
# Tests: prepare_boltz_input
# ---------------------------------------------------------------------------

class TestPrepareBoltzInput:
    def test_creates_input_file(self, tmp_path: Path):
        path = prepare_boltz_input(
            enzyme_id="enzyme_A",
            sequence="MVLSPADKTNVKAAWGKVGA",
            ligand_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            cofactors=["NAD+"],
            output_dir=tmp_path / "boltz_input",
        )
        assert path.exists()
        assert path.suffix == ".fasta"

    def test_file_contains_protein_section(self, tmp_path: Path):
        path = prepare_boltz_input(
            enzyme_id="enzyme_A",
            sequence="MVLSPADKTNVKAAWGKVGA",
            ligand_smiles="CC(=O)O",
            cofactors=[],
            output_dir=tmp_path,
        )
        content = path.read_text()
        assert ">protein|enzyme_A" in content
        assert "MVLSPADKTNVKAAWGKVGA" in content

    def test_file_contains_ligand_smiles(self, tmp_path: Path):
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        path = prepare_boltz_input(
            enzyme_id="enzyme_A",
            sequence="ACDEF",
            ligand_smiles=smiles,
            cofactors=[],
            output_dir=tmp_path,
        )
        content = path.read_text()
        assert ">ligand|SMILES" in content
        assert smiles in content

    def test_file_contains_cofactors(self, tmp_path: Path):
        path = prepare_boltz_input(
            enzyme_id="enzyme_A",
            sequence="ACDEF",
            ligand_smiles="C",
            cofactors=["NAD+", "FAD"],
            output_dir=tmp_path,
        )
        content = path.read_text()
        assert ">cofactor|NAD+" in content
        assert ">cofactor|FAD" in content

    def test_no_cofactors(self, tmp_path: Path):
        path = prepare_boltz_input(
            enzyme_id="enzyme_A",
            sequence="ACDEF",
            ligand_smiles="C",
            cofactors=[],
            output_dir=tmp_path,
        )
        content = path.read_text()
        assert "cofactor" not in content

    def test_creates_output_directory(self, tmp_path: Path):
        out = tmp_path / "nested" / "dir"
        prepare_boltz_input(
            enzyme_id="enzyme_A",
            sequence="ACDEF",
            ligand_smiles="C",
            cofactors=[],
            output_dir=out,
        )
        assert out.exists()

    def test_correct_file_format_order(self, tmp_path: Path):
        """Verify protein > ligand > cofactor ordering in the file."""
        path = prepare_boltz_input(
            enzyme_id="enzyme_A",
            sequence="ACDEF",
            ligand_smiles="CCO",
            cofactors=["NAD+"],
            output_dir=tmp_path,
        )
        content = path.read_text()
        protein_pos = content.index(">protein|")
        ligand_pos = content.index(">ligand|")
        cofactor_pos = content.index(">cofactor|")
        assert protein_pos < ligand_pos < cofactor_pos


# ---------------------------------------------------------------------------
# Tests: run_boltz2
# ---------------------------------------------------------------------------

class TestRunBoltz2:
    @patch("pipeline.stage2_wt_structures.subprocess.run")
    def test_constructs_correct_command(self, mock_run: MagicMock, tmp_path: Path):
        out_dir = tmp_path / "output"
        _create_mock_boltz_output(out_dir)

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_fasta = tmp_path / "input.fasta"
        input_fasta.write_text(">protein|test\nACDEF\n>ligand|SMILES\nCCO\n")

        run_boltz2(input_fasta, out_dir, recycling_steps=10, num_seeds=5)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "boltz"
        assert cmd[1] == "predict"
        assert str(input_fasta) in cmd
        assert "--recycling_steps" in cmd
        assert "10" in cmd
        assert "--num_seeds" in cmd
        assert "5" in cmd
        assert "--output_dir" in cmd
        assert "--use_msa_server" in cmd
        assert "false" in cmd

    @patch("pipeline.stage2_wt_structures.subprocess.run")
    def test_raises_on_nonzero_exit(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="CUDA out of memory"
        )
        input_fasta = tmp_path / "input.fasta"
        input_fasta.write_text(">protein|test\nACDEF\n")

        with pytest.raises(RuntimeError, match="Boltz-2 prediction failed"):
            run_boltz2(input_fasta, tmp_path / "output")

    @patch("pipeline.stage2_wt_structures.subprocess.run")
    def test_returns_pdb_and_json_paths(self, mock_run: MagicMock, tmp_path: Path):
        out_dir = tmp_path / "output"
        _create_mock_boltz_output(out_dir, num_seeds=3)

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_fasta = tmp_path / "input.fasta"
        input_fasta.write_text(">protein|test\nACDEF\n")

        result = run_boltz2(input_fasta, out_dir)

        assert "pdb_files" in result
        assert "confidence_jsons" in result
        assert len(result["pdb_files"]) == 3
        assert len(result["confidence_jsons"]) == 3
        assert all(p.suffix == ".pdb" for p in result["pdb_files"])
        assert all(p.suffix == ".json" for p in result["confidence_jsons"])

    @patch("pipeline.stage2_wt_structures.subprocess.run")
    def test_custom_boltz_bin(self, mock_run: MagicMock, tmp_path: Path):
        out_dir = tmp_path / "output"
        _create_mock_boltz_output(out_dir)

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_fasta = tmp_path / "input.fasta"
        input_fasta.write_text(">protein|test\nACDEF\n")

        run_boltz2(input_fasta, out_dir, boltz_bin="/opt/boltz/bin/boltz")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/opt/boltz/bin/boltz"


# ---------------------------------------------------------------------------
# Tests: select_best_model
# ---------------------------------------------------------------------------

class TestSelectBestModel:
    def test_picks_highest_iptm(self, tmp_path: Path):
        _create_mock_boltz_output(tmp_path, num_seeds=3)
        # Seeds have iPTM 0.60, 0.65, 0.70 — best is model_2

        best_pdb, confidence = select_best_model(tmp_path)

        assert confidence["iptm"] == 0.70
        assert best_pdb.exists()
        assert best_pdb.suffix == ".pdb"

    def test_returns_confidence_dict_with_expected_keys(self, tmp_path: Path):
        _create_mock_boltz_output(tmp_path, num_seeds=1)

        _, confidence = select_best_model(tmp_path)

        assert "plddt" in confidence
        assert "iptm" in confidence
        assert "ptm" in confidence

    def test_raises_if_no_confidence_files(self, tmp_path: Path):
        tmp_path.mkdir(exist_ok=True)
        # No files at all
        with pytest.raises(FileNotFoundError, match="No confidence JSON"):
            select_best_model(tmp_path)

    def test_single_model(self, tmp_path: Path):
        _create_mock_boltz_output(tmp_path, num_seeds=1)

        best_pdb, confidence = select_best_model(tmp_path)

        assert confidence["iptm"] == 0.60
        assert best_pdb.exists()

    def test_handles_nested_output_dirs(self, tmp_path: Path):
        """Boltz-2 may place outputs in nested directories."""
        nested = tmp_path / "boltz_results" / "predictions"
        _create_mock_boltz_output(nested, num_seeds=2)

        best_pdb, confidence = select_best_model(tmp_path)

        assert confidence["iptm"] == 0.65
        assert best_pdb.exists()


# ---------------------------------------------------------------------------
# Tests: run_boltz2_for_complex
# ---------------------------------------------------------------------------

class TestRunBoltz2ForComplex:
    @patch("pipeline.stage2_wt_structures.run_boltz2")
    def test_returns_expected_keys(self, mock_boltz: MagicMock, tmp_path: Path):
        # Setup: create mock output files that select_best_model will find
        pred_dir = (
            tmp_path / "work" / "enzyme_A_substrate" / "predictions"
        )
        _create_mock_boltz_output(pred_dir, num_seeds=2)

        mock_boltz.return_value = {
            "pdb_files": list(pred_dir.glob("*.pdb")),
            "confidence_jsons": list(pred_dir.glob("*.json")),
        }

        config = _make_config(tmp_path)

        result = run_boltz2_for_complex(
            enzyme_id="enzyme_A",
            sequence="MVLSPADKTNVKAAWGKVGA",
            ligand_smiles="CC(=O)O",
            cofactors=["NAD+"],
            output_dir=tmp_path / "work",
            ligand_type="substrate",
            config=config,
        )

        assert result["enzyme_id"] == "enzyme_A"
        assert result["ligand_type"] == "substrate"
        assert "pdb_path" in result
        assert "confidence" in result
        assert "iptm" in result["confidence"]

    @patch("pipeline.stage2_wt_structures.run_boltz2")
    def test_calls_boltz2_with_config_params(self, mock_boltz: MagicMock, tmp_path: Path):
        pred_dir = tmp_path / "work" / "enzyme_A_substrate" / "predictions"
        _create_mock_boltz_output(pred_dir, num_seeds=1)

        mock_boltz.return_value = {
            "pdb_files": list(pred_dir.glob("*.pdb")),
            "confidence_jsons": list(pred_dir.glob("*.json")),
        }

        config = _make_config(tmp_path)
        config["stages"]["stage_02_wt_structures"]["recycling_steps"] = 15
        config["stages"]["stage_02_wt_structures"]["num_seeds"] = 3

        run_boltz2_for_complex(
            enzyme_id="enzyme_A",
            sequence="ACDEF",
            ligand_smiles="C",
            cofactors=[],
            output_dir=tmp_path / "work",
            ligand_type="substrate",
            config=config,
        )

        mock_boltz.assert_called_once()
        _, kwargs = mock_boltz.call_args
        assert kwargs["recycling_steps"] == 15
        assert kwargs["num_seeds"] == 3


# ---------------------------------------------------------------------------
# Tests: run_stage
# ---------------------------------------------------------------------------

class TestRunStage:
    @patch("pipeline.stage2_wt_structures.run_boltz2_for_complex")
    def test_creates_output_directory_structure(
        self, mock_complex: MagicMock, tmp_path: Path
    ):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        # Create a dummy PDB that can be copied
        dummy_pdb = tmp_path / "dummy.pdb"
        dummy_pdb.write_text("ATOM      1  CA  ALA A   1\nEND\n")

        mock_complex.return_value = {
            "enzyme_id": "enzyme_A",
            "ligand_type": "substrate",
            "pdb_path": str(dummy_pdb),
            "confidence": {"plddt": 0.85, "iptm": 0.72, "ptm": 0.78},
        }

        run_stage(config)

        substrate_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
        product_dir = base_dir / "stage_02_wt_structures" / "product_complexes"
        assert substrate_dir.exists()
        assert product_dir.exists()

    @patch("pipeline.stage2_wt_structures.run_boltz2_for_complex")
    def test_processes_both_substrate_and_product(
        self, mock_complex: MagicMock, tmp_path: Path
    ):
        config = _make_config(tmp_path)

        dummy_pdb = tmp_path / "dummy.pdb"
        dummy_pdb.write_text("ATOM      1  CA  ALA A   1\nEND\n")

        call_count = {"substrate": 0, "product": 0}

        def fake_complex(enzyme_id, sequence, ligand_smiles, cofactors,
                         output_dir, ligand_type, config):
            call_count[ligand_type] += 1
            return {
                "enzyme_id": enzyme_id,
                "ligand_type": ligand_type,
                "pdb_path": str(dummy_pdb),
                "confidence": {"plddt": 0.85, "iptm": 0.72, "ptm": 0.78},
            }

        mock_complex.side_effect = fake_complex

        results = run_stage(config)

        # 2 enzymes × 2 ligand types = 4 calls
        assert mock_complex.call_count == 4
        # Each type called once per enzyme
        assert call_count["substrate"] == 2
        assert call_count["product"] == 2

    @patch("pipeline.stage2_wt_structures.run_boltz2_for_complex")
    def test_copies_pdb_to_final_location(
        self, mock_complex: MagicMock, tmp_path: Path
    ):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        dummy_pdb = tmp_path / "dummy.pdb"
        dummy_pdb.write_text("ATOM      1  CA  ALA A   1\nEND\n")

        def fake_complex(enzyme_id, sequence, ligand_smiles, cofactors,
                         output_dir, ligand_type, config):
            return {
                "enzyme_id": enzyme_id,
                "ligand_type": ligand_type,
                "pdb_path": str(dummy_pdb),
                "confidence": {"plddt": 0.85, "iptm": 0.72, "ptm": 0.78},
            }

        mock_complex.side_effect = fake_complex

        results = run_stage(config)

        # Check PDBs were copied
        sub_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
        prod_dir = base_dir / "stage_02_wt_structures" / "product_complexes"
        assert (sub_dir / "enzyme_A_substrate.pdb").exists()
        assert (prod_dir / "enzyme_A_product.pdb").exists()
        assert (sub_dir / "enzyme_B_substrate.pdb").exists()
        assert (prod_dir / "enzyme_B_product.pdb").exists()

    @patch("pipeline.stage2_wt_structures.run_boltz2_for_complex")
    def test_writes_confidence_jsons(
        self, mock_complex: MagicMock, tmp_path: Path
    ):
        config = _make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        dummy_pdb = tmp_path / "dummy.pdb"
        dummy_pdb.write_text("ATOM      1  CA  ALA A   1\nEND\n")

        confidence = {"plddt": 0.85, "iptm": 0.72, "ptm": 0.78}

        def fake_complex(enzyme_id, sequence, ligand_smiles, cofactors,
                         output_dir, ligand_type, config):
            return {
                "enzyme_id": enzyme_id,
                "ligand_type": ligand_type,
                "pdb_path": str(dummy_pdb),
                "confidence": confidence,
            }

        mock_complex.side_effect = fake_complex

        run_stage(config)

        json_path = (
            base_dir / "stage_02_wt_structures" / "substrate_complexes"
            / "enzyme_A_substrate_confidence.json"
        )
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["iptm"] == 0.72

    @patch("pipeline.stage2_wt_structures.run_boltz2_for_complex")
    def test_returns_results_for_all_enzymes(
        self, mock_complex: MagicMock, tmp_path: Path
    ):
        config = _make_config(tmp_path)

        dummy_pdb = tmp_path / "dummy.pdb"
        dummy_pdb.write_text("ATOM      1  CA  ALA A   1\nEND\n")

        mock_complex.return_value = {
            "enzyme_id": "enzyme_A",
            "ligand_type": "substrate",
            "pdb_path": str(dummy_pdb),
            "confidence": {"plddt": 0.85, "iptm": 0.72, "ptm": 0.78},
        }

        results = run_stage(config)

        assert "enzyme_A" in results
        assert "enzyme_B" in results

    @patch("pipeline.stage2_wt_structures.run_boltz2_for_complex")
    def test_handles_boltz_failure_gracefully(
        self, mock_complex: MagicMock, tmp_path: Path
    ):
        config = _make_config(tmp_path)
        mock_complex.side_effect = RuntimeError("Boltz-2 crashed")

        # Should not raise — errors are logged and enzyme is skipped
        results = run_stage(config)

        # Results should still have enzyme keys, but without pdb paths
        for enzyme_id in results:
            assert "substrate_pdb" not in results[enzyme_id]

    @patch("pipeline.stage2_wt_structures.run_boltz2_for_complex")
    def test_results_contain_confidence_data(
        self, mock_complex: MagicMock, tmp_path: Path
    ):
        config = _make_config(tmp_path)

        dummy_pdb = tmp_path / "dummy.pdb"
        dummy_pdb.write_text("ATOM      1  CA  ALA A   1\nEND\n")

        def fake_complex(enzyme_id, sequence, ligand_smiles, cofactors,
                         output_dir, ligand_type, config):
            return {
                "enzyme_id": enzyme_id,
                "ligand_type": ligand_type,
                "pdb_path": str(dummy_pdb),
                "confidence": {"plddt": 0.85, "iptm": 0.72, "ptm": 0.78},
            }

        mock_complex.side_effect = fake_complex

        results = run_stage(config)

        for enzyme_id, data in results.items():
            assert "confidences" in data
            assert "substrate" in data["confidences"]
            assert "product" in data["confidences"]
            assert data["confidences"]["substrate"]["iptm"] == 0.72
