"""Tests for Stage 5: ESMFold Pre-screening (pipeline/stage5_esmfold.py).

All tests run without ESMFold model, transformers, or GPU.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from pipeline.stage5_esmfold import (
    STAGE_NAME,
    STAGE_NUM,
    INPUT_DESCRIPTION,
    OUTPUT_DESCRIPTION,
    load_esmfold_model,
    predict_structure_esmfold,
    batch_predict,
    calc_ca_rmsd_from_strings,
    apply_filters,
    run_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_pdb_string(
    coords: list[tuple[float, float, float]],
    bfactors: list[float] | None = None,
) -> str:
    """Build a minimal PDB string with CA atoms at the given coordinates."""
    lines: list[str] = []
    for i, (x, y, z) in enumerate(coords, start=1):
        bfac = bfactors[i - 1] if bfactors else 0.0
        lines.append(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def _write_pdb_file(path: Path, pdb_string: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pdb_string)
    return path


def _write_fasta(path: Path, sequences: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for header, seq in sequences.items():
            fh.write(f">{header}\n{seq}\n")
    return path


def _make_stage5_config(tmp_path: Path, enzyme_ids: list[str] | None = None) -> dict:
    """Build a minimal pipeline config for testing run_stage."""
    if enzyme_ids is None:
        enzyme_ids = ["enzyme_A", "enzyme_B"]

    base_dir = tmp_path / "pipeline"
    base_dir.mkdir(exist_ok=True)

    # Create input FASTA files (from stage 4)
    input_dir = base_dir / "stage_04_ligandmpnn"
    input_dir.mkdir(parents=True)
    for eid in enzyme_ids:
        _write_fasta(
            input_dir / f"{eid}_designed.fasta",
            {
                f"{eid}_var1": "MKTLYKE" * 5,
                f"{eid}_var2": "AVILWFG" * 5,
                f"{eid}_var3": "GSTCMPN" * 5,
            },
        )

    # Create WT PDB files (from stage 2)
    wt_dir = base_dir / "stage_02_wt_structures" / "substrate_complexes"
    wt_dir.mkdir(parents=True)
    wt_pdb_str = _minimal_pdb_string(
        [(float(i), 0.0, 0.0) for i in range(35)],
        bfactors=[80.0] * 35,
    )
    for eid in enzyme_ids:
        _write_pdb_file(wt_dir / f"{eid}.pdb", wt_pdb_str)

    config = {
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
            "esmfold_env": str(tmp_path / "esmfold_env"),
        },
        "stages": {
            "stage_02_wt_structures": {"output_dir": "stage_02_wt_structures"},
            "stage_04_ligandmpnn": {"output_dir": "stage_04_ligandmpnn"},
            "stage_05_esmfold": {
                "output_dir": "stage_05_esmfold_prescreen",
                "plddt_hard_floor": 60,
                "ca_rmsd_max": 5.0,
                "device": "cpu",
            },
        },
        "enzymes": {
            eid: {
                "name": eid.replace("_", " ").title(),
                "length": 35,
                "sequence_file": f"inputs/sequences/{eid}.fasta",
                "substrate_smiles": "C",
                "product_smiles": "C",
                "fixed_residues": [],
                "hotspot_residues": [],
                "contigs": "A1-35",
                "cofactors": [],
            }
            for eid in enzyme_ids
        },
    }
    return config


# ---------------------------------------------------------------------------
# Tests: module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_stage_name(self):
        assert STAGE_NAME == "Stage 5: ESMFold Pre-screening"

    def test_stage_num(self):
        assert STAGE_NUM == "5"

    def test_input_description(self):
        assert "FASTA" in INPUT_DESCRIPTION
        assert "Stage 4" in INPUT_DESCRIPTION

    def test_output_description(self):
        assert "stage_05_esmfold_prescreen" in OUTPUT_DESCRIPTION
        assert "plddt_scores.csv" in OUTPUT_DESCRIPTION


# ---------------------------------------------------------------------------
# Tests: load_esmfold_model
# ---------------------------------------------------------------------------

class TestLoadEsmfoldModel:
    def setup_method(self):
        """Reset the module-level cache before each test."""
        import pipeline.stage5_esmfold as mod
        mod._cached_model = None
        mod._cached_tokenizer = None

    @patch("pipeline.stage5_esmfold.EsmForProteinFolding", create=True)
    @patch("pipeline.stage5_esmfold.AutoTokenizer", create=True)
    def test_loads_model_and_tokenizer(self, mock_tok_cls, mock_model_cls):
        """Should import and load ESMFold model via transformers."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_tokenizer = MagicMock()

        with patch.dict("sys.modules", {
            "transformers": MagicMock(
                AutoTokenizer=MagicMock(from_pretrained=MagicMock(return_value=mock_tokenizer)),
                EsmForProteinFolding=MagicMock(from_pretrained=MagicMock(return_value=mock_model)),
            ),
        }):
            with patch(
                "pipeline.stage5_esmfold.load_esmfold_model.__module__",
                "pipeline.stage5_esmfold",
            ):
                import pipeline.stage5_esmfold as mod
                mod._cached_model = None
                mod._cached_tokenizer = None

                # Manually mock the import inside the function
                with patch.object(mod, "_cached_model", None), \
                     patch.object(mod, "_cached_tokenizer", None):
                    # Patch the imports that happen inside load_esmfold_model
                    mock_esm_model = MagicMock()
                    mock_esm_model.to.return_value = mock_esm_model
                    mock_esm_tokenizer = MagicMock()

                    transformers_mock = MagicMock()
                    transformers_mock.AutoTokenizer.from_pretrained.return_value = mock_esm_tokenizer
                    transformers_mock.EsmForProteinFolding.from_pretrained.return_value = mock_esm_model

                    with patch.dict("sys.modules", {"transformers": transformers_mock}):
                        model, tok = mod.load_esmfold_model("cpu")

                    assert model is mock_esm_model
                    assert tok is mock_esm_tokenizer

    def test_caches_model(self):
        """Second call should return cached model without re-loading."""
        import pipeline.stage5_esmfold as mod

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        mod._cached_model = fake_model
        mod._cached_tokenizer = fake_tokenizer

        model, tok = mod.load_esmfold_model("cpu")
        assert model is fake_model
        assert tok is fake_tokenizer


# ---------------------------------------------------------------------------
# Tests: calc_ca_rmsd_from_strings
# ---------------------------------------------------------------------------

class TestCalcCaRmsdFromStrings:
    def test_identical_structures_rmsd_zero(self):
        pdb = _minimal_pdb_string([(0, 0, 0), (3.8, 0, 0), (7.6, 0, 0)])
        rmsd = calc_ca_rmsd_from_strings(pdb, pdb)
        assert rmsd == pytest.approx(0.0, abs=1e-4)

    def test_shifted_structure_nonzero_rmsd(self):
        pdb1 = _minimal_pdb_string([(0, 0, 0), (3.8, 0, 0)])
        # After optimal superposition of 2 atoms, any rigid shift → RMSD 0
        # But rotation-free shift of CA atoms with different relative positions
        pdb2 = _minimal_pdb_string([(0, 0, 0), (4.8, 0, 0)])
        # Two 1D arrays: [0, 3.8] vs [0, 4.8]
        # After superposition (centroid alignment): RMSD = 0.5
        rmsd = calc_ca_rmsd_from_strings(pdb1, pdb2)
        assert rmsd == pytest.approx(0.5, abs=0.1)

    def test_different_lengths_truncates(self):
        """Structures with different residue counts should use shorter length."""
        pdb_short = _minimal_pdb_string([(0, 0, 0), (3.8, 0, 0)])
        pdb_long = _minimal_pdb_string([(0, 0, 0), (3.8, 0, 0), (7.6, 0, 0)])
        rmsd = calc_ca_rmsd_from_strings(pdb_short, pdb_long)
        assert rmsd == pytest.approx(0.0, abs=1e-4)

    def test_empty_structures_rmsd_zero(self):
        rmsd = calc_ca_rmsd_from_strings("END\n", "END\n")
        assert rmsd == 0.0


# ---------------------------------------------------------------------------
# Tests: apply_filters
# ---------------------------------------------------------------------------

class TestApplyFilters:
    @pytest.fixture
    def wt_pdb(self, tmp_path: Path) -> Path:
        pdb_str = _minimal_pdb_string([(float(i), 0.0, 0.0) for i in range(10)])
        return _write_pdb_file(tmp_path / "wt.pdb", pdb_str)

    def test_passes_good_sequence(self, wt_pdb: Path):
        """High pLDDT + low RMSD → passes."""
        pdb_str = _minimal_pdb_string([(float(i), 0.0, 0.0) for i in range(10)])
        passed, info = apply_filters("seq1", pdb_str, 85.0, wt_pdb)
        assert passed is True
        assert info["passed_plddt_filter"] is True
        assert info["passed_rmsd_filter"] is True
        assert info["passed_all_filters"] is True
        assert info["rmsd_to_wt"] == pytest.approx(0.0, abs=1e-4)

    def test_rejects_low_plddt(self, wt_pdb: Path):
        """pLDDT < 60 → hard floor reject."""
        pdb_str = _minimal_pdb_string([(float(i), 0.0, 0.0) for i in range(10)])
        passed, info = apply_filters("seq2", pdb_str, 45.0, wt_pdb)
        assert passed is False
        assert info["passed_plddt_filter"] is False
        assert info["passed_all_filters"] is False
        assert "pLDDT" in info["reason"]

    def test_rejects_high_rmsd(self, wt_pdb: Path):
        """RMSD > 5A → hard filter reject."""
        # Shift all coords significantly
        pdb_str = _minimal_pdb_string(
            [(float(i) * 10.0, float(i) * 10.0, 0.0) for i in range(10)]
        )
        passed, info = apply_filters("seq3", pdb_str, 85.0, wt_pdb)
        assert passed is False
        assert info["passed_rmsd_filter"] is False
        assert info["passed_all_filters"] is False
        assert "RMSD" in info["reason"]

    def test_plddt_at_boundary_passes(self, wt_pdb: Path):
        """pLDDT exactly at 60 → should pass (floor is strict <)."""
        pdb_str = _minimal_pdb_string([(float(i), 0.0, 0.0) for i in range(10)])
        passed, info = apply_filters("seq4", pdb_str, 60.0, wt_pdb)
        assert passed is True
        assert info["passed_plddt_filter"] is True

    def test_plddt_just_below_boundary_fails(self, wt_pdb: Path):
        """pLDDT = 59.9 → should fail."""
        pdb_str = _minimal_pdb_string([(float(i), 0.0, 0.0) for i in range(10)])
        passed, info = apply_filters("seq5", pdb_str, 59.9, wt_pdb)
        assert passed is False
        assert info["passed_plddt_filter"] is False

    def test_custom_thresholds(self, wt_pdb: Path):
        """Custom thresholds should override defaults."""
        pdb_str = _minimal_pdb_string([(float(i), 0.0, 0.0) for i in range(10)])
        # pLDDT = 55 would fail default (60) but pass with floor=50
        passed, info = apply_filters("seq6", pdb_str, 55.0, wt_pdb, plddt_floor=50.0)
        assert passed is True

    def test_plddt_filter_short_circuits_rmsd(self, wt_pdb: Path):
        """When pLDDT fails, RMSD should not be computed (rmsd_to_wt = None)."""
        pdb_str = _minimal_pdb_string([(float(i), 0.0, 0.0) for i in range(10)])
        passed, info = apply_filters("seq7", pdb_str, 30.0, wt_pdb)
        assert passed is False
        assert info["rmsd_to_wt"] is None


# ---------------------------------------------------------------------------
# Tests: batch_predict (mocked model)
# ---------------------------------------------------------------------------

class TestBatchPredict:
    def _make_mock_model_and_tokenizer(self, plddt_value: float = 80.0):
        """Create mock model/tokenizer that returns predictable results."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        return mock_model, mock_tokenizer

    @patch("pipeline.stage5_esmfold.predict_structure_esmfold")
    def test_processes_all_sequences(self, mock_predict):
        """All sequences should be processed."""
        mock_predict.side_effect = lambda seq, m, t, d: {
            "plddt": 80.0,
            "pdb_string": "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n",
        }

        sequences = {"s1": "MKTLY", "s2": "AVILD", "s3": "GSTCM"}
        results = batch_predict(sequences, MagicMock(), MagicMock(), batch_size=2, device="cpu")

        assert len(results) == 3
        assert all(r["plddt"] == 80.0 for r in results.values())

    @patch("pipeline.stage5_esmfold.predict_structure_esmfold")
    def test_handles_oom_reduces_batch_size(self, mock_predict):
        """On OOM, batch size should be halved and retried."""
        call_count = [0]

        def predict_with_oom(seq, m, t, d):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("CUDA out of memory")
            return {
                "plddt": 75.0,
                "pdb_string": "END\n",
            }

        mock_predict.side_effect = predict_with_oom

        sequences = {"s1": "MKTLY", "s2": "AVILD"}
        results = batch_predict(
            sequences, MagicMock(), MagicMock(), batch_size=4, device="cpu"
        )

        assert len(results) == 2

    @patch("pipeline.stage5_esmfold.predict_structure_esmfold")
    def test_skips_sequence_on_persistent_oom(self, mock_predict):
        """If single sequence causes OOM, skip it and continue."""
        def always_oom_first(seq, m, t, d):
            if seq == "MKTLY":
                raise RuntimeError("CUDA out of memory")
            return {"plddt": 70.0, "pdb_string": "END\n"}

        mock_predict.side_effect = always_oom_first

        sequences = {"s1": "MKTLY", "s2": "AVILD"}
        results = batch_predict(
            sequences, MagicMock(), MagicMock(), batch_size=1, device="cpu"
        )

        # s1 skipped, s2 processed
        assert "s2" in results
        assert "s1" not in results

    @patch("pipeline.stage5_esmfold.predict_structure_esmfold")
    def test_empty_sequences_returns_empty(self, mock_predict):
        results = batch_predict({}, MagicMock(), MagicMock(), device="cpu")
        assert results == {}
        mock_predict.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: run_stage (fully mocked)
# ---------------------------------------------------------------------------

class TestRunStage:
    def _mock_batch_predict(self, sequences, model, tokenizer, batch_size=8, device="cuda"):
        """Fake batch_predict returning reasonable predictions."""
        results = {}
        for i, (sid, seq) in enumerate(sequences.items()):
            # Alternate: var1 → high pLDDT, var2 → low pLDDT, var3 → medium
            if "var1" in sid:
                plddt = 85.0
            elif "var2" in sid:
                plddt = 45.0  # below floor
            else:
                plddt = 72.0
            coords = [(float(j), 0.0, 0.0) for j in range(35)]
            results[sid] = {
                "plddt": plddt,
                "pdb_string": _minimal_pdb_string(coords, bfactors=[plddt] * 35),
            }
        return results

    @patch("pipeline.stage5_esmfold.load_esmfold_model")
    @patch("pipeline.stage5_esmfold.batch_predict")
    def test_writes_filtered_fasta(self, mock_bp, mock_load, tmp_path: Path):
        """Passing sequences should be written to filtered FASTA."""
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_bp.side_effect = self._mock_batch_predict

        config = _make_stage5_config(tmp_path)
        results = run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        output_dir = base_dir / "stage_05_esmfold_prescreen"

        for eid in ("enzyme_A", "enzyme_B"):
            fasta_path = output_dir / f"{eid}_filtered.fasta"
            assert fasta_path.exists()
            content = fasta_path.read_text()
            # var1 (pLDDT 85) and var3 (pLDDT 72) should pass pLDDT floor
            assert f"{eid}_var1" in content
            assert f"{eid}_var3" in content
            # var2 (pLDDT 45) should be filtered out
            assert f"{eid}_var2" not in content

    @patch("pipeline.stage5_esmfold.load_esmfold_model")
    @patch("pipeline.stage5_esmfold.batch_predict")
    def test_writes_scores_csv_with_all_sequences(self, mock_bp, mock_load, tmp_path: Path):
        """Scores CSV should include ALL sequences, even filtered ones."""
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_bp.side_effect = self._mock_batch_predict

        config = _make_stage5_config(tmp_path)
        run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        scores_csv = base_dir / "stage_05_esmfold_prescreen" / "plddt_scores.csv"
        assert scores_csv.exists()

        with open(scores_csv) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # 2 enzymes × 3 sequences = 6 rows
        assert len(rows) == 6

        # Check required columns
        required_cols = {
            "sequence_id", "enzyme_id", "mean_plddt", "rmsd_to_wt",
            "passed_plddt_filter", "passed_rmsd_filter", "passed_all_filters",
        }
        assert required_cols.issubset(set(rows[0].keys()))

        # Check that filtered-out sequences are included
        seq_ids = {r["sequence_id"] for r in rows}
        for eid in ("enzyme_A", "enzyme_B"):
            assert f"{eid}_var2" in seq_ids  # this one was filtered

    @patch("pipeline.stage5_esmfold.load_esmfold_model")
    @patch("pipeline.stage5_esmfold.batch_predict")
    def test_retention_rate_logged(self, mock_bp, mock_load, tmp_path: Path):
        """Return dict should contain correct retention rates."""
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_bp.side_effect = self._mock_batch_predict

        config = _make_stage5_config(tmp_path)
        results = run_stage(config)

        assert "retention_rates" in results
        assert "retained" in results
        assert "filtered" in results

        for eid in ("enzyme_A", "enzyme_B"):
            # 2 pass (var1, var3), 1 fail (var2) → 66.7%
            assert results["retained"][eid] == 2
            assert results["filtered"][eid] == 1
            assert results["retention_rates"][eid] == pytest.approx(66.7, abs=0.1)

    @patch("pipeline.stage5_esmfold.load_esmfold_model")
    @patch("pipeline.stage5_esmfold.batch_predict")
    def test_writes_pdb_files(self, mock_bp, mock_load, tmp_path: Path):
        """PDB files should be written for all predicted structures."""
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_bp.side_effect = self._mock_batch_predict

        config = _make_stage5_config(tmp_path)
        run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        structures_dir = base_dir / "stage_05_esmfold_prescreen" / "structures"
        assert structures_dir.exists()

        pdb_files = list(structures_dir.glob("*.pdb"))
        # 2 enzymes × 3 sequences = 6 PDB files
        assert len(pdb_files) == 6

    @patch("pipeline.stage5_esmfold.load_esmfold_model")
    @patch("pipeline.stage5_esmfold.batch_predict")
    def test_skips_enzyme_without_input_fasta(self, mock_bp, mock_load, tmp_path: Path):
        """Enzyme with no input FASTA should be skipped gracefully."""
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_bp.side_effect = self._mock_batch_predict

        config = _make_stage5_config(tmp_path, enzyme_ids=["enzyme_A", "enzyme_B"])
        base_dir = Path(config["global"]["base_dir"])

        # Remove enzyme_B input
        fasta_path = base_dir / "stage_04_ligandmpnn" / "enzyme_B_designed.fasta"
        fasta_path.unlink()

        results = run_stage(config)

        assert "enzyme_A" in results["retained"]
        assert "enzyme_B" not in results["retained"]

    @patch("pipeline.stage5_esmfold.load_esmfold_model")
    @patch("pipeline.stage5_esmfold.batch_predict")
    def test_creates_output_directories(self, mock_bp, mock_load, tmp_path: Path):
        """Output directories should be created."""
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_bp.side_effect = self._mock_batch_predict

        config = _make_stage5_config(tmp_path)
        run_stage(config)

        base_dir = Path(config["global"]["base_dir"])
        output_dir = base_dir / "stage_05_esmfold_prescreen"
        assert output_dir.exists()
        assert (output_dir / "structures").exists()

    @patch("pipeline.stage5_esmfold.load_esmfold_model")
    @patch("pipeline.stage5_esmfold.batch_predict")
    def test_single_enzyme(self, mock_bp, mock_load, tmp_path: Path):
        """Should work with a single enzyme."""
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_bp.side_effect = self._mock_batch_predict

        config = _make_stage5_config(tmp_path, enzyme_ids=["enzyme_A"])
        results = run_stage(config)

        assert "enzyme_A" in results["retained"]
        assert len(results["retained"]) == 1
