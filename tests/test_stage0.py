"""Tests for Stage 0: Bottleneck Identification.

All tests run without real COBRApy or CatPred installations.
"""
from __future__ import annotations

import csv
import importlib
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"


def _make_config(tmp_path: Path, extra_stages: dict | None = None) -> dict:
    """Return a minimal valid pipeline config dict pointing at *tmp_path*."""
    base = {
        "global": {
            "project_name": "test",
            "base_dir": str(tmp_path),
            "log_dir": str(tmp_path / "logs"),
            "results_dir": str(tmp_path / "results"),
            "round_number": 1,
        },
        "databases": {
            "colabfold_db_dir": str(tmp_path),
            "mmseqs2_db_dir": str(tmp_path),
        },
        "tools": {
            "catpred_dir": str(tmp_path / "fake_catpred"),
        },
        "stages": {
            "stage_00_bottleneck": {
                "output_dir": "stage_00_bottleneck",
                "fcc_threshold": 0.3,
                **(extra_stages or {}),
            },
        },
        "enzymes": {
            "enzyme_A": {
                "name": "Enzyme A",
                "length": 428,
                "sequence_file": "inputs/sequences/enzyme_A.fasta",
                "substrate_smiles": "CCO",
                "product_smiles": "CC=O",
                "fixed_residues": [],
                "hotspot_residues": [],
                "contigs": "A1-428",
                "cofactors": [],
            },
            "enzyme_B": {
                "name": "Enzyme B",
                "length": 512,
                "sequence_file": "inputs/sequences/enzyme_B.fasta",
                "substrate_smiles": "CC=O",
                "product_smiles": "CC(=O)O",
                "fixed_residues": [],
                "hotspot_residues": [],
                "contigs": "A1-512",
                "cofactors": [],
            },
            "enzyme_C": {
                "name": "Enzyme C",
                "length": 348,
                "sequence_file": "inputs/sequences/enzyme_C.fasta",
                "substrate_smiles": "CC(=O)O",
                "product_smiles": "C(=O)O",
                "fixed_residues": [],
                "hotspot_residues": [],
                "contigs": "A1-348",
                "cofactors": [],
            },
            "enzyme_D": {
                "name": "Enzyme D",
                "length": 295,
                "sequence_file": "inputs/sequences/enzyme_D.fasta",
                "substrate_smiles": "C(=O)O",
                "product_smiles": "O=C=O",
                "fixed_residues": [],
                "hotspot_residues": [],
                "contigs": "A1-295",
                "cofactors": [],
            },
        },
    }
    return base


# ---------------------------------------------------------------------------
# Import the module
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _ensure_pipeline_importable():
    """Make sure the pipeline package is importable."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def stage0():
    """Import (or reimport) the stage0_bottleneck module."""
    import pipeline.stage0_bottleneck as mod
    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# Test module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_stage_name(self, stage0):
        assert stage0.STAGE_NAME == "Stage 0: Bottleneck Identification"

    def test_stage_num(self, stage0):
        assert stage0.STAGE_NUM == "0"

    def test_input_description(self, stage0):
        assert "FASTA" in stage0.INPUT_DESCRIPTION
        assert "YAML" in stage0.INPUT_DESCRIPTION

    def test_output_description(self, stage0):
        assert "bottleneck_results.csv" in stage0.OUTPUT_DESCRIPTION


# ---------------------------------------------------------------------------
# Test run_catpred
# ---------------------------------------------------------------------------

class TestRunCatpred:
    def test_fallback_when_catpred_missing(self, stage0, tmp_path):
        """When CatPred directory doesn't exist, return placeholder values."""
        result = stage0.run_catpred(
            sequence="MAKKL",
            substrate_smiles="CCO",
            enzyme_id="enzyme_A",
            catpred_dir=tmp_path / "nonexistent",
        )
        assert "kcat" in result
        assert "km" in result
        assert "kcat_unit" in result
        assert result["kcat_unit"] == "s^-1"
        assert result["km_unit"] == "mM"
        assert isinstance(result["kcat"], float)
        assert isinstance(result["km"], float)

    def test_successful_catpred_call(self, stage0, tmp_path):
        """Simulate a successful CatPred subprocess call."""
        catpred_dir = tmp_path / "CatPred"
        predict_script = catpred_dir / "catpred" / "predict.py"
        predict_script.parent.mkdir(parents=True)
        predict_script.write_text("# dummy")

        fake_result = {"kcat": 42.0, "km": 1.5, "kcat_unit": "s^-1", "km_unit": "mM"}

        with mock.patch("subprocess.run") as mock_run, \
             mock.patch("builtins.open", mock.mock_open(read_data='{"kcat": 42.0, "km": 1.5}')), \
             mock.patch("json.load", return_value=fake_result):
            mock_run.return_value = mock.MagicMock(returncode=0)
            result = stage0.run_catpred("MAKKL", "CCO", "enzyme_A", catpred_dir)

        assert result["kcat"] == 42.0
        assert result["km"] == 1.5


# ---------------------------------------------------------------------------
# Test FCC calculation with mock cobra model
# ---------------------------------------------------------------------------

class TestCalculateFCC:
    def _make_mock_model(self, stage0, base_flux=10.0, perturbed_plus=11.0, perturbed_minus=9.0):
        """Build a mock cobra model that returns predictable optimization results."""
        mock_model = mock.MagicMock()

        # Mock reaction
        mock_rxn = mock.MagicMock()
        mock_rxn.id = "rxn_enzyme_A"
        mock_rxn.upper_bound = 1000.0
        mock_model.reactions.get_by_id.return_value = mock_rxn

        # Optimization returns different values depending on upper_bound
        call_count = [0]
        def fake_optimize():
            sol = mock.MagicMock()
            sol.status = "optimal"
            c = call_count[0]
            call_count[0] += 1
            if c == 0:
                sol.objective_value = base_flux  # baseline
            elif c == 1:
                sol.objective_value = perturbed_plus  # +10%
            else:
                sol.objective_value = perturbed_minus  # -10%
            return sol

        mock_model.optimize.side_effect = fake_optimize
        return mock_model

    def test_fcc_calculation(self, stage0):
        """FCC = (J+ - J-) / (2 * perturbation * J_baseline)."""
        model = self._make_mock_model(stage0, base_flux=10.0, perturbed_plus=11.0, perturbed_minus=9.0)
        fcc = stage0.calculate_fcc(model, "enzyme_A", "rxn_enzyme_A", perturbation=0.1)
        # Expected: (11 - 9) / (2 * 0.1 * 10) = 2.0 / 2.0 = 1.0
        assert abs(fcc - 1.0) < 1e-6

    def test_fcc_zero_baseline(self, stage0):
        """FCC should be 0 when baseline flux is zero."""
        model = self._make_mock_model(stage0, base_flux=0.0, perturbed_plus=0.0, perturbed_minus=0.0)
        fcc = stage0.calculate_fcc(model, "enzyme_A", "rxn_enzyme_A")
        assert fcc == 0.0

    def test_fcc_missing_reaction(self, stage0):
        """FCC should be 0 when reaction is not in the model."""
        mock_model = mock.MagicMock()
        mock_model.reactions.get_by_id.side_effect = KeyError("not found")
        fcc = stage0.calculate_fcc(mock_model, "enzyme_X", "rxn_nonexistent")
        assert fcc == 0.0

    def test_fcc_partial_control(self, stage0):
        """FCC should be 0.5 for partial control."""
        model = self._make_mock_model(stage0, base_flux=10.0, perturbed_plus=10.5, perturbed_minus=9.5)
        fcc = stage0.calculate_fcc(model, "enzyme_A", "rxn_enzyme_A", perturbation=0.1)
        # Expected: (10.5 - 9.5) / (2 * 0.1 * 10) = 1.0 / 2.0 = 0.5
        assert abs(fcc - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Test run_fba with mock model
# ---------------------------------------------------------------------------

class TestRunFBA:
    def test_run_fba_optimal(self, stage0):
        mock_model = mock.MagicMock()
        mock_sol = mock.MagicMock()
        mock_sol.status = "optimal"
        mock_sol.fluxes = {"rxn1": 5.0, "rxn2": 3.0}

        rxn1 = mock.MagicMock()
        rxn1.id = "rxn1"
        rxn2 = mock.MagicMock()
        rxn2.id = "rxn2"
        mock_model.reactions = [rxn1, rxn2]
        mock_model.optimize.return_value = mock_sol

        fluxes = stage0.run_fba(mock_model)
        assert fluxes == {"rxn1": 5.0, "rxn2": 3.0}

    def test_run_fba_infeasible(self, stage0):
        mock_model = mock.MagicMock()
        mock_sol = mock.MagicMock()
        mock_sol.status = "infeasible"
        mock_model.optimize.return_value = mock_sol

        fluxes = stage0.run_fba(mock_model)
        assert fluxes == {}


# ---------------------------------------------------------------------------
# Test run_stage with all mocked dependencies
# ---------------------------------------------------------------------------

class TestRunStage:
    def test_produces_csv_with_correct_columns(self, stage0, tmp_path):
        """run_stage should produce a CSV with required columns."""
        config = _make_config(tmp_path)
        (tmp_path / "stage_00_bottleneck").mkdir(parents=True, exist_ok=True)

        # Mock away cobra (use placeholder FCCs)
        with mock.patch.object(stage0, "_COBRA_AVAILABLE", False):
            output = stage0.run_stage(config)

        assert output.exists()
        with open(output) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 4
        expected_cols = {"enzyme_id", "fcc", "kcat_pred", "km_pred", "priority", "notes"}
        assert set(rows[0].keys()) == expected_cols

    def test_sorted_by_fcc_descending(self, stage0, tmp_path):
        """Results should be sorted by FCC descending."""
        config = _make_config(tmp_path)

        with mock.patch.object(stage0, "_COBRA_AVAILABLE", False):
            output = stage0.run_stage(config)

        with open(output) as fh:
            rows = list(csv.DictReader(fh))

        fccs = [float(r["fcc"]) for r in rows]
        assert fccs == sorted(fccs, reverse=True)

    def test_priority_assignment(self, stage0, tmp_path):
        """FCC > 0.3 → primary, otherwise secondary."""
        config = _make_config(tmp_path)

        with mock.patch.object(stage0, "_COBRA_AVAILABLE", False):
            output = stage0.run_stage(config)

        with open(output) as fh:
            rows = list(csv.DictReader(fh))

        for row in rows:
            fcc = float(row["fcc"])
            if fcc > 0.3:
                assert row["priority"] == "primary", f"{row['enzyme_id']} FCC={fcc} should be primary"
            else:
                assert row["priority"] == "secondary", f"{row['enzyme_id']} FCC={fcc} should be secondary"

    def test_all_enzymes_present(self, stage0, tmp_path):
        """All 4 enzymes should appear in the output."""
        config = _make_config(tmp_path)

        with mock.patch.object(stage0, "_COBRA_AVAILABLE", False):
            output = stage0.run_stage(config)

        with open(output) as fh:
            rows = list(csv.DictReader(fh))

        enzyme_ids = {r["enzyme_id"] for r in rows}
        assert enzyme_ids == {"enzyme_A", "enzyme_B", "enzyme_C", "enzyme_D"}

    def test_no_enzymes_raises(self, stage0, tmp_path):
        """run_stage should raise RuntimeError if no enzymes configured."""
        config = _make_config(tmp_path)
        config["enzymes"] = {}

        with pytest.raises(RuntimeError, match="No enzymes"):
            with mock.patch.object(stage0, "_COBRA_AVAILABLE", False):
                stage0.run_stage(config)

    def test_placeholder_notes(self, stage0, tmp_path):
        """When using placeholders, notes should indicate that."""
        config = _make_config(tmp_path)

        with mock.patch.object(stage0, "_COBRA_AVAILABLE", False):
            output = stage0.run_stage(config)

        with open(output) as fh:
            rows = list(csv.DictReader(fh))

        for row in rows:
            assert "placeholder" in row["notes"].lower()


# ---------------------------------------------------------------------------
# Test graceful degradation when cobra not available
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_cobra_import_error_uses_placeholders(self, stage0, tmp_path):
        """When _COBRA_AVAILABLE is False, use placeholder FCC values."""
        config = _make_config(tmp_path)

        with mock.patch.object(stage0, "_COBRA_AVAILABLE", False):
            output = stage0.run_stage(config)

        assert output.exists()
        with open(output) as fh:
            rows = list(csv.DictReader(fh))

        # Should still produce valid results
        assert len(rows) == 4
        fccs = {r["enzyme_id"]: float(r["fcc"]) for r in rows}
        assert fccs["enzyme_A"] == 0.4
        assert fccs["enzyme_B"] == 0.3
        assert fccs["enzyme_D"] == 0.1

    def test_cobra_failure_falls_back(self, stage0, tmp_path):
        """If COBRApy FBA raises, fall back to placeholders."""
        config = _make_config(tmp_path)

        with mock.patch.object(stage0, "_COBRA_AVAILABLE", True), \
             mock.patch.object(stage0, "load_yeast_gem", side_effect=Exception("model broke")):
            output = stage0.run_stage(config)

        assert output.exists()
        with open(output) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4


# ---------------------------------------------------------------------------
# Test main() CLI
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_calls_run_stage(self, stage0, tmp_path):
        """main() should call run_stage and produce output."""
        config = _make_config(tmp_path)
        config_path = tmp_path / "config.yaml"

        import yaml
        with open(config_path, "w") as fh:
            yaml.dump(config, fh)

        (tmp_path / "logs").mkdir(exist_ok=True)
        (tmp_path / "stage_00_bottleneck").mkdir(exist_ok=True)

        with mock.patch.object(stage0, "_COBRA_AVAILABLE", False):
            stage0.main(config_path)

        output_csv = tmp_path / "stage_00_bottleneck" / "bottleneck_results.csv"
        assert output_csv.exists()

    def test_cli_entry_point(self, stage0, tmp_path):
        """_cli() should parse args and call main()."""
        config_path = tmp_path / "config.yaml"

        with mock.patch.object(stage0, "main") as mock_main, \
             mock.patch("sys.argv", ["prog", "--config", str(config_path)]):
            stage0._cli()

        mock_main.assert_called_once_with(config_path)


# ---------------------------------------------------------------------------
# Test load_yeast_gem
# ---------------------------------------------------------------------------

class TestLoadYeastGem:
    def test_builds_toy_model_when_file_missing(self, stage0, tmp_path):
        """When SBML file doesn't exist, build a toy model."""
        if not stage0._COBRA_AVAILABLE:
            pytest.skip("COBRApy not installed")
        model = stage0.load_yeast_gem(tmp_path / "nonexistent.xml")
        assert model is not None
        assert hasattr(model, "optimize")

    def test_raises_without_cobra(self, stage0, tmp_path):
        """Should raise ImportError when cobra is not available."""
        with mock.patch.object(stage0, "_COBRA_AVAILABLE", False):
            with pytest.raises(ImportError, match="COBRApy"):
                stage0.load_yeast_gem(tmp_path / "model.xml")
