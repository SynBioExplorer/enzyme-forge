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
    """Return a minimal valid pipeline config dict pointing at *tmp_path*.

    By default ``enabled=True`` so that tests for the active stage work
    without modification.  Pass ``extra_stages={"enabled": False}`` to test
    the disabled path explicitly.
    """
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
                "enabled": True,  # tests default to enabled; override via extra_stages
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


# ---------------------------------------------------------------------------
# Test disabled stage (enabled: false)
# ---------------------------------------------------------------------------

class TestStageDisabled:
    def test_disabled_returns_equal_priority(self, stage0, tmp_path):
        """When enabled=false, all enzymes get FCC=1.0 and priority='equal'."""
        config = _make_config(tmp_path, extra_stages={"enabled": False})
        output = stage0.run_stage(config)

        assert output.exists()
        with open(output) as fh:
            rows = list(csv.DictReader(fh))

        assert len(rows) == 4
        for row in rows:
            assert float(row["fcc"]) == 1.0, f"{row['enzyme_id']} fcc should be 1.0"
            assert row["priority"] == "equal", f"{row['enzyme_id']} priority should be 'equal'"

    def test_disabled_has_all_enzymes(self, stage0, tmp_path):
        """Disabled stage CSV must contain all 4 enzymes."""
        config = _make_config(tmp_path, extra_stages={"enabled": False})
        output = stage0.run_stage(config)

        with open(output) as fh:
            rows = list(csv.DictReader(fh))

        assert {r["enzyme_id"] for r in rows} == {
            "enzyme_A", "enzyme_B", "enzyme_C", "enzyme_D"
        }

    def test_disabled_does_not_call_cobra(self, stage0, tmp_path):
        """When disabled, COBRApy/CatPred should not be touched."""
        config = _make_config(tmp_path, extra_stages={"enabled": False})

        with mock.patch.object(stage0, "load_yeast_gem") as mock_gem, \
             mock.patch.object(stage0, "run_catpred") as mock_cat:
            stage0.run_stage(config)

        mock_gem.assert_not_called()
        mock_cat.assert_not_called()

    def test_disabled_csv_has_correct_columns(self, stage0, tmp_path):
        """Disabled output CSV must have all standard columns."""
        config = _make_config(tmp_path, extra_stages={"enabled": False})
        output = stage0.run_stage(config)

        with open(output) as fh:
            rows = list(csv.DictReader(fh))

        expected_cols = {"enzyme_id", "fcc", "kcat_pred", "km_pred", "priority", "notes"}
        assert set(rows[0].keys()) == expected_cols

    def test_enabled_true_runs_normally(self, stage0, tmp_path):
        """When enabled=true, stage runs the full analysis."""
        config = _make_config(tmp_path, extra_stages={"enabled": True})

        with mock.patch.object(stage0, "_COBRA_AVAILABLE", False):
            output = stage0.run_stage(config)

        with open(output) as fh:
            rows = list(csv.DictReader(fh))

        # In normal mode priority is primary/secondary, not equal
        priorities = {r["priority"] for r in rows}
        assert priorities <= {"primary", "secondary"}


# ---------------------------------------------------------------------------
# Test TFBA method
# ---------------------------------------------------------------------------

class TestTFBAMethod:
    def test_run_tfba_function_exists(self, stage0):
        """run_tfba() function must be defined."""
        assert hasattr(stage0, "run_tfba"), "run_tfba function not found"
        assert callable(stage0.run_tfba)

    def test_method_tfba_uses_run_tfba(self, stage0, tmp_path):
        """When method='tfba', run_stage calls run_tfba for FCC computation."""
        config = _make_config(tmp_path, extra_stages={"method": "tfba"})

        fake_fcc = {
            "enzyme_A": 0.55,
            "enzyme_B": 0.40,
            "enzyme_C": 0.20,
            "enzyme_D": 0.10,
        }

        with mock.patch.object(stage0, "run_tfba", return_value=fake_fcc) as mock_tfba, \
             mock.patch.object(stage0, "_COBRA_AVAILABLE", True), \
             mock.patch.object(stage0, "load_yeast_gem"):
            output = stage0.run_stage(config)

        mock_tfba.assert_called_once()
        assert output.exists()
        with open(output) as fh:
            rows = list(csv.DictReader(fh))
        fcc_map = {r["enzyme_id"]: float(r["fcc"]) for r in rows}
        assert abs(fcc_map["enzyme_A"] - 0.55) < 1e-3

    def test_method_fba_uses_calculate_fcc(self, stage0, tmp_path):
        """When method='fba' (explicit), run_stage uses the standard calculate_fcc path."""
        config = _make_config(tmp_path, extra_stages={"method": "fba"})

        with mock.patch.object(stage0, "_COBRA_AVAILABLE", False):
            output = stage0.run_stage(config)

        assert output.exists()
        with open(output) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4

    def test_run_tfba_returns_dict(self, stage0):
        """run_tfba() must return a dict mapping enzyme_id to float FCC."""
        mock_model = mock.MagicMock()
        reaction_map = {
            "enzyme_A": "rxn_enzyme_A",
            "enzyme_B": "rxn_enzyme_B",
            "enzyme_C": "rxn_enzyme_C",
            "enzyme_D": "rxn_enzyme_D",
        }
        enzyme_ids = list(reaction_map.keys())

        with mock.patch.object(stage0, "_EQUILIBRATOR_AVAILABLE", False, create=True):
            result = stage0.run_tfba(mock_model, enzyme_ids, reaction_map)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(enzyme_ids)
        for v in result.values():
            assert isinstance(v, float)

    def test_tfba_falls_back_when_equilibrator_missing(self, stage0, tmp_path):
        """When equilibrator_api is unavailable, tfba falls back to standard FBA FCCs."""
        config = _make_config(tmp_path, extra_stages={"method": "tfba"})

        # Ensure equilibrator is flagged unavailable
        with mock.patch.object(stage0, "_COBRA_AVAILABLE", True), \
             mock.patch.object(stage0, "load_yeast_gem"), \
             mock.patch.object(stage0, "_EQUILIBRATOR_AVAILABLE", False, create=True), \
             mock.patch.object(stage0, "calculate_fcc", return_value=0.25) as mock_fcc:
            output = stage0.run_stage(config)

        assert output.exists()


# ---------------------------------------------------------------------------
# Test ensemble method
# ---------------------------------------------------------------------------

class TestEnsembleMethod:
    def test_run_ensemble_function_exists(self, stage0):
        """run_ensemble() function must be defined."""
        assert hasattr(stage0, "run_ensemble"), "run_ensemble function not found"
        assert callable(stage0.run_ensemble)

    def test_method_ensemble_produces_output(self, stage0, tmp_path):
        """When method='ensemble', run_stage produces a valid CSV."""
        config = _make_config(
            tmp_path, extra_stages={"method": "ensemble", "ensemble_n": 10}
        )

        fake_ensemble_result = {
            "enzyme_A": {"fcc_median": 0.55, "fcc_ci_low": 0.40, "fcc_ci_high": 0.70},
            "enzyme_B": {"fcc_median": 0.35, "fcc_ci_low": 0.20, "fcc_ci_high": 0.50},
            "enzyme_C": {"fcc_median": 0.20, "fcc_ci_low": 0.10, "fcc_ci_high": 0.30},
            "enzyme_D": {"fcc_median": 0.10, "fcc_ci_low": 0.05, "fcc_ci_high": 0.15},
        }

        with mock.patch.object(stage0, "run_ensemble", return_value=fake_ensemble_result) as mock_ens, \
             mock.patch.object(stage0, "_COBRA_AVAILABLE", True), \
             mock.patch.object(stage0, "load_yeast_gem"):
            output = stage0.run_stage(config)

        mock_ens.assert_called_once()
        assert output.exists()

    def test_ensemble_csv_has_ci_columns(self, stage0, tmp_path):
        """Ensemble output CSV must include fcc_ci_low and fcc_ci_high columns."""
        config = _make_config(
            tmp_path, extra_stages={"method": "ensemble", "ensemble_n": 10}
        )

        fake_ensemble_result = {
            "enzyme_A": {"fcc_median": 0.55, "fcc_ci_low": 0.40, "fcc_ci_high": 0.70},
            "enzyme_B": {"fcc_median": 0.35, "fcc_ci_low": 0.20, "fcc_ci_high": 0.50},
            "enzyme_C": {"fcc_median": 0.20, "fcc_ci_low": 0.10, "fcc_ci_high": 0.30},
            "enzyme_D": {"fcc_median": 0.10, "fcc_ci_low": 0.05, "fcc_ci_high": 0.15},
        }

        with mock.patch.object(stage0, "run_ensemble", return_value=fake_ensemble_result), \
             mock.patch.object(stage0, "_COBRA_AVAILABLE", True), \
             mock.patch.object(stage0, "load_yeast_gem"):
            output = stage0.run_stage(config)

        with open(output) as fh:
            rows = list(csv.DictReader(fh))

        assert "fcc_ci_low" in rows[0], "Missing fcc_ci_low column in ensemble output"
        assert "fcc_ci_high" in rows[0], "Missing fcc_ci_high column in ensemble output"

    def test_ensemble_uses_median_for_fcc(self, stage0, tmp_path):
        """Ensemble mode uses fcc_median as the fcc value for ranking."""
        config = _make_config(
            tmp_path, extra_stages={"method": "ensemble", "ensemble_n": 5}
        )

        fake_ensemble_result = {
            "enzyme_A": {"fcc_median": 0.55, "fcc_ci_low": 0.40, "fcc_ci_high": 0.70},
            "enzyme_B": {"fcc_median": 0.35, "fcc_ci_low": 0.20, "fcc_ci_high": 0.50},
            "enzyme_C": {"fcc_median": 0.20, "fcc_ci_low": 0.10, "fcc_ci_high": 0.30},
            "enzyme_D": {"fcc_median": 0.10, "fcc_ci_low": 0.05, "fcc_ci_high": 0.15},
        }

        with mock.patch.object(stage0, "run_ensemble", return_value=fake_ensemble_result), \
             mock.patch.object(stage0, "_COBRA_AVAILABLE", True), \
             mock.patch.object(stage0, "load_yeast_gem"):
            output = stage0.run_stage(config)

        with open(output) as fh:
            rows = list(csv.DictReader(fh))

        fcc_map = {r["enzyme_id"]: float(r["fcc"]) for r in rows}
        assert abs(fcc_map["enzyme_A"] - 0.55) < 1e-3

    def test_run_ensemble_returns_dict_structure(self, stage0):
        """run_ensemble() must return dict with fcc_median, fcc_ci_low, fcc_ci_high."""
        mock_model = mock.MagicMock()
        reaction_map = {
            "enzyme_A": "rxn_enzyme_A",
            "enzyme_B": "rxn_enzyme_B",
        }
        enzyme_ids = list(reaction_map.keys())
        kinetics = {
            "enzyme_A": {"kcat": 12.5, "km": 0.45},
            "enzyme_B": {"kcat": 8.3, "km": 0.82},
        }

        with mock.patch.object(stage0, "calculate_fcc", return_value=0.5):
            result = stage0.run_ensemble(
                mock_model, enzyme_ids, reaction_map, kinetics, n_iterations=5
            )

        assert isinstance(result, dict)
        for eid in enzyme_ids:
            assert eid in result
            assert "fcc_median" in result[eid]
            assert "fcc_ci_low" in result[eid]
            assert "fcc_ci_high" in result[eid]
            assert isinstance(result[eid]["fcc_median"], float)

    def test_run_ensemble_respects_n_iterations(self, stage0):
        """run_ensemble() samples kinetic parameters n_iterations times."""
        mock_model = mock.MagicMock()
        reaction_map = {"enzyme_A": "rxn_enzyme_A"}
        enzyme_ids = ["enzyme_A"]
        kinetics = {"enzyme_A": {"kcat": 12.5, "km": 0.45}}

        fcc_call_counts = []
        def counting_fcc(*args, **kwargs):
            fcc_call_counts.append(1)
            return 0.4

        with mock.patch.object(stage0, "calculate_fcc", side_effect=counting_fcc):
            stage0.run_ensemble(
                mock_model, enzyme_ids, reaction_map, kinetics, n_iterations=7
            )

        assert len(fcc_call_counts) == 7, (
            f"Expected 7 FCC calls (one per iteration), got {len(fcc_call_counts)}"
        )
