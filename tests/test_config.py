"""Tests for pipeline.config — YAML config loader with schema validation."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pipeline.config import get_enzyme_config, load_config, resolve_paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _minimal_config() -> dict:
    """Return a minimal valid pipeline config dict."""
    return {
        "global": {
            "project_name": "test_project",
            "base_dir": "/tmp/test_pipeline",
        },
        "databases": {
            "colabfold_db_dir": "/data/colabfold",
            "mmseqs2_db_dir": "/data/mmseqs2",
        },
        "tools": {},
        "stages": {},
        "enzymes": {
            "enzyme_A": {
                "name": "Enzyme A",
                "length": 428,
                "sequence_file": "inputs/sequences/enzyme_A.fasta",
                "substrate_smiles": "CCO",
                "product_smiles": "CC=O",
                "fixed_residues": [101, 202],
                "hotspot_residues": [100, 101, 102],
                "contigs": "A1-428",
                "cofactors": ["NAD+"],
            },
            "enzyme_B": {
                "name": "Enzyme B",
                "length": 512,
                "sequence_file": "inputs/sequences/enzyme_B.fasta",
                "substrate_smiles": "CC=O",
                "product_smiles": "CC(=O)O",
                "fixed_residues": [55, 89],
                "hotspot_residues": [54, 55, 56, 88, 89],
                "contigs": "A1-512",
                "cofactors": [],
            },
        },
    }


@pytest.fixture()
def valid_config_path(tmp_path: Path) -> Path:
    """Write a minimal valid config and return its path."""
    cfg_file = tmp_path / "pipeline_config.yaml"
    cfg_file.write_text(yaml.dump(_minimal_config()))
    return cfg_file


@pytest.fixture()
def valid_config(valid_config_path: Path) -> dict:
    """Load and return a valid config dict."""
    return load_config(valid_config_path)


# ---------------------------------------------------------------------------
# load_config — happy path
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_yaml(self, valid_config_path: Path) -> None:
        cfg = load_config(valid_config_path)
        assert cfg["global"]["project_name"] == "test_project"

    def test_fills_defaults(self, valid_config_path: Path) -> None:
        cfg = load_config(valid_config_path)
        assert cfg["global"]["round_number"] == 1
        assert cfg["global"]["log_dir"] == "logs"
        assert cfg["global"]["results_dir"] == "results"

    def test_preserves_explicit_round(self, tmp_path: Path) -> None:
        data = _minimal_config()
        data["global"]["round_number"] = 3
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.dump(data))
        cfg = load_config(cfg_file)
        assert cfg["global"]["round_number"] == 3

    def test_enzyme_section_present(self, valid_config: dict) -> None:
        assert "enzyme_A" in valid_config["enzymes"]
        assert "enzyme_B" in valid_config["enzymes"]

    # --- error paths ---

    def test_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/path/config.yaml"))

    def test_raises_value_error_on_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_config(empty)

    def test_raises_value_error_missing_global(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.dump({"databases": {}, "tools": {}, "stages": {}, "enzymes": {}}))
        with pytest.raises(ValueError, match="validation failed"):
            load_config(bad)

    def test_raises_value_error_missing_enzyme_keys(self, tmp_path: Path) -> None:
        data = _minimal_config()
        data["enzymes"]["enzyme_A"] = {"name": "incomplete"}  # missing required keys
        cfg_file = tmp_path / "bad_enzyme.yaml"
        cfg_file.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="validation failed"):
            load_config(cfg_file)

    def test_raises_value_error_bad_enzyme_length(self, tmp_path: Path) -> None:
        data = _minimal_config()
        data["enzymes"]["enzyme_A"]["length"] = -1
        cfg_file = tmp_path / "neg_len.yaml"
        cfg_file.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="validation failed"):
            load_config(cfg_file)


# ---------------------------------------------------------------------------
# load_config — enzyme YAML file references
# ---------------------------------------------------------------------------

class TestLoadConfigEnzymeRefs:
    def test_loads_enzyme_from_separate_yaml(self, tmp_path: Path) -> None:
        enzyme_data = {
            "name": "Enzyme X",
            "length": 100,
            "sequence_file": "inputs/sequences/enzyme_X.fasta",
            "substrate_smiles": "C",
            "product_smiles": "CO",
            "fixed_residues": [10],
            "hotspot_residues": [9, 10, 11],
            "contigs": "A1-100",
            "cofactors": [],
        }
        enzyme_file = tmp_path / "enzyme_X.yaml"
        enzyme_file.write_text(yaml.dump(enzyme_data))

        data = _minimal_config()
        data["enzymes"]["enzyme_X"] = str(enzyme_file)
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.dump(data))

        cfg = load_config(cfg_file)
        assert cfg["enzymes"]["enzyme_X"]["name"] == "Enzyme X"
        assert cfg["enzymes"]["enzyme_X"]["length"] == 100

    def test_raises_for_missing_enzyme_yaml_ref(self, tmp_path: Path) -> None:
        data = _minimal_config()
        data["enzymes"]["enzyme_X"] = "nonexistent.yaml"
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.dump(data))
        with pytest.raises(FileNotFoundError, match="enzyme_X"):
            load_config(cfg_file)


# ---------------------------------------------------------------------------
# get_enzyme_config
# ---------------------------------------------------------------------------

class TestGetEnzymeConfig:
    def test_returns_correct_enzyme(self, valid_config: dict) -> None:
        ea = get_enzyme_config(valid_config, "enzyme_A")
        assert ea["name"] == "Enzyme A"
        assert ea["length"] == 428
        assert ea["substrate_smiles"] == "CCO"

    def test_returns_different_enzymes(self, valid_config: dict) -> None:
        ea = get_enzyme_config(valid_config, "enzyme_A")
        eb = get_enzyme_config(valid_config, "enzyme_B")
        assert ea["length"] != eb["length"]

    def test_raises_key_error_unknown_enzyme(self, valid_config: dict) -> None:
        with pytest.raises(KeyError, match="enzyme_Z"):
            get_enzyme_config(valid_config, "enzyme_Z")

    def test_enzyme_has_all_required_keys(self, valid_config: dict) -> None:
        ea = get_enzyme_config(valid_config, "enzyme_A")
        required = {
            "name", "length", "sequence_file", "substrate_smiles",
            "product_smiles", "fixed_residues", "hotspot_residues",
            "contigs", "cofactors",
        }
        assert required.issubset(ea.keys())


# ---------------------------------------------------------------------------
# resolve_paths
# ---------------------------------------------------------------------------

class TestResolvePaths:
    def test_converts_relative_to_absolute(self, valid_config: dict) -> None:
        base = Path("/project/root")
        resolved = resolve_paths(valid_config, base)
        # log_dir was "logs" (relative) → should be absolute
        log_dir = resolved["global"]["log_dir"]
        assert isinstance(log_dir, Path)
        assert log_dir.is_absolute()
        assert str(log_dir).endswith("logs")

    def test_absolute_paths_unchanged(self, valid_config: dict) -> None:
        base = Path("/project/root")
        resolved = resolve_paths(valid_config, base)
        db_dir = resolved["databases"]["colabfold_db_dir"]
        assert isinstance(db_dir, Path)
        assert str(db_dir) == "/data/colabfold"

    def test_enzyme_sequence_file_resolved(self, valid_config: dict) -> None:
        base = Path("/project/root")
        resolved = resolve_paths(valid_config, base)
        seq = resolved["enzymes"]["enzyme_A"]["sequence_file"]
        assert isinstance(seq, Path)
        assert seq.is_absolute()

    def test_does_not_mutate_original(self, valid_config: dict) -> None:
        base = Path("/project/root")
        original_log = valid_config["global"]["log_dir"]
        resolve_paths(valid_config, base)
        assert valid_config["global"]["log_dir"] == original_log
        assert isinstance(valid_config["global"]["log_dir"], str)

    def test_base_dir_resolved(self, valid_config: dict) -> None:
        base = Path("/project/root")
        resolved = resolve_paths(valid_config, base)
        bd = resolved["global"]["base_dir"]
        assert isinstance(bd, Path)
        assert bd.is_absolute()
