"""Tests for Stage 1: MSA Generation (pipeline/stage1_msa.py).

All tests run without colabfold, MMseqs2, or GPU.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.stage1_msa import (
    STAGE_NAME,
    STAGE_NUM,
    INPUT_DESCRIPTION,
    OUTPUT_DESCRIPTION,
    parse_a3m,
    validate_a3m,
    parse_a3m_stats,
    run_colabfold_search,
    run_stage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_A3M = """\
>query_enzyme
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
>hit_1
MVLSPADKTNVKAAWGKVGA-AGEYGAEALERMFLSFPTTKTYFPHFDLSH
>hit_2
MVLS--DKTNVKAAWGKVGAHAGEYGAEALERMFL---TTKTYFPHFDLSH
>hit_3
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDlshinsert
"""

SHALLOW_A3M = """\
>query
ACDEF
>hit1
ACD-F
"""

EMPTY_A3M = """\
>query
ACDEF
"""


@pytest.fixture
def a3m_file(tmp_path: Path) -> Path:
    """Write a minimal A3M file and return its path."""
    p = tmp_path / "test.a3m"
    p.write_text(MINIMAL_A3M)
    return p


@pytest.fixture
def shallow_a3m_file(tmp_path: Path) -> Path:
    p = tmp_path / "shallow.a3m"
    p.write_text(SHALLOW_A3M)
    return p


@pytest.fixture
def single_seq_a3m(tmp_path: Path) -> Path:
    p = tmp_path / "single.a3m"
    p.write_text(EMPTY_A3M)
    return p


# ---------------------------------------------------------------------------
# Tests: parse_a3m
# ---------------------------------------------------------------------------

class TestParseA3m:
    def test_parses_correct_number_of_sequences(self, a3m_file: Path):
        query, seqs = parse_a3m(a3m_file)
        assert len(seqs) == 4  # query + 3 hits

    def test_query_is_first_sequence(self, a3m_file: Path):
        query, seqs = parse_a3m(a3m_file)
        assert query.startswith("MVLSPADKTNVKAAWGKVGA")

    def test_single_sequence(self, single_seq_a3m: Path):
        query, seqs = parse_a3m(single_seq_a3m)
        assert len(seqs) == 1
        assert query == "ACDEF"


# ---------------------------------------------------------------------------
# Tests: validate_a3m
# ---------------------------------------------------------------------------

class TestValidateA3m:
    def test_returns_correct_depth(self, a3m_file: Path):
        stats = validate_a3m(a3m_file)
        assert stats["depth"] == 4

    def test_returns_nonzero_coverage(self, a3m_file: Path):
        stats = validate_a3m(a3m_file)
        assert 0.0 < stats["coverage"] <= 1.0

    def test_returns_neff(self, a3m_file: Path):
        stats = validate_a3m(a3m_file)
        assert stats["neff"] > 0

    def test_returns_path(self, a3m_file: Path):
        stats = validate_a3m(a3m_file)
        assert stats["path"] == str(a3m_file)

    def test_warns_on_low_depth(self, shallow_a3m_file: Path):
        """Depth 2 < default min_depth 10 -> should warn."""
        stats = validate_a3m(shallow_a3m_file, min_depth=10)
        assert stats["depth"] == 2

    def test_single_sequence_depth(self, single_seq_a3m: Path):
        stats = validate_a3m(single_seq_a3m)
        assert stats["depth"] == 1

    def test_coverage_is_float(self, a3m_file: Path):
        stats = validate_a3m(a3m_file)
        assert isinstance(stats["coverage"], float)


# ---------------------------------------------------------------------------
# Tests: parse_a3m_stats
# ---------------------------------------------------------------------------

class TestParseA3mStats:
    def test_includes_per_position_coverage(self, a3m_file: Path):
        stats = parse_a3m_stats(a3m_file)
        assert "per_position_coverage" in stats
        assert isinstance(stats["per_position_coverage"], list)
        assert len(stats["per_position_coverage"]) > 0

    def test_per_position_values_in_range(self, a3m_file: Path):
        stats = parse_a3m_stats(a3m_file)
        for val in stats["per_position_coverage"]:
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Tests: run_colabfold_search
# ---------------------------------------------------------------------------

class TestRunColabfoldSearch:
    @patch("pipeline.stage1_msa.subprocess.run")
    def test_constructs_correct_command(self, mock_run: MagicMock, tmp_path: Path):
        """Verify the subprocess command contains expected arguments."""
        # Setup: create a fake .a3m output so the function can find it
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        (out_dir / "query.a3m").write_text(">q\nACDEF\n")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        query = tmp_path / "query.fasta"
        query.write_text(">q\nACDEF\n")
        db_dir = tmp_path / "db"

        run_colabfold_search(query, db_dir, out_dir)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "colabfold_search" in cmd[0]
        assert "--mmseqs" in cmd
        assert "--db-load-mode" in cmd
        assert "2" in cmd
        assert str(query) in cmd
        assert str(db_dir) in cmd
        assert str(out_dir) in cmd

    @patch("pipeline.stage1_msa.subprocess.run")
    def test_raises_on_nonzero_exit(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="search failed"
        )

        query = tmp_path / "query.fasta"
        query.write_text(">q\nACDEF\n")
        db_dir = tmp_path / "db"
        out_dir = tmp_path / "output"

        with pytest.raises(RuntimeError, match="colabfold_search failed"):
            run_colabfold_search(query, db_dir, out_dir)

    @patch("pipeline.stage1_msa.subprocess.run")
    def test_returns_a3m_path(self, mock_run: MagicMock, tmp_path: Path):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        expected = out_dir / "result.a3m"
        expected.write_text(">q\nACDEF\n")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        query = tmp_path / "query.fasta"
        query.write_text(">q\nACDEF\n")

        result = run_colabfold_search(query, tmp_path / "db", out_dir)
        assert result.suffix == ".a3m"
        assert result.exists()

    @patch("pipeline.stage1_msa.subprocess.run")
    def test_raises_if_no_a3m_produced(self, mock_run: MagicMock, tmp_path: Path):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        # No .a3m file created

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        query = tmp_path / "query.fasta"
        query.write_text(">q\nACDEF\n")

        with pytest.raises(RuntimeError, match="no .a3m files"):
            run_colabfold_search(query, tmp_path / "db", out_dir)


# ---------------------------------------------------------------------------
# Tests: run_stage
# ---------------------------------------------------------------------------

class TestRunStage:
    def _make_config(self, tmp_path: Path) -> dict:
        """Create a minimal config for testing run_stage."""
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
                "stage_01_msa": {"output_dir": "stage_01_msa"},
            },
            "enzymes": {
                "enzyme_A": {
                    "name": "Enzyme A",
                    "length": 428,
                    "sequence_file": "inputs/sequences/enzyme_A.fasta",
                    "substrate_smiles": "C",
                    "product_smiles": "C",
                    "fixed_residues": [],
                    "hotspot_residues": [],
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
                    "hotspot_residues": [],
                    "contigs": "A1-512",
                    "cofactors": [],
                },
            },
        }

    @patch("pipeline.stage1_msa.run_colabfold_search")
    def test_calls_search_for_each_enzyme(self, mock_search: MagicMock, tmp_path: Path):
        config = self._make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        # Mock returns a fake .a3m path with valid content
        def fake_search(query_fasta, db_dir, output_dir, **kwargs):
            a3m = output_dir / "result.a3m"
            a3m.write_text(">q\nMVLSPADKTNVKAAWGKVGA\n>h1\nMVLSPADKTNVKAAWGKVGA\n")
            return a3m

        mock_search.side_effect = fake_search

        results = run_stage(config)

        assert mock_search.call_count == 2
        assert "enzyme_A" in results
        assert "enzyme_B" in results

    @patch("pipeline.stage1_msa.run_colabfold_search")
    def test_creates_output_directory(self, mock_search: MagicMock, tmp_path: Path):
        config = self._make_config(tmp_path)
        base_dir = Path(config["global"]["base_dir"])

        def fake_search(query_fasta, db_dir, output_dir, **kwargs):
            a3m = output_dir / "result.a3m"
            a3m.write_text(">q\nACDEF\n>h1\nACDEF\n")
            return a3m

        mock_search.side_effect = fake_search

        run_stage(config)

        output_dir = base_dir / "stage_01_msa"
        assert output_dir.exists()
        assert output_dir.is_dir()

    @patch("pipeline.stage1_msa.run_colabfold_search")
    def test_handles_search_failure_gracefully(self, mock_search: MagicMock, tmp_path: Path):
        config = self._make_config(tmp_path)
        mock_search.side_effect = RuntimeError("search failed")

        # Should not raise — errors are logged and enzyme is skipped
        results = run_stage(config)
        assert results == {}

    @patch("pipeline.stage1_msa.run_colabfold_search")
    def test_results_contain_a3m_paths(self, mock_search: MagicMock, tmp_path: Path):
        config = self._make_config(tmp_path)

        def fake_search(query_fasta, db_dir, output_dir, **kwargs):
            a3m = output_dir / "result.a3m"
            a3m.write_text(">q\nACDEF\n>h1\nACDEF\n")
            return a3m

        mock_search.side_effect = fake_search

        results = run_stage(config)
        for eid, path in results.items():
            assert path.endswith(".a3m")


# ---------------------------------------------------------------------------
# Tests: module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_stage_name(self):
        assert STAGE_NAME == "Stage 1: MSA Generation"

    def test_stage_num(self):
        assert STAGE_NUM == "1"

    def test_input_description(self):
        assert "FASTA" in INPUT_DESCRIPTION

    def test_output_description(self):
        assert ".a3m" in OUTPUT_DESCRIPTION
