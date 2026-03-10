"""Tests for pipeline.utils module."""

import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeline.utils import (
    calc_ca_rmsd,
    check_vram_available,
    get_vram_usage,
    md5_checksum,
    merge_scores,
    read_fasta,
    read_pdb,
    read_scores,
    setup_logging,
    write_fasta,
    write_scores,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

class TestFastaIO:
    def test_read_fasta(self):
        seqs = read_fasta(FIXTURES / "sample.fasta")
        assert isinstance(seqs, dict)
        assert len(seqs) == 2
        assert "EnzymeA_WT" in seqs
        assert "EnzymeB_WT" in seqs
        # Sequences should contain only amino acid letters
        for seq in seqs.values():
            assert len(seq) > 0
            assert all(c.isalpha() for c in seq)

    def test_write_read_roundtrip(self, tmp_path):
        original = {"SeqA": "MKLIVFAGPGDPIGMG", "SeqB": "MSTPKELIAEVDAGIR"}
        out = tmp_path / "test.fasta"
        write_fasta(original, out)
        assert out.exists()
        recovered = read_fasta(out)
        assert recovered == original


# ---------------------------------------------------------------------------
# PDB I/O
# ---------------------------------------------------------------------------

class TestPdbIO:
    def test_read_pdb(self):
        data = read_pdb(FIXTURES / "sample.pdb")
        assert "atoms" in data
        assert "residues" in data
        assert "chains" in data
        assert "coords" in data
        assert len(data["atoms"]) > 0
        assert len(data["chains"]) >= 1
        # Should have at least 20 CA atoms
        assert data["coords"].shape[0] >= 20
        assert data["coords"].shape[1] == 3

    def test_calc_ca_rmsd_self(self):
        pdb = FIXTURES / "sample.pdb"
        rmsd = calc_ca_rmsd(pdb, pdb)
        assert isinstance(rmsd, float)
        assert rmsd == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Scores CSV I/O
# ---------------------------------------------------------------------------

class TestScoresIO:
    def test_read_scores(self):
        scores = read_scores(FIXTURES / "sample_scores.csv")
        assert isinstance(scores, list)
        assert len(scores) == 5
        assert "sequence_id" in scores[0]
        assert "plddt" in scores[0]
        assert "rmsd" in scores[0]

    def test_write_read_roundtrip(self, tmp_path):
        original = [
            {"sequence_id": "A1", "plddt": 90.0, "rmsd": 0.5},
            {"sequence_id": "A2", "plddt": 85.0, "rmsd": 1.0},
        ]
        out = tmp_path / "scores.csv"
        write_scores(original, out)
        assert out.exists()
        recovered = read_scores(out)
        assert len(recovered) == 2
        assert recovered[0]["sequence_id"] == "A1"
        assert recovered[0]["plddt"] == pytest.approx(90.0)

    def test_merge_scores(self, tmp_path):
        scores_a = [
            {"sequence_id": "A1", "plddt": 90.0},
            {"sequence_id": "A2", "plddt": 85.0},
        ]
        scores_b = [
            {"sequence_id": "A1", "rmsd": 0.5},
            {"sequence_id": "A2", "rmsd": 1.0},
            {"sequence_id": "A3", "rmsd": 2.0},
        ]
        file_a = tmp_path / "a.csv"
        file_b = tmp_path / "b.csv"
        write_scores(scores_a, file_a)
        write_scores(scores_b, file_b)

        merged_path = tmp_path / "merged.csv"
        merge_scores([file_a, file_b], merged_path)
        merged = read_scores(merged_path)

        # Should have all 3 sequence_ids (outer join)
        ids = {r["sequence_id"] for r in merged}
        assert ids == {"A1", "A2", "A3"}
        # A1 should have both plddt and rmsd
        a1 = [r for r in merged if r["sequence_id"] == "A1"][0]
        assert a1["plddt"] == pytest.approx(90.0)
        assert a1["rmsd"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_setup_logging(self, tmp_path):
        # Use unique logger name to avoid handler accumulation
        logger = setup_logging("test_stage_unique", tmp_path, level=logging.DEBUG)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "pipeline.test_stage_unique"
        logger.info("Test message")
        log_file = tmp_path / "test_stage_unique.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content


# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------

class TestVRAM:
    def test_get_vram_usage_returns_dict(self):
        usage = get_vram_usage()
        assert isinstance(usage, dict)
        assert "total_gb" in usage
        assert "used_gb" in usage
        assert "free_gb" in usage
        assert "utilization_pct" in usage
        for v in usage.values():
            assert isinstance(v, float)

    def test_check_vram_available_returns_bool(self):
        result = check_vram_available(1.0)
        assert isinstance(result, bool)

    def test_get_vram_usage_with_mock_nvidia_smi(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "24576, 8192, 16384, 45"

        with patch("pipeline.utils.subprocess.run", return_value=mock_result):
            usage = get_vram_usage()
            assert usage["total_gb"] == pytest.approx(24.0)
            assert usage["used_gb"] == pytest.approx(8.0)
            assert usage["free_gb"] == pytest.approx(16.0)
            assert usage["utilization_pct"] == pytest.approx(45.0)

    def test_check_vram_available_with_mock(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "98304, 4096, 94208, 10"

        with patch("pipeline.utils.subprocess.run", return_value=mock_result):
            assert check_vram_available(90.0) is True
            assert check_vram_available(95.0) is False


# ---------------------------------------------------------------------------
# MD5 checksum
# ---------------------------------------------------------------------------

class TestMD5:
    def test_md5_checksum(self):
        checksum = md5_checksum(FIXTURES / "sample.fasta")
        assert isinstance(checksum, str)
        assert len(checksum) == 32
        # Same file should always produce same checksum
        assert checksum == md5_checksum(FIXTURES / "sample.fasta")

    def test_md5_different_files(self):
        c1 = md5_checksum(FIXTURES / "sample.fasta")
        c2 = md5_checksum(FIXTURES / "sample_scores.csv")
        assert c1 != c2
