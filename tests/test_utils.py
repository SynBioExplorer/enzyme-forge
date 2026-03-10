"""Tests for pipeline.utils module."""

import gzip
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.utils import (
    bf16_autocast,
    calc_ca_rmsd,
    check_vram_available,
    compress_directory,
    compress_file,
    count_mutations,
    decompress_file,
    get_vram_usage,
    group_by_length,
    IncrementalCheckpoint,
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


# ---------------------------------------------------------------------------
# bf16_autocast
# ---------------------------------------------------------------------------

class TestBf16Autocast:
    """Tests for the bf16_autocast context manager."""

    def test_noop_when_torch_missing(self):
        """bf16_autocast must work as a no-op when torch is not installed."""
        # Temporarily hide torch from the import system
        real_torch = sys.modules.pop("torch", None)
        try:
            # Also prevent 'import torch' inside the context manager from
            # finding a cached module
            with patch.dict(sys.modules, {"torch": None}):
                with bf16_autocast():
                    executed = True
        finally:
            if real_torch is not None:
                sys.modules["torch"] = real_torch

        assert executed

    def test_delegates_to_torch_autocast_when_available(self):
        """bf16_autocast must call torch.amp.autocast with bfloat16."""
        # Build a minimal fake torch module
        fake_torch = types.ModuleType("torch")
        # bfloat16 sentinel
        fake_torch.bfloat16 = "bfloat16_sentinel"

        entered = {}

        class _FakeAutocast:
            def __init__(self, device, dtype):
                entered["device"] = device
                entered["dtype"] = dtype

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        fake_amp = types.ModuleType("torch.amp")
        fake_amp.autocast = _FakeAutocast
        fake_torch.amp = fake_amp

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with bf16_autocast(device="cuda"):
                pass

        assert entered["device"] == "cuda"
        assert entered["dtype"] == "bfloat16_sentinel"

    def test_custom_device_forwarded(self):
        """bf16_autocast must forward a custom device string to torch.amp.autocast."""
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = "bf16"

        captured = {}

        class _Cap:
            def __init__(self, device, dtype):
                captured["device"] = device

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        fake_amp = types.ModuleType("torch.amp")
        fake_amp.autocast = _Cap
        fake_torch.amp = fake_amp

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with bf16_autocast(device="cpu"):
                pass

        assert captured["device"] == "cpu"


# ---------------------------------------------------------------------------
# IncrementalCheckpoint
# ---------------------------------------------------------------------------

class TestIncrementalCheckpoint:
    def test_starts_empty_when_no_file(self, tmp_path):
        ckpt = IncrementalCheckpoint(tmp_path / "ckpt.csv", interval=5)
        assert ckpt.results == []
        assert ckpt.get_completed_ids() == set()

    def test_add_result_and_manual_save(self, tmp_path):
        path = tmp_path / "ckpt.csv"
        ckpt = IncrementalCheckpoint(path, interval=100)
        ckpt.add_result({"sequence_id": "s1", "plddt": 88.0})
        ckpt.save()
        assert path.exists()
        recovered = read_scores(path)
        assert len(recovered) == 1
        assert recovered[0]["sequence_id"] == "s1"

    def test_auto_save_triggers_at_interval(self, tmp_path):
        path = tmp_path / "ckpt.csv"
        ckpt = IncrementalCheckpoint(path, interval=3)
        for i in range(3):
            ckpt.add_result({"sequence_id": f"s{i}", "plddt": float(i)})
        # After exactly 3 additions, auto-save should have fired
        assert path.exists()
        assert len(read_scores(path)) == 3

    def test_auto_save_does_not_trigger_before_interval(self, tmp_path):
        path = tmp_path / "ckpt.csv"
        ckpt = IncrementalCheckpoint(path, interval=10)
        for i in range(5):
            ckpt.add_result({"sequence_id": f"s{i}", "plddt": float(i)})
        # Only 5 of 10 added – no auto-save yet
        assert not path.exists()

    def test_get_completed_ids(self, tmp_path):
        path = tmp_path / "ckpt.csv"
        ckpt = IncrementalCheckpoint(path, interval=100)
        ckpt.add_result({"sequence_id": "alpha", "score": 1.0})
        ckpt.add_result({"sequence_id": "beta", "score": 2.0})
        assert ckpt.get_completed_ids() == {"alpha", "beta"}

    def test_resume_loads_existing_file(self, tmp_path):
        path = tmp_path / "ckpt.csv"
        # Create a pre-existing checkpoint
        write_scores([{"sequence_id": "prev", "plddt": 99.0}], path)

        ckpt = IncrementalCheckpoint(path, interval=100)
        assert "prev" in ckpt.get_completed_ids()

    def test_resume_skips_completed(self, tmp_path):
        path = tmp_path / "ckpt.csv"
        write_scores(
            [{"sequence_id": "done1", "plddt": 90.0},
             {"sequence_id": "done2", "plddt": 91.0}],
            path,
        )
        ckpt = IncrementalCheckpoint(path, interval=100)
        completed = ckpt.get_completed_ids()
        all_ids = {"done1", "done2", "new1"}
        remaining = all_ids - completed
        assert remaining == {"new1"}

    def test_corrupted_checkpoint_does_not_crash(self, tmp_path):
        """IncrementalCheckpoint must not raise even when the checkpoint file
        cannot be parsed.  It either loads partial data or falls back to an
        empty result list after emitting a UserWarning."""
        path = tmp_path / "ckpt.csv"
        # Write genuinely unparseable binary content
        path.write_bytes(b"\xff\xfe" + b"\x00" * 50)
        # Should not raise regardless of whether pandas can parse the file
        try:
            ckpt = IncrementalCheckpoint(path, interval=5)
            _ = ckpt.get_completed_ids()
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"IncrementalCheckpoint raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# File compression helpers
# ---------------------------------------------------------------------------

class TestCompression:
    def test_compress_file_creates_gz(self, tmp_path):
        src = tmp_path / "test.pdb"
        src.write_text("ATOM data here\n")
        gz = compress_file(src)
        assert gz == src.with_suffix(".pdb.gz")
        assert gz.exists()
        assert not src.exists()  # original deleted

    def test_decompress_file_recovers_content(self, tmp_path):
        src = tmp_path / "test.pdb"
        original_content = "ATOM data here\n"
        src.write_text(original_content)
        gz = compress_file(src)
        recovered_path = decompress_file(gz)
        assert recovered_path == src
        assert recovered_path.read_text() == original_content

    def test_compress_decompress_roundtrip_binary(self, tmp_path):
        src = tmp_path / "data.bin"
        payload = bytes(range(256)) * 100
        src.write_bytes(payload)
        gz = compress_file(src)
        out = decompress_file(gz)
        assert out.read_bytes() == payload

    def test_decompress_rejects_non_gz(self, tmp_path):
        bad = tmp_path / "file.txt"
        bad.write_text("hello")
        with pytest.raises(ValueError, match=".gz"):
            decompress_file(bad)

    def test_compress_directory_compresses_matching_files(self, tmp_path):
        pdb1 = tmp_path / "a.pdb"
        pdb2 = tmp_path / "b.pdb"
        txt = tmp_path / "note.txt"
        pdb1.write_text("ATOM 1\n")
        pdb2.write_text("ATOM 2\n")
        txt.write_text("keep me\n")

        compress_directory(tmp_path, pattern="*.pdb")

        assert (tmp_path / "a.pdb.gz").exists()
        assert (tmp_path / "b.pdb.gz").exists()
        assert not (tmp_path / "a.pdb").exists()
        assert not (tmp_path / "b.pdb").exists()
        # Non-matching file untouched
        assert txt.exists()

    def test_compress_directory_custom_pattern(self, tmp_path):
        fa = tmp_path / "seqs.fasta"
        fa.write_text(">A\nMKL\n")
        compress_directory(tmp_path, pattern="*.fasta")
        assert (tmp_path / "seqs.fasta.gz").exists()


# ---------------------------------------------------------------------------
# Mutation counting
# ---------------------------------------------------------------------------

class TestCountMutations:
    def test_identical_sequences_zero(self):
        assert count_mutations("ACDEF", "ACDEF") == 0

    def test_all_different(self):
        assert count_mutations("AAAAA", "CCCCC") == 5

    def test_single_mutation(self):
        assert count_mutations("ACDEF", "ACDEF"[:2] + "X" + "ACDEF"[3:]) == 1

    def test_mixed(self):
        assert count_mutations("MKLIVF", "MKXIVF") == 1

    def test_empty_sequences(self):
        assert count_mutations("", "") == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            count_mutations("ABC", "AB")

    def test_case_sensitive(self):
        # Different case counts as a mutation
        assert count_mutations("MKL", "mkl") == 3


# ---------------------------------------------------------------------------
# group_by_length
# ---------------------------------------------------------------------------

class TestGroupByLength:
    def test_empty_returns_empty(self):
        assert group_by_length({}) == []

    def test_single_sequence(self):
        buckets = group_by_length({"s1": "MKL"}, bucket_size=10)
        assert len(buckets) == 1
        assert buckets[0] == [("s1", "MKL")]

    def test_bucket_size_respected(self):
        seqs = {f"s{i}": "A" * (i + 1) for i in range(10)}
        buckets = group_by_length(seqs, bucket_size=4)
        # 10 sequences / 4 per bucket = 3 buckets (4+4+2)
        assert len(buckets) == 3
        assert len(buckets[0]) == 4
        assert len(buckets[1]) == 4
        assert len(buckets[2]) == 2

    def test_sorted_by_length_within_bucket(self):
        seqs = {"long": "A" * 50, "short": "A" * 5, "medium": "A" * 25}
        buckets = group_by_length(seqs, bucket_size=10)
        assert len(buckets) == 1
        lengths = [len(seq) for _, seq in buckets[0]]
        assert lengths == sorted(lengths)

    def test_all_sequences_covered(self):
        seqs = {f"s{i}": "M" * (i + 1) for i in range(17)}
        buckets = group_by_length(seqs, bucket_size=5)
        total = sum(len(b) for b in buckets)
        assert total == 17

    def test_default_bucket_size_is_10(self):
        seqs = {f"s{i}": "A" * i for i in range(1, 25)}
        buckets = group_by_length(seqs)
        # Default bucket_size=10 => ceil(24/10)=3 buckets
        assert len(buckets) == 3

    def test_returns_list_of_list_of_tuples(self):
        seqs = {"a": "MK", "b": "LIV"}
        buckets = group_by_length(seqs, bucket_size=5)
        assert isinstance(buckets, list)
        assert isinstance(buckets[0], list)
        assert all(isinstance(item, tuple) for item in buckets[0])
