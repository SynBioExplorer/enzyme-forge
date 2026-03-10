"""Tests for pipeline.setup_databases — no network or GPU required."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest import mock

import pytest

from pipeline.setup_databases import (
    DATABASE_URLS,
    check_disk_space,
    download_file,
    extract_archive,
    is_already_indexed,
    main,
    mmseqs2_createdb,
    mmseqs2_createindex,
    setup_databases,
    verify_checksum,
)


# ---------------------------------------------------------------------------
# check_disk_space
# ---------------------------------------------------------------------------

class TestCheckDiskSpace:
    def test_returns_true_when_sufficient(self, tmp_path: Path) -> None:
        # tmp_path should have many GB free; 0.001 GB = ~1 MB is safe.
        assert check_disk_space(tmp_path, required_gb=0.001) is True

    def test_returns_false_when_insufficient(self, tmp_path: Path) -> None:
        # Requesting an absurdly large amount should always fail.
        assert check_disk_space(tmp_path, required_gb=999_999_999) is False

    def test_handles_nonexistent_path(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does" / "not" / "exist"
        # Should walk up to an existing parent and still succeed.
        assert check_disk_space(nonexistent, required_gb=0.001) is True


# ---------------------------------------------------------------------------
# is_already_indexed
# ---------------------------------------------------------------------------

class TestIsAlreadyIndexed:
    def test_true_when_idx_files_exist(self, tmp_path: Path) -> None:
        db_prefix = tmp_path / "mydb"
        (tmp_path / "mydb.idx").touch()
        (tmp_path / "mydb.idx.index").touch()
        assert is_already_indexed(db_prefix) is True

    def test_false_when_missing_idx(self, tmp_path: Path) -> None:
        db_prefix = tmp_path / "mydb"
        # Only one of the two files.
        (tmp_path / "mydb.idx").touch()
        assert is_already_indexed(db_prefix) is False

    def test_false_when_no_files(self, tmp_path: Path) -> None:
        db_prefix = tmp_path / "mydb"
        assert is_already_indexed(db_prefix) is False


# ---------------------------------------------------------------------------
# verify_checksum
# ---------------------------------------------------------------------------

class TestVerifyChecksum:
    def test_md5_match(self, tmp_path: Path) -> None:
        content = b"hello checksum world"
        file = tmp_path / "test.bin"
        file.write_bytes(content)
        expected_md5 = hashlib.md5(content).hexdigest()
        assert verify_checksum(file, expected_md5=expected_md5) is True

    def test_md5_mismatch(self, tmp_path: Path) -> None:
        file = tmp_path / "test.bin"
        file.write_bytes(b"data")
        assert verify_checksum(file, expected_md5="0" * 32) is False

    def test_sha256_match(self, tmp_path: Path) -> None:
        content = b"sha256 test data"
        file = tmp_path / "test.bin"
        file.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert verify_checksum(file, expected_sha256=expected) is True

    def test_raises_when_no_checksum(self, tmp_path: Path) -> None:
        file = tmp_path / "test.bin"
        file.write_bytes(b"x")
        with pytest.raises(ValueError, match="At least one"):
            verify_checksum(file)


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------

class TestDownloadFile:
    def test_calls_aria2c_when_available(self, tmp_path: Path) -> None:
        dest = tmp_path / "archive.tar.gz"
        with (
            mock.patch("pipeline.setup_databases.shutil.which", return_value="/usr/bin/aria2c"),
            mock.patch("pipeline.setup_databases.subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(returncode=0)
            download_file("http://example.com/file.tar.gz", dest, resume=True)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "aria2c"
            assert "--continue=true" in cmd
            assert "http://example.com/file.tar.gz" in cmd

    def test_falls_back_to_wget(self, tmp_path: Path) -> None:
        dest = tmp_path / "archive.tar.gz"

        def fake_which(name: str) -> str | None:
            return "/usr/bin/wget" if name == "wget" else None

        with (
            mock.patch("pipeline.setup_databases.shutil.which", side_effect=fake_which),
            mock.patch("pipeline.setup_databases.subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(returncode=0)
            download_file("http://example.com/f.tar.gz", dest, resume=True)
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "wget"
            assert "--continue" in cmd

    def test_raises_on_failure(self, tmp_path: Path) -> None:
        dest = tmp_path / "archive.tar.gz"
        with (
            mock.patch("pipeline.setup_databases.shutil.which", return_value="/usr/bin/aria2c"),
            mock.patch("pipeline.setup_databases.subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(returncode=1, stderr="connection refused")
            with pytest.raises(RuntimeError, match="Download failed"):
                download_file("http://example.com/f.tar.gz", dest)

    def test_raises_when_no_downloader(self, tmp_path: Path) -> None:
        dest = tmp_path / "archive.tar.gz"
        with mock.patch("pipeline.setup_databases.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Neither aria2c nor wget"):
                download_file("http://example.com/f.tar.gz", dest)


# ---------------------------------------------------------------------------
# mmseqs2_createdb
# ---------------------------------------------------------------------------

class TestMmseqs2Createdb:
    def test_constructs_correct_command(self, tmp_path: Path) -> None:
        fasta = tmp_path / "input.fasta"
        fasta.touch()
        db_path = tmp_path / "db" / "mydb"

        with mock.patch("pipeline.setup_databases.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            mmseqs2_createdb(fasta, db_path, mmseqs2_bin="mmseqs")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd == ["mmseqs", "createdb", str(fasta), str(db_path)]

    def test_raises_on_failure(self, tmp_path: Path) -> None:
        fasta = tmp_path / "input.fasta"
        fasta.touch()
        db_path = tmp_path / "db" / "mydb"
        with mock.patch("pipeline.setup_databases.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stderr="error")
            with pytest.raises(RuntimeError, match="mmseqs createdb failed"):
                mmseqs2_createdb(fasta, db_path)


# ---------------------------------------------------------------------------
# mmseqs2_createindex
# ---------------------------------------------------------------------------

class TestMmseqs2Createindex:
    def test_gpu_flag(self, tmp_path: Path) -> None:
        db_path = tmp_path / "mydb"
        tmp_dir = tmp_path / "tmp"
        with mock.patch("pipeline.setup_databases.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            mmseqs2_createindex(db_path, tmp_dir, gpu=True)
            cmd = mock_run.call_args[0][0]
            assert "--gpu" in cmd
            assert "1" in cmd

    def test_no_gpu_flag(self, tmp_path: Path) -> None:
        db_path = tmp_path / "mydb"
        tmp_dir = tmp_path / "tmp"
        with mock.patch("pipeline.setup_databases.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            mmseqs2_createindex(db_path, tmp_dir, gpu=False)
            cmd = mock_run.call_args[0][0]
            assert "--gpu" not in cmd


# ---------------------------------------------------------------------------
# setup_databases — orchestration
# ---------------------------------------------------------------------------

class TestSetupDatabases:
    def test_skips_when_already_indexed(self, tmp_path: Path) -> None:
        """If index files exist, setup_databases should skip download."""
        db_dir = tmp_path / "dbs"
        # Pre-create index files for uniref30.
        uniref_dir = db_dir / "uniref30"
        uniref_dir.mkdir(parents=True)
        (uniref_dir / "uniref30.idx").touch()
        (uniref_dir / "uniref30.idx.index").touch()

        config: dict = {"databases": {}, "tools": {}}
        status = setup_databases(config, db_dir, databases=["uniref30"])
        assert status["uniref30"] == "already_indexed"

    def test_dry_run(self, tmp_path: Path) -> None:
        db_dir = tmp_path / "dbs"
        config: dict = {"databases": {}, "tools": {}}
        status = setup_databases(config, db_dir, dry_run=True, databases=["uniref30"])
        assert status["uniref30"] == "dry_run"

    def test_raises_on_insufficient_disk_space(self, tmp_path: Path) -> None:
        db_dir = tmp_path / "dbs"
        config: dict = {"databases": {}, "tools": {}}
        with mock.patch("pipeline.setup_databases.check_disk_space", return_value=False):
            with pytest.raises(RuntimeError, match="Insufficient disk space"):
                setup_databases(config, db_dir, databases=["uniref30"])

    def test_unknown_database(self, tmp_path: Path) -> None:
        db_dir = tmp_path / "dbs"
        config: dict = {"databases": {}, "tools": {}}
        status = setup_databases(config, db_dir, databases=["nonexistent"])
        assert status["nonexistent"] == "unknown"


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_parses_config_and_db_dir(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "global:\n  project_name: test\n  base_dir: .\n"
            "databases:\n  colabfold_db_dir: /data/db\n  mmseqs2_db_dir: /data/mmseqs\n"
            "tools: {}\nstages: {}\nenzymes: {}\n"
        )
        db_dir = tmp_path / "databases"

        with mock.patch("pipeline.setup_databases.setup_databases") as mock_setup:
            mock_setup.return_value = {"uniref30": "indexed"}
            main(["--config", str(config_file), "--db-dir", str(db_dir)])
            mock_setup.assert_called_once()
            call_kwargs = mock_setup.call_args
            assert call_kwargs.kwargs["db_dir"] == db_dir

    def test_force_flag(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "global:\n  project_name: test\n  base_dir: .\n"
            "databases:\n  colabfold_db_dir: /data/db\n  mmseqs2_db_dir: /data/mmseqs\n"
            "tools: {}\nstages: {}\nenzymes: {}\n"
        )

        with mock.patch("pipeline.setup_databases.setup_databases") as mock_setup:
            mock_setup.return_value = {}
            main(["--config", str(config_file), "--db-dir", str(tmp_path), "--force"])
            assert mock_setup.call_args.kwargs["force"] is True

    def test_dry_run_flag(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "global:\n  project_name: test\n  base_dir: .\n"
            "databases:\n  colabfold_db_dir: /data/db\n  mmseqs2_db_dir: /data/mmseqs\n"
            "tools: {}\nstages: {}\nenzymes: {}\n"
        )

        with mock.patch("pipeline.setup_databases.setup_databases") as mock_setup:
            mock_setup.return_value = {}
            main(["--config", str(config_file), "--db-dir", str(tmp_path), "--dry-run"])
            assert mock_setup.call_args.kwargs["dry_run"] is True

    def test_databases_selection(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "global:\n  project_name: test\n  base_dir: .\n"
            "databases:\n  colabfold_db_dir: /data/db\n  mmseqs2_db_dir: /data/mmseqs\n"
            "tools: {}\nstages: {}\nenzymes: {}\n"
        )

        with mock.patch("pipeline.setup_databases.setup_databases") as mock_setup:
            mock_setup.return_value = {}
            main([
                "--config", str(config_file),
                "--db-dir", str(tmp_path),
                "--databases", "uniref30",
            ])
            assert mock_setup.call_args.kwargs["databases"] == ["uniref30"]
