"""Download and index ColabFoldDB + UniRef30 for MMseqs2-GPU on local NVMe.

Supports aria2c/wget with resume, checksum verification, MMseqs2 createdb/createindex,
disk space checks, and idempotent operation (skip if already indexed).

CLI: python pipeline/setup_databases.py --config config.yaml --db-dir /data/databases
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

try:
    from pipeline.utils import setup_logging
except ImportError:
    def setup_logging(stage_name: str, log_dir: Path | None = None, **kwargs: Any) -> logging.Logger:
        logging.basicConfig(
            level=kwargs.get("level", logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )
        return logging.getLogger(stage_name)

try:
    from pipeline.config import load_config
except ImportError:
    import yaml

    def load_config(config_path: Path) -> dict[str, Any]:
        with open(config_path) as f:
            return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Database definitions
# ---------------------------------------------------------------------------

DATABASE_URLS: dict[str, str] = {
    "uniref30": "http://wwwuser.gwdg.de/~compbiol/colabfold/current/uniref30_2302.tar.gz",
    "colabfolddb": "http://wwwuser.gwdg.de/~compbiol/colabfold/current/colabfold_envdb_202108.tar.gz",
}

# Default expected sizes (GB) for disk-space checks (archive + extracted + index).
DATABASE_REQUIRED_GB: dict[str, float] = {
    "uniref30": 500.0,
    "colabfolddb": 900.0,
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def check_disk_space(path: Path, required_gb: float) -> bool:
    """Return True if *path* has at least *required_gb* GB free."""
    path = Path(path)
    # Walk up until an existing directory is found.
    check_path = path
    while not check_path.exists():
        check_path = check_path.parent
    usage = shutil.disk_usage(check_path)
    free_gb = usage.free / (1024 ** 3)
    logger.info("Disk space at %s: %.1f GB free, %.1f GB required", check_path, free_gb, required_gb)
    return free_gb >= required_gb


def download_file(url: str, dest_path: Path, resume: bool = True) -> Path:
    """Download *url* to *dest_path* using aria2c (preferred) or wget.

    Args:
        url: Remote URL to download.
        dest_path: Local destination file path.
        resume: If True, resume partial downloads.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: If the download fails.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Try aria2c first, fall back to wget.
    if shutil.which("aria2c"):
        cmd = [
            "aria2c",
            "--dir", str(dest_path.parent),
            "--out", dest_path.name,
            "--file-allocation=falloc",
            "--max-connection-per-server=8",
            "--split=8",
        ]
        if resume:
            cmd.append("--continue=true")
        cmd.append(url)
    elif shutil.which("wget"):
        cmd = ["wget", "-O", str(dest_path)]
        if resume:
            cmd.append("--continue")
        cmd.append(url)
    else:
        raise RuntimeError("Neither aria2c nor wget found on PATH")

    logger.info("Downloading %s -> %s", url, dest_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Clean up partial download on failure.
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(
            f"Download failed (exit {result.returncode}): {result.stderr}"
        )

    return dest_path


def verify_checksum(
    file_path: Path,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
) -> bool:
    """Verify file integrity against expected checksum(s).

    At least one of *expected_md5* or *expected_sha256* must be provided.

    Returns:
        True if the checksum(s) match.

    Raises:
        ValueError: If no expected checksum is provided.
    """
    if expected_md5 is None and expected_sha256 is None:
        raise ValueError("At least one expected checksum must be provided")

    file_path = Path(file_path)
    buf_size = 1024 * 1024  # 1 MB

    if expected_md5 is not None:
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(buf_size):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected_md5:
            logger.error("MD5 mismatch for %s: expected %s, got %s", file_path, expected_md5, actual)
            return False
        logger.info("MD5 verified for %s", file_path)

    if expected_sha256 is not None:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(buf_size):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected_sha256:
            logger.error("SHA256 mismatch for %s: expected %s, got %s", file_path, expected_sha256, actual)
            return False
        logger.info("SHA256 verified for %s", file_path)

    return True


def extract_archive(archive_path: Path, dest_dir: Path) -> Path:
    """Extract a .tar.gz archive to *dest_dir*.

    Returns:
        Path to the destination directory.
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s -> %s", archive_path, dest_dir)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(path=dest_dir)
    return dest_dir


def mmseqs2_createdb(fasta_path: Path, db_path: Path, mmseqs2_bin: str = "mmseqs") -> None:
    """Run ``mmseqs createdb`` to convert FASTA to MMseqs2 database.

    Args:
        fasta_path: Path to input FASTA file.
        db_path: Path for the output MMseqs2 database prefix.
        mmseqs2_bin: Path to mmseqs2 binary.

    Raises:
        RuntimeError: If the command fails.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [mmseqs2_bin, "createdb", str(fasta_path), str(db_path)]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mmseqs createdb failed: {result.stderr}")
    logger.info("mmseqs createdb complete: %s", db_path)


def mmseqs2_createindex(
    db_path: Path,
    tmp_dir: Path,
    mmseqs2_bin: str = "mmseqs",
    gpu: bool = True,
) -> None:
    """Run ``mmseqs createindex`` on an MMseqs2 database.

    Args:
        db_path: Path to the MMseqs2 database prefix.
        tmp_dir: Path to temporary directory for indexing.
        mmseqs2_bin: Path to mmseqs2 binary.
        gpu: If True, pass ``--gpu 1`` for GPU-accelerated indexing.

    Raises:
        RuntimeError: If the command fails.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [mmseqs2_bin, "createindex", str(db_path), str(tmp_dir)]
    if gpu:
        cmd.extend(["--gpu", "1"])
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mmseqs createindex failed: {result.stderr}")
    logger.info("mmseqs createindex complete: %s", db_path)


def is_already_indexed(db_path: Path) -> bool:
    """Check whether an MMseqs2 database at *db_path* has been indexed.

    Looks for ``.idx`` and ``.idx.index`` files alongside the database prefix.
    """
    db_path = Path(db_path)
    idx = db_path.parent / (db_path.name + ".idx")
    idx_index = db_path.parent / (db_path.name + ".idx.index")
    exists = idx.exists() and idx_index.exists()
    if exists:
        logger.info("Database already indexed: %s", db_path)
    return exists


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def setup_databases(
    config: dict[str, Any],
    db_dir: Path,
    force: bool = False,
    dry_run: bool = False,
    databases: list[str] | None = None,
) -> dict[str, str]:
    """Download, verify, extract, and index databases.

    Args:
        config: Pipeline config dictionary.
        db_dir: Root directory for databases.
        force: Re-download even if already present.
        dry_run: Only report what would be done.
        databases: List of database keys to process (default: all).

    Returns:
        Dict mapping database name to status string.
    """
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    if databases is None:
        databases = list(DATABASE_URLS.keys())

    db_config = config.get("databases", {})
    checksums = db_config.get("checksums", {})
    mmseqs2_bin = config.get("tools", {}).get("mmseqs2_bin", "mmseqs")

    status: dict[str, str] = {}

    for db_name in databases:
        if db_name not in DATABASE_URLS:
            logger.warning("Unknown database: %s — skipping", db_name)
            status[db_name] = "unknown"
            continue

        db_subdir = db_dir / db_name
        db_prefix = db_subdir / db_name

        # Check idempotency.
        if not force and is_already_indexed(db_prefix):
            status[db_name] = "already_indexed"
            continue

        if dry_run:
            logger.info("[DRY RUN] Would download and index %s", db_name)
            status[db_name] = "dry_run"
            continue

        # Disk-space check.
        required_gb = DATABASE_REQUIRED_GB.get(db_name, 500.0)
        if not check_disk_space(db_dir, required_gb):
            raise RuntimeError(
                f"Insufficient disk space for {db_name}: need {required_gb} GB"
            )

        url = DATABASE_URLS[db_name]
        archive_name = url.rsplit("/", 1)[-1]
        archive_path = db_dir / archive_name

        # Download.
        try:
            download_file(url, archive_path, resume=True)
        except RuntimeError:
            # Clean up partial file.
            if archive_path.exists():
                archive_path.unlink()
            raise

        # Verify checksum if available.
        db_checksums = checksums.get(db_name, {})
        expected_md5 = db_checksums.get("md5")
        expected_sha256 = db_checksums.get("sha256")
        if expected_md5 or expected_sha256:
            if not verify_checksum(archive_path, expected_md5=expected_md5, expected_sha256=expected_sha256):
                archive_path.unlink()
                raise RuntimeError(f"Checksum verification failed for {db_name}")

        # Extract.
        extract_archive(archive_path, db_subdir)

        # Find FASTA files for createdb.
        fasta_files = list(db_subdir.glob("*.fasta")) + list(db_subdir.glob("*.fa"))
        if fasta_files:
            mmseqs2_createdb(fasta_files[0], db_prefix, mmseqs2_bin=mmseqs2_bin)

        # Create index.
        tmp_dir = db_dir / "tmp" / db_name
        mmseqs2_createindex(db_prefix, tmp_dir, mmseqs2_bin=mmseqs2_bin, gpu=True)

        status[db_name] = "indexed"
        logger.info("Database %s setup complete", db_name)

    return status


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point for database setup."""
    parser = argparse.ArgumentParser(
        description="Download and index ColabFoldDB + UniRef30 for MMseqs2-GPU.",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to pipeline YAML configuration file.",
    )
    parser.add_argument(
        "--db-dir", type=Path, default=None,
        help="Override database directory from config.",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Re-download even if databases already exist.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Show what would be downloaded without doing it.",
    )
    parser.add_argument(
        "--databases", nargs="+", default=None,
        choices=["colabfolddb", "uniref30"],
        help="Which databases to download (default: both).",
    )
    args = parser.parse_args(argv)

    log = setup_logging("setup_databases", log_dir=Path("logs"))

    config = load_config(args.config)

    db_dir = args.db_dir
    if db_dir is None:
        db_dir = Path(config.get("databases", {}).get("colabfold_db_dir", "/data/databases"))

    log.info("Database directory: %s", db_dir)

    status = setup_databases(
        config=config,
        db_dir=db_dir,
        force=args.force,
        dry_run=args.dry_run,
        databases=args.databases,
    )

    for db_name, result in status.items():
        log.info("  %s: %s", db_name, result)


if __name__ == "__main__":
    main()
