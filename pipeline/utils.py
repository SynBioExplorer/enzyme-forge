"""Common utilities for the enzyme engineering pipeline.

Implements protocols from contracts/utils_protocol.py:
FASTA I/O, PDB I/O, scores CSV I/O, structured logging,
VRAM monitoring, and MD5 checksum helpers.

Hardware optimisation utilities (bf16_autocast, IncrementalCheckpoint,
compress_file, decompress_file, compress_directory, count_mutations,
group_by_length) are appended at the end of this module.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import shutil
import subprocess
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Superimposer


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

def read_fasta(fasta_path: Path) -> dict[str, str]:
    """Read a FASTA file and return {header: sequence} dict."""
    sequences: dict[str, str] = {}
    current_header: str | None = None
    current_seq: list[str] = []

    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            sequences[current_header] = "".join(current_seq)

    return sequences


def write_fasta(sequences: dict[str, str], output_path: Path) -> None:
    """Write sequences to a FASTA file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for header, seq in sequences.items():
            fh.write(f">{header}\n")
            # Write sequence in 80-char lines
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")


# ---------------------------------------------------------------------------
# PDB I/O
# ---------------------------------------------------------------------------

_pdb_parser = PDBParser(QUIET=True)


def read_pdb(pdb_path: Path) -> dict:
    """Read a PDB file and return parsed structure data.

    Returns:
        Dictionary with keys:
        - atoms: list of dicts (name, resname, chain, resseq, coord)
        - residues: list of residue identifiers
        - chains: list of chain IDs
        - coords: numpy array of CA positions (N x 3)
    """
    structure = _pdb_parser.get_structure("struct", str(pdb_path))
    model = structure[0]

    atoms: list[dict] = []
    residues: list[str] = []
    chains: list[str] = []
    ca_coords: list[list[float]] = []

    seen_chains: set[str] = set()
    seen_residues: set[tuple] = set()

    for chain in model:
        chain_id = chain.id
        if chain_id not in seen_chains:
            chains.append(chain_id)
            seen_chains.add(chain_id)

        for residue in chain:
            het = residue.get_id()[0]
            if het.strip() and het != " ":
                continue  # skip HETATMs

            res_key = (chain_id, residue.get_id()[1], residue.get_resname())
            if res_key not in seen_residues:
                residues.append(f"{chain_id}:{residue.get_resname()}{residue.get_id()[1]}")
                seen_residues.add(res_key)

            for atom in residue:
                atoms.append({
                    "name": atom.get_name(),
                    "resname": residue.get_resname(),
                    "chain": chain_id,
                    "resseq": residue.get_id()[1],
                    "coord": atom.get_vector().get_array().tolist(),
                })
                if atom.get_name() == "CA":
                    ca_coords.append(atom.get_vector().get_array().tolist())

    return {
        "atoms": atoms,
        "residues": residues,
        "chains": chains,
        "coords": np.array(ca_coords, dtype=np.float64),
    }


def write_pdb(structure: dict, output_path: Path) -> None:
    """Write structure data to a PDB file.

    Accepts the dict format returned by read_pdb.
    If the dict contains a '_biopython_structure' key (Bio.PDB Structure),
    it will be written directly. Otherwise, writes atoms from the 'atoms' list.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "_biopython_structure" in structure:
        io = PDBIO()
        io.set_structure(structure["_biopython_structure"])
        io.save(str(output_path))
        return

    # Write from atoms list
    with open(output_path, "w") as fh:
        for i, atom in enumerate(structure.get("atoms", []), start=1):
            coord = atom["coord"]
            fh.write(
                f"ATOM  {i:5d}  {atom['name']:<3s} {atom['resname']:>3s} "
                f"{atom['chain']}{atom['resseq']:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00\n"
            )
        fh.write("END\n")


def calc_ca_rmsd(
    pdb_path_1: Path,
    pdb_path_2: Path,
    residue_subset: list[int] | None = None,
) -> float:
    """Calculate Calpha RMSD between two structures using Bio.PDB.Superimposer."""
    parser = PDBParser(QUIET=True)
    struct1 = parser.get_structure("s1", str(pdb_path_1))
    struct2 = parser.get_structure("s2", str(pdb_path_2))

    def _get_ca_atoms(structure, subset=None):
        cas = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    het = residue.get_id()[0]
                    if het.strip() and het != " ":
                        continue
                    if subset is not None and residue.get_id()[1] not in subset:
                        continue
                    for atom in residue:
                        if atom.get_name() == "CA":
                            cas.append(atom)
            break  # first model only
        return cas

    subset_set = set(residue_subset) if residue_subset else None
    ca1 = _get_ca_atoms(struct1, subset_set)
    ca2 = _get_ca_atoms(struct2, subset_set)

    # Use the shorter list length to handle different chain lengths
    n = min(len(ca1), len(ca2))
    if n == 0:
        return 0.0

    ca1 = ca1[:n]
    ca2 = ca2[:n]

    sup = Superimposer()
    sup.set_atoms(ca1, ca2)
    return float(sup.rms)


# ---------------------------------------------------------------------------
# Scores CSV I/O
# ---------------------------------------------------------------------------

def read_scores(csv_path: Path) -> list[dict]:
    """Read a scores CSV file and return list of dicts."""
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


def write_scores(scores: list[dict], output_path: Path) -> None:
    """Write scores to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(scores)
    df.to_csv(output_path, index=False)


def merge_scores(score_files: list[Path], output_path: Path) -> None:
    """Merge multiple score CSV files into one, joining on sequence_id."""
    if not score_files:
        return

    dfs = [pd.read_csv(f) for f in score_files]
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="sequence_id", how="outer")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(
    stage_name: str,
    log_dir: Path,
    level: int = logging.INFO,
    json_format: bool = True,
) -> logging.Logger:
    """Set up structured logging for a pipeline stage."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"pipeline.{stage_name}")
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # File handler
    log_file = log_dir / f"{stage_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    if json_format:
        formatter = _JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------

def get_vram_usage() -> dict[str, float]:
    """Get current VRAM usage. Tries nvidia-smi, then pynvml, then returns zeros."""
    # Try nvidia-smi first
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 4:
                total_mb = float(parts[0].strip())
                used_mb = float(parts[1].strip())
                free_mb = float(parts[2].strip())
                util_pct = float(parts[3].strip())
                return {
                    "total_gb": total_mb / 1024.0,
                    "used_gb": used_mb / 1024.0,
                    "free_gb": free_mb / 1024.0,
                    "utilization_pct": util_pct,
                }
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Try pynvml
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        total_gb = mem_info.total / (1024**3)
        used_gb = mem_info.used / (1024**3)
        free_gb = mem_info.free / (1024**3)
        pynvml.nvmlShutdown()
        return {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "utilization_pct": float(util.gpu),
        }
    except Exception:
        pass

    # Fallback: no GPU available
    warnings.warn("No GPU detected. Returning zero VRAM usage.", stacklevel=2)
    return {
        "total_gb": 0.0,
        "used_gb": 0.0,
        "free_gb": 0.0,
        "utilization_pct": 0.0,
    }


def check_vram_available(required_gb: float) -> bool:
    """Check if sufficient VRAM is available."""
    usage = get_vram_usage()
    if usage["total_gb"] == 0.0:
        return False
    return usage["free_gb"] >= required_gb


# ---------------------------------------------------------------------------
# MD5 checksum helper
# ---------------------------------------------------------------------------

def md5_checksum(file_path: Path) -> str:
    """Compute MD5 checksum of a file for reproducibility tracking."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


# ---------------------------------------------------------------------------
# Hardware optimisation utilities
# ---------------------------------------------------------------------------

@contextmanager
def bf16_autocast(device: str = "cuda") -> Generator[None, None, None]:
    """Context manager for BF16 mixed precision inference on Blackwell GPUs.

    Falls back to a no-op context when PyTorch is not installed so that the
    rest of the pipeline can still import and run on CPU-only machines.

    Args:
        device: PyTorch device string, e.g. ``'cuda'`` or ``'cpu'``.

    Example::

        with bf16_autocast():
            logits = model(inputs)
    """
    try:
        import torch  # noqa: PLC0415

        with torch.amp.autocast(device, dtype=torch.bfloat16):
            yield
    except ImportError:
        yield  # No-op if torch not available


class IncrementalCheckpoint:
    """Save results incrementally to allow crash recovery.

    Results are persisted to a CSV file every *interval* additions so that a
    pipeline stage can resume from the last checkpoint rather than restarting
    from scratch.

    Args:
        checkpoint_path: Path to the CSV file used for persistent storage.
        interval: Number of ``add_result`` calls between automatic saves.

    Example::

        ckpt = IncrementalCheckpoint(Path("run/ckpt.csv"), interval=100)
        completed = ckpt.get_completed_ids()
        for seq_id, seq in sequences.items():
            if seq_id in completed:
                continue
            result = run_inference(seq_id, seq)
            ckpt.add_result(result)
    """

    def __init__(self, checkpoint_path: Path, interval: int = 200) -> None:
        self.path = Path(checkpoint_path)
        self.interval = interval
        self.results: list[dict] = []
        self._unsaved_count: int = 0
        self._load_existing()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_existing(self) -> None:
        """Load previously checkpointed results on resume."""
        if self.path.exists():
            try:
                existing = read_scores(self.path)
                self.results = existing
            except Exception:
                # Corrupted checkpoint – start fresh but keep the file for
                # manual inspection.
                warnings.warn(
                    f"Could not read checkpoint {self.path}. Starting fresh.",
                    stacklevel=2,
                )
                self.results = []
        else:
            self.results = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_completed_ids(self) -> set:
        """Return set of already-completed sequence_ids."""
        return {r["sequence_id"] for r in self.results if "sequence_id" in r}

    def add_result(self, result: dict) -> None:
        """Add a result and auto-save if *interval* results have accumulated.

        Args:
            result: Dictionary containing at minimum a ``sequence_id`` key.
        """
        self.results.append(result)
        self._unsaved_count += 1
        if self._unsaved_count >= self.interval:
            self.save()
            self._unsaved_count = 0

    def save(self) -> None:
        """Force save current results to CSV."""
        if not self.results:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_scores(self.results, self.path)


# ---------------------------------------------------------------------------
# File compression helpers
# ---------------------------------------------------------------------------

def compress_file(path: Path) -> Path:
    """Gzip compress a file, returning the ``.gz`` path, then delete original.

    Args:
        path: Path to the file to compress.

    Returns:
        Path to the newly created ``.gz`` file.
    """
    path = Path(path)
    gz_path = path.with_suffix(path.suffix + ".gz")
    with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    path.unlink()
    return gz_path


def decompress_file(gz_path: Path) -> Path:
    """Decompress a ``.gz`` file, returning the path to the decompressed file.

    The ``.gz`` extension is stripped to derive the output path.

    Args:
        gz_path: Path to the ``.gz`` file.

    Returns:
        Path to the decompressed file.
    """
    gz_path = Path(gz_path)
    if gz_path.suffix != ".gz":
        raise ValueError(f"Expected a .gz file, got: {gz_path}")
    out_path = gz_path.with_suffix("")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


def compress_directory(dir_path: Path, pattern: str = "*.pdb") -> None:
    """Gzip compress all files matching *pattern* inside *dir_path*.

    Each matched file is compressed in-place (original deleted, ``.gz`` kept).

    Args:
        dir_path: Directory to search (non-recursive).
        pattern: Glob pattern, e.g. ``"*.pdb"`` or ``"*.fasta"``.
    """
    dir_path = Path(dir_path)
    for file_path in sorted(dir_path.glob(pattern)):
        compress_file(file_path)


# ---------------------------------------------------------------------------
# Mutation counting
# ---------------------------------------------------------------------------

def count_mutations(seq1: str, seq2: str) -> int:
    """Count the number of positions where *seq1* and *seq2* differ.

    Args:
        seq1: First amino-acid sequence string.
        seq2: Second amino-acid sequence string (must be same length as *seq1*).

    Returns:
        Integer count of mismatching positions.

    Raises:
        ValueError: If the sequences have different lengths.
    """
    if len(seq1) != len(seq2):
        raise ValueError(
            f"Sequences must have the same length ({len(seq1)} != {len(seq2)})"
        )
    return sum(a != b for a, b in zip(seq1, seq2))


# ---------------------------------------------------------------------------
# Batch inference helper
# ---------------------------------------------------------------------------

def group_by_length(
    sequences: dict[str, str],
    bucket_size: int = 10,
) -> list[list[tuple[str, str]]]:
    """Group sequences into length buckets for efficient GPU batching.

    Sequences are sorted by length and then chunked into groups of
    *bucket_size* so that each batch contains sequences of similar length,
    reducing padding overhead.

    Args:
        sequences: Mapping of ``{sequence_id: sequence}`` strings.
        bucket_size: Maximum number of sequences per bucket.

    Returns:
        List of buckets; each bucket is a list of ``(sequence_id, sequence)``
        tuples sorted by ascending sequence length within the bucket.

    Example::

        buckets = group_by_length(seqs, bucket_size=8)
        for batch in buckets:
            run_batch(batch)
    """
    if not sequences:
        return []

    # Sort all entries by sequence length ascending
    sorted_items = sorted(sequences.items(), key=lambda kv: len(kv[1]))

    buckets: list[list[tuple[str, str]]] = []
    for i in range(0, len(sorted_items), bucket_size):
        buckets.append(sorted_items[i : i + bucket_size])

    return buckets
