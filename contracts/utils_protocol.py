"""
Contract: PipelineUtilsProtocol
Version: da599e4 (commit hash when contract was created)
Generated: 2026-03-10T17:00:00Z

Defines common utility functions shared across all pipeline stages:
FASTA/PDB I/O, CSV helpers, logging setup, VRAM monitoring.
"""
from typing import Protocol
from pathlib import Path
import logging


class FastaIOProtocol(Protocol):
    """Interface for reading and writing FASTA files."""

    @staticmethod
    def read_fasta(fasta_path: Path) -> dict[str, str]:
        """Read a FASTA file and return {header: sequence} dict.

        Args:
            fasta_path: Path to the FASTA file.

        Returns:
            Dictionary mapping sequence IDs to sequences.
        """
        ...

    @staticmethod
    def write_fasta(sequences: dict[str, str], output_path: Path) -> None:
        """Write sequences to a FASTA file.

        Args:
            sequences: Dictionary mapping sequence IDs to sequences.
            output_path: Path to write the FASTA file.
        """
        ...


class StructureIOProtocol(Protocol):
    """Interface for reading and writing structure files (PDB/CIF)."""

    @staticmethod
    def read_pdb(pdb_path: Path) -> dict:
        """Read a PDB file and return parsed structure data.

        Args:
            pdb_path: Path to the PDB/CIF file.

        Returns:
            Dictionary with keys: atoms, residues, chains, coords (numpy array).
        """
        ...

    @staticmethod
    def write_pdb(structure: dict, output_path: Path) -> None:
        """Write structure data to a PDB file.

        Args:
            structure: Parsed structure dictionary.
            output_path: Path for the output PDB file.
        """
        ...

    @staticmethod
    def calc_ca_rmsd(pdb_path_1: Path, pdb_path_2: Path,
                     residue_subset: list[int] | None = None) -> float:
        """Calculate Calpha RMSD between two structures.

        Args:
            pdb_path_1: Path to first PDB file.
            pdb_path_2: Path to second PDB file.
            residue_subset: Optional list of residue indices to compare.

        Returns:
            RMSD value in Angstroms.
        """
        ...


class ScoreIOProtocol(Protocol):
    """Interface for reading and writing score CSV files."""

    @staticmethod
    def read_scores(csv_path: Path) -> list[dict]:
        """Read a scores CSV file.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            List of dicts with keys: sequence_id, metric_name, score, plus extras.
        """
        ...

    @staticmethod
    def write_scores(scores: list[dict], output_path: Path) -> None:
        """Write scores to a CSV file.

        Args:
            scores: List of score dictionaries.
            output_path: Path for the output CSV file.
        """
        ...

    @staticmethod
    def merge_scores(score_files: list[Path], output_path: Path) -> None:
        """Merge multiple score CSV files into one, joining on sequence_id.

        Args:
            score_files: List of CSV file paths to merge.
            output_path: Path for the merged output CSV.
        """
        ...


class LoggingProtocol(Protocol):
    """Interface for structured logging setup."""

    @staticmethod
    def setup_logging(
        stage_name: str,
        log_dir: Path,
        level: int = logging.INFO,
        json_format: bool = True,
    ) -> logging.Logger:
        """Set up structured logging for a pipeline stage.

        Args:
            stage_name: Name of the stage (e.g., "stage1_msa").
            log_dir: Directory for log files.
            level: Logging level.
            json_format: If True, use JSON log format.

        Returns:
            Configured logger instance.
        """
        ...


class VRAMMonitorProtocol(Protocol):
    """Interface for GPU VRAM monitoring."""

    @staticmethod
    def get_vram_usage() -> dict[str, float]:
        """Get current VRAM usage.

        Returns:
            Dict with keys: total_gb, used_gb, free_gb, utilization_pct.
        """
        ...

    @staticmethod
    def check_vram_available(required_gb: float) -> bool:
        """Check if sufficient VRAM is available.

        Args:
            required_gb: Required VRAM in GB.

        Returns:
            True if sufficient VRAM is available.
        """
        ...
