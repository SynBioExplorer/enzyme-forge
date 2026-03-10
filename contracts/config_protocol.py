"""
Contract: PipelineConfigProtocol
Version: da599e4 (commit hash when contract was created)
Generated: 2026-03-10T17:00:00Z

Defines the configuration loading and validation interface used by all pipeline stages.
Every stage module imports the config loader to access YAML-based configuration.
"""
from typing import Protocol, Any
from pathlib import Path


class PipelineConfigProtocol(Protocol):
    """Interface for loading and validating pipeline configuration."""

    @staticmethod
    def load_config(config_path: Path) -> dict[str, Any]:
        """Load and validate a YAML config file.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Validated configuration dictionary.

        Raises:
            FileNotFoundError: If config file does not exist.
            ValueError: If config fails schema validation.
        """
        ...

    @staticmethod
    def get_enzyme_config(config: dict[str, Any], enzyme_id: str) -> dict[str, Any]:
        """Extract per-enzyme configuration.

        Args:
            config: Full pipeline config dictionary.
            enzyme_id: Enzyme identifier (e.g., "enzyme_A").

        Returns:
            Dictionary with keys: name, length, sequence_file, substrate_smiles,
            product_smiles, fixed_residues, hotspot_residues, contigs, cofactors.

        Raises:
            KeyError: If enzyme_id not found in config.
        """
        ...

    @staticmethod
    def resolve_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
        """Resolve all relative paths in config to absolute paths.

        Args:
            config: Configuration dictionary with potentially relative paths.
            base_dir: Base directory for path resolution.

        Returns:
            Config with all paths resolved to absolute Path objects.
        """
        ...
