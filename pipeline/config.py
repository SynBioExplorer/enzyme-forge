"""YAML configuration loader with schema validation for the enzyme engineering pipeline.

Implements PipelineConfigProtocol from contracts/config_protocol.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft7Validator, ValidationError

# ---------------------------------------------------------------------------
# JSON Schema for pipeline_config.yaml
# ---------------------------------------------------------------------------

# Keys whose string values should be resolved as filesystem paths.
_PATH_KEYS = frozenset({
    "base_dir",
    "log_dir",
    "results_dir",
    "colabfold_db_dir",
    "mmseqs2_db_dir",
    "uniref30_db",
    "colabfolddb",
    "boltz2_env",
    "ligandmpnn_dir",
    "esmfold_env",
    "rfdiffusion3_foundry_bin",
    "colabfold_env",
    "protenix_env",
    "output_dir",
    "sequence_file",
})

ENZYME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "length", "sequence_file", "substrate_smiles",
                  "product_smiles", "fixed_residues", "hotspot_residues",
                  "contigs", "cofactors"],
    "properties": {
        "name": {"type": "string"},
        "length": {"type": "integer", "minimum": 1},
        "sequence_file": {"type": "string"},
        "substrate_smiles": {"type": "string"},
        "product_smiles": {"type": "string"},
        "fixed_residues": {"type": "array", "items": {"type": "integer"}},
        "hotspot_residues": {"type": "array", "items": {"type": "integer"}},
        "contigs": {"type": "string"},
        "cofactors": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": True,
}

PIPELINE_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["global", "databases", "tools", "stages", "enzymes"],
    "properties": {
        "global": {
            "type": "object",
            "required": ["project_name", "base_dir"],
            "properties": {
                "project_name": {"type": "string"},
                "base_dir": {"type": "string"},
                "log_dir": {"type": "string"},
                "results_dir": {"type": "string"},
                "round_number": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
        "databases": {
            "type": "object",
            "required": ["colabfold_db_dir", "mmseqs2_db_dir"],
            "properties": {
                "colabfold_db_dir": {"type": "string"},
                "mmseqs2_db_dir": {"type": "string"},
                "uniref30_db": {"type": "string"},
                "colabfolddb": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "tools": {
            "type": "object",
            "properties": {
                "boltz2_env": {"type": "string"},
                "ligandmpnn_dir": {"type": "string"},
                "esmfold_env": {"type": "string"},
                "rfdiffusion3_foundry_bin": {"type": "string"},
                "colabfold_env": {"type": "string"},
                "protenix_env": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "stages": {
            "type": "object",
            "additionalProperties": True,
        },
        "enzymes": {
            "type": "object",
            "additionalProperties": ENZYME_SCHEMA,
        },
    },
    "additionalProperties": True,
}

# ---------------------------------------------------------------------------
# Defaults applied after validation
# ---------------------------------------------------------------------------

_GLOBAL_DEFAULTS: dict[str, Any] = {
    "log_dir": "logs",
    "results_dir": "results",
    "round_number": 1,
}


# ---------------------------------------------------------------------------
# Public API — satisfies PipelineConfigProtocol
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Validated configuration dictionary with defaults filled.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        ValueError: If the YAML content fails schema validation.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        raise ValueError("Config file is empty")

    # Inline enzyme YAML references: if an enzyme value is a string path,
    # load that file and merge it in.
    enzymes = data.get("enzymes", {})
    for eid, evalue in list(enzymes.items()):
        if isinstance(evalue, str):
            enzyme_path = Path(evalue)
            if not enzyme_path.is_absolute():
                enzyme_path = config_path.parent / enzyme_path
            if not enzyme_path.exists():
                raise FileNotFoundError(
                    f"Enzyme config file not found: {enzyme_path} "
                    f"(referenced by enzymes.{eid})"
                )
            with open(enzyme_path, "r") as efh:
                enzymes[eid] = yaml.safe_load(efh)

    # Validate against schema
    validator = Draft7Validator(PIPELINE_CONFIG_SCHEMA)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        messages = []
        for err in errors:
            path = ".".join(str(p) for p in err.absolute_path) or "<root>"
            messages.append(f"  {path}: {err.message}")
        raise ValueError(
            "Config validation failed:\n" + "\n".join(messages)
        )

    # Apply defaults
    glb = data["global"]
    for key, default in _GLOBAL_DEFAULTS.items():
        glb.setdefault(key, default)

    return data


def get_enzyme_config(config: dict[str, Any], enzyme_id: str) -> dict[str, Any]:
    """Extract per-enzyme configuration.

    Args:
        config: Full pipeline config dictionary.
        enzyme_id: Enzyme identifier (e.g. ``"enzyme_A"``).

    Returns:
        Dictionary with per-enzyme keys.

    Raises:
        KeyError: If *enzyme_id* is not found in config.
    """
    enzymes = config.get("enzymes", {})
    if enzyme_id not in enzymes:
        raise KeyError(
            f"Enzyme '{enzyme_id}' not found in config. "
            f"Available: {list(enzymes.keys())}"
        )
    return enzymes[enzyme_id]


def resolve_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve all relative paths in *config* to absolute paths.

    String values whose key is in the known path-keys set are converted to
    absolute :class:`~pathlib.Path` objects relative to *base_dir*.

    Args:
        config: Configuration dictionary (not mutated — a deep copy is returned).
        base_dir: Base directory for resolving relative paths.

    Returns:
        A new config dict with path strings replaced by absolute ``Path`` objects.
    """
    base_dir = Path(base_dir).resolve()
    out = copy.deepcopy(config)
    _resolve_paths_recursive(out, base_dir)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_paths_recursive(obj: Any, base_dir: Path) -> None:
    """Walk *obj* in-place, converting path-like string values."""
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            val = obj[key]
            if isinstance(val, str) and key in _PATH_KEYS:
                p = Path(val)
                if not p.is_absolute():
                    p = base_dir / p
                obj[key] = p.resolve()
            elif isinstance(val, (dict, list)):
                _resolve_paths_recursive(val, base_dir)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                _resolve_paths_recursive(item, base_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI: validate a config file."""
    parser = argparse.ArgumentParser(
        description="Validate enzyme engineering pipeline configuration."
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to pipeline_config.yaml",
    )
    parser.add_argument(
        "--validate", action="store_true", default=True,
        help="Validate config (default behaviour).",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Config OK — project: {cfg['global']['project_name']}")
    print(f"  Enzymes: {list(cfg['enzymes'].keys())}")
    print(f"  Round:   {cfg['global']['round_number']}")


if __name__ == "__main__":
    main()
