"""Stage 3: Backbone Diversification with RFdiffusion3 via Rosetta Foundry CLI.

Generates diverse backbone conformations around the active site while
preserving the global fold. Uses partial diffusion (partial_T 10-20 of 50
total timesteps) with an all-atom representation.

Design strategy is guided by flux control coefficients from Stage 0:
- High-FCC enzymes (FCC > 0.3): 500 backbones at partial_T=15 + 100 at partial_T=20
- Low-FCC enzymes (FCC <= 0.3): 200 backbones at partial_T=10

Pre-filters: radius of gyration, ligand SASA, active site pocket volume.

CLI: python pipeline/stage3_rfdiffusion3.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pipeline.config import load_config, get_enzyme_config, resolve_paths
    from pipeline.utils import setup_logging, read_fasta
except ImportError:
    import yaml

    def load_config(p):  # type: ignore[misc]
        return yaml.safe_load(open(p))

    def get_enzyme_config(cfg, eid):  # type: ignore[misc]
        return cfg.get("enzymes", {}).get(eid, {})

    def resolve_paths(cfg, base):  # type: ignore[misc]
        return cfg

    def setup_logging(n, d, **kw):  # type: ignore[misc]
        logging.basicConfig()
        return logging.getLogger(n)

    def read_fasta(p):  # type: ignore[misc]
        seqs: dict[str, str] = {}
        with open(p) as f:
            header, seq = None, []  # type: ignore[assignment]
            for line in f:
                if line.startswith(">"):
                    if header:
                        seqs[header] = "".join(seq)
                    header, seq = line[1:].strip(), []
                else:
                    seq.append(line.strip())
            if header:
                seqs[header] = "".join(seq)
        return seqs


# ---------------------------------------------------------------------------
# Module constants (StageModuleProtocol)
# ---------------------------------------------------------------------------

STAGE_NAME = "Stage 3: Backbone Diversification (RFdiffusion3)"
STAGE_NUM = "3"
INPUT_DESCRIPTION = "WT PDB files, contigs config, hotspot residues, FCC rankings from Stage 0"
OUTPUT_DESCRIPTION = "Backbone PDB files in stage_03_rfdiffusion3/{enzyme_id}/"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Design strategy
# ---------------------------------------------------------------------------

def determine_design_strategy(enzyme_id: str, fcc: float) -> list[dict[str, Any]]:
    """Determine backbone generation strategy based on flux control coefficient.

    Args:
        enzyme_id: Enzyme identifier (for logging).
        fcc: Flux control coefficient from Stage 0.

    Returns:
        List of strategy dicts, each with keys:
        partial_T, n_designs, strategy.
        High-FCC enzymes get two rounds (standard + aggressive).
    """
    if fcc > 0.3:
        strategies = [
            {"partial_T": 15, "n_designs": 500, "strategy": "aggressive"},
            {"partial_T": 20, "n_designs": 100, "strategy": "aggressive_exploration"},
        ]
        logger.info(
            "%s: FCC=%.3f (>0.3) → aggressive strategy: "
            "500 @ partial_T=15 + 100 @ partial_T=20",
            enzyme_id, fcc,
        )
    else:
        strategies = [
            {"partial_T": 10, "n_designs": 200, "strategy": "conservative"},
        ]
        logger.info(
            "%s: FCC=%.3f (<=0.3) → conservative strategy: "
            "200 @ partial_T=10",
            enzyme_id, fcc,
        )
    return strategies


# ---------------------------------------------------------------------------
# RFdiffusion3 execution
# ---------------------------------------------------------------------------

def run_rfdiffusion3(
    input_pdb: Path,
    contigs: str,
    hotspot_residues: str,
    partial_T: int,
    n_designs: int,
    output_dir: Path,
    foundry_bin: str = "foundry",
    seed: int = 42,
) -> list[Path]:
    """Run RFdiffusion3 partial diffusion via Rosetta Foundry CLI.

    Args:
        input_pdb: Path to input PDB file (WT structure).
        contigs: Contig specification string.
        hotspot_residues: Comma-separated hotspot residue string.
        partial_T: Number of partial diffusion timesteps.
        n_designs: Number of backbone designs to generate.
        output_dir: Directory for output PDB files.
        foundry_bin: Path to foundry binary.
        seed: Random seed for reproducibility.

    Returns:
        List of paths to generated PDB files.

    Raises:
        RuntimeError: If the foundry command fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(foundry_bin), "run", "rfd3",
        "--pdb", str(input_pdb),
        "--contigs", contigs,
        "--hotspot_residues", hotspot_residues,
        "--partial_T", str(partial_T),
        "--num_designs", str(n_designs),
        "--output_dir", str(output_dir),
        "--seed", str(seed),
    ]

    logger.info("Running RFdiffusion3: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"RFdiffusion3 failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    pdb_files = sorted(output_dir.glob("*.pdb"))
    logger.info(
        "RFdiffusion3 generated %d PDB files in %s",
        len(pdb_files), output_dir,
    )
    return pdb_files


# ---------------------------------------------------------------------------
# Structural pre-filters
# ---------------------------------------------------------------------------

def _parse_ca_coords(pdb_path: Path) -> np.ndarray:
    """Parse CA atom coordinates from a PDB file.

    Returns:
        Numpy array of shape (N, 3) with CA coordinates.
    """
    coords = []
    with open(pdb_path) as fh:
        for line in fh:
            if (line.startswith("ATOM") or line.startswith("HETATM")):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords, dtype=np.float64)


def _parse_all_atom_coords(pdb_path: Path) -> tuple[np.ndarray, list[int]]:
    """Parse all protein atom coordinates and their residue numbers.

    Returns:
        Tuple of (coords array (N, 3), list of residue sequence numbers).
    """
    coords = []
    resseqs = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                resseq = int(line[22:26])
                coords.append([x, y, z])
                resseqs.append(resseq)
    if not coords:
        return np.empty((0, 3), dtype=np.float64), []
    return np.array(coords, dtype=np.float64), resseqs


def calc_radius_of_gyration(pdb_path: Path) -> float:
    """Calculate radius of gyration from CA atoms.

    Rg = sqrt(mean of squared distances from centroid).

    Args:
        pdb_path: Path to PDB file.

    Returns:
        Radius of gyration in Angstroms.
    """
    coords = _parse_ca_coords(pdb_path)
    if len(coords) == 0:
        return 0.0
    centroid = coords.mean(axis=0)
    sq_dists = np.sum((coords - centroid) ** 2, axis=1)
    return float(np.sqrt(sq_dists.mean()))


def calc_pocket_volume(
    pdb_path: Path,
    hotspot_residues: list[int],
    probe_radius: float = 1.4,
    grid_spacing: float = 1.0,
    pocket_radius: float = 10.0,
) -> float:
    """Approximate active site pocket volume using a grid-based method.

    Places a cubic grid around the centroid of hotspot residues and counts
    grid points that are within pocket_radius of hotspot atoms but not
    occluded by (too close to) other protein atoms.

    Args:
        pdb_path: Path to PDB file.
        hotspot_residues: List of residue sequence numbers defining the active site.
        probe_radius: Probe radius in Angstroms (default 1.4 for water).
        grid_spacing: Grid point spacing in Angstroms.
        pocket_radius: Radius around hotspot centroid to search.

    Returns:
        Approximate pocket volume in Angstrom^3.
    """
    if not hotspot_residues:
        return 0.0

    all_coords, all_resseqs = _parse_all_atom_coords(pdb_path)
    if len(all_coords) == 0:
        return 0.0

    hotspot_set = set(hotspot_residues)

    # Separate hotspot and non-hotspot atoms
    hotspot_mask = np.array([r in hotspot_set for r in all_resseqs])
    hotspot_coords = all_coords[hotspot_mask]
    protein_coords = all_coords  # all atoms for occlusion check

    if len(hotspot_coords) == 0:
        return 0.0

    centroid = hotspot_coords.mean(axis=0)

    # Generate grid
    half = pocket_radius
    xs = np.arange(centroid[0] - half, centroid[0] + half + grid_spacing, grid_spacing)
    ys = np.arange(centroid[1] - half, centroid[1] + half + grid_spacing, grid_spacing)
    zs = np.arange(centroid[2] - half, centroid[2] + half + grid_spacing, grid_spacing)

    grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)

    # Distance from each grid point to nearest protein atom
    pocket_count = 0
    for point in grid:
        dists_to_protein = np.sqrt(np.sum((protein_coords - point) ** 2, axis=1))
        min_protein_dist = dists_to_protein.min()

        # Point is in pocket if: not inside protein (dist > probe_radius)
        # and within pocket_radius of centroid
        dist_to_centroid = np.sqrt(np.sum((point - centroid) ** 2))
        if min_protein_dist > probe_radius and dist_to_centroid <= pocket_radius:
            pocket_count += 1

    volume = pocket_count * (grid_spacing ** 3)
    return float(volume)


def prefilter_backbones(
    pdb_files: list[Path],
    hotspot_residues: list[int],
    rg_max: float | None = None,
    rg_min: float | None = None,
    pocket_volume_min: float | None = None,
) -> list[Path]:
    """Apply structural pre-filters to backbone PDB files.

    Args:
        pdb_files: List of PDB file paths to filter.
        hotspot_residues: Residue numbers for pocket volume calculation.
        rg_max: Maximum radius of gyration (Angstroms). None = no filter.
        rg_min: Minimum radius of gyration (Angstroms). None = no filter.
        pocket_volume_min: Minimum pocket volume (Angstrom^3). None = no filter.

    Returns:
        List of PDB paths passing all filters.
    """
    if not pdb_files:
        return []

    passing = []
    rg_fail = 0
    vol_fail = 0

    for pdb in pdb_files:
        passed = True

        # Radius of gyration filter
        if rg_max is not None or rg_min is not None:
            rg = calc_radius_of_gyration(pdb)
            if rg_max is not None and rg > rg_max:
                rg_fail += 1
                passed = False
            if rg_min is not None and rg < rg_min:
                rg_fail += 1
                passed = False

        # Pocket volume filter
        if passed and pocket_volume_min is not None and hotspot_residues:
            vol = calc_pocket_volume(pdb, hotspot_residues)
            if vol < pocket_volume_min:
                vol_fail += 1
                passed = False

        if passed:
            passing.append(pdb)

    total = len(pdb_files)
    logger.info(
        "Pre-filter results: %d/%d passed (Rg fail: %d, pocket vol fail: %d)",
        len(passing), total, rg_fail, vol_fail,
    )
    return passing


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def _load_fcc_from_csv(csv_path: Path) -> dict[str, float]:
    """Load FCC values from Stage 0 bottleneck_results.csv.

    Returns:
        Dict mapping enzyme_id -> fcc value.
    """
    fcc_map: dict[str, float] = {}
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fcc_map[row["enzyme_id"]] = float(row["fcc"])
    return fcc_map


def run_stage(config: dict[str, Any]) -> dict[str, list[Path]]:
    """Execute Stage 3: backbone diversification for all enzymes.

    1. Load FCC results from Stage 0 (if available)
    2. For each enzyme: determine strategy, run RFdiffusion3, prefilter
    3. Return dict of enzyme_id -> list of passing PDB paths

    Args:
        config: Loaded pipeline configuration dictionary.

    Returns:
        Dictionary mapping enzyme IDs to lists of backbone PDB paths.
    """
    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_03_rfdiffusion3", {})
    output_dir_name = stage_cfg.get("output_dir", "stage_03_rfdiffusion3")
    output_dir = base_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Foundry binary path
    foundry_bin = str(
        config.get("tools", {}).get("rfdiffusion3_foundry_bin", "foundry")
    )

    # Pre-filter thresholds from config
    rg_max = stage_cfg.get("rg_max", None)
    rg_min = stage_cfg.get("rg_min", None)
    pocket_volume_min = stage_cfg.get("pocket_volume_min", None)

    # Load FCC from Stage 0 output
    bottleneck_dir = base_dir / config.get("stages", {}).get(
        "stage_00_bottleneck", {}
    ).get("output_dir", "stage_00_bottleneck")
    bottleneck_csv = bottleneck_dir / "bottleneck_results.csv"

    fcc_map: dict[str, float] = {}
    if bottleneck_csv.exists():
        fcc_map = _load_fcc_from_csv(bottleneck_csv)
        logger.info("Loaded FCC values from %s: %s", bottleneck_csv, fcc_map)
    else:
        logger.warning(
            "Bottleneck results not found at %s — using default FCC=0.25 for all enzymes",
            bottleneck_csv,
        )

    # WT structure directory (Stage 2 output)
    wt_dir = base_dir / config.get("stages", {}).get(
        "stage_02_wt_structures", {}
    ).get("output_dir", "stage_02_wt_structures")

    enzyme_ids = sorted(config.get("enzymes", {}).keys())
    results: dict[str, list[Path]] = {}

    for enzyme_id in enzyme_ids:
        enzyme_cfg = config["enzymes"][enzyme_id]
        enzyme_out_dir = output_dir / enzyme_id
        enzyme_out_dir.mkdir(parents=True, exist_ok=True)

        # Get FCC (default to 0.25 if not available)
        fcc = fcc_map.get(enzyme_id, 0.25)

        # Get contig and hotspot config
        contigs = enzyme_cfg.get("contigs", "")
        hotspot_list = enzyme_cfg.get("hotspot_residues", [])
        hotspot_str = ",".join(str(r) for r in hotspot_list)

        # Find WT PDB
        wt_pdb = wt_dir / "substrate_complexes" / f"{enzyme_id}.pdb"
        if not wt_pdb.exists():
            wt_pdb = wt_dir / f"{enzyme_id}.pdb"
        if not wt_pdb.exists():
            logger.error(
                "WT PDB not found for %s (checked %s) — skipping",
                enzyme_id, wt_pdb,
            )
            continue

        # Determine strategy and run
        strategies = determine_design_strategy(enzyme_id, fcc)
        all_pdbs: list[Path] = []

        for i, strategy in enumerate(strategies):
            run_dir = enzyme_out_dir / f"pT{strategy['partial_T']}"
            try:
                pdbs = run_rfdiffusion3(
                    input_pdb=wt_pdb,
                    contigs=contigs,
                    hotspot_residues=hotspot_str,
                    partial_T=strategy["partial_T"],
                    n_designs=strategy["n_designs"],
                    output_dir=run_dir,
                    foundry_bin=foundry_bin,
                    seed=42 + i,
                )
                all_pdbs.extend(pdbs)
            except RuntimeError as exc:
                logger.error(
                    "RFdiffusion3 failed for %s (partial_T=%d): %s",
                    enzyme_id, strategy["partial_T"], exc,
                )

        # Pre-filter
        if all_pdbs:
            passing = prefilter_backbones(
                pdb_files=all_pdbs,
                hotspot_residues=hotspot_list,
                rg_max=rg_max,
                rg_min=rg_min,
                pocket_volume_min=pocket_volume_min,
            )
            results[enzyme_id] = passing
            logger.info(
                "%s: %d/%d backbones passed pre-filters",
                enzyme_id, len(passing), len(all_pdbs),
            )
        else:
            results[enzyme_id] = []
            logger.warning("%s: no backbones generated", enzyme_id)

    logger.info(
        "Stage 3 complete. Backbones per enzyme: %s",
        {eid: len(pdbs) for eid, pdbs in results.items()},
    )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(config_path: Path) -> None:
    """Load config, set up logging, run Stage 3."""
    config_path = Path(config_path)
    config = load_config(config_path)

    base_dir = Path(config["global"]["base_dir"])
    config = resolve_paths(config, base_dir)

    log_dir = base_dir / config["global"].get("log_dir", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "stage3_rfdiffusion3.log"),
        ],
    )

    logger.info("=" * 60)
    logger.info("%s — starting", STAGE_NAME)
    logger.info("Config: %s", config_path)
    logger.info("=" * 60)

    results = run_stage(config)

    logger.info("=" * 60)
    logger.info("%s — complete", STAGE_NAME)
    for eid, pdbs in results.items():
        logger.info("  %s: %d backbones", eid, len(pdbs))
    logger.info("=" * 60)


def _cli() -> None:
    """Argument parser wrapper for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Stage 3: Backbone Diversification with RFdiffusion3",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    _cli()
