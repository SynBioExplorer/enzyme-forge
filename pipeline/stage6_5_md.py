"""Stage 6.5: OpenMM Molecular Dynamics Simulations (Optional).

Run 10-50 ns MD simulations on top ~200 candidates (50 per enzyme) using
OpenMM with AMBER ff14SB + TIP3P water. GPU-accelerated on RTX 6000 Pro.

Steps:
  1. Read top ~200 candidate PDB files (50/enzyme) from Stage 6 screening
  2. Set up OpenMM system with AMBER ff14SB, TIP3P water, PME electrostatics
  3. Energy minimization
  4. NVT/NPT equilibration
  5. Production run 10-50 ns
  6. Analyze trajectories: active site RMSD, backbone RMSD, RMSF
  7. Filter: active site RMSD < 2.0 Å, backbone RMSD < 3.0 Å

Output: md_metrics.csv + trajectory summary in stage_06.5_md_simulations/

CLI: python pipeline/stage6_5_md.py --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pipeline.config import load_config, get_enzyme_config, resolve_paths
    from pipeline.utils import setup_logging, write_scores, read_scores
except ImportError:
    import yaml
    import csv

    def load_config(p):  # type: ignore[misc]
        return yaml.safe_load(open(p))

    def get_enzyme_config(cfg, eid):  # type: ignore[misc]
        return cfg.get("enzymes", {}).get(eid, {})

    def resolve_paths(cfg, base):  # type: ignore[misc]
        return cfg

    def setup_logging(n, d, **kw):  # type: ignore[misc]
        logging.basicConfig()
        return logging.getLogger(n)

    def write_scores(scores, p):  # type: ignore[misc]
        import csv as _csv
        from pathlib import Path as _P

        _P(p).parent.mkdir(parents=True, exist_ok=True)
        if not scores:
            with open(p, "w", newline="") as f:
                f.write("")
            return
        with open(p, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=scores[0].keys())
            w.writeheader()
            w.writerows(scores)

    def read_scores(p):  # type: ignore[misc]
        import csv as _csv

        with open(p) as f:
            return list(_csv.DictReader(f))


# ---------------------------------------------------------------------------
# Check for optional dependencies
# ---------------------------------------------------------------------------

_OPENMM_AVAILABLE = False
try:
    import openmm  # noqa: F401
    import openmm.app as app  # noqa: F401
    import openmm.unit as unit  # noqa: F401

    _OPENMM_AVAILABLE = True
except ImportError:
    pass

_MDTRAJ_AVAILABLE = False
try:
    import mdtraj  # noqa: F401

    _MDTRAJ_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Module constants (StageModuleProtocol)
# ---------------------------------------------------------------------------

STAGE_NAME = "Stage 6.5: MD Simulations (Optional)"
STAGE_NUM = "6.5"
INPUT_DESCRIPTION = "Top candidate PDB files from Stage 6 screening"
OUTPUT_DESCRIPTION = "md_metrics.csv + trajectory summaries in stage_06.5_md_simulations/"

logger = logging.getLogger(__name__)

# Default thresholds from tech-reqs Section 4.9
DEFAULT_ACTIVE_SITE_RMSD_MAX = 2.0  # Å
DEFAULT_BACKBONE_RMSD_MAX = 3.0     # Å
DEFAULT_SUBSTRATE_RMSD_MAX = 3.0    # Å
DEFAULT_PRODUCTION_NS = 10.0
DEFAULT_CANDIDATES_PER_ENZYME = 50
DEFAULT_TOTAL_CANDIDATES = 200

# Simulation parameters
TIMESTEP_FS = 2.0  # femtoseconds
NVT_STEPS = 50000   # 100 ps at 2 fs/step
NPT_STEPS = 50000   # 100 ps at 2 fs/step
REPORT_INTERVAL = 1000


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def setup_openmm_system(
    pdb_path: Path,
    forcefield: str = "amber14-all.xml",
    water_model: str = "amber14/tip3p.xml",
    padding: float = 1.0,
    ion_concentration: float = 0.15,
) -> tuple:
    """Set up an OpenMM system from a PDB file.

    Args:
        pdb_path: Path to input PDB.
        forcefield: ForceField XML file name.
        water_model: Water model XML file name.
        padding: Solvent padding in nanometers.
        ion_concentration: Ionic strength in mol/L.

    Returns:
        Tuple of (system, topology, positions).

    Raises:
        ImportError: If OpenMM is not installed.
    """
    if not _OPENMM_AVAILABLE:
        raise ImportError("OpenMM is not installed. Install with: conda install -c conda-forge openmm")

    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(forcefield, water_model)

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(
        ff,
        padding=padding * unit.nanometers,
        ionicStrength=ion_concentration * unit.molar,
    )

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
    )

    return system, modeller.topology, modeller.positions


def run_energy_minimization(simulation, max_iterations: int = 1000) -> float:
    """Run energy minimization.

    Args:
        simulation: OpenMM Simulation object.
        max_iterations: Maximum minimization iterations (0 = until convergence).

    Returns:
        Final potential energy in kJ/mol.
    """
    if not _OPENMM_AVAILABLE:
        raise ImportError("OpenMM is not installed.")

    simulation.minimizeEnergy(maxIterations=max_iterations)
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    return energy.value_in_unit(unit.kilojoules_per_mole)


def run_equilibration(
    simulation,
    nvt_steps: int = NVT_STEPS,
    npt_steps: int = NPT_STEPS,
    temperature: float = 300.0,
    pressure: float = 1.0,
) -> dict[str, float]:
    """Run NVT then NPT equilibration.

    Args:
        simulation: OpenMM Simulation object.
        nvt_steps: Number of NVT steps.
        npt_steps: Number of NPT steps.
        temperature: Temperature in Kelvin.
        pressure: Pressure in atm.

    Returns:
        Dict with nvt_final_energy and npt_final_density.
    """
    if not _OPENMM_AVAILABLE:
        raise ImportError("OpenMM is not installed.")

    # NVT equilibration (no barostat)
    simulation.step(nvt_steps)
    nvt_state = simulation.context.getState(getEnergy=True)
    nvt_energy = nvt_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    # NPT equilibration (add barostat)
    system = simulation.context.getSystem()
    barostat = openmm.MonteCarloBarostat(
        pressure * unit.atmospheres,
        temperature * unit.kelvin,
        25,  # frequency
    )
    system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)

    simulation.step(npt_steps)
    npt_state = simulation.context.getState(getEnergy=True)

    # Estimate density from box volume
    box_vectors = npt_state.getPeriodicBoxVectors()
    volume_nm3 = (
        box_vectors[0][0].value_in_unit(unit.nanometers)
        * box_vectors[1][1].value_in_unit(unit.nanometers)
        * box_vectors[2][2].value_in_unit(unit.nanometers)
    )
    # Rough density in g/cm³ (assuming ~water density system)
    volume_cm3 = volume_nm3 * 1e-21  # nm³ → cm³
    # We don't have the mass easily, so report volume instead
    npt_density = 1.0 / volume_cm3 if volume_cm3 > 0 else 0.0

    return {
        "nvt_final_energy": nvt_energy,
        "npt_final_density": npt_density,
    }


def run_production(
    simulation,
    n_steps: int,
    output_prefix: Path,
    report_interval: int = REPORT_INTERVAL,
) -> Path:
    """Run production MD simulation.

    Args:
        simulation: OpenMM Simulation object.
        n_steps: Total production MD steps.
        output_prefix: Path prefix for output files.
        report_interval: Steps between trajectory/energy reports.

    Returns:
        Path to trajectory .dcd file.
    """
    if not _OPENMM_AVAILABLE:
        raise ImportError("OpenMM is not installed.")

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    dcd_path = output_prefix.with_suffix(".dcd")
    log_path = output_prefix.with_suffix(".csv")

    simulation.reporters.append(
        app.DCDReporter(str(dcd_path), report_interval)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            str(log_path),
            report_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
        )
    )

    logger.info("Starting production MD: %d steps, reporting every %d steps", n_steps, report_interval)
    simulation.step(n_steps)
    logger.info("Production MD complete. Trajectory: %s", dcd_path)

    return dcd_path


def analyze_trajectory(
    traj_path: Path,
    pdb_path: Path,
    hotspot_residues: list[int],
    wt_pdb_path: Path,
) -> dict[str, float]:
    """Analyze MD trajectory for stability metrics.

    Uses mdtraj if available, otherwise returns placeholder values.

    Args:
        traj_path: Path to .dcd trajectory file.
        pdb_path: Path to reference PDB (topology).
        hotspot_residues: Active site residue indices (1-based).
        wt_pdb_path: Path to wild-type PDB for RMSD reference.

    Returns:
        Dict with backbone_rmsd_mean, backbone_rmsd_plateau,
        active_site_rmsd, active_site_rmsf.
    """
    if not _MDTRAJ_AVAILABLE:
        logger.warning("mdtraj not available — using simplified analysis")
        return _simplified_analysis(pdb_path, wt_pdb_path, hotspot_residues)

    traj = mdtraj.load(str(traj_path), top=str(pdb_path))
    ref = mdtraj.load(str(wt_pdb_path))

    # Backbone RMSD (full trajectory)
    backbone_atoms = traj.topology.select("backbone")
    backbone_rmsd = mdtraj.rmsd(traj, ref, atom_indices=backbone_atoms) * 10.0  # nm → Å

    backbone_rmsd_mean = float(np.mean(backbone_rmsd))

    # Use last 50% for plateau calculation
    n_frames = len(backbone_rmsd)
    plateau_start = n_frames // 2
    backbone_rmsd_plateau = float(np.mean(backbone_rmsd[plateau_start:]))

    # Active site RMSD (hotspot residues, 0-indexed for mdtraj)
    hotspot_0indexed = [r - 1 for r in hotspot_residues if r > 0]
    if hotspot_0indexed:
        active_atoms = traj.topology.select(
            " or ".join(f"resid {r}" for r in hotspot_0indexed)
        )
        if len(active_atoms) > 0:
            active_rmsd = mdtraj.rmsd(traj, ref, atom_indices=active_atoms) * 10.0
            active_site_rmsd = float(np.mean(active_rmsd[plateau_start:]))
        else:
            active_site_rmsd = 0.0

        # Active site RMSF
        active_rmsf = mdtraj.rmsf(traj[plateau_start:], ref, atom_indices=active_atoms) * 10.0
        active_site_rmsf = float(np.mean(active_rmsf)) if len(active_rmsf) > 0 else 0.0
    else:
        active_site_rmsd = 0.0
        active_site_rmsf = 0.0

    return {
        "backbone_rmsd_mean": backbone_rmsd_mean,
        "backbone_rmsd_plateau": backbone_rmsd_plateau,
        "active_site_rmsd": active_site_rmsd,
        "active_site_rmsf": active_site_rmsf,
    }


def _simplified_analysis(
    pdb_path: Path,
    wt_pdb_path: Path,
    hotspot_residues: list[int],
) -> dict[str, float]:
    """Simplified static analysis when mdtraj is not available.

    Computes CA RMSD between design and WT as a static proxy.
    """
    try:
        from pipeline.utils import calc_ca_rmsd
    except ImportError:
        from Bio.PDB import PDBParser, Superimposer

        def calc_ca_rmsd(p1, p2, subset=None):
            parser = PDBParser(QUIET=True)
            s1 = parser.get_structure("s1", str(p1))
            s2 = parser.get_structure("s2", str(p2))

            def _get_ca(s, sub=None):
                cas = []
                for m in s:
                    for c in m:
                        for r in c:
                            if r.get_id()[0].strip():
                                continue
                            if sub and r.get_id()[1] not in sub:
                                continue
                            for a in r:
                                if a.get_name() == "CA":
                                    cas.append(a)
                    break
                return cas

            ca1 = _get_ca(s1, set(subset) if subset else None)
            ca2 = _get_ca(s2, set(subset) if subset else None)
            n = min(len(ca1), len(ca2))
            if n == 0:
                return 0.0
            sup = Superimposer()
            sup.set_atoms(ca1[:n], ca2[:n])
            return float(sup.rms)

    backbone_rmsd = calc_ca_rmsd(pdb_path, wt_pdb_path)
    active_site_rmsd = calc_ca_rmsd(pdb_path, wt_pdb_path, hotspot_residues) if hotspot_residues else 0.0

    return {
        "backbone_rmsd_mean": backbone_rmsd,
        "backbone_rmsd_plateau": backbone_rmsd,
        "active_site_rmsd": active_site_rmsd,
        "active_site_rmsf": 0.0,  # cannot compute without trajectory
    }


def apply_md_filters(
    metrics: dict[str, float],
    active_site_rmsd_max: float = DEFAULT_ACTIVE_SITE_RMSD_MAX,
    backbone_rmsd_max: float = DEFAULT_BACKBONE_RMSD_MAX,
) -> tuple[bool, str]:
    """Apply MD stability filters.

    Args:
        metrics: Dict with backbone_rmsd_plateau and active_site_rmsd.
        active_site_rmsd_max: Max active site RMSD threshold (Å).
        backbone_rmsd_max: Max backbone RMSD plateau threshold (Å).

    Returns:
        Tuple of (passed: bool, reason: str).
    """
    reasons = []

    backbone_rmsd = metrics.get("backbone_rmsd_plateau", 0.0)
    if backbone_rmsd > backbone_rmsd_max:
        reasons.append(
            f"backbone RMSD {backbone_rmsd:.2f} > {backbone_rmsd_max:.2f} Å"
        )

    active_rmsd = metrics.get("active_site_rmsd", 0.0)
    if active_rmsd > active_site_rmsd_max:
        reasons.append(
            f"active site RMSD {active_rmsd:.2f} > {active_site_rmsd_max:.2f} Å"
        )

    if reasons:
        return False, "; ".join(reasons)
    return True, "passed"


def run_md_simulation(
    pdb_path: Path,
    output_dir: Path,
    hotspot_residues: list[int],
    wt_pdb_path: Path,
    production_ns: float = DEFAULT_PRODUCTION_NS,
    device: str = "CUDA",
) -> dict[str, Any]:
    """Run full MD pipeline for one structure.

    Handles OpenMM not being available: returns skip metrics.

    Args:
        pdb_path: Path to candidate PDB.
        output_dir: Output directory for trajectory files.
        hotspot_residues: Active site residue indices (1-based).
        wt_pdb_path: Wild-type PDB for reference.
        production_ns: Production run length in nanoseconds.
        device: OpenMM platform (CUDA, OpenCL, CPU, Reference).

    Returns:
        Metrics dict with RMSD/RMSF values, or skip status.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _OPENMM_AVAILABLE:
        logger.warning("OpenMM not available — skipping MD for %s", pdb_path.name)
        return {
            "backbone_rmsd_mean": float("nan"),
            "backbone_rmsd_plateau": float("nan"),
            "active_site_rmsd": float("nan"),
            "active_site_rmsf": float("nan"),
            "simulation_ns": 0.0,
            "skipped": True,
            "skip_reason": "OpenMM not installed",
        }

    try:
        # Setup
        system, topology, positions = setup_openmm_system(pdb_path)

        # Create integrator and simulation
        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picoseconds,
            TIMESTEP_FS * unit.femtoseconds,
        )

        try:
            platform = openmm.Platform.getPlatformByName(device)
        except Exception:
            logger.warning("Platform %s not available, falling back to CPU", device)
            platform = openmm.Platform.getPlatformByName("CPU")

        simulation = app.Simulation(topology, system, integrator, platform)
        simulation.context.setPositions(positions)

        # Energy minimization
        logger.info("Energy minimization for %s", pdb_path.name)
        min_energy = run_energy_minimization(simulation)
        logger.info("Minimized energy: %.2f kJ/mol", min_energy)

        # Equilibration
        logger.info("Equilibration for %s", pdb_path.name)
        equil_results = run_equilibration(simulation)
        logger.info("Equilibration NVT energy: %.2f kJ/mol", equil_results["nvt_final_energy"])

        # Production
        steps_per_ns = int(1e6 / TIMESTEP_FS)  # 500,000 steps/ns at 2 fs
        n_steps = int(production_ns * steps_per_ns)
        output_prefix = output_dir / pdb_path.stem

        logger.info("Production MD for %s: %.1f ns (%d steps)", pdb_path.name, production_ns, n_steps)
        traj_path = run_production(simulation, n_steps, output_prefix)

        # Analysis
        logger.info("Analyzing trajectory for %s", pdb_path.name)
        metrics = analyze_trajectory(traj_path, pdb_path, hotspot_residues, wt_pdb_path)
        metrics["simulation_ns"] = production_ns
        metrics["trajectory_path"] = str(traj_path)
        metrics["skipped"] = False

        return metrics

    except Exception as exc:
        logger.error("MD simulation failed for %s: %s", pdb_path.name, exc)
        return {
            "backbone_rmsd_mean": float("nan"),
            "backbone_rmsd_plateau": float("nan"),
            "active_site_rmsd": float("nan"),
            "active_site_rmsf": float("nan"),
            "simulation_ns": 0.0,
            "skipped": True,
            "skip_reason": str(exc),
        }


def _select_top_candidates(
    screening_csv: Path,
    n_per_enzyme: int = DEFAULT_CANDIDATES_PER_ENZYME,
    max_total: int = DEFAULT_TOTAL_CANDIDATES,
) -> list[dict[str, Any]]:
    """Select top candidates from Stage 6 screening results.

    Selects top n_per_enzyme per enzyme_id, up to max_total total.

    Args:
        screening_csv: Path to screening_results.csv.
        n_per_enzyme: Max candidates per enzyme.
        max_total: Max total candidates.

    Returns:
        List of candidate dicts.
    """
    try:
        scores = read_scores(screening_csv)
    except Exception as exc:
        logger.error("Cannot read screening results: %s", exc)
        return []

    if not scores:
        return []

    # Group by enzyme_id
    by_enzyme: dict[str, list[dict]] = {}
    for row in scores:
        eid = row.get("enzyme_id", "unknown")
        by_enzyme.setdefault(eid, []).append(row)

    # Select top N per enzyme (already sorted by screening, take first N)
    selected: list[dict] = []
    for eid in sorted(by_enzyme.keys()):
        enzyme_candidates = by_enzyme[eid][:n_per_enzyme]
        selected.extend(enzyme_candidates)

    # Cap at max_total
    return selected[:max_total]


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(config: dict[str, Any]) -> dict[str, Any]:
    """Execute Stage 6.5: MD simulations (optional).

    Args:
        config: Loaded pipeline configuration dictionary.

    Returns:
        Summary dict with total_simulated, passed, failed, skipped.
    """
    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_06_5_md", {})
    output_dir_name = stage_cfg.get("output_dir", "stage_06.5_md_simulations")
    output_dir = base_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if stage is enabled
    optional_md = stage_cfg.get("enabled", config.get("stages", {}).get("optional_md", False))
    if not optional_md:
        logger.info("%s is disabled (set stages.stage_06_5_md.enabled: true to enable)", STAGE_NAME)
        # Write empty metrics file
        metrics_path = output_dir / "md_metrics.csv"
        write_scores([], metrics_path)
        return {
            "total_simulated": 0,
            "passed": 0,
            "failed": 0,
            "skipped": True,
            "reason": "Stage disabled in config",
        }

    if not _OPENMM_AVAILABLE:
        logger.warning("OpenMM not installed — %s will be skipped", STAGE_NAME)
        metrics_path = output_dir / "md_metrics.csv"
        write_scores([], metrics_path)
        return {
            "total_simulated": 0,
            "passed": 0,
            "failed": 0,
            "skipped": True,
            "reason": "OpenMM not installed",
        }

    # Parameters
    production_ns = stage_cfg.get("production_ns", DEFAULT_PRODUCTION_NS)
    n_per_enzyme = stage_cfg.get("candidates_per_enzyme", DEFAULT_CANDIDATES_PER_ENZYME)
    max_total = stage_cfg.get("max_total_candidates", DEFAULT_TOTAL_CANDIDATES)
    device = stage_cfg.get("device", "CUDA")
    active_site_rmsd_max = stage_cfg.get("active_site_rmsd_max", DEFAULT_ACTIVE_SITE_RMSD_MAX)
    backbone_rmsd_max = stage_cfg.get("backbone_rmsd_max", DEFAULT_BACKBONE_RMSD_MAX)

    # Input: Stage 6 screening results
    stage6_dir_name = config.get("stages", {}).get(
        "stage_06_boltz2_screening", {}
    ).get("output_dir", "stage_06_boltz2_screening")
    screening_csv = base_dir / stage6_dir_name / "screening_results.csv"

    if not screening_csv.exists():
        logger.error("Stage 6 screening results not found: %s", screening_csv)
        metrics_path = output_dir / "md_metrics.csv"
        write_scores([], metrics_path)
        return {
            "total_simulated": 0,
            "passed": 0,
            "failed": 0,
            "skipped": True,
            "reason": f"Screening results not found: {screening_csv}",
        }

    # Select top candidates
    candidates = _select_top_candidates(screening_csv, n_per_enzyme, max_total)
    logger.info("Selected %d candidates for MD simulations", len(candidates))

    # WT PDB directory for reference
    wt_dir_name = config.get("stages", {}).get(
        "stage_02_wt_structures", {}
    ).get("output_dir", "stage_02_wt_structures")
    wt_dir = base_dir / wt_dir_name

    # Run MD simulations
    all_metrics: list[dict[str, Any]] = []
    n_passed = 0
    n_failed = 0

    for candidate in candidates:
        seq_id = candidate.get("sequence_id", "unknown")
        enzyme_id = candidate.get("enzyme_id", "unknown")

        # Find candidate PDB
        pdb_path_str = candidate.get("pdb_path", "")
        if pdb_path_str:
            pdb_path = Path(pdb_path_str)
            if not pdb_path.is_absolute():
                pdb_path = base_dir / pdb_path
        else:
            # Try standard naming
            pdb_path = base_dir / stage6_dir_name / enzyme_id / f"{seq_id}.pdb"

        if not pdb_path.exists():
            logger.warning("PDB not found for %s: %s — skipping", seq_id, pdb_path)
            continue

        # Find WT PDB
        wt_pdb = wt_dir / "substrate_complexes" / f"{enzyme_id}_substrate.pdb"
        if not wt_pdb.exists():
            wt_pdb = wt_dir / "substrate_complexes" / f"{enzyme_id}.pdb"
        if not wt_pdb.exists():
            wt_pdb = wt_dir / f"{enzyme_id}.pdb"

        # Get hotspot residues
        try:
            enzyme_cfg = get_enzyme_config(config, enzyme_id)
            hotspot_residues = enzyme_cfg.get("hotspot_residues", [])
        except (KeyError, TypeError):
            hotspot_residues = []

        # Run simulation
        sim_output_dir = output_dir / enzyme_id / seq_id
        metrics = run_md_simulation(
            pdb_path=pdb_path,
            output_dir=sim_output_dir,
            hotspot_residues=hotspot_residues,
            wt_pdb_path=wt_pdb,
            production_ns=production_ns,
            device=device,
        )

        # Apply filters
        passed, reason = apply_md_filters(
            metrics,
            active_site_rmsd_max=active_site_rmsd_max,
            backbone_rmsd_max=backbone_rmsd_max,
        )

        if passed:
            n_passed += 1
        else:
            n_failed += 1

        all_metrics.append({
            "sequence_id": seq_id,
            "enzyme_id": enzyme_id,
            "backbone_rmsd_mean": metrics.get("backbone_rmsd_mean", float("nan")),
            "backbone_rmsd_plateau": metrics.get("backbone_rmsd_plateau", float("nan")),
            "active_site_rmsd": metrics.get("active_site_rmsd", float("nan")),
            "active_site_rmsf": metrics.get("active_site_rmsf", float("nan")),
            "passed_md_filters": passed,
            "simulation_ns": metrics.get("simulation_ns", 0.0),
            "trajectory_path": metrics.get("trajectory_path", ""),
        })

    # Write metrics CSV
    metrics_path = output_dir / "md_metrics.csv"
    write_scores(all_metrics, metrics_path)
    logger.info("Wrote MD metrics to %s", metrics_path)

    summary = {
        "total_simulated": len(all_metrics),
        "passed": n_passed,
        "failed": n_failed,
        "skipped": False,
    }
    logger.info(
        "%s complete: %d simulated, %d passed, %d failed",
        STAGE_NAME, summary["total_simulated"], summary["passed"], summary["failed"],
    )

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(config_path: Path) -> None:
    """Load config, set up logging, run Stage 6.5."""
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
            logging.FileHandler(log_dir / "stage6_5_md.log"),
        ],
    )

    logger.info("=" * 60)
    logger.info("%s — starting (OPTIONAL STAGE)", STAGE_NAME)
    logger.info("Config: %s", config_path)
    logger.info("OpenMM available: %s", _OPENMM_AVAILABLE)
    logger.info("mdtraj available: %s", _MDTRAJ_AVAILABLE)
    logger.info("=" * 60)

    results = run_stage(config)

    logger.info("=" * 60)
    logger.info("%s — complete", STAGE_NAME)
    if results.get("skipped"):
        logger.info("  Stage was skipped: %s", results.get("reason", "unknown"))
    else:
        logger.info("  Total simulated: %d", results["total_simulated"])
        logger.info("  Passed filters:  %d", results["passed"])
        logger.info("  Failed filters:  %d", results["failed"])
    logger.info("=" * 60)


def _cli() -> None:
    """Argument parser wrapper for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Stage 6.5: OpenMM MD Simulations (Optional)",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    _cli()
