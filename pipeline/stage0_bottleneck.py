"""Stage 0: Metabolic bottleneck identification using COBRApy + yeast-GEM.

Determines which enzyme(s) exert the most flux control over pathway output
before committing GPU time to design. Uses flux control coefficients (FCCs)
computed by perturbing each enzyme's Vmax ±10%.

CLI: python pipeline/stage0_bottleneck.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from pipeline.config import load_config, get_enzyme_config, resolve_paths
except ImportError:
    import yaml

    def load_config(p):  # type: ignore[misc]
        return yaml.safe_load(open(p))

    def get_enzyme_config(cfg, eid):  # type: ignore[misc]
        return cfg["enzymes"][eid]

    def resolve_paths(cfg, base):  # type: ignore[misc]
        return cfg

# ---------------------------------------------------------------------------
# Module-level constants (StageModuleProtocol)
# ---------------------------------------------------------------------------

STAGE_NAME = "Stage 0: Bottleneck Identification"
STAGE_NUM = "0"
INPUT_DESCRIPTION = "WT FASTA sequences, pipeline config YAML, yeast-GEM SBML model"
OUTPUT_DESCRIPTION = "bottleneck_results.csv with FCC ranking"

# ---------------------------------------------------------------------------
# COBRApy import with graceful fallback
# ---------------------------------------------------------------------------

_COBRA_AVAILABLE = True
try:
    import cobra  # type: ignore[import-untyped]
    from cobra.io import read_sbml_model  # type: ignore[import-untyped]
except ImportError:
    _COBRA_AVAILABLE = False
    cobra = None  # type: ignore[assignment]
    read_sbml_model = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Placeholder kinetic values when CatPred is unavailable
_PLACEHOLDER_KINETICS: dict[str, dict[str, float]] = {
    "enzyme_A": {"kcat": 12.5, "km": 0.45},
    "enzyme_B": {"kcat": 8.3, "km": 0.82},
    "enzyme_C": {"kcat": 25.1, "km": 0.31},
    "enzyme_D": {"kcat": 15.7, "km": 0.55},
}

# Placeholder FCC values when COBRApy is unavailable
_PLACEHOLDER_FCC: dict[str, float] = {
    "enzyme_A": 0.4,
    "enzyme_B": 0.3,
    "enzyme_C": 0.2,
    "enzyme_D": 0.1,
}


# ---------------------------------------------------------------------------
# CatPred interface
# ---------------------------------------------------------------------------

def run_catpred(
    sequence: str,
    substrate_smiles: str,
    enzyme_id: str,
    catpred_dir: Path,
) -> dict[str, Any]:
    """Predict kcat and Km for a sequence/substrate pair using CatPred.

    Calls CatPred via subprocess. Falls back to placeholder values if
    CatPred is not installed or the subprocess fails.

    Returns:
        Dictionary with keys: kcat, km, kcat_unit, km_unit.
    """
    predict_script = catpred_dir / "catpred" / "predict.py"
    if not predict_script.exists():
        logger.warning(
            "CatPred not found at %s — using placeholder kinetics for %s",
            predict_script,
            enzyme_id,
        )
        fallback = _PLACEHOLDER_KINETICS.get(
            enzyme_id, {"kcat": 10.0, "km": 0.5}
        )
        return {
            "kcat": fallback["kcat"],
            "km": fallback["km"],
            "kcat_unit": "s^-1",
            "km_unit": "mM",
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / f"{enzyme_id}_catpred.json"
        cmd = [
            sys.executable,
            str(predict_script),
            "--sequence", sequence,
            "--substrate", substrate_smiles,
            "--output", str(output_json),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            with open(output_json) as fh:
                result = json.load(fh)
            return {
                "kcat": float(result.get("kcat", 10.0)),
                "km": float(result.get("km", 0.5)),
                "kcat_unit": result.get("kcat_unit", "s^-1"),
                "km_unit": result.get("km_unit", "mM"),
            }
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "CatPred failed for %s: %s — using placeholder kinetics",
                enzyme_id, exc,
            )
            fallback = _PLACEHOLDER_KINETICS.get(
                enzyme_id, {"kcat": 10.0, "km": 0.5}
            )
            return {
                "kcat": fallback["kcat"],
                "km": fallback["km"],
                "kcat_unit": "s^-1",
                "km_unit": "mM",
            }


# ---------------------------------------------------------------------------
# COBRApy model loading
# ---------------------------------------------------------------------------

def load_yeast_gem(gem_path: Path) -> Any:
    """Load the yeast-GEM SBML model, or create a toy 4-reaction model.

    Args:
        gem_path: Path to the yeast-GEM SBML file.

    Returns:
        A cobra.Model instance.
    """
    if not _COBRA_AVAILABLE:
        raise ImportError("COBRApy is not installed")

    if gem_path.exists():
        logger.info("Loading yeast-GEM from %s", gem_path)
        return read_sbml_model(str(gem_path))

    logger.warning(
        "yeast-GEM not found at %s — building toy 4-reaction pathway model",
        gem_path,
    )
    return _build_toy_model()


def _build_toy_model() -> Any:
    """Build a simplified 4-enzyme linear pathway model for testing."""
    model = cobra.Model("toy_pathway")

    # Metabolites
    sub1 = cobra.Metabolite("sub1_c", name="Substrate 1", compartment="c")
    int1 = cobra.Metabolite("int1_c", name="Intermediate 1", compartment="c")
    int2 = cobra.Metabolite("int2_c", name="Intermediate 2", compartment="c")
    int3 = cobra.Metabolite("int3_c", name="Intermediate 3", compartment="c")
    prod = cobra.Metabolite("prod_c", name="Final product", compartment="c")

    # Reactions (enzyme A → B → C → D)
    rxn_a = cobra.Reaction("rxn_enzyme_A")
    rxn_a.name = "Enzyme A reaction"
    rxn_a.lower_bound = 0
    rxn_a.upper_bound = 1000
    rxn_a.add_metabolites({sub1: -1, int1: 1})

    rxn_b = cobra.Reaction("rxn_enzyme_B")
    rxn_b.name = "Enzyme B reaction"
    rxn_b.lower_bound = 0
    rxn_b.upper_bound = 1000
    rxn_b.add_metabolites({int1: -1, int2: 1})

    rxn_c = cobra.Reaction("rxn_enzyme_C")
    rxn_c.name = "Enzyme C reaction"
    rxn_c.lower_bound = 0
    rxn_c.upper_bound = 1000
    rxn_c.add_metabolites({int2: -1, int3: 1})

    rxn_d = cobra.Reaction("rxn_enzyme_D")
    rxn_d.name = "Enzyme D reaction"
    rxn_d.lower_bound = 0
    rxn_d.upper_bound = 1000
    rxn_d.add_metabolites({int3: -1, prod: 1})

    # Exchange reactions (substrate in, product out)
    rxn_sub_ex = cobra.Reaction("EX_sub1")
    rxn_sub_ex.lower_bound = -10
    rxn_sub_ex.upper_bound = 0
    rxn_sub_ex.add_metabolites({sub1: -1})

    rxn_prod_ex = cobra.Reaction("EX_prod")
    rxn_prod_ex.lower_bound = 0
    rxn_prod_ex.upper_bound = 1000
    rxn_prod_ex.add_metabolites({prod: -1})

    model.add_reactions([rxn_a, rxn_b, rxn_c, rxn_d, rxn_sub_ex, rxn_prod_ex])
    model.objective = "EX_prod"

    return model


# ---------------------------------------------------------------------------
# FBA and FCC calculation
# ---------------------------------------------------------------------------

def run_fba(model: Any) -> dict[str, float]:
    """Run flux balance analysis on the model.

    Returns:
        Dictionary mapping reaction IDs to flux values.
    """
    solution = model.optimize()
    if solution.status != "optimal":
        logger.warning("FBA solution status: %s", solution.status)
        return {}
    return {rxn.id: solution.fluxes[rxn.id] for rxn in model.reactions}


def calculate_fcc(
    model: Any,
    enzyme_id: str,
    reaction_id: str,
    perturbation: float = 0.1,
) -> float:
    """Calculate flux control coefficient for an enzyme reaction.

    Uses central difference approximation:
        FCC = (dJ/J) / (dVmax/Vmax)

    where J is the objective flux and Vmax is the reaction upper bound.

    Args:
        model: cobra.Model instance.
        enzyme_id: Enzyme identifier (for logging).
        reaction_id: Reaction ID in the model to perturb.
        perturbation: Fractional perturbation (default 0.1 = ±10%).

    Returns:
        Flux control coefficient (float).
    """
    try:
        rxn = model.reactions.get_by_id(reaction_id)
    except KeyError:
        logger.warning("Reaction %s not found in model", reaction_id)
        return 0.0

    original_ub = rxn.upper_bound

    # Baseline flux
    baseline_sol = model.optimize()
    if baseline_sol.status != "optimal":
        return 0.0
    j_baseline = baseline_sol.objective_value
    if abs(j_baseline) < 1e-12:
        return 0.0

    # Perturb +
    rxn.upper_bound = original_ub * (1 + perturbation)
    sol_plus = model.optimize()
    j_plus = sol_plus.objective_value if sol_plus.status == "optimal" else j_baseline

    # Perturb -
    rxn.upper_bound = original_ub * (1 - perturbation)
    sol_minus = model.optimize()
    j_minus = sol_minus.objective_value if sol_minus.status == "optimal" else j_baseline

    # Restore
    rxn.upper_bound = original_ub

    # Central difference: FCC = (J+ - J-) / (2 * perturbation * J_baseline)
    dj = j_plus - j_minus
    fcc = dj / (2 * perturbation * j_baseline)

    logger.info(
        "FCC for %s (rxn %s): %.4f  (J-=%.4f, J0=%.4f, J+=%.4f)",
        enzyme_id, reaction_id, fcc, j_minus, j_baseline, j_plus,
    )
    return fcc


# ---------------------------------------------------------------------------
# Main stage logic
# ---------------------------------------------------------------------------

def run_stage(config: dict[str, Any]) -> Path:
    """Execute Stage 0: bottleneck identification.

    1. Load yeast-GEM (or toy model)
    2. Run CatPred for each enzyme
    3. Run FBA
    4. Calculate FCCs for all 4 enzymes
    5. Rank by FCC, assign priority
    6. Write bottleneck_results.csv

    Returns:
        Path to the output CSV file.
    """
    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_00_bottleneck", {})
    fcc_threshold = stage_cfg.get("fcc_threshold", 0.3)
    output_dir_name = stage_cfg.get("output_dir", "stage_00_bottleneck")
    output_dir = base_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    enzyme_ids = sorted(config.get("enzymes", {}).keys())
    if not enzyme_ids:
        raise RuntimeError("No enzymes defined in config")

    # Map enzyme_id -> reaction_id in the cobra model
    reaction_map = {eid: f"rxn_{eid}" for eid in enzyme_ids}

    # --- CatPred kinetics ---
    catpred_dir = Path(config.get("tools", {}).get("catpred_dir", "/opt/tools/CatPred"))
    kinetics: dict[str, dict[str, Any]] = {}
    for eid in enzyme_ids:
        ecfg = config["enzymes"][eid]
        seq_file = Path(ecfg.get("sequence_file", ""))
        sequence = _read_first_sequence(base_dir / seq_file) if seq_file.suffix else ""
        substrate = ecfg.get("substrate_smiles", "")
        kinetics[eid] = run_catpred(sequence, substrate, eid, catpred_dir)

    # --- FBA and FCC ---
    use_cobra = _COBRA_AVAILABLE
    fcc_values: dict[str, float] = {}

    if use_cobra:
        gem_path = Path(
            config.get("tools", {}).get(
                "yeast_gem_path",
                base_dir / "inputs" / "yeast-GEM.xml",
            )
        )
        try:
            model = load_yeast_gem(gem_path)
            fluxes = run_fba(model)
            logger.info("FBA complete — %d reactions, objective=%.4f",
                        len(fluxes),
                        model.optimize().objective_value if fluxes else 0)
            for eid in enzyme_ids:
                rxn_id = reaction_map[eid]
                fcc_values[eid] = calculate_fcc(model, eid, rxn_id)
        except Exception as exc:
            logger.warning("COBRApy FBA failed: %s — using placeholder FCCs", exc)
            use_cobra = False

    if not use_cobra:
        logger.warning(
            "COBRApy not available or FBA failed — using placeholder FCC values"
        )
        for eid in enzyme_ids:
            fcc_values[eid] = _PLACEHOLDER_FCC.get(eid, 0.15)

    # --- Build results ---
    results: list[dict[str, Any]] = []
    for eid in enzyme_ids:
        fcc = fcc_values[eid]
        kin = kinetics[eid]
        priority = "primary" if fcc > fcc_threshold else "secondary"
        notes_parts = []
        if not use_cobra:
            notes_parts.append("placeholder FCC (COBRApy unavailable)")
        if kin.get("kcat") == _PLACEHOLDER_KINETICS.get(eid, {}).get("kcat"):
            notes_parts.append("placeholder kinetics (CatPred unavailable)")
        results.append({
            "enzyme_id": eid,
            "fcc": round(fcc, 4),
            "kcat_pred": round(kin["kcat"], 2),
            "km_pred": round(kin["km"], 3),
            "priority": priority,
            "notes": "; ".join(notes_parts) if notes_parts else "",
        })

    # Sort by FCC descending
    results.sort(key=lambda r: r["fcc"], reverse=True)

    # --- Write CSV ---
    output_csv = output_dir / "bottleneck_results.csv"
    fieldnames = ["enzyme_id", "fcc", "kcat_pred", "km_pred", "priority", "notes"]
    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Bottleneck results written to %s", output_csv)
    for r in results:
        logger.info(
            "  %s: FCC=%.4f, kcat=%.2f s^-1, Km=%.3f mM → %s",
            r["enzyme_id"], r["fcc"], r["kcat_pred"], r["km_pred"], r["priority"],
        )

    return output_csv


def _read_first_sequence(fasta_path: Path) -> str:
    """Read the first sequence from a FASTA file."""
    if not fasta_path.exists():
        return ""
    lines: list[str] = []
    started = False
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if started:
                    break
                started = True
                continue
            if started:
                lines.append(line)
    return "".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(config_path: Path) -> None:
    """Load config, set up logging, run Stage 0, log summary."""
    config_path = Path(config_path)
    config = load_config(config_path)

    # Set up logging
    log_dir = Path(config["global"].get("log_dir", "logs"))
    base_dir = Path(config["global"]["base_dir"])
    if not log_dir.is_absolute():
        log_dir = base_dir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "stage0_bottleneck.log"),
        ],
    )

    logger.info("=" * 60)
    logger.info("%s — starting", STAGE_NAME)
    logger.info("Config: %s", config_path)
    logger.info("=" * 60)

    output_csv = run_stage(config)

    logger.info("=" * 60)
    logger.info("%s — complete", STAGE_NAME)
    logger.info("Output: %s", output_csv)
    logger.info("=" * 60)


def _cli() -> None:
    """Argument parser wrapper for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Stage 0: Metabolic bottleneck identification",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    _cli()
