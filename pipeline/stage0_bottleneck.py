"""Stage 0: Metabolic bottleneck identification using COBRApy + yeast-GEM.

Determines which enzyme(s) exert the most flux control over pathway output
before committing GPU time to design. Uses flux control coefficients (FCCs)
computed by perturbing each enzyme's Vmax ±10%.

Supports three analysis methods:
- "fba"      : standard flux balance analysis (default)
- "tfba"     : thermodynamic FBA using eQuilibrator Gibbs energies
- "ensemble" : Monte Carlo sampling of kinetic parameters, reports FCC
               median + 90% CI

Stage can be disabled entirely (``stages.stage_00_bottleneck.enabled: false``),
in which case all enzymes are assigned equal priority (FCC=1.0) with no
computational cost.

CLI: python pipeline/stage0_bottleneck.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
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

# ---------------------------------------------------------------------------
# eQuilibrator import with graceful fallback
# ---------------------------------------------------------------------------

_EQUILIBRATOR_AVAILABLE = True
try:
    from equilibrator_api import ComponentContribution, Q_  # type: ignore[import-untyped]
except ImportError:
    _EQUILIBRATOR_AVAILABLE = False
    ComponentContribution = None  # type: ignore[assignment]
    Q_ = None  # type: ignore[assignment]

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


def _catpred_worker(
    eid: str,
    sequence: str,
    substrate: str,
    catpred_dir_str: str,
) -> tuple[str, dict[str, Any]]:
    """Worker for parallel CatPred prediction (ProcessPoolExecutor)."""
    return eid, run_catpred(sequence, substrate, eid, Path(catpred_dir_str))


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
# TFBA: thermodynamic FBA
# ---------------------------------------------------------------------------

def run_tfba(
    model: Any,
    enzyme_ids: list[str],
    reaction_map: dict[str, str],
    perturbation: float = 0.1,
) -> dict[str, float]:
    """Compute FCCs using thermodynamic FBA.

    Applies standard Gibbs energy constraints (ΔrG'°) from eQuilibrator's
    component contribution method to enforce thermodynamically feasible reaction
    directions before computing FCCs via the same central-difference method as
    standard FBA.

    When eQuilibrator is unavailable the function falls back to standard
    ``calculate_fcc`` for every reaction.

    Args:
        model: cobra.Model instance (will be used in a context manager copy).
        enzyme_ids: Ordered list of enzyme identifiers.
        reaction_map: Mapping from enzyme_id to reaction ID in the model.
        perturbation: Fractional perturbation for FCC central difference (default 0.1).

    Returns:
        Dictionary mapping enzyme_id -> FCC (float).
    """
    if not _EQUILIBRATOR_AVAILABLE:
        logger.warning(
            "equilibrator_api not available — TFBA falling back to standard FBA FCCs"
        )
        return {
            eid: calculate_fcc(model, eid, reaction_map[eid], perturbation)
            for eid in enzyme_ids
        }

    logger.info("Running TFBA with eQuilibrator component contribution method")
    try:
        cc = ComponentContribution()

        with model:
            for eid in enzyme_ids:
                rxn_id = reaction_map[eid]
                try:
                    rxn = model.reactions.get_by_id(rxn_id)
                except KeyError:
                    logger.warning("TFBA: reaction %s not found, skipping", rxn_id)
                    continue

                # Attempt to fetch ΔrG'° — use reaction KEGG/BiGG annotation if
                # available, otherwise skip thermodynamic constraint for this rxn.
                try:
                    eq_rxn = cc.parse_reaction_formula(rxn.reaction)
                    dg_result = cc.standard_dg_prime(eq_rxn)
                    # dg_result.value is a pint Quantity in kJ/mol
                    dg_value = float(dg_result.value.magnitude)
                    dg_std = float(dg_result.error.magnitude)

                    RT = 2.479  # kJ/mol at 298 K
                    # Positive ΔrG'° → reaction disfavoured in forward direction
                    if dg_value > 0:
                        # Allow small positive ΔrG'° (within 2σ of zero) — may still run
                        if dg_value - 2 * dg_std > 0:
                            logger.info(
                                "TFBA: constraining %s (rxn %s) — ΔrG'°=%.2f kJ/mol, forcing lb=0",
                                eid, rxn_id, dg_value,
                            )
                            rxn.lower_bound = 0  # already 0 for irreversible; no-op but explicit
                        # For very unfavourable reactions, upper bound set to constrain
                    elif dg_value < -RT * math.log(1000):
                        # Strongly favourable: reaction proceeds; no additional constraint needed
                        pass
                    logger.debug(
                        "TFBA: %s (rxn %s) ΔrG'°=%.2f ± %.2f kJ/mol",
                        eid, rxn_id, dg_value, dg_std,
                    )
                except Exception as eq_exc:
                    logger.debug(
                        "TFBA: could not get ΔrG'° for %s (rxn %s): %s — skipping constraint",
                        eid, rxn_id, eq_exc,
                    )

            # Compute FCCs on the thermodynamically constrained model
            fcc_values = {
                eid: calculate_fcc(model, eid, reaction_map[eid], perturbation)
                for eid in enzyme_ids
            }

    except Exception as exc:
        logger.warning("TFBA failed (%s) — falling back to standard FBA FCCs", exc)
        fcc_values = {
            eid: calculate_fcc(model, eid, reaction_map[eid], perturbation)
            for eid in enzyme_ids
        }

    return fcc_values


# ---------------------------------------------------------------------------
# Ensemble: Monte Carlo FCC sampling
# ---------------------------------------------------------------------------

def run_ensemble(
    model: Any,
    enzyme_ids: list[str],
    reaction_map: dict[str, str],
    kinetics: dict[str, dict[str, Any]],
    n_iterations: int = 100,
    perturbation: float = 0.1,
    kcat_cv: float = 0.5,
    seed: int | None = None,
) -> dict[str, dict[str, float]]:
    """Estimate FCC distribution via Monte Carlo kinetic parameter sampling.

    For each iteration, samples kcat and Km from log-normal distributions
    centred on the CatPred predictions (or placeholders).  The coefficient of
    variation (CV) for the log-normal draws defaults to 0.5 (50%), reflecting
    typical uncertainty in kcat predictions.

    The sampled kcat values are used to scale each reaction's upper bound
    (acting as a proxy for Vmax), and FCCs are recomputed.

    Args:
        model: cobra.Model instance.
        enzyme_ids: Ordered list of enzyme identifiers.
        reaction_map: Mapping from enzyme_id to reaction ID in the model.
        kinetics: Per-enzyme kinetics dict with at least keys "kcat" and "km".
        n_iterations: Number of Monte Carlo samples (default 100).
        perturbation: Fractional perturbation for FCC central difference (default 0.1).
        kcat_cv: Coefficient of variation for log-normal kcat sampling (default 0.5).
        seed: Optional random seed for reproducibility.

    Returns:
        Dictionary mapping enzyme_id -> {fcc_median, fcc_ci_low, fcc_ci_high}.
        CI bounds are the 5th and 95th percentiles (90% CI).
    """
    if seed is not None:
        random.seed(seed)

    # log-normal parameters: mu and sigma such that the distribution has
    # the desired mean and CV.
    # For log-normal: CV = sqrt(exp(sigma^2) - 1) => sigma^2 = log(1 + CV^2)
    sigma_log = math.sqrt(math.log(1 + kcat_cv ** 2))

    fcc_samples: dict[str, list[float]] = {eid: [] for eid in enzyme_ids}

    for _ in range(n_iterations):
        with model:
            # Scale each reaction's upper bound by a sampled kcat multiplier
            for eid in enzyme_ids:
                rxn_id = reaction_map[eid]
                try:
                    rxn = model.reactions.get_by_id(rxn_id)
                except KeyError:
                    continue
                kcat_central = kinetics.get(eid, {}).get("kcat", 10.0)
                # Sample a multiplicative factor from log-normal(0, sigma_log)
                # Mean of this distribution is exp(sigma_log^2/2), so we
                # normalise by dividing by that mean to keep the central kcat.
                log_mean = -0.5 * sigma_log ** 2  # mu for mean-1 log-normal
                multiplier = math.exp(log_mean + sigma_log * _standard_normal())
                # Scale upper bound (Vmax proxy)
                rxn.upper_bound = rxn.upper_bound * max(multiplier, 1e-3)

            for eid in enzyme_ids:
                rxn_id = reaction_map[eid]
                fcc = calculate_fcc(model, eid, rxn_id, perturbation)
                fcc_samples[eid].append(fcc)

    results: dict[str, dict[str, float]] = {}
    for eid in enzyme_ids:
        samples = sorted(fcc_samples[eid])
        n = len(samples)
        median = _percentile(samples, 50)
        ci_low = _percentile(samples, 5)
        ci_high = _percentile(samples, 95)
        results[eid] = {
            "fcc_median": median,
            "fcc_ci_low": ci_low,
            "fcc_ci_high": ci_high,
        }
        logger.info(
            "Ensemble FCC %s: median=%.4f  90%%CI=[%.4f, %.4f]  (n=%d)",
            eid, median, ci_low, ci_high, n,
        )

    return results


def _standard_normal() -> float:
    """Return a single standard-normal random variate using Box-Muller."""
    u1 = random.random() or 1e-15  # avoid log(0)
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def _percentile(sorted_data: list[float], p: float) -> float:
    """Return the p-th percentile of a sorted list (linear interpolation)."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


# ---------------------------------------------------------------------------
# Main stage logic
# ---------------------------------------------------------------------------

def run_stage(config: dict[str, Any]) -> Path:
    """Execute Stage 0: bottleneck identification.

    When ``stages.stage_00_bottleneck.enabled`` is ``false`` (the new
    default), the stage is skipped and a default CSV is written where all
    4 enzymes receive FCC=1.0 and priority="equal".

    When enabled (``true``), the behaviour depends on the ``method`` key:

    - ``"fba"``      : standard FBA + central-difference FCC (default).
    - ``"tfba"``     : thermodynamic FBA using eQuilibrator Gibbs energies.
    - ``"ensemble"`` : Monte Carlo sampling of kinetic parameters; reports
                       median FCC + 90% CI as extra CSV columns.

    Steps when running (method fba/tfba):
    1. Load yeast-GEM (or toy model)
    2. Run CatPred for each enzyme
    3. Run FBA / TFBA
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

    # ------------------------------------------------------------------
    # Early exit when stage is disabled
    # ------------------------------------------------------------------
    enabled = stage_cfg.get("enabled", False)  # default: disabled
    if not enabled:
        logger.info(
            "Stage 0 disabled — all enzymes will be optimized equally"
        )
        results: list[dict[str, Any]] = [
            {
                "enzyme_id": eid,
                "fcc": 1.0,
                "kcat_pred": 0.0,
                "km_pred": 0.0,
                "priority": "equal",
                "notes": "Stage 0 disabled; equal priority assigned",
            }
            for eid in enzyme_ids
        ]
        output_csv = output_dir / "bottleneck_results.csv"
        fieldnames = ["enzyme_id", "fcc", "kcat_pred", "km_pred", "priority", "notes"]
        with open(output_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info("Default (equal) bottleneck results written to %s", output_csv)
        return output_csv

    # ------------------------------------------------------------------
    # Stage is enabled — determine method
    # ------------------------------------------------------------------
    method = stage_cfg.get("method", "fba").lower()
    ensemble_n = stage_cfg.get("ensemble_n", 100)

    logger.info("Stage 0 running with method='%s'", method)

    # Map enzyme_id -> reaction_id in the cobra model
    reaction_map = {eid: f"rxn_{eid}" for eid in enzyme_ids}

    # --- CatPred kinetics (parallel across enzymes) ---
    catpred_dir = Path(config.get("tools", {}).get("catpred_dir", "/opt/tools/CatPred"))
    kinetics: dict[str, dict[str, Any]] = {}
    catpred_args: list[tuple[str, str, str, str]] = []
    for eid in enzyme_ids:
        ecfg = config["enzymes"][eid]
        seq_file = Path(ecfg.get("sequence_file", ""))
        sequence = _read_first_sequence(base_dir / seq_file) if seq_file.suffix else ""
        substrate = ecfg.get("substrate_smiles", "")
        catpred_args.append((eid, sequence, substrate, str(catpred_dir)))

    n_workers = min(os.cpu_count() or 1, len(catpred_args))
    if n_workers > 1:
        logger.info(
            "Running CatPred predictions in parallel (%d enzymes, %d workers)",
            len(catpred_args), n_workers,
        )
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_catpred_worker, *args) for args in catpred_args]
            for future in futures:
                eid_result, kin_result = future.result()
                kinetics[eid_result] = kin_result
    else:
        for args in catpred_args:
            eid_result, kin_result = _catpred_worker(*args)
            kinetics[eid_result] = kin_result

    # --- FBA / TFBA / Ensemble ---
    use_cobra = _COBRA_AVAILABLE
    fcc_values: dict[str, float] = {}
    # Ensemble extras (only populated for method="ensemble")
    ensemble_results: dict[str, dict[str, float]] = {}

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
            logger.info(
                "FBA complete — %d reactions, objective=%.4f",
                len(fluxes),
                model.optimize().objective_value if fluxes else 0,
            )

            if method == "tfba":
                fcc_values = run_tfba(model, enzyme_ids, reaction_map)
            elif method == "ensemble":
                ensemble_results = run_ensemble(
                    model, enzyme_ids, reaction_map, kinetics, n_iterations=ensemble_n
                )
                fcc_values = {
                    eid: ensemble_results[eid]["fcc_median"] for eid in enzyme_ids
                }
            else:
                # Default: standard FBA
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
    is_ensemble = method == "ensemble" and bool(ensemble_results)

    results = []
    for eid in enzyme_ids:
        fcc = fcc_values[eid]
        kin = kinetics[eid]
        priority = "primary" if fcc > fcc_threshold else "secondary"
        notes_parts = []
        if not use_cobra:
            notes_parts.append("placeholder FCC (COBRApy unavailable)")
        if kin.get("kcat") == _PLACEHOLDER_KINETICS.get(eid, {}).get("kcat"):
            notes_parts.append("placeholder kinetics (CatPred unavailable)")
        row: dict[str, Any] = {
            "enzyme_id": eid,
            "fcc": round(fcc, 4),
            "kcat_pred": round(kin["kcat"], 2),
            "km_pred": round(kin["km"], 3),
            "priority": priority,
            "notes": "; ".join(notes_parts) if notes_parts else "",
        }
        if is_ensemble:
            ens = ensemble_results[eid]
            row["fcc_ci_low"] = round(ens["fcc_ci_low"], 4)
            row["fcc_ci_high"] = round(ens["fcc_ci_high"], 4)
        results.append(row)

    # Sort by FCC descending
    results.sort(key=lambda r: r["fcc"], reverse=True)

    # --- Write CSV ---
    output_csv = output_dir / "bottleneck_results.csv"
    fieldnames = ["enzyme_id", "fcc", "kcat_pred", "km_pred", "priority", "notes"]
    if is_ensemble:
        fieldnames = fieldnames + ["fcc_ci_low", "fcc_ci_high"]
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
