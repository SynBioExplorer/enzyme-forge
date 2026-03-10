"""Stage 7: Cross-validation with Protenix-v1 and Pareto-front ranking.

Validates top candidates with an independent structure predictor (Protenix-v1)
and composes a final ranking using multi-metric Pareto-front selection.

Steps:
1. Read top 200 candidates (50/enzyme) + all scores CSVs from Stages 4-6.5
2. Run Protenix-v1 structure prediction
3. Discard designs where Boltz-2 vs Protenix-v1 Ca RMSD > 2.0 A
4. Run CatPred + RealKcat for kcat/Km predictions
5. Collect all 7-8 metrics
6. Compute Pareto front
7. Select 48-96 diverse designs per enzyme spanning the front

Output: final_rankings.csv + pareto_front.csv in stage_07_crossval_ranking/

CLI: python pipeline/stage7_ranking.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pipeline.config import load_config, get_enzyme_config, resolve_paths
    from pipeline.utils import (
        setup_logging,
        read_fasta,
        write_fasta,
        write_scores,
        read_scores,
        merge_scores,
    )
except ImportError:
    import csv as _csv

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
            h, s = None, []  # type: ignore[assignment]
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if h:
                        seqs[h] = "".join(s)
                    h, s = line[1:].split()[0], []
                elif h:
                    s.append(line)
            if h:
                seqs[h] = "".join(s)
        return seqs

    def write_fasta(seqs, p):  # type: ignore[misc]
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for h, s in seqs.items():
                f.write(f">{h}\n{s}\n")

    def write_scores(scores, p):  # type: ignore[misc]
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        if not scores:
            return
        import csv

        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=scores[0].keys())
            w.writeheader()
            w.writerows(scores)

    def read_scores(p):  # type: ignore[misc]
        import csv

        with open(p) as f:
            return list(csv.DictReader(f))

    def merge_scores(files, out):  # type: ignore[misc]
        pass


# ---------------------------------------------------------------------------
# Module constants (StageModuleProtocol)
# ---------------------------------------------------------------------------

STAGE_NAME = "Stage 7: Cross-validation and Pareto Ranking"
STAGE_NUM = "7"
INPUT_DESCRIPTION = "Top 200 candidates + all scores CSVs from Stages 4-6.5"
OUTPUT_DESCRIPTION = (
    "final_rankings.csv + pareto_front.csv in stage_07_crossval_ranking/"
)

logger = logging.getLogger(__name__)

# Metric directions: "max" = higher is better, "min" = lower is better
METRIC_DIRECTIONS: dict[str, str] = {
    "esmfold_plddt": "max",
    "active_site_rmsd": "min",
    "ligandmpnn_confidence": "max",
    "zs_fitness_score": "max",
    "product_inhibition_ratio": "max",
    "catpred_kcat": "max",
    "boltz2_substrate_affinity": "max",
    "md_stability_rmsd": "min",
    "ts_affinity": "max",
    "ts_substrate_ratio": "max",
}

# Columns required in the final output
REQUIRED_OUTPUT_COLUMNS = [
    "sequence_id",
    "enzyme_id",
    "sequence",
    "esmfold_plddt",
    "active_site_rmsd",
    "ligandmpnn_confidence",
    "zs_fitness_score",
    "product_inhibition_ratio",
    "catpred_kcat",
    "boltz2_substrate_affinity",
    "md_stability_rmsd",
    "ts_affinity",
    "ts_substrate_ratio",
    "protenix_plddt",
    "crossval_ca_rmsd",
    "crossval_passed",
    "pareto_rank",
]


# ---------------------------------------------------------------------------
# Protenix-v1 structure prediction
# ---------------------------------------------------------------------------

def run_protenix(
    sequence: str,
    output_dir: Path,
    protenix_bin: str = "protenix",
) -> dict[str, Any]:
    """Run Protenix-v1 structure prediction via subprocess.

    Args:
        sequence: Amino-acid sequence string.
        output_dir: Directory to write outputs.
        protenix_bin: Path to protenix binary.

    Returns:
        {"pdb_path": str, "plddt": float, "iptm": float}

    Raises:
        RuntimeError: If Protenix fails with non-zero exit code.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        protenix_bin,
        "predict",
        "--sequence", sequence,
        "--output", str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Protenix failed (exit {result.returncode}): {result.stderr}"
        )

    # Parse output: look for PDB file and confidence scores
    pdb_files = sorted(output_dir.glob("*.pdb"))
    if not pdb_files:
        raise RuntimeError(f"No PDB files found in {output_dir}")

    pdb_path = pdb_files[0]

    # Parse confidence from JSON sidecar or stdout
    confidence = _parse_protenix_confidence(output_dir, result.stdout)

    return {
        "pdb_path": str(pdb_path),
        "plddt": confidence.get("plddt", 0.0),
        "iptm": confidence.get("iptm", 0.0),
    }


def _parse_protenix_confidence(
    output_dir: Path, stdout: str
) -> dict[str, float]:
    """Parse Protenix confidence scores from output directory or stdout."""
    # Try JSON sidecar first
    json_files = sorted(output_dir.glob("*confidence*.json")) + sorted(
        output_dir.glob("*scores*.json")
    )
    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
            return {
                "plddt": float(data.get("plddt", data.get("mean_plddt", 0.0))),
                "iptm": float(data.get("iptm", 0.0)),
            }
        except (json.JSONDecodeError, ValueError, KeyError):
            continue

    # Try parsing stdout
    plddt = 0.0
    iptm = 0.0
    for line in stdout.splitlines():
        line_lower = line.lower()
        if "plddt" in line_lower:
            try:
                plddt = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        if "iptm" in line_lower:
            try:
                iptm = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass

    return {"plddt": plddt, "iptm": iptm}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    boltz2_pdb: Path,
    protenix_pdb: Path,
    ca_rmsd_threshold: float = 2.0,
) -> tuple[bool, float]:
    """Calculate Ca RMSD between Boltz-2 and Protenix-v1 predictions.

    Args:
        boltz2_pdb: Path to Boltz-2 predicted PDB.
        protenix_pdb: Path to Protenix-v1 predicted PDB.
        ca_rmsd_threshold: Maximum acceptable Ca RMSD (default 2.0 A).

    Returns:
        Tuple of (passed: bool, rmsd: float).
    """
    from Bio.PDB import PDBParser, Superimposer

    parser = PDBParser(QUIET=True)

    struct1 = parser.get_structure("boltz2", str(boltz2_pdb))
    struct2 = parser.get_structure("protenix", str(protenix_pdb))

    def _get_ca(struct):
        cas = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    het = residue.get_id()[0]
                    if het.strip() and het != " ":
                        continue
                    for atom in residue:
                        if atom.get_name() == "CA":
                            cas.append(atom)
            break  # first model only
        return cas

    ca1 = _get_ca(struct1)
    ca2 = _get_ca(struct2)

    n = min(len(ca1), len(ca2))
    if n == 0:
        return False, float("inf")

    ca1 = ca1[:n]
    ca2 = ca2[:n]

    sup = Superimposer()
    sup.set_atoms(ca1, ca2)
    rmsd = float(sup.rms)

    passed = rmsd <= ca_rmsd_threshold
    return passed, rmsd


# ---------------------------------------------------------------------------
# CatPred + RealKcat
# ---------------------------------------------------------------------------

def run_catpred(
    sequence: str,
    substrate_smiles: str,
    catpred_dir: Path,
) -> dict[str, float]:
    """Run CatPred for kcat + Km prediction via subprocess.

    Args:
        sequence: Amino-acid sequence.
        substrate_smiles: SMILES string for the substrate.
        catpred_dir: Path to CatPred installation.

    Returns:
        {"kcat": float, "km": float}
    """
    catpred_dir = Path(catpred_dir)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(
            {"sequence": sequence, "substrate_smiles": substrate_smiles},
            tmp,
        )
        input_path = tmp.name

    try:
        cmd = [
            "python",
            str(catpred_dir / "predict.py"),
            "--input", input_path,
            "--mode", "kcat_km",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.warning(
                "CatPred failed (exit %d): %s", result.returncode, result.stderr
            )
            return {"kcat": float("nan"), "km": float("nan")}

        # Parse output
        output = json.loads(result.stdout)
        return {
            "kcat": float(output.get("kcat", float("nan"))),
            "km": float(output.get("km", float("nan"))),
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as exc:
        logger.warning("CatPred error: %s", exc)
        return {"kcat": float("nan"), "km": float("nan")}
    finally:
        Path(input_path).unlink(missing_ok=True)


def run_realkcat(
    sequence: str,
    wt_sequence: str,
    mutations: list[str],
    realkcat_dir: Path,
) -> dict[str, float]:
    """Run RealKcat for mutation-aware kcat prediction via subprocess.

    Args:
        sequence: Mutant amino-acid sequence.
        wt_sequence: Wild-type amino-acid sequence.
        mutations: List of mutation strings (e.g. ["A100G", "K200R"]).
        realkcat_dir: Path to RealKcat installation.

    Returns:
        {"kcat_ratio": float} — predicted kcat relative to WT.
    """
    realkcat_dir = Path(realkcat_dir)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(
            {
                "sequence": sequence,
                "wt_sequence": wt_sequence,
                "mutations": mutations,
            },
            tmp,
        )
        input_path = tmp.name

    try:
        cmd = [
            "python",
            str(realkcat_dir / "predict.py"),
            "--input", input_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.warning(
                "RealKcat failed (exit %d): %s",
                result.returncode,
                result.stderr,
            )
            return {"kcat_ratio": float("nan")}

        output = json.loads(result.stdout)
        return {"kcat_ratio": float(output.get("kcat_ratio", float("nan")))}
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as exc:
        logger.warning("RealKcat error: %s", exc)
        return {"kcat_ratio": float("nan")}
    finally:
        Path(input_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def collect_all_metrics(
    candidate_id: str,
    stage_outputs: dict[str, list[dict]],
) -> dict[str, Any]:
    """Collect all ranking metrics for one candidate from stage output data.

    Args:
        candidate_id: Sequence identifier.
        stage_outputs: Dict of stage name to list of score dicts.
            Keys: "stage_04", "stage_04_5", "stage_05", "stage_06",
                  "stage_06_5", "stage_07_catpred"

    Returns:
        Unified metrics dict for one candidate.
    """
    metrics: dict[str, Any] = {"sequence_id": candidate_id}

    # Helper to look up a candidate in a stage's score list
    def _lookup(stage_key: str, field: str, default: Any = None) -> Any:
        rows = stage_outputs.get(stage_key, [])
        for row in rows:
            sid = row.get("sequence_id", "")
            if sid == candidate_id:
                val = row.get(field, default)
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return default
                return default
        return default

    # 1. ESMFold pLDDT (Stage 5) — higher is better
    metrics["esmfold_plddt"] = _lookup("stage_05", "mean_plddt", 0.0)

    # 2. Active site heavy-atom RMSD (Stage 6) — lower is better
    metrics["active_site_rmsd"] = _lookup(
        "stage_06", "active_site_rmsd", float("nan")
    )

    # 3. LigandMPNN confidence (Stage 4) — higher is better
    metrics["ligandmpnn_confidence"] = _lookup(
        "stage_04", "ligandmpnn_score", 0.0
    )

    # 4. Zero-shot fitness ensemble score (Stage 4.5) — higher is better
    metrics["zs_fitness_score"] = _lookup(
        "stage_04_5", "ensemble_score", 0.0
    )

    # 5. Product inhibition ratio (Stage 6) — higher is better
    metrics["product_inhibition_ratio"] = _lookup(
        "stage_06", "product_inhibition_ratio", float("nan")
    )

    # 6. CatPred predicted kcat (Stage 7) — higher is better
    metrics["catpred_kcat"] = _lookup(
        "stage_07_catpred", "kcat", float("nan")
    )

    # 7. Boltz-2 substrate affinity (Stage 6) — higher is better
    metrics["boltz2_substrate_affinity"] = _lookup(
        "stage_06", "substrate_affinity", float("nan")
    )

    # 8. MD stability metrics (Stage 6.5, if available) — lower RMSD better
    metrics["md_stability_rmsd"] = _lookup(
        "stage_06_5", "mean_rmsd", float("nan")
    )

    # 9. Transition state affinity (Stage 6) — tighter TS binding is better
    metrics["ts_affinity"] = _lookup(
        "stage_06", "ts_affinity", float("nan")
    )

    # 10. TS / substrate affinity ratio (Stage 6) — higher means better TS
    #     stabilization relative to substrate
    metrics["ts_substrate_ratio"] = _lookup(
        "stage_06", "ts_substrate_ratio", float("nan")
    )

    return metrics


# ---------------------------------------------------------------------------
# Pareto front computation (NSGA-II style fast non-dominated sort)
# ---------------------------------------------------------------------------

def compute_pareto_front(
    candidates: list[dict],
    metric_directions: dict[str, str] | None = None,
) -> list[dict]:
    """Compute Pareto front ranks for all candidates using fast non-dominated sort.

    Args:
        candidates: List of candidate dicts, each with metric values.
        metric_directions: {"metric_name": "max"|"min"}.
            If None, uses the module-level METRIC_DIRECTIONS.

    Returns:
        Input list with "pareto_rank" added to each candidate dict.
        Rank 1 = non-dominated (Pareto-optimal).
    """
    if not candidates:
        return []

    if metric_directions is None:
        metric_directions = METRIC_DIRECTIONS

    # Identify available metrics (present and non-NaN in at least one candidate)
    available_metrics = []
    for metric, direction in metric_directions.items():
        values = []
        for c in candidates:
            v = c.get(metric)
            if v is not None:
                try:
                    fv = float(v)
                    if not np.isnan(fv):
                        values.append(fv)
                except (ValueError, TypeError):
                    pass
        if values:
            available_metrics.append(metric)

    if not available_metrics:
        for c in candidates:
            c["pareto_rank"] = 1
        return candidates

    n = len(candidates)

    # Build objective matrix: convert to "higher is better" for all
    obj_matrix = np.zeros((n, len(available_metrics)))
    for j, metric in enumerate(available_metrics):
        direction = metric_directions.get(metric, "max")
        for i, c in enumerate(candidates):
            val = c.get(metric)
            try:
                fv = float(val) if val is not None else float("nan")
            except (ValueError, TypeError):
                fv = float("nan")

            if np.isnan(fv):
                # Assign worst possible value for NaN
                fv = -np.inf if direction == "max" else np.inf

            # Negate "min" objectives so higher is always better
            obj_matrix[i, j] = fv if direction == "max" else -fv

    # Fast non-dominated sort
    ranks = _fast_non_dominated_sort(obj_matrix)

    for i, c in enumerate(candidates):
        c["pareto_rank"] = int(ranks[i])

    return candidates


def _fast_non_dominated_sort(objectives: np.ndarray) -> np.ndarray:
    """NSGA-II fast non-dominated sort.

    Args:
        objectives: (n_candidates, n_objectives) array.
            All objectives are assumed to be maximized.

    Returns:
        Array of ranks (1-based), shape (n_candidates,).
    """
    n = objectives.shape[0]
    ranks = np.zeros(n, dtype=int)

    # domination_count[i]: number of solutions that dominate i
    domination_count = np.zeros(n, dtype=int)
    # dominated_set[i]: set of solutions that i dominates
    dominated_set: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            # Check if i dominates j or vice versa
            diff = objectives[i] - objectives[j]
            i_dom_j = np.all(diff >= 0) and np.any(diff > 0)
            j_dom_i = np.all(diff <= 0) and np.any(diff < 0)

            if i_dom_j:
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif j_dom_i:
                dominated_set[j].append(i)
                domination_count[i] += 1

    # Build fronts
    current_front = [i for i in range(n) if domination_count[i] == 0]
    rank = 1

    while current_front:
        for i in current_front:
            ranks[i] = rank

        next_front = []
        for i in current_front:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)

        current_front = next_front
        rank += 1

    return ranks


# ---------------------------------------------------------------------------
# Normalize metrics to 0-1 range
# ---------------------------------------------------------------------------

def _normalize_metrics(
    candidates: list[dict],
    metrics: list[str],
) -> list[dict]:
    """Normalize metric values to [0, 1] range for diversity computation."""
    if not candidates or not metrics:
        return candidates

    for metric in metrics:
        vals = []
        for c in candidates:
            v = c.get(metric)
            try:
                fv = float(v) if v is not None else float("nan")
            except (ValueError, TypeError):
                fv = float("nan")
            vals.append(fv)

        finite_vals = [v for v in vals if np.isfinite(v)]
        if not finite_vals:
            continue

        vmin = min(finite_vals)
        vmax = max(finite_vals)
        rng = vmax - vmin

        norm_key = f"{metric}_norm"
        for i, c in enumerate(candidates):
            if np.isfinite(vals[i]) and rng > 0:
                c[norm_key] = (vals[i] - vmin) / rng
            else:
                c[norm_key] = 0.5  # midpoint for NaN

    return candidates


# ---------------------------------------------------------------------------
# Diverse design selection
# ---------------------------------------------------------------------------

def select_diverse_designs(
    pareto_candidates: list[dict],
    all_candidates: list[dict],
    n_designs: int = 72,
) -> list[dict]:
    """Select diverse designs spanning the Pareto front.

    Strategy:
    - Include all rank-1 (Pareto-optimal) candidates
    - Fill remaining slots from rank 2, 3, etc.
    - Use metric-space diversity to ensure coverage
    - Reserve ~20% of slots for WT-adjacent (conservative) designs

    Args:
        pareto_candidates: Candidates with pareto_rank == 1.
        all_candidates: All candidates with pareto_rank assigned.
        n_designs: Target number of designs to select (default 72).

    Returns:
        List of n_designs selected candidate dicts.
    """
    if not all_candidates:
        return []

    if len(all_candidates) <= n_designs:
        for c in all_candidates:
            c["selected"] = True
        return list(all_candidates)

    selected: list[dict] = []
    selected_ids: set[str] = set()

    # Reserve ~20% for WT-adjacent designs
    n_wt_adjacent = max(1, n_designs // 5)
    n_pareto = n_designs - n_wt_adjacent

    # Sort all candidates by pareto_rank
    sorted_candidates = sorted(
        all_candidates, key=lambda c: c.get("pareto_rank", 999)
    )

    # 1. Select WT-adjacent (conservative) designs first
    # WT-adjacent = designs with "wt" in backbone or low mutation count
    wt_adjacent = [
        c for c in sorted_candidates
        if _is_wt_adjacent(c)
    ]

    for c in wt_adjacent[:n_wt_adjacent]:
        if c.get("sequence_id") not in selected_ids:
            c["selected"] = True
            c["selection_reason"] = "wt_adjacent"
            selected.append(c)
            selected_ids.add(c["sequence_id"])

    remaining_budget = n_designs - len(selected)

    # 2. Select Pareto-optimal (rank 1) candidates with diversity
    rank1 = [
        c for c in sorted_candidates
        if c.get("pareto_rank") == 1
        and c.get("sequence_id") not in selected_ids
    ]

    if rank1:
        diverse_rank1 = _select_diverse_subset(
            rank1, min(len(rank1), remaining_budget)
        )
        for c in diverse_rank1:
            if c.get("sequence_id") not in selected_ids:
                c["selected"] = True
                c["selection_reason"] = "pareto_rank_1"
                selected.append(c)
                selected_ids.add(c["sequence_id"])

    remaining_budget = n_designs - len(selected)

    # 3. Fill from subsequent ranks
    rank = 2
    max_rank = max(
        (c.get("pareto_rank", 1) for c in all_candidates), default=1
    )

    while remaining_budget > 0 and rank <= max_rank:
        rank_candidates = [
            c for c in sorted_candidates
            if c.get("pareto_rank") == rank
            and c.get("sequence_id") not in selected_ids
        ]

        if rank_candidates:
            n_from_rank = min(len(rank_candidates), remaining_budget)
            diverse_from_rank = _select_diverse_subset(
                rank_candidates, n_from_rank
            )
            for c in diverse_from_rank:
                if c.get("sequence_id") not in selected_ids:
                    c["selected"] = True
                    c["selection_reason"] = f"pareto_rank_{rank}"
                    selected.append(c)
                    selected_ids.add(c["sequence_id"])

        remaining_budget = n_designs - len(selected)
        rank += 1

    return selected[:n_designs]


def _is_wt_adjacent(candidate: dict) -> bool:
    """Check if a candidate is WT-adjacent (conservative design)."""
    backbone = str(candidate.get("backbone", "")).lower()
    design_branch = str(candidate.get("design_branch", "")).lower()
    return "wt" in backbone or "wt" in design_branch


def _select_diverse_subset(
    candidates: list[dict],
    n: int,
) -> list[dict]:
    """Select n diverse candidates using greedy farthest-point sampling in metric space."""
    if len(candidates) <= n:
        return list(candidates)

    # Build feature vectors from available metric values
    available_metrics = [
        k for k in METRIC_DIRECTIONS
        if any(
            _safe_float(c.get(k)) is not None
            for c in candidates
        )
    ]

    if not available_metrics:
        # No metrics to compute diversity — just take first n
        return candidates[:n]

    # Build matrix
    matrix = np.zeros((len(candidates), len(available_metrics)))
    for j, metric in enumerate(available_metrics):
        vals = []
        for c in candidates:
            v = _safe_float(c.get(metric))
            vals.append(v if v is not None else 0.0)
        arr = np.array(vals)
        # Normalize
        rng = arr.max() - arr.min()
        if rng > 0:
            matrix[:, j] = (arr - arr.min()) / rng
        else:
            matrix[:, j] = 0.5

    # Greedy farthest-point sampling
    selected_indices = [0]  # start with first candidate
    for _ in range(n - 1):
        # Compute min distance from each unselected to nearest selected
        min_dists = np.full(len(candidates), np.inf)
        for si in selected_indices:
            dists = np.linalg.norm(matrix - matrix[si], axis=1)
            min_dists = np.minimum(min_dists, dists)

        # Mask already-selected
        for si in selected_indices:
            min_dists[si] = -1.0

        next_idx = int(np.argmax(min_dists))
        selected_indices.append(next_idx)

    return [candidates[i] for i in selected_indices]


def _safe_float(val: Any) -> float | None:
    """Safely convert a value to float, returning None on failure or NaN."""
    if val is None:
        return None
    try:
        fv = float(val)
        return fv if np.isfinite(fv) else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(config: dict[str, Any]) -> dict[str, Any]:
    """Execute Stage 7: Cross-validation and Pareto ranking.

    Steps:
    1. Select top 200 candidates from stage_06 results
    2. Run Protenix cross-validation
    3. Run CatPred + RealKcat
    4. Collect all metrics, compute Pareto front
    5. Select 48-96 diverse designs per enzyme
    6. Write final_rankings.csv and pareto_front.csv

    Args:
        config: Loaded pipeline configuration dict.

    Returns:
        Summary dict with keys: total_selected, per_enzyme, output_files.
    """
    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_07_crossval", {})
    output_dir_name = stage_cfg.get("output_dir", "stage_07_crossval_ranking")
    output_dir = base_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    top_n = int(stage_cfg.get("top_n_candidates", 200))
    ca_rmsd_threshold = float(
        stage_cfg.get("protenix_ca_rmsd_agreement", 2.0)
    )
    protenix_bin = str(
        config.get("tools", {}).get("protenix_env", "protenix")
    )
    catpred_dir = Path(
        config.get("tools", {}).get("catpred_dir", "/opt/tools/CatPred")
    )
    realkcat_dir = Path(
        config.get("tools", {}).get("realkcat_dir", "/opt/tools/RealKcat")
    )

    enzyme_ids = sorted(config.get("enzymes", {}).keys())
    n_per_enzyme = top_n // max(len(enzyme_ids), 1)

    # --- Load stage input data ---
    stage_data = _load_stage_data(config, base_dir)

    # --- Select top candidates per enzyme ---
    all_candidates: list[dict] = []
    for enzyme_id in enzyme_ids:
        enzyme_candidates = _select_top_candidates(
            enzyme_id, stage_data, n_per_enzyme
        )
        all_candidates.extend(enzyme_candidates)

    logger.info("Selected %d candidates for cross-validation", len(all_candidates))

    # --- Run Protenix cross-validation ---
    stage_06_dir = base_dir / config.get("stages", {}).get(
        "stage_06_boltz2_screening", {}
    ).get("output_dir", "stage_06_boltz2_screening")

    protenix_dir = output_dir / "protenix_predictions"
    protenix_dir.mkdir(parents=True, exist_ok=True)

    catpred_scores: list[dict] = []

    for candidate in all_candidates:
        seq_id = candidate["sequence_id"]
        enzyme_id = candidate.get("enzyme_id", "unknown")

        # Run Protenix
        try:
            pred_dir = protenix_dir / seq_id
            protenix_result = run_protenix(
                candidate.get("sequence", ""),
                pred_dir,
                protenix_bin=protenix_bin,
            )

            candidate["protenix_plddt"] = protenix_result["plddt"]

            # Cross-validate against Boltz-2
            boltz2_pdb = stage_06_dir / "structures" / f"{seq_id}.pdb"
            if boltz2_pdb.exists():
                passed, rmsd = cross_validate(
                    boltz2_pdb,
                    Path(protenix_result["pdb_path"]),
                    ca_rmsd_threshold=ca_rmsd_threshold,
                )
                candidate["crossval_ca_rmsd"] = rmsd
                candidate["crossval_passed"] = passed
            else:
                candidate["crossval_ca_rmsd"] = float("nan")
                candidate["crossval_passed"] = True
                logger.warning(
                    "Boltz-2 PDB not found for %s — skipping cross-val",
                    seq_id,
                )

        except (RuntimeError, OSError) as exc:
            logger.warning("Protenix failed for %s: %s", seq_id, exc)
            candidate["protenix_plddt"] = float("nan")
            candidate["crossval_ca_rmsd"] = float("nan")
            candidate["crossval_passed"] = True  # don't discard on failure

        # Run CatPred
        try:
            enzyme_cfg = config.get("enzymes", {}).get(enzyme_id, {})
            substrate_smiles = enzyme_cfg.get("substrate_smiles", "")
            catpred_result = run_catpred(
                candidate.get("sequence", ""),
                substrate_smiles,
                catpred_dir,
            )
            candidate["catpred_kcat"] = catpred_result["kcat"]
            catpred_scores.append({
                "sequence_id": seq_id,
                "kcat": catpred_result["kcat"],
                "km": catpred_result.get("km", float("nan")),
            })
        except Exception as exc:
            logger.warning("CatPred failed for %s: %s", seq_id, exc)
            candidate["catpred_kcat"] = float("nan")

    # Store catpred scores in stage_data for metric collection
    stage_data["stage_07_catpred"] = catpred_scores

    # --- Filter out cross-validation failures ---
    passing = [c for c in all_candidates if c.get("crossval_passed", True)]
    discarded = len(all_candidates) - len(passing)
    logger.info(
        "Cross-validation: %d passed, %d discarded (RMSD > %.1f A)",
        len(passing),
        discarded,
        ca_rmsd_threshold,
    )

    # --- Collect all metrics ---
    for candidate in passing:
        metrics = collect_all_metrics(candidate["sequence_id"], stage_data)
        candidate.update(metrics)

    # --- Compute Pareto front ---
    passing = compute_pareto_front(passing, METRIC_DIRECTIONS)

    # --- Select diverse designs per enzyme ---
    final_selected: list[dict] = []
    per_enzyme_counts: dict[str, int] = {}

    for enzyme_id in enzyme_ids:
        enzyme_passing = [
            c for c in passing if c.get("enzyme_id") == enzyme_id
        ]
        enzyme_pareto = [
            c for c in enzyme_passing if c.get("pareto_rank") == 1
        ]

        # Target: 48-96 per enzyme, default 72
        n_target = min(72, len(enzyme_passing))
        n_target = max(n_target, min(48, len(enzyme_passing)))

        enzyme_selected = select_diverse_designs(
            enzyme_pareto, enzyme_passing, n_designs=n_target
        )

        final_selected.extend(enzyme_selected)
        per_enzyme_counts[enzyme_id] = len(enzyme_selected)

        logger.info(
            "%s: %d/%d selected (%d Pareto-optimal)",
            enzyme_id,
            len(enzyme_selected),
            len(enzyme_passing),
            len(enzyme_pareto),
        )

    # --- Write output files ---
    final_rankings_path = output_dir / "final_rankings.csv"
    pareto_front_path = output_dir / "pareto_front.csv"

    # Ensure all required columns exist
    for c in final_selected:
        for col in REQUIRED_OUTPUT_COLUMNS:
            c.setdefault(col, "")

    write_scores(final_selected, final_rankings_path)
    logger.info(
        "Wrote final_rankings.csv: %d designs", len(final_selected)
    )

    # Pareto front only (rank 1)
    pareto_only = [
        c for c in passing if c.get("pareto_rank") == 1
    ]
    for c in pareto_only:
        for col in REQUIRED_OUTPUT_COLUMNS:
            c.setdefault(col, "")
    write_scores(pareto_only, pareto_front_path)
    logger.info(
        "Wrote pareto_front.csv: %d Pareto-optimal designs", len(pareto_only)
    )

    return {
        "total_selected": len(final_selected),
        "per_enzyme": per_enzyme_counts,
        "total_discarded": discarded,
        "total_pareto_optimal": len(pareto_only),
        "output_files": {
            "final_rankings": str(final_rankings_path),
            "pareto_front": str(pareto_front_path),
        },
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_stage_data(
    config: dict[str, Any],
    base_dir: Path,
) -> dict[str, list[dict]]:
    """Load score CSVs from Stages 4-6.5."""
    stage_data: dict[str, list[dict]] = {}

    stage_csv_map = {
        "stage_04": (
            "stage_04_ligandmpnn",
            "ligandmpnn_scores.csv",
        ),
        "stage_04_5": (
            "stage_04_5_fitness",
            "fitness_scores.csv",
        ),
        "stage_05": (
            "stage_05_esmfold",
            "plddt_scores.csv",
        ),
        "stage_06": (
            "stage_06_boltz2_screening",
            "boltz2_scores.csv",
        ),
        "stage_06_5": (
            "stage_06_5_md",
            "md_scores.csv",
        ),
    }

    for key, (stage_config_key, csv_name) in stage_csv_map.items():
        stage_output_dir = config.get("stages", {}).get(
            stage_config_key, {}
        ).get("output_dir", stage_config_key)
        csv_path = base_dir / stage_output_dir / csv_name

        if csv_path.exists():
            try:
                stage_data[key] = read_scores(csv_path)
                logger.info(
                    "Loaded %d rows from %s", len(stage_data[key]), csv_path
                )
            except Exception as exc:
                logger.warning("Failed to read %s: %s", csv_path, exc)
                stage_data[key] = []
        else:
            logger.info("Score file not found (optional): %s", csv_path)
            stage_data[key] = []

    return stage_data


def _select_top_candidates(
    enzyme_id: str,
    stage_data: dict[str, list[dict]],
    n: int,
) -> list[dict]:
    """Select top N candidates for an enzyme based on available stage data.

    Ranks by: ESMFold pLDDT (Stage 5) as primary sort key.
    """
    # Find candidates for this enzyme from Stage 5 (filtered pool)
    stage_05 = stage_data.get("stage_05", [])
    enzyme_candidates = [
        row for row in stage_05
        if row.get("enzyme_id") == enzyme_id
        and str(row.get("passed_all_filters", "True")).lower() == "true"
    ]

    # Sort by pLDDT descending
    enzyme_candidates.sort(
        key=lambda r: float(r.get("mean_plddt", 0)),
        reverse=True,
    )

    # Take top N
    top = enzyme_candidates[:n]

    # Enrich with sequence data from Stage 4
    stage_04 = stage_data.get("stage_04", [])
    seq_lookup = {
        row["sequence_id"]: row
        for row in stage_04
        if "sequence_id" in row
    }

    result = []
    for row in top:
        candidate = dict(row)
        sid = candidate.get("sequence_id", "")
        candidate["enzyme_id"] = enzyme_id

        # Merge Stage 4 data
        if sid in seq_lookup:
            s4 = seq_lookup[sid]
            candidate.setdefault("sequence", s4.get("sequence", ""))
            candidate.setdefault("backbone", s4.get("backbone_pdb", ""))
            candidate.setdefault(
                "design_branch", s4.get("design_branch", "")
            )

        result.append(candidate)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(config_path: Path) -> None:
    """Load config, set up logging, run Stage 7."""
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
            logging.FileHandler(log_dir / "stage7_ranking.log"),
        ],
    )

    logger.info("=" * 60)
    logger.info("%s — starting", STAGE_NAME)
    logger.info("Config: %s", config_path)
    logger.info("=" * 60)

    results = run_stage(config)

    logger.info("=" * 60)
    logger.info("%s — complete", STAGE_NAME)
    logger.info("  Total selected: %d", results["total_selected"])
    logger.info("  Discarded (cross-val): %d", results["total_discarded"])
    logger.info(
        "  Pareto-optimal: %d", results["total_pareto_optimal"]
    )
    for eid, count in results["per_enzyme"].items():
        logger.info("  %s: %d designs", eid, count)
    logger.info("=" * 60)


def _cli() -> None:
    """Argument parser wrapper for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Stage 7: Cross-validation and Pareto Ranking",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    _cli()
