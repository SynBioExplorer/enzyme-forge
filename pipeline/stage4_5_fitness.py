"""Stage 4.5: Zero-shot Fitness Ensemble Scoring.

Scores all designed sequences for evolutionary plausibility using an
ensemble of three zero-shot fitness predictors:
  1. TranceptEVE L — autoregressive + EVE variational model
  2. ESM-2 log-likelihood — protein language model pseudo-likelihood
  3. GEMME — MSA-based evolutionary conservation

Ensemble score is a soft weighting factor for Stage 7 ranking (NOT a filter).

CLI: python pipeline/stage4_5_fitness.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pipeline.config import load_config, get_enzyme_config, resolve_paths
    from pipeline.utils import setup_logging, read_fasta, write_scores
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
                line = line.strip()
                if line.startswith(">"):
                    if header:
                        seqs[header] = "".join(seq)
                    header, seq = line[1:].split()[0], []
                else:
                    seq.append(line)
            if header:
                seqs[header] = "".join(seq)
        return seqs

    def write_scores(scores, p):  # type: ignore[misc]
        import csv as _csv
        if not scores:
            return
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=scores[0].keys())
            w.writeheader()
            w.writerows(scores)


# ---------------------------------------------------------------------------
# Module constants (StageModuleProtocol)
# ---------------------------------------------------------------------------

STAGE_NAME = "Stage 4.5: Zero-shot Fitness Ensemble"
STAGE_NUM = "4.5"
INPUT_DESCRIPTION = "Designed FASTA sequences, .a3m MSA files from Stage 1"
OUTPUT_DESCRIPTION = "fitness_scores.csv in stage_04.5_fitness_scores/"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def compute_mutations(sequence: str, wt_sequence: str) -> list[str]:
    """Return mutations between a designed sequence and WT.

    Each mutation is formatted as ``"A42G"`` (WT_aa, 1-indexed position,
    designed_aa). Only substitutions at aligned positions are reported;
    length differences beyond the shorter sequence are ignored.

    Args:
        sequence: Designed amino acid sequence.
        wt_sequence: Wild-type amino acid sequence.

    Returns:
        List of mutation strings, e.g. ``["A42G", "L100V"]``.
    """
    mutations: list[str] = []
    for i, (wt_aa, des_aa) in enumerate(zip(wt_sequence, sequence)):
        if wt_aa != des_aa:
            mutations.append(f"{wt_aa}{i + 1}{des_aa}")
    return mutations


# ---------------------------------------------------------------------------
# TranceptEVE scorer
# ---------------------------------------------------------------------------

def run_trancepteve(
    sequences: dict[str, str],
    wt_sequence: str,
    msa_path: Path,
    trancepteve_dir: Path,
    model_type: str = "L",
) -> dict[str, float]:
    """Score sequences with TranceptEVE via subprocess.

    Writes a temporary FASTA, calls ``trancepteve/score_sequences.py``,
    and parses the output CSV.

    Args:
        sequences: ``{sequence_id: amino_acid_sequence}`` dict.
        wt_sequence: Wild-type sequence for the enzyme.
        msa_path: Path to ``.a3m`` MSA file from Stage 1.
        trancepteve_dir: Path to TranceptEVE installation.
        model_type: Model size (default ``"L"``).

    Returns:
        ``{sequence_id: score}`` dict. Returns zeros with a warning if
        TranceptEVE is not available.
    """
    if not sequences:
        return {}

    score_script = trancepteve_dir / "score_sequences.py"
    if not score_script.exists():
        logger.warning(
            "TranceptEVE not found at %s — returning zero scores", score_script
        )
        return {sid: 0.0 for sid in sequences}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write input FASTA
            input_fasta = tmpdir / "input.fasta"
            with open(input_fasta, "w") as fh:
                for sid, seq in sequences.items():
                    fh.write(f">{sid}\n{seq}\n")

            output_csv = tmpdir / "scores.csv"

            cmd = [
                "python", str(score_script),
                "--model_type", model_type,
                "--MSA_data_folder", str(msa_path.parent),
                "--sequence_file", str(input_fasta),
                "--output", str(output_csv),
            ]

            logger.info("Running TranceptEVE: %s", " ".join(cmd))
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )

            if result.returncode != 0:
                logger.error(
                    "TranceptEVE failed (exit %d): %s",
                    result.returncode, result.stderr[:500],
                )
                return {sid: 0.0 for sid in sequences}

            # Parse output
            scores: dict[str, float] = {}
            with open(output_csv) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    sid = row.get("sequence_id", row.get("id", ""))
                    score_val = row.get("score", row.get("avg_score", "0"))
                    try:
                        scores[sid] = float(score_val)
                    except (ValueError, TypeError):
                        scores[sid] = 0.0

            # Fill missing
            for sid in sequences:
                if sid not in scores:
                    scores[sid] = 0.0

            return scores

    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.error("TranceptEVE execution error: %s", exc)
        return {sid: 0.0 for sid in sequences}


# ---------------------------------------------------------------------------
# ESM-2 log-likelihood scorer
# ---------------------------------------------------------------------------

def run_esm2_likelihood(
    sequences: dict[str, str],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute pseudo-log-likelihood for each sequence using ESM-2.

    Uses the masked marginals approach: for each position, mask it and
    compute the log-probability of the true token. The per-sequence score
    is the sum of log-probs (unnormalised) divided by sequence length.

    Args:
        sequences: ``{sequence_id: amino_acid_sequence}`` dict.
        model_name: Hugging Face model identifier.
        batch_size: Batch size for inference (reduced on OOM).
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        ``{sequence_id: log_likelihood_score}`` dict.
    """
    if not sequences:
        return {}

    try:
        import torch
        from transformers import AutoTokenizer, EsmForMaskedLM
    except ImportError:
        logger.warning("transformers/torch not available — returning NaN ESM-2 scores")
        return {sid: float("nan") for sid in sequences}

    # Device selection with CUDA OOM fallback
    if device == "cuda":
        try:
            import torch as _torch
            if not _torch.cuda.is_available():
                device = "cpu"
                logger.info("CUDA not available, falling back to CPU")
        except Exception:
            device = "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
    except Exception as exc:
        logger.error("Failed to load ESM-2 model: %s", exc)
        return {sid: float("nan") for sid in sequences}

    scores: dict[str, float] = {}
    seq_items = list(sequences.items())

    for batch_start in range(0, len(seq_items), batch_size):
        batch = seq_items[batch_start : batch_start + batch_size]

        for sid, seq in batch:
            try:
                score = _esm2_masked_marginals(seq, model, tokenizer, device)
                scores[sid] = score
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning(
                        "CUDA OOM for %s (len=%d), retrying on CPU", sid, len(seq)
                    )
                    try:
                        import torch as _torch
                        _torch.cuda.empty_cache()
                        score = _esm2_masked_marginals(
                            seq, model.cpu(), tokenizer, "cpu"
                        )
                        model = model.to(device)
                        scores[sid] = score
                    except Exception as exc2:
                        logger.error("ESM-2 scoring failed for %s: %s", sid, exc2)
                        scores[sid] = float("nan")
                else:
                    logger.error("ESM-2 scoring failed for %s: %s", sid, exc)
                    scores[sid] = float("nan")

    return scores


def _esm2_masked_marginals(
    sequence: str,
    model: Any,
    tokenizer: Any,
    device: str,
) -> float:
    """Compute pseudo-log-likelihood via masked marginals for one sequence.

    For each position i, mask it, run forward pass, and accumulate
    the log-probability of the correct token.

    Returns:
        Mean log-probability per residue position.
    """
    import torch

    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)
    # input_ids shape: (1, L+2) — [CLS] seq... [EOS]

    seq_len = len(sequence)
    total_log_prob = 0.0

    mask_token_id = tokenizer.mask_token_id

    with torch.no_grad():
        for i in range(seq_len):
            masked_ids = input_ids.clone()
            # Position i in sequence corresponds to i+1 in token ids (after CLS)
            token_pos = i + 1
            original_token = masked_ids[0, token_pos].item()
            masked_ids[0, token_pos] = mask_token_id

            outputs = model(masked_ids)
            logits = outputs.logits[0, token_pos]  # (vocab_size,)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            total_log_prob += log_probs[original_token].item()

    return total_log_prob / seq_len


# ---------------------------------------------------------------------------
# GEMME scorer
# ---------------------------------------------------------------------------

def run_gemme(
    sequences: dict[str, str],
    msa_path: Path,
    wt_sequence: str,
    gemme_script: str = "gemme.py",
    output_dir: Path | None = None,
) -> dict[str, float]:
    """Score sequences with GEMME MSA-based conservation.

    Computes mutations relative to WT, writes a variant file, and calls
    GEMME via subprocess. Falls back to ``None`` scores if GEMME is
    unavailable.

    Args:
        sequences: ``{sequence_id: amino_acid_sequence}`` dict.
        msa_path: Path to ``.a3m`` MSA file.
        wt_sequence: Wild-type amino acid sequence.
        gemme_script: Path or command for the GEMME script.
        output_dir: Optional output directory for GEMME results.

    Returns:
        ``{sequence_id: gemme_score}`` dict. Values are ``None`` if GEMME
        is not available.
    """
    if not sequences:
        return {}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            if output_dir is None:
                output_dir = tmpdir

            # Write variants file: each line is a mutation list for one sequence
            variants_file = tmpdir / "variants.txt"
            sid_order: list[str] = []
            with open(variants_file, "w") as fh:
                for sid, seq in sequences.items():
                    muts = compute_mutations(seq, wt_sequence)
                    if muts:
                        fh.write(",".join(muts) + "\n")
                        sid_order.append(sid)
                    else:
                        # Identical to WT — score = 0 (neutral)
                        sid_order.append(sid)
                        fh.write("WT\n")

            cmd = [
                "python", gemme_script,
                "--msa", str(msa_path),
                "--variants", str(variants_file),
                "--output", str(output_dir / "gemme_scores.csv"),
            ]

            logger.info("Running GEMME: %s", " ".join(cmd))
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800
            )

            if result.returncode != 0:
                logger.warning(
                    "GEMME failed (exit %d): %s — returning None scores",
                    result.returncode, result.stderr[:500],
                )
                return {sid: None for sid in sequences}

            # Parse output
            scores: dict[str, float] = {}
            gemme_csv = output_dir / "gemme_scores.csv"
            if gemme_csv.exists():
                with open(gemme_csv) as fh:
                    reader = csv.DictReader(fh)
                    for row_idx, row in enumerate(reader):
                        score_val = row.get("score", row.get("gemme_score", None))
                        sid = sid_order[row_idx] if row_idx < len(sid_order) else None
                        if sid and score_val is not None:
                            try:
                                scores[sid] = float(score_val)
                            except (ValueError, TypeError):
                                scores[sid] = None

            # Fill missing
            for sid in sequences:
                if sid not in scores:
                    scores[sid] = None

            return scores

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("GEMME not available: %s — returning None scores", exc)
        return {sid: None for sid in sequences}


# ---------------------------------------------------------------------------
# Ensemble scoring
# ---------------------------------------------------------------------------

def compute_ensemble_score(scores: dict[str, dict]) -> dict[str, float]:
    """Compute normalised ensemble score from individual predictor scores.

    Each predictor's scores are min-max normalised to [0, 1] across all
    sequences, then averaged (skipping ``None`` / ``NaN`` values).

    Args:
        scores: ``{sequence_id: {"trancepteve": float|None, "esm2": float|None,
                  "gemme": float|None}}``

    Returns:
        ``{sequence_id: ensemble_score}`` with values in [0, 1].
    """
    if not scores:
        return {}

    predictors = ["trancepteve", "esm2", "gemme"]

    # Collect valid values per predictor for normalisation
    pred_values: dict[str, list[float]] = {p: [] for p in predictors}
    for sid, pscores in scores.items():
        for pred in predictors:
            val = pscores.get(pred)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                pred_values[pred].append(val)

    # Min-max ranges
    pred_min: dict[str, float] = {}
    pred_max: dict[str, float] = {}
    for pred in predictors:
        vals = pred_values[pred]
        if vals:
            pred_min[pred] = min(vals)
            pred_max[pred] = max(vals)
        else:
            pred_min[pred] = 0.0
            pred_max[pred] = 0.0

    def _normalise(val: float | None, pred: str) -> float | None:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        rng = pred_max[pred] - pred_min[pred]
        if rng == 0:
            return 0.5  # all values identical
        return (val - pred_min[pred]) / rng

    ensemble: dict[str, float] = {}
    for sid, pscores in scores.items():
        normed = []
        for pred in predictors:
            n = _normalise(pscores.get(pred), pred)
            if n is not None:
                normed.append(n)
        if normed:
            ensemble[sid] = sum(normed) / len(normed)
        else:
            ensemble[sid] = float("nan")

    return ensemble


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(config: dict[str, Any]) -> Path:
    """Execute Stage 4.5: zero-shot fitness ensemble scoring.

    1. Read designed FASTAs from stage_04_ligandmpnn/
    2. Read MSA .a3m files from stage_01_msa/ (per enzyme)
    3. For each enzyme: run all 3 scorers, compute ensemble
    4. Write fitness_scores.csv to stage_04.5_fitness_scores/

    Args:
        config: Loaded pipeline configuration dictionary.

    Returns:
        Path to the output fitness_scores.csv file.
    """
    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_04_5_fitness", {})

    # Output directory
    output_dir_name = stage_cfg.get("output_dir", "stage_04.5_fitness_scores")
    output_dir = base_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Input directories
    ligandmpnn_dir = base_dir / config.get("stages", {}).get(
        "stage_04_ligandmpnn", {}
    ).get("output_dir", "stage_04_ligandmpnn")

    msa_dir = base_dir / config.get("stages", {}).get(
        "stage_01_msa", {}
    ).get("output_dir", "stage_01_msa")

    # Tool paths
    trancepteve_dir = Path(
        config.get("tools", {}).get("trancepteve_dir", "trancepteve")
    )
    gemme_script = config.get("tools", {}).get("gemme_script", "gemme.py")

    # ESM-2 config
    esm2_model = stage_cfg.get("esm2_model", "facebook/esm2_t33_650M_UR50D")
    esm2_batch_size = stage_cfg.get("esm2_batch_size", 32)
    esm2_device = stage_cfg.get("esm2_device", "cuda")

    enzyme_ids = sorted(config.get("enzymes", {}).keys())
    all_scores: list[dict[str, Any]] = []

    for enzyme_id in enzyme_ids:
        enzyme_cfg = config["enzymes"][enzyme_id]
        wt_seq_file = enzyme_cfg.get("sequence_file", "")
        wt_sequence = ""

        # Load WT sequence
        if wt_seq_file and Path(str(wt_seq_file)).exists():
            wt_seqs = read_fasta(Path(str(wt_seq_file)))
            if wt_seqs:
                wt_sequence = next(iter(wt_seqs.values()))

        # Load designed sequences from LigandMPNN output
        enzyme_fasta = ligandmpnn_dir / enzyme_id / "designed_sequences.fasta"
        if not enzyme_fasta.exists():
            # Try alternative naming
            enzyme_fasta = ligandmpnn_dir / f"{enzyme_id}_designed.fasta"
        if not enzyme_fasta.exists():
            # Try globbing
            candidates = list(ligandmpnn_dir.glob(f"{enzyme_id}*.fasta"))
            if candidates:
                enzyme_fasta = candidates[0]
            else:
                logger.warning(
                    "No designed sequences found for %s in %s — skipping",
                    enzyme_id, ligandmpnn_dir,
                )
                continue

        sequences = read_fasta(enzyme_fasta)
        if not sequences:
            logger.warning("Empty FASTA for %s — skipping", enzyme_id)
            continue

        logger.info(
            "%s: scoring %d designed sequences", enzyme_id, len(sequences)
        )

        # MSA path
        msa_path = msa_dir / enzyme_id / f"{enzyme_id}.a3m"
        if not msa_path.exists():
            msa_path = msa_dir / f"{enzyme_id}.a3m"

        # --- Run scorers ---
        # TranceptEVE
        trancepteve_scores = run_trancepteve(
            sequences, wt_sequence, msa_path, trancepteve_dir,
        )

        # ESM-2
        esm2_scores = run_esm2_likelihood(
            sequences,
            model_name=esm2_model,
            batch_size=esm2_batch_size,
            device=esm2_device,
        )

        # GEMME
        gemme_scores = run_gemme(
            sequences, msa_path, wt_sequence, gemme_script=gemme_script,
        )

        # --- Collect per-sequence scores ---
        per_seq_scores: dict[str, dict] = {}
        for sid in sequences:
            per_seq_scores[sid] = {
                "trancepteve": trancepteve_scores.get(sid),
                "esm2": esm2_scores.get(sid),
                "gemme": gemme_scores.get(sid),
            }

        # Ensemble
        ensemble_scores = compute_ensemble_score(per_seq_scores)

        # Build output rows
        for sid, seq in sequences.items():
            n_mutations = len(compute_mutations(seq, wt_sequence)) if wt_sequence else 0
            row = {
                "sequence_id": sid,
                "enzyme_id": enzyme_id,
                "trancepteve_score": trancepteve_scores.get(sid),
                "esm2_ll": esm2_scores.get(sid),
                "gemme_score": gemme_scores.get(sid),
                "ensemble_score": ensemble_scores.get(sid, float("nan")),
                "n_mutations_from_wt": n_mutations,
            }
            all_scores.append(row)

    # Write output CSV
    output_csv = output_dir / "fitness_scores.csv"
    write_scores(all_scores, output_csv)

    logger.info(
        "Stage 4.5 complete — wrote %d rows to %s",
        len(all_scores), output_csv,
    )
    return output_csv


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(config_path: Path) -> None:
    """Load config, set up logging, run Stage 4.5."""
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
            logging.FileHandler(log_dir / "stage4_5_fitness.log"),
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
        description="Stage 4.5: Zero-shot Fitness Ensemble Scoring",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    _cli()
