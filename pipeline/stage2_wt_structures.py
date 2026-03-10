"""Stage 2: Wild-type Structure Prediction with Boltz-2.

Predicts holo structures of all target enzymes complexed with substrates,
products, and cofactors using Boltz-2. For each enzyme, both substrate and
product complexes are generated (product structures needed for Stage 6
product inhibition screening).

CLI: python pipeline/stage2_wt_structures.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from pipeline.config import load_config, get_enzyme_config, resolve_paths
    from pipeline.utils import setup_logging, read_fasta, write_fasta
except ImportError:
    import yaml
    import logging

    def load_config(p):
        return yaml.safe_load(open(p))

    def get_enzyme_config(cfg, eid):
        return cfg.get("enzymes", {}).get(eid, {})

    def resolve_paths(cfg, base):
        return cfg

    def setup_logging(n, d, **kw):
        logging.basicConfig()
        return logging.getLogger(n)

    def read_fasta(p):
        seqs = {}
        with open(p) as f:
            h, s = None, []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if h:
                        seqs[h] = "".join(s)
                    h, s = line[1:], []
                elif h:
                    s.append(line)
            if h:
                seqs[h] = "".join(s)
        return seqs

    def write_fasta(seqs, p):
        with open(p, "w") as f:
            for h, s in seqs.items():
                f.write(f">{h}\n{s}\n")


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
STAGE_NAME = "Stage 2: Wild-type Structure Prediction"
STAGE_NUM = "2"
INPUT_DESCRIPTION = (
    "WT FASTA sequences, substrate SMILES, product SMILES, cofactor definitions"
)
OUTPUT_DESCRIPTION = (
    "PDB files in stage_02_wt_structures/{substrate,product}_complexes/ "
    "with confidence JSONs"
)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def prepare_boltz_input(
    enzyme_id: str,
    sequence: str,
    ligand_smiles: str,
    cofactors: list[str],
    output_dir: Path,
) -> Path:
    """Write a Boltz-2 input FASTA file with sequence + ligand SMILES + cofactors.

    Boltz-2 uses a FASTA-like input format::

        >protein|sequence
        MKTLLLTLVVVTIVCLDLGYTLS...
        >ligand|SMILES
        CC(=O)OC1=CC=CC=C1C(=O)O
        >cofactor|name
        NAD+

    Args:
        enzyme_id: Enzyme identifier.
        sequence: Amino acid sequence.
        ligand_smiles: SMILES string for the ligand.
        cofactors: List of cofactor name strings.
        output_dir: Directory to write the input file.

    Returns:
        Path to the written input FASTA file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_dir / f"{enzyme_id}_boltz_input.fasta"

    with open(input_path, "w") as fh:
        # Protein sequence
        fh.write(f">protein|{enzyme_id}\n")
        for i in range(0, len(sequence), 80):
            fh.write(sequence[i : i + 80] + "\n")

        # Ligand SMILES
        fh.write(f">ligand|SMILES\n")
        fh.write(f"{ligand_smiles}\n")

        # Cofactors
        for cofactor in cofactors:
            fh.write(f">cofactor|{cofactor}\n")
            fh.write(f"{cofactor}\n")

    return input_path


def run_boltz2(
    input_fasta: Path,
    output_dir: Path,
    recycling_steps: int = 10,
    num_seeds: int = 5,
    boltz_bin: str = "boltz",
) -> dict:
    """Run Boltz-2 prediction on an input FASTA file.

    Constructs and executes::

        boltz predict input.fasta \\
            --recycling_steps 10 \\
            --num_seeds 5 \\
            --output_dir output/ \\
            --use_msa_server false

    Args:
        input_fasta: Path to the Boltz-2 input file.
        output_dir: Directory for prediction outputs.
        recycling_steps: Number of recycling iterations (default 10).
        num_seeds: Number of random seeds (default 5).
        boltz_bin: Name or path of the boltz binary.

    Returns:
        Dict with ``"pdb_files"`` and ``"confidence_jsons"`` lists of Paths.

    Raises:
        RuntimeError: If boltz predict exits with non-zero return code.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(boltz_bin),
        "predict",
        str(input_fasta),
        "--recycling_steps", str(recycling_steps),
        "--num_seeds", str(num_seeds),
        "--output_dir", str(output_dir),
        "--use_msa_server", "false",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Boltz-2 prediction failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    pdb_files = sorted(output_dir.glob("**/*.pdb"))
    confidence_jsons = sorted(output_dir.glob("**/*confidence*.json"))

    return {
        "pdb_files": pdb_files,
        "confidence_jsons": confidence_jsons,
    }


def select_best_model(output_dir: Path) -> tuple[Path, dict]:
    """Select the best model from Boltz-2 outputs by highest iPTM score.

    Reads all confidence JSON files in *output_dir*, selects the model with
    the highest ``iptm`` score, and returns its PDB path and confidence dict.

    Args:
        output_dir: Directory containing Boltz-2 prediction outputs.

    Returns:
        Tuple of (pdb_path, confidence_dict).
        confidence_dict has keys: plddt, iptm, ptm (and possibly others).

    Raises:
        FileNotFoundError: If no confidence JSON files are found.
    """
    confidence_files = sorted(Path(output_dir).glob("**/*confidence*.json"))
    if not confidence_files:
        raise FileNotFoundError(
            f"No confidence JSON files found in {output_dir}"
        )

    best_iptm = -1.0
    best_confidence: dict = {}
    best_pdb: Path | None = None

    for conf_file in confidence_files:
        with open(conf_file) as fh:
            conf_data = json.load(fh)

        iptm = conf_data.get("iptm", 0.0)
        if iptm > best_iptm:
            best_iptm = iptm
            best_confidence = conf_data

            # Find the corresponding PDB file (same stem, .pdb extension)
            # Boltz-2 typically names them with matching stems
            stem = conf_file.stem.replace("_confidence", "").replace("confidence_", "")
            pdb_candidates = list(conf_file.parent.glob(f"{stem}*.pdb"))
            if not pdb_candidates:
                # Fallback: look for any PDB in the same directory
                pdb_candidates = sorted(conf_file.parent.glob("*.pdb"))
            if pdb_candidates:
                best_pdb = pdb_candidates[0]

    if best_pdb is None:
        # Final fallback: pick any PDB in the output dir
        all_pdbs = sorted(Path(output_dir).glob("**/*.pdb"))
        if all_pdbs:
            best_pdb = all_pdbs[0]
        else:
            raise FileNotFoundError(
                f"No PDB files found in {output_dir}"
            )

    return best_pdb, best_confidence


def run_boltz2_for_complex(
    enzyme_id: str,
    sequence: str,
    ligand_smiles: str,
    cofactors: list[str],
    output_dir: Path,
    ligand_type: str,
    config: dict,
) -> dict:
    """Full workflow for one enzyme + one ligand type (substrate or product).

    Prepares Boltz-2 input, runs prediction, selects best model.

    Args:
        enzyme_id: Enzyme identifier.
        sequence: Amino acid sequence.
        ligand_smiles: SMILES string for the ligand.
        cofactors: List of cofactor names.
        output_dir: Base output directory for this complex.
        ligand_type: Either ``"substrate"`` or ``"product"``.
        config: Pipeline configuration dictionary.

    Returns:
        Dict with keys: enzyme_id, ligand_type, pdb_path, confidence.
    """
    import logging

    logger = logging.getLogger(f"pipeline.{STAGE_NUM}")

    # Get stage-specific config
    stage_cfg = config.get("stages", {}).get("stage_02_wt_structures", {})
    recycling_steps = stage_cfg.get("recycling_steps", 10)
    num_seeds = stage_cfg.get("num_seeds", 5)
    boltz_bin = config.get("tools", {}).get("boltz_bin", "boltz")

    # Prepare Boltz-2 input
    complex_dir = Path(output_dir) / f"{enzyme_id}_{ligand_type}"
    input_fasta = prepare_boltz_input(
        enzyme_id=enzyme_id,
        sequence=sequence,
        ligand_smiles=ligand_smiles,
        cofactors=cofactors,
        output_dir=complex_dir,
    )

    logger.info(
        "Running Boltz-2 for %s %s complex (recycling=%d, seeds=%d)",
        enzyme_id, ligand_type, recycling_steps, num_seeds,
    )

    # Run Boltz-2
    boltz_output = run_boltz2(
        input_fasta=input_fasta,
        output_dir=complex_dir / "predictions",
        recycling_steps=recycling_steps,
        num_seeds=num_seeds,
        boltz_bin=boltz_bin,
    )

    # Select best model
    best_pdb, confidence = select_best_model(complex_dir / "predictions")

    logger.info(
        "%s %s — best model: iPTM=%.3f, pLDDT=%.3f, pdb=%s",
        enzyme_id, ligand_type,
        confidence.get("iptm", 0.0),
        confidence.get("plddt", 0.0),
        best_pdb.name,
    )

    return {
        "enzyme_id": enzyme_id,
        "ligand_type": ligand_type,
        "pdb_path": str(best_pdb),
        "confidence": confidence,
    }


def run_stage(config: dict) -> dict:
    """Run Stage 2: Wild-type structure prediction for all enzymes.

    For each enzyme, runs Boltz-2 on both substrate and product complexes.
    Creates output directories and copies best PDB + confidence JSON to
    final locations.

    Args:
        config: Loaded pipeline configuration dictionary.

    Returns:
        Dict mapping enzyme_id -> {
            "substrate_pdb": Path,
            "product_pdb": Path,
            "confidences": {"substrate": dict, "product": dict},
        }
    """
    import logging

    logger = logging.getLogger(f"pipeline.{STAGE_NUM}")

    base_dir = Path(config["global"]["base_dir"])
    stage_cfg = config.get("stages", {}).get("stage_02_wt_structures", {})
    output_dir = base_dir / stage_cfg.get("output_dir", "stage_02_wt_structures")

    # Create output subdirectories
    substrate_dir = output_dir / "substrate_complexes"
    product_dir = output_dir / "product_complexes"
    substrate_dir.mkdir(parents=True, exist_ok=True)
    product_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    for enzyme_id, enzyme_cfg in config["enzymes"].items():
        logger.info("Processing %s (%s)", enzyme_id, enzyme_cfg.get("name", enzyme_id))

        # Read WT sequence
        seq_file = enzyme_cfg["sequence_file"]
        if not isinstance(seq_file, Path):
            seq_file = Path(seq_file)
            if not seq_file.is_absolute():
                seq_file = base_dir / seq_file

        sequences = read_fasta(seq_file)
        if not sequences:
            logger.error("No sequences found in %s for %s", seq_file, enzyme_id)
            continue

        # Use the first sequence
        sequence = next(iter(sequences.values()))
        cofactors = enzyme_cfg.get("cofactors", [])

        enzyme_results: dict = {"confidences": {}}

        # Run substrate complex
        for ligand_type, smiles_key, final_dir in [
            ("substrate", "substrate_smiles", substrate_dir),
            ("product", "product_smiles", product_dir),
        ]:
            ligand_smiles = enzyme_cfg.get(smiles_key, "")
            if not ligand_smiles:
                logger.warning(
                    "No %s SMILES for %s — skipping %s complex",
                    ligand_type, enzyme_id, ligand_type,
                )
                continue

            try:
                result = run_boltz2_for_complex(
                    enzyme_id=enzyme_id,
                    sequence=sequence,
                    ligand_smiles=ligand_smiles,
                    cofactors=cofactors,
                    output_dir=output_dir / "work",
                    ligand_type=ligand_type,
                    config=config,
                )
            except (RuntimeError, FileNotFoundError) as exc:
                logger.error(
                    "Boltz-2 failed for %s %s: %s", enzyme_id, ligand_type, exc
                )
                continue

            # Copy best PDB to final location
            src_pdb = Path(result["pdb_path"])
            dst_pdb = final_dir / f"{enzyme_id}_{ligand_type}.pdb"
            shutil.copy2(src_pdb, dst_pdb)

            # Write confidence JSON alongside the PDB
            dst_json = final_dir / f"{enzyme_id}_{ligand_type}_confidence.json"
            with open(dst_json, "w") as fh:
                json.dump(result["confidence"], fh, indent=2)

            enzyme_results[f"{ligand_type}_pdb"] = str(dst_pdb)
            enzyme_results["confidences"][ligand_type] = result["confidence"]

            logger.info(
                "%s %s complex saved to %s", enzyme_id, ligand_type, dst_pdb,
            )

        results[enzyme_id] = enzyme_results

    logger.info("Stage 2 complete. Structures predicted for: %s", list(results.keys()))
    return results


def main(config_path: Path) -> None:
    """Entry point: load config, run WT structure prediction, log results."""
    config = load_config(config_path)
    base_dir = Path(config["global"]["base_dir"])
    config = resolve_paths(config, base_dir)

    log_dir = base_dir / config["global"].get("log_dir", "logs")
    logger = setup_logging("stage2_wt_structures", log_dir)

    logger.info("Starting %s", STAGE_NAME)
    results = run_stage(config)
    logger.info("Finished %s — outputs: %s", STAGE_NAME, results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)
