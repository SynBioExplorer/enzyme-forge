"""
Contract: StageInterfaceProtocol
Version: da599e4 (commit hash when contract was created)
Generated: 2026-03-10T17:00:00Z

Defines the standard interface that every pipeline stage module must implement.
Each stage is a standalone Python script with a main() function and CLI interface.

File I/O Contracts (stage inputs/outputs):
    Stage 0: reads config YAML, WT FASTA -> writes bottleneck_results.csv, priority_list.csv
    Stage 1: reads config YAML, WT FASTA -> writes .a3m MSA files per enzyme
    Stage 2: reads config YAML, WT FASTA, SMILES -> writes PDB files (substrate + product complexes)
    Stage 3: reads config YAML, WT PDB -> writes backbone PDB files per enzyme
    Stage 4: reads config YAML, backbone PDBs / WT PDBs -> writes designed FASTA + scores CSV
    Stage 4.5: reads config YAML, designed FASTA, MSA .a3m -> writes fitness_scores.csv
    Stage 5: reads config YAML, designed FASTA -> writes filtered FASTA + plddt_scores.csv + PDB
    Stage 6: reads config YAML, filtered FASTA, SMILES -> writes screening_results.csv + PDB
    Stage 6.5: reads config YAML, top PDB files -> writes md_metrics.csv
    Stage 7: reads config YAML, filtered FASTA, all scores CSVs -> writes final_rankings.csv
    Stage 7.5: reads config YAML, ranked FASTA, PDB files -> writes glycosylation_report.csv + modified FASTA
    Stage 8: reads config YAML, final FASTA -> writes DNA FASTA + plate_maps.csv
"""
from typing import Protocol
from pathlib import Path


class StageModuleProtocol(Protocol):
    """Every stage module must implement these."""

    STAGE_NAME: str
    """Human-readable stage name, e.g., 'Stage 1: MSA Generation'."""

    STAGE_NUM: str
    """Stage number string, e.g., '1', '4.5', '6.5'."""

    INPUT_DESCRIPTION: str
    """Description of expected input files and formats."""

    OUTPUT_DESCRIPTION: str
    """Description of output files and formats."""

    def main(self, config_path: Path) -> None:
        """Run the stage with the given configuration.

        Args:
            config_path: Path to the pipeline YAML config file.
                         The config contains all parameters, paths, and
                         enzyme-specific settings.

        Raises:
            FileNotFoundError: If required input files are missing.
            RuntimeError: If the stage fails during execution.
        """
        ...
