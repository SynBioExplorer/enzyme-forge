"""Tests for Stage 4.5: Zero-shot Fitness Ensemble (pipeline/stage4_5_fitness.py).

All tests run without ESM-2 models, GPU, TranceptEVE, or GEMME installed.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import pytest

from pipeline.stage4_5_fitness import (
    STAGE_NAME,
    STAGE_NUM,
    INPUT_DESCRIPTION,
    OUTPUT_DESCRIPTION,
    compute_mutations,
    compute_ensemble_score,
    run_esm2_likelihood,
    run_trancepteve,
    run_gemme,
    run_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> dict:
    """Create a minimal config for testing."""
    base_dir = tmp_path / "pipeline"
    base_dir.mkdir()

    # Create input directories
    seq_dir = base_dir / "inputs" / "sequences"
    seq_dir.mkdir(parents=True)

    ligandmpnn_dir = base_dir / "stage_04_ligandmpnn"
    msa_dir = base_dir / "stage_01_msa"

    for eid in ("enzyme_A", "enzyme_B"):
        # WT sequence files
        fasta = seq_dir / f"{eid}.fasta"
        fasta.write_text(f">{eid}\nMVLSPADKTNVKAAWGKVGA\n")

        # Designed sequences (LigandMPNN output)
        enz_lmpnn = ligandmpnn_dir / eid
        enz_lmpnn.mkdir(parents=True)
        designed_fasta = enz_lmpnn / "designed_sequences.fasta"
        designed_fasta.write_text(
            f">{eid}_design_001\nMVLSPADKTNVKAAWGKVGA\n"
            f">{eid}_design_002\nMVLSPADKTNVKAAWGKVGG\n"
            f">{eid}_design_003\nMVLTPADKTNVKAAWGKVGA\n"
        )

        # MSA .a3m files
        enz_msa = msa_dir / eid
        enz_msa.mkdir(parents=True)
        a3m = enz_msa / f"{eid}.a3m"
        a3m.write_text(f">{eid}\nMVLSPADKTNVKAAWGKVGA\n")

    return {
        "global": {
            "project_name": "test",
            "base_dir": str(base_dir),
            "log_dir": "logs",
            "round_number": 1,
        },
        "databases": {
            "colabfold_db_dir": str(tmp_path / "db"),
            "mmseqs2_db_dir": str(tmp_path / "mmseqs"),
        },
        "tools": {
            "trancepteve_dir": str(tmp_path / "trancepteve"),
            "gemme_script": "gemme.py",
        },
        "stages": {
            "stage_04_5_fitness": {
                "output_dir": "stage_04.5_fitness_scores",
                "esm2_model": "facebook/esm2_t33_650M_UR50D",
                "esm2_batch_size": 4,
                "esm2_device": "cpu",
            },
            "stage_04_ligandmpnn": {
                "output_dir": "stage_04_ligandmpnn",
            },
            "stage_01_msa": {
                "output_dir": "stage_01_msa",
            },
        },
        "enzymes": {
            "enzyme_A": {
                "name": "Enzyme A",
                "length": 20,
                "sequence_file": str(seq_dir / "enzyme_A.fasta"),
                "substrate_smiles": "CCO",
                "product_smiles": "CC=O",
                "fixed_residues": [1, 5, 10],
                "hotspot_residues": [5, 10, 15],
                "contigs": "A1-20",
                "cofactors": ["NAD+"],
            },
            "enzyme_B": {
                "name": "Enzyme B",
                "length": 20,
                "sequence_file": str(seq_dir / "enzyme_B.fasta"),
                "substrate_smiles": "CC=O",
                "product_smiles": "CC(O)=O",
                "fixed_residues": [2, 8],
                "hotspot_residues": [8, 12],
                "contigs": "A1-20",
                "cofactors": ["FAD"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_stage_name(self):
        assert "4.5" in STAGE_NAME
        assert "Fitness" in STAGE_NAME or "fitness" in STAGE_NAME.lower()

    def test_stage_num(self):
        assert STAGE_NUM == "4.5"

    def test_input_description(self):
        assert INPUT_DESCRIPTION  # non-empty

    def test_output_description(self):
        assert "fitness_scores" in OUTPUT_DESCRIPTION


# ---------------------------------------------------------------------------
# compute_mutations
# ---------------------------------------------------------------------------

class TestComputeMutations:
    def test_no_mutations(self):
        assert compute_mutations("MVLSPAD", "MVLSPAD") == []

    def test_single_mutation(self):
        result = compute_mutations("MVLSPAD", "MVLSAAD")
        assert result == ["A5P"]  # position 5 (1-indexed), A→P

    def test_multiple_mutations(self):
        wt = "MVLSPAD"
        des = "AVLSAAD"  # M1A, P5A
        result = compute_mutations(des, wt)
        assert "M1A" in result
        assert "P5A" in result
        assert len(result) == 2

    def test_all_mutated(self):
        wt = "AAA"
        des = "GGG"
        result = compute_mutations(des, wt)
        assert result == ["A1G", "A2G", "A3G"]

    def test_different_lengths_uses_shorter(self):
        # Only aligns to min(len(wt), len(designed))
        wt = "MVLS"
        des = "MVLT"  # S4T only
        result = compute_mutations(des, wt)
        assert result == ["S4T"]

    def test_empty_sequences(self):
        assert compute_mutations("", "") == []

    def test_1indexed_positions(self):
        result = compute_mutations("G", "A")
        assert result == ["A1G"]


# ---------------------------------------------------------------------------
# compute_ensemble_score
# ---------------------------------------------------------------------------

class TestComputeEnsembleScore:
    def test_all_scores_present(self):
        scores = {
            "seq1": {"trancepteve": 0.8, "esm2": 0.6, "gemme": 0.4},
            "seq2": {"trancepteve": 0.2, "esm2": 0.4, "gemme": 0.6},
        }
        result = compute_ensemble_score(scores)
        assert "seq1" in result
        assert "seq2" in result
        # All values should be in [0, 1]
        for val in result.values():
            assert 0.0 <= val <= 1.0

    def test_none_values_skipped(self):
        scores = {
            "seq1": {"trancepteve": 1.0, "esm2": None, "gemme": None},
            "seq2": {"trancepteve": 0.0, "esm2": None, "gemme": None},
        }
        result = compute_ensemble_score(scores)
        assert result["seq1"] == 1.0  # max normalised
        assert result["seq2"] == 0.0  # min normalised

    def test_nan_values_skipped(self):
        scores = {
            "seq1": {"trancepteve": 1.0, "esm2": float("nan"), "gemme": 0.5},
        }
        result = compute_ensemble_score(scores)
        assert not math.isnan(result["seq1"])

    def test_all_none_returns_nan(self):
        scores = {
            "seq1": {"trancepteve": None, "esm2": None, "gemme": None},
        }
        result = compute_ensemble_score(scores)
        assert math.isnan(result["seq1"])

    def test_partial_scoring(self):
        """Ensemble still computed when some scorers are unavailable."""
        scores = {
            "seq1": {"trancepteve": 0.5, "esm2": None, "gemme": 0.8},
            "seq2": {"trancepteve": 0.3, "esm2": None, "gemme": 0.2},
        }
        result = compute_ensemble_score(scores)
        assert "seq1" in result
        assert "seq2" in result
        # Both should get valid scores
        assert not math.isnan(result["seq1"])
        assert not math.isnan(result["seq2"])

    def test_empty_input(self):
        assert compute_ensemble_score({}) == {}

    def test_identical_values_normalise_to_half(self):
        scores = {
            "seq1": {"trancepteve": 5.0, "esm2": 5.0, "gemme": 5.0},
        }
        result = compute_ensemble_score(scores)
        # All identical → normalised to 0.5
        assert result["seq1"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# run_esm2_likelihood (mocked)
# ---------------------------------------------------------------------------

class TestRunEsm2Likelihood:
    def test_with_mock_transformers(self):
        """ESM-2 scoring works with fully mocked torch + transformers."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        sequences = {"seq1": "MVLS", "seq2": "AVLS"}

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.mask_token_id = 32

        def mock_tokenize(seq, return_tensors=None, add_special_tokens=True):
            ids = torch.tensor([[0] + list(range(4, 4 + len(seq))) + [2]])
            return {"input_ids": ids}

        mock_tokenizer.side_effect = mock_tokenize

        mock_tokenizer_cls = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock model: return logits tensor from forward pass
        vocab_size = 33

        def model_forward(input_ids):
            seq_len = input_ids.shape[1]
            logits = torch.zeros(1, seq_len, vocab_size)
            logits[0, :, :] = -10.0
            for i in range(seq_len):
                logits[0, i, i % vocab_size] = 0.0
            result = MagicMock()
            result.logits = logits
            return result

        mock_model = MagicMock()
        mock_model.side_effect = model_forward
        mock_model.__call__ = model_forward
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        # Make __call__ work via MagicMock protocol
        mock_model.return_value = model_forward(torch.zeros(1, 10, dtype=torch.long))

        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        with patch("pipeline.stage4_5_fitness.AutoTokenizer", mock_tokenizer_cls), \
             patch("pipeline.stage4_5_fitness.EsmForMaskedLM", mock_model_cls):
            result = run_esm2_likelihood(sequences, device="cpu")

        assert "seq1" in result
        assert "seq2" in result
        assert isinstance(result["seq1"], float)

    def test_missing_transformers_returns_nan(self):
        """When transformers is not importable, returns NaN scores."""
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            with patch(
                "pipeline.stage4_5_fitness.run_esm2_likelihood",
                wraps=run_esm2_likelihood,
            ):
                # Simulate import failure by making the import raise
                original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

                def mock_import(name, *args, **kwargs):
                    if name in ("torch", "transformers"):
                        raise ImportError(f"No module named '{name}'")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    result = run_esm2_likelihood({"s1": "MVLS"}, device="cpu")

                assert "s1" in result
                assert math.isnan(result["s1"])

    def test_empty_sequences(self):
        assert run_esm2_likelihood({}) == {}


# ---------------------------------------------------------------------------
# run_trancepteve (mocked)
# ---------------------------------------------------------------------------

class TestRunTrancepteve:
    def test_missing_script_returns_zeros(self, tmp_path):
        """Falls back gracefully when TranceptEVE is not installed."""
        sequences = {"seq1": "MVLS", "seq2": "AVLS"}
        msa_path = tmp_path / "msa" / "test.a3m"
        msa_path.parent.mkdir(parents=True)
        msa_path.write_text(">wt\nMVLS\n")

        result = run_trancepteve(
            sequences,
            wt_sequence="MVLS",
            msa_path=msa_path,
            trancepteve_dir=tmp_path / "nonexistent_trancepteve",
        )

        assert result == {"seq1": 0.0, "seq2": 0.0}

    def test_subprocess_failure_returns_zeros(self, tmp_path):
        """Returns zero scores when subprocess fails."""
        # Create the script file so the existence check passes
        script_dir = tmp_path / "trancepteve"
        script_dir.mkdir()
        script = script_dir / "score_sequences.py"
        script.write_text("# dummy")

        sequences = {"seq1": "MVLS"}
        msa_path = tmp_path / "test.a3m"
        msa_path.write_text(">wt\nMVLS\n")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="error"
            )
            result = run_trancepteve(
                sequences,
                wt_sequence="MVLS",
                msa_path=msa_path,
                trancepteve_dir=script_dir,
            )

        assert result == {"seq1": 0.0}

    def test_empty_sequences(self, tmp_path):
        result = run_trancepteve(
            {},
            wt_sequence="MVLS",
            msa_path=tmp_path / "test.a3m",
            trancepteve_dir=tmp_path,
        )
        assert result == {}

    def test_successful_parse(self, tmp_path):
        """Parses TranceptEVE output CSV correctly."""
        script_dir = tmp_path / "trancepteve"
        script_dir.mkdir()
        script = script_dir / "score_sequences.py"
        script.write_text("# dummy")

        sequences = {"seq1": "MVLS", "seq2": "AVLS"}
        msa_path = tmp_path / "test.a3m"
        msa_path.write_text(">wt\nMVLS\n")

        def mock_run_side_effect(cmd, **kwargs):
            # Write a fake output CSV to the --output path
            output_idx = cmd.index("--output") + 1
            output_path = Path(cmd[output_idx])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["sequence_id", "score"])
                writer.writeheader()
                writer.writerow({"sequence_id": "seq1", "score": "0.85"})
                writer.writerow({"sequence_id": "seq2", "score": "-0.12"})
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=mock_run_side_effect):
            result = run_trancepteve(
                sequences,
                wt_sequence="MVLS",
                msa_path=msa_path,
                trancepteve_dir=script_dir,
            )

        assert result["seq1"] == pytest.approx(0.85)
        assert result["seq2"] == pytest.approx(-0.12)


# ---------------------------------------------------------------------------
# run_gemme (mocked)
# ---------------------------------------------------------------------------

class TestRunGemme:
    def test_gemme_not_available_returns_none(self):
        """Returns None scores when GEMME script is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError("gemme not found")):
            result = run_gemme(
                {"seq1": "AVLS"},
                msa_path=Path("/tmp/test.a3m"),
                wt_sequence="MVLS",
            )
        assert result == {"seq1": None}

    def test_empty_sequences(self):
        assert run_gemme({}, Path("/tmp/test.a3m"), "MVLS") == {}


# ---------------------------------------------------------------------------
# run_stage (fully mocked scorers)
# ---------------------------------------------------------------------------

class TestRunStage:
    def test_produces_csv_with_required_columns(self, tmp_path):
        """run_stage writes a CSV with the correct columns."""
        config = _make_config(tmp_path)

        with patch(
            "pipeline.stage4_5_fitness.run_trancepteve"
        ) as mock_trancepteve, patch(
            "pipeline.stage4_5_fitness.run_esm2_likelihood"
        ) as mock_esm2, patch(
            "pipeline.stage4_5_fitness.run_gemme"
        ) as mock_gemme:
            # Return mock scores for all sequences
            def trancepteve_side_effect(sequences, *a, **kw):
                return {sid: 0.5 for sid in sequences}

            def esm2_side_effect(sequences, *a, **kw):
                return {sid: -2.0 for sid in sequences}

            def gemme_side_effect(sequences, *a, **kw):
                return {sid: -0.3 for sid in sequences}

            mock_trancepteve.side_effect = trancepteve_side_effect
            mock_esm2.side_effect = esm2_side_effect
            mock_gemme.side_effect = gemme_side_effect

            output_csv = run_stage(config)

        assert output_csv.exists()

        # Read and validate CSV
        with open(output_csv) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) > 0

        required_columns = {
            "sequence_id",
            "enzyme_id",
            "trancepteve_score",
            "esm2_ll",
            "gemme_score",
            "ensemble_score",
            "n_mutations_from_wt",
        }
        actual_columns = set(rows[0].keys())
        assert required_columns.issubset(actual_columns), (
            f"Missing columns: {required_columns - actual_columns}"
        )

    def test_handles_partial_scorer_failure(self, tmp_path):
        """CSV is still written when some scorers return None/NaN."""
        config = _make_config(tmp_path)

        with patch(
            "pipeline.stage4_5_fitness.run_trancepteve"
        ) as mock_trancepteve, patch(
            "pipeline.stage4_5_fitness.run_esm2_likelihood"
        ) as mock_esm2, patch(
            "pipeline.stage4_5_fitness.run_gemme"
        ) as mock_gemme:
            # TranceptEVE works, ESM-2 returns NaN, GEMME returns None
            mock_trancepteve.side_effect = lambda seqs, *a, **kw: {
                sid: 0.5 for sid in seqs
            }
            mock_esm2.side_effect = lambda seqs, *a, **kw: {
                sid: float("nan") for sid in seqs
            }
            mock_gemme.side_effect = lambda seqs, *a, **kw: {
                sid: None for sid in seqs
            }

            output_csv = run_stage(config)

        assert output_csv.exists()

        with open(output_csv) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # All rows should still have ensemble_score (computed from trancepteve only)
        assert len(rows) > 0
        for row in rows:
            assert "ensemble_score" in row

    def test_correct_number_of_rows(self, tmp_path):
        """CSV has one row per designed sequence across all enzymes."""
        config = _make_config(tmp_path)

        with patch(
            "pipeline.stage4_5_fitness.run_trancepteve"
        ) as mock_t, patch(
            "pipeline.stage4_5_fitness.run_esm2_likelihood"
        ) as mock_e, patch(
            "pipeline.stage4_5_fitness.run_gemme"
        ) as mock_g:
            mock_t.side_effect = lambda seqs, *a, **kw: {s: 0.1 for s in seqs}
            mock_e.side_effect = lambda seqs, *a, **kw: {s: -1.0 for s in seqs}
            mock_g.side_effect = lambda seqs, *a, **kw: {s: -0.5 for s in seqs}

            output_csv = run_stage(config)

        with open(output_csv) as fh:
            rows = list(csv.DictReader(fh))

        # 2 enzymes × 3 designed sequences each = 6 rows
        assert len(rows) == 6

    def test_enzyme_id_column_populated(self, tmp_path):
        """Each row has a non-empty enzyme_id."""
        config = _make_config(tmp_path)

        with patch(
            "pipeline.stage4_5_fitness.run_trancepteve"
        ) as mock_t, patch(
            "pipeline.stage4_5_fitness.run_esm2_likelihood"
        ) as mock_e, patch(
            "pipeline.stage4_5_fitness.run_gemme"
        ) as mock_g:
            mock_t.side_effect = lambda seqs, *a, **kw: {s: 0.0 for s in seqs}
            mock_e.side_effect = lambda seqs, *a, **kw: {s: 0.0 for s in seqs}
            mock_g.side_effect = lambda seqs, *a, **kw: {s: 0.0 for s in seqs}

            output_csv = run_stage(config)

        with open(output_csv) as fh:
            rows = list(csv.DictReader(fh))

        enzyme_ids = {row["enzyme_id"] for row in rows}
        assert "enzyme_A" in enzyme_ids
        assert "enzyme_B" in enzyme_ids


# ---------------------------------------------------------------------------
# Ensemble with partial scorers
# ---------------------------------------------------------------------------

class TestEnsemblePartialScorers:
    def test_ensemble_with_one_scorer(self):
        """Ensemble score is computed even with only one available scorer."""
        scores = {
            "seq1": {"trancepteve": 0.8, "esm2": None, "gemme": None},
            "seq2": {"trancepteve": 0.2, "esm2": None, "gemme": None},
        }
        result = compute_ensemble_score(scores)
        # seq1 should be higher than seq2 (normalised)
        assert result["seq1"] > result["seq2"]
        assert result["seq1"] == pytest.approx(1.0)
        assert result["seq2"] == pytest.approx(0.0)

    def test_ensemble_with_two_scorers(self):
        """Ensemble score uses available scorers, skipping None."""
        scores = {
            "seq1": {"trancepteve": 1.0, "esm2": -1.0, "gemme": None},
            "seq2": {"trancepteve": 0.0, "esm2": -2.0, "gemme": None},
        }
        result = compute_ensemble_score(scores)
        assert not math.isnan(result["seq1"])
        assert not math.isnan(result["seq2"])
        # seq1 has higher scores in both available predictors
        assert result["seq1"] > result["seq2"]
