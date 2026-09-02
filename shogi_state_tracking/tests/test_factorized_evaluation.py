import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FactorizedEvaluationTest(unittest.TestCase):
    @staticmethod
    def record():
        state = ["<TURN_BLACK>", "<BOARD_BLACK>", "<K>", "<SQ_5i>", "<P>", "<SQ_7g>",
                 "<BOARD_WHITE>", "<K>", "<SQ_5a>", "<HAND_BLACK>", "<HAND_WHITE>"]
        return {
            "game_id": "g",
            "game_result": 1,
            "terminal_encoding": "eos_on_complete_decisive_game_v1",
            "terminal_token": "<EOS>",
            "initial_sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "move_tokens": ["7g7f"],
            "move_annotations": [{"eligible": True, "piece": "<P>", "source": "<SQ_7g>"}],
            "start_candidates": [{"start_ply": 0, "state_prompt_tokens": state, "position_scope": "unseen_position"}],
            "evaluation_steps": [{"ply": 0, "target_move": "7g7f", "legal_moves": ["7g7f"], "legal_sources_by_piece": {"<P>": ["<SQ_7g>"]}}],
            "legal_drop_available_by_ply": [False],
            "promotion_choice_available_by_ply": [False],
            "position_scope_by_ply": ["unseen_position"], "trajectory_scope": "strict_unseen_position",
        }

    def test_stage_1_implicit_initial_omits_the_stored_prompt(self):
        from factorized_prompt import factorized_vocabulary_tokens
        from factorized_prompt_data import FactorizedPromptSequenceDataset

        vocab = {token: index for index, token in enumerate(factorized_vocabulary_tokens())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(self.record()) + "\n", encoding="utf-8")
            implicit = FactorizedPromptSequenceDataset(
                str(path), vocab, state_prompt_mode="implicit_initial", start_selection="fixed_initial",
                randomize_each_epoch=False,
            )[0]
            explicit = FactorizedPromptSequenceDataset(
                str(path), vocab, state_prompt_mode="explicit", start_selection="fixed_initial",
                randomize_each_epoch=False,
            )[0]
        self.assertEqual(implicit["start_ply"], 0)
        self.assertLess(len(implicit["input_ids"]), len(explicit["input_ids"]))
        self.assertEqual(int(implicit["move_boundary_mask"].sum()), 1)

    def test_ap_annotates_every_eligible_normal_move(self):
        from factorized_prompt import factorized_vocabulary_tokens
        from factorized_prompt_data import FactorizedPromptSequenceDataset

        vocab = {token: index for index, token in enumerate(factorized_vocabulary_tokens())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(self.record()) + "\n", encoding="utf-8")
            example = FactorizedPromptSequenceDataset(
                str(path), vocab, annotation_mode="ap", annotation_probability=1.0,
                randomize_each_epoch=False,
            )[0]
            with self.assertRaisesRegex(ValueError, "ap mode requires"):
                FactorizedPromptSequenceDataset(
                    str(path), vocab, annotation_mode="ap", annotation_probability=0.5,
                    randomize_each_epoch=False,
                )
        self.assertEqual(int(example["hint_target_mask"].sum()), 1)

    def test_standard_initial_sfen_requires_all_nine_pawns(self):
        from factorized_prompt_data import is_standard_initial_sfen

        self.assertTrue(is_standard_initial_sfen(self.record()["initial_sfen"]))
        self.assertFalse(is_standard_initial_sfen(
            "lnsgkgsnl/1r5b1/p1ppppppp/9/9/9/P1PPPPPPP/1B5R1/LNSGKGSNL b - 1"
        ))

    def test_dataset_builder_validates_and_packages_jsonl(self):
        from build_factorized_prompt_dataset import copy_split

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source.jsonl", root / "evaluation.jsonl"
            source.write_text(json.dumps(self.record()) + "\n", encoding="utf-8")
            metrics = copy_split(source, destination, "evaluation")
            self.assertEqual(metrics["records"], 1)
            self.assertEqual(metrics["moves"], 1)
            self.assertTrue(destination.is_file())

    def test_tiny_llama_and_vanilla_training_loss(self):
        from factorized_prompt import factorized_vocabulary_tokens
        from factorized_prompt_data import FactorizedPromptSequenceDataset
        from models import ModelConfig, build_model
        from new_prompt_data import collate_new_prompt_sequences
        from train_new_prompt import loss_for_batch

        record = self.record()
        vocab = {token: index for index, token in enumerate(factorized_vocabulary_tokens())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            example = FactorizedPromptSequenceDataset(
                str(path), vocab, annotation_mode="rap", annotation_probability=1.0,
                randomize_each_epoch=False,
            )[0]
            batch = collate_new_prompt_sequences([example], vocab["<PAD>"], 64)
        for model_type in ("vanilla", "llama"):
            config = ModelConfig(vocab_size=len(vocab), max_seq_len=64, d_model=16, n_layers=1, n_heads=4, d_ff=32, dropout=0.0)
            model = build_model(model_type, config)
            loss, metrics = loss_for_batch(model, batch, torch.device("cpu"))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertAlmostEqual(float(batch["move_unit_weight"].sum()), 2.0)
            self.assertEqual(int(batch["move_boundary_mask"].sum()), 1)
            self.assertEqual(int(metrics["hint"]["targets"]), 1)
            self.assertEqual(int(metrics["eos"]["targets"]), 1)

    def test_factorized_loss_preserves_action_normalization_and_scales_rap_by_count(self):
        """q=0の指手損失を保ち，RAP NLLの和だけを同じ分子へ加える．"""
        from train_new_prompt import loss_for_batch

        class PositionModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None, recurrent_mask=None, output_hidden_states=False):
                del attention_mask, recurrent_mask, output_hidden_states
                logits = torch.zeros((*input_ids.shape, 5), dtype=torch.float32)
                logits[0, 0, 1] = 2.0
                logits[0, 1, 2] = -1.0
                logits[0, 2, 3] = 1.0
                return SimpleNamespace(logits=logits)

        labels = torch.tensor([[1, 2, 3]])
        weights = torch.tensor([[1.0, 1.0, 0.25]])
        batch = {
            "input_ids": torch.tensor([[0, 1, 2]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.bool),
            "recurrent_mask": torch.zeros((1, 3), dtype=torch.bool),
            "labels": labels,
            "loss_weights": weights,
            "move_target_mask": torch.tensor([[True, True, False]]),
            "hint_target_mask": torch.tensor([[False, False, True]]),
            "eos_target_mask": torch.zeros((1, 3), dtype=torch.bool),
            "move_unit_weight": torch.tensor([[1.0, 1.0, 0.0]]),
            "move_boundary_mask": torch.tensor([[False, True, False]]),
        }
        model = PositionModel()
        loss, metrics = loss_for_batch(model, batch, torch.device("cpu"))
        token_nll = torch.nn.functional.cross_entropy(
            model(batch["input_ids"]).logits.reshape(-1, 5), labels.reshape(-1), reduction="none"
        ).reshape_as(labels)
        # 2 subtokenで1指手，RAP重み0.25．分母は従来どおり指手数1である．
        expected = (token_nll[0, :2].sum() + 0.25 * token_nll[0, 2]) / 1.0
        self.assertTrue(torch.allclose(loss.detach(), expected))
        self.assertTrue(torch.allclose(metrics["combined_weight"], torch.tensor(1.0)))

    def test_canonical_nll_masks_rap_piece_logits_except_after_drop(self):
        from evaluate_factorized_moves import canonical_nll_for_component
        from factorized_prompt import DROP_TOKEN, PIECE_TOKENS, factorized_vocabulary_tokens

        vocabulary = {
            token: index for index, token in enumerate(factorized_vocabulary_tokens())
        }
        vector = torch.zeros(len(vocabulary))
        square_target = vocabulary["<SQ_7g>"]
        nll = canonical_nll_for_component(vector, square_target, [], vocabulary)
        self.assertAlmostEqual(nll, math.log(len(vocabulary) - len(PIECE_TOKENS)), places=6)
        drop_piece_target = vocabulary["<P>"]
        drop_nll = canonical_nll_for_component(
            vector, drop_piece_target, [vocabulary[DROP_TOKEN]], vocabulary
        )
        self.assertAlmostEqual(drop_nll, math.log(len(vocabulary)), places=6)

    def test_complete_move_definition_handles_normal_promotion_and_drop(self):
        from evaluate_factorized_moves import move_piece_and_kind, move_piece_group

        self.assertEqual(
            move_piece_and_kind("7g7f", {"eligible": True, "piece": "<P>"}),
            ("<P>", "normal"),
        )
        self.assertEqual(
            move_piece_and_kind("2b3c+", {"eligible": True, "piece": "<B>"}),
            ("<B>", "promotion"),
        )
        self.assertEqual(move_piece_and_kind("P*5e", {"eligible": False}), ("<P>", "drop"))
        self.assertEqual(move_piece_group("<R>"), "major")
        self.assertEqual(move_piece_group("<S>"), "minor")
        self.assertEqual(move_piece_group("<K>"), "king")

    def test_drop_relevance_matching_requires_same_held_count_and_legal_drop(self):
        from factorized_drop_relevance import select_anchors_and_controls

        def position(game, ply, move, count, legal, in_check=0):
            hands = [0] * 14
            hands[0] = count
            return {
                "game_id": game, "ply": ply, "move": move,
                "is_drop": "*" in move, "side": 0, "hands": hands,
                "in_check": in_check, "legal_moves": legal,
                "query_position": 2 * ply + 2,
                "event_markers": {"0:0": [2]}, "all_move_markers": list(range(2, 2 * ply + 2, 2)),
            }

        anchor = position("drop-game", 12, "P*5e", 2, ["P*5e", "P*4e"])
        wrong_count = position("wrong-count", 11, "7g7f", 1, ["P*5e"])
        no_legal_drop = position("no-legal", 11, "7g7f", 2, ["7g7f"])
        matched = position("matched", 14, "2g2f", 2, ["P*5e", "P*4e"])
        pairs, summary = select_anchors_and_controls(
            [anchor, wrong_count, no_legal_drop, matched], 10, 7
        )
        self.assertEqual(summary["matched_pairs"], 1)
        self.assertEqual(pairs[0]["control"]["game_id"], "matched")

    def test_llama_attention_observation_and_empty_ablation_preserve_logits(self):
        from evaluate_factorized_drop_attention import (
            forward_with_edge_ablation,
            selected_attention_rows,
        )
        from models import ModelConfig, build_model

        torch.manual_seed(3)
        config = ModelConfig(
            vocab_size=32, max_seq_len=16, d_model=16,
            n_layers=2, n_heads=4, d_ff=32, dropout=0.0,
        )
        model = build_model("llama", config).eval()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        observed_rows, observed_logits = selected_attention_rows(model, ids, 4)
        baseline = model(ids).logits
        empty = forward_with_edge_ablation(model, ids, 4, [], {0, 1})
        self.assertEqual(len(observed_rows), 2)
        self.assertEqual(tuple(observed_rows[0].shape), (4, 5))
        self.assertTrue(torch.allclose(observed_logits, baseline, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(empty, baseline, atol=1e-6, rtol=1e-6))

        masked = forward_with_edge_ablation(model, ids, 4, [1], {0, 1})
        self.assertFalse(torch.allclose(masked[:, -1], baseline[:, -1]))

    def test_attention_ablation_cluster_interval_resamples_games(self):
        from evaluate_factorized_drop_attention import (
            clustered_contrast_interval,
            finish_ablation_contrasts,
        )

        records = [
            {"game_id": "g1", "probability_change_difference": -0.20,
             "log_probability_change_difference": -0.40},
            {"game_id": "g1", "probability_change_difference": -0.10,
             "log_probability_change_difference": -0.20},
            {"game_id": "g2", "probability_change_difference": -0.30,
             "log_probability_change_difference": -0.60},
        ]
        interval = clustered_contrast_interval(records, "probability_change_difference", 7, repetitions=100)
        self.assertEqual(interval["clusters"], 2)
        self.assertLessEqual(interval["lower"], -0.20)
        self.assertGreaterEqual(interval["upper"], -0.20)

        summary = finish_ablation_contrasts({"drop:all:after_drop": records}, 7)
        result = summary["drop:all:after_drop"]
        self.assertAlmostEqual(result["probability_change_difference"], -0.20)
        self.assertEqual(result["probability_change_difference_clustered_95ci"]["clusters"], 2)

    def test_chess_protocol_instance_uses_true_start_and_end_prompts(self):
        from evaluate_factorized_chess_protocol import make_instance

        record = {
            "game_id": "chess-compatible",
            "move_tokens": ["8h3c"],
            "move_annotations": [{"eligible": True, "piece": "<B>", "source": "<SQ_8h>"}],
            "evaluation_steps": [{
                "ply": 0,
                "target_move": "8h3c",
                "legal_moves": ["8h3c", "8h4d", "2h2f"],
                "legal_sources_by_piece": {
                    "<B>": ["<SQ_8h>"],
                    "<R>": ["<SQ_2h>"],
                },
            }],
        }
        instance = make_instance(record, 0, ["<SQ_7g>", "<SQ_7f>"], "vanilla", 7)
        self.assertEqual(instance["tasks"]["start_actual"]["prompt"][-1], "<B>")
        self.assertEqual(instance["tasks"]["start_actual"]["exact"], "<SQ_8h>")
        self.assertEqual(instance["tasks"]["end_actual"]["prompt"][-1], "<SQ_8h>")
        self.assertEqual(instance["tasks"]["end_actual"]["exact"], "<SQ_3c>")
        self.assertEqual(instance["tasks"]["start_other"]["legal"], ["<SQ_2h>"])
        self.assertEqual(instance["tasks"]["end_other"]["legal"], ["<SQ_2f>"])
        self.assertEqual(instance["tasks"]["start_actual"]["piece"], "<B>")
        self.assertEqual(instance["tasks"]["end_other"]["piece"], "<R>")

    def test_chess_protocol_excludes_pawns_drops_and_promotion_branches(self):
        from evaluate_factorized_chess_protocol import make_instance

        base = {
            "game_id": "excluded",
            "move_tokens": ["7g7f"],
            "move_annotations": [{"eligible": True, "piece": "<P>", "source": "<SQ_7g>"}],
            "evaluation_steps": [{
                "ply": 0, "target_move": "7g7f", "legal_moves": ["7g7f"],
                "legal_sources_by_piece": {"<P>": ["<SQ_7g>"]},
            }],
        }
        self.assertIsNone(make_instance(base, 0, [], "vanilla", 7))
        promoted = json.loads(json.dumps(base))
        promoted["move_tokens"] = ["8h3c"]
        promoted["move_annotations"] = [{"eligible": True, "piece": "<B>", "source": "<SQ_8h>"}]
        promoted["evaluation_steps"] = [{
            "ply": 0, "target_move": "8h3c", "legal_moves": ["8h3c", "8h3c+"],
            "legal_sources_by_piece": {"<B>": ["<SQ_8h>"]},
        }]
        self.assertIsNone(make_instance(promoted, 0, [], "vanilla", 7))

        other_promotes = json.loads(json.dumps(promoted))
        other_promotes["evaluation_steps"] = [{
            "ply": 0,
            "target_move": "8h3c",
            "legal_moves": ["8h3c", "2h2f", "2h2f+"],
            "legal_sources_by_piece": {"<B>": ["<SQ_8h>"], "<R>": ["<SQ_2h>"]},
        }]
        instance = make_instance(other_promotes, 0, [], "vanilla", 7)
        self.assertIsNotNone(instance)
        self.assertNotIn("end_other", instance["tasks"])

    def test_chess_protocol_scores_over_full_vocabulary(self):
        from evaluate_factorized_chess_protocol import score_next_token

        # 非座標token 0が最大なら，座標だけに絞った場合と違ってExM/LgMは失敗する．
        logits = torch.tensor([9.0, 8.0, 7.0, 6.0])
        score = score_next_token(logits, exact_id=1, legal_ids=[1, 2], square_id_set={1, 2, 3})
        self.assertEqual(score["exact_move_correct"], 0)
        self.assertEqual(score["legal_move_correct"], 0)
        self.assertEqual(score["square_top1"], 0)
        self.assertEqual(score["legal_r_precision"], 0.5)

    def test_chess_protocol_cardinality_baselines_and_strata(self):
        from evaluate_factorized_chess_protocol import (
            add_scores,
            empty_task_metrics,
            legal_count_bin,
            piece_group,
            summarize_task,
        )

        totals = empty_task_metrics()
        examples = [
            (1, "<B>", 1, 1.0),
            (3, "<R>", 0, 2.0 / 3.0),
            (8, "<S>", 1, 0.5),
        ]
        for legal_count, piece, legal_correct, r_precision in examples:
            score = {
                "legal_move_correct": legal_correct,
                "square_top1": 1,
                "exact_move_correct": legal_correct,
                "legal_r_precision": r_precision,
            }
            for target in (
                totals["overall"],
                totals["by_legal_set_cardinality"][legal_count_bin(legal_count)],
                totals["by_piece_group"][piece_group(piece)],
                totals["by_piece"][piece],
            ):
                add_scores(target, score, exact_id=7, legal_count=legal_count)

        summary = summarize_task(totals, vocabulary_size=125)
        self.assertAlmostEqual(summary["legal_set_cardinality"]["mean"], 4.0)
        self.assertEqual(summary["legal_set_cardinality"]["median"], 3.0)
        self.assertAlmostEqual(
            summary["chance_baselines"]["uniform_81_squares_legal_accuracy"], 4.0 / 81.0
        )
        self.assertAlmostEqual(
            summary["chance_baselines"]["uniform_125_vocabulary_legal_accuracy"], 4.0 / 125.0
        )
        self.assertAlmostEqual(
            summary["chance_baselines"]["uniform_legal_set_exact_accuracy"],
            (1.0 + 1.0 / 3.0 + 1.0 / 8.0) / 3.0,
        )
        self.assertEqual(summary["by_legal_set_cardinality"]["1"]["queries"], 1)
        self.assertEqual(summary["by_legal_set_cardinality"]["2_3"]["queries"], 1)
        self.assertEqual(summary["by_legal_set_cardinality"]["4_7"]["queries"], 0)
        self.assertEqual(summary["by_legal_set_cardinality"]["8_plus"]["queries"], 1)
        self.assertEqual(summary["by_piece_group"]["major"]["queries"], 2)
        self.assertEqual(summary["by_piece_group"]["minor"]["queries"], 1)

    def test_chess_protocol_geometry_and_pseudo_legal_oracles(self):
        from evaluate_factorized_chess_protocol import add_scores, empty_metrics, oracle_destination_sets, summarize

        labels = ["<EMPTY>"] * 81

        def place(square, label):
            file_index = int(square[0]) - 1
            rank_index = "abcdefghi".index(square[1])
            labels[file_index * 9 + rank_index] = label

        place("5e", "<B_R>")
        place("5h", "<B_P>")
        place("5c", "<W_S>")
        place("2e", "<B_G>")
        place("7e", "<W_B>")
        geometry, pseudo = oracle_destination_sets(
            {"probe_targets": {"board_labels_cshogi_order": labels}}, "<SQ_5e>", "<R>"
        )
        self.assertEqual(len(geometry), 16)
        self.assertEqual(len(pseudo), 8)
        self.assertIn("<SQ_5c>", pseudo)
        self.assertNotIn("<SQ_5b>", pseudo)
        self.assertNotIn("<SQ_5h>", pseudo)

        metrics = empty_metrics()
        score = {
            "legal_move_correct": 1,
            "square_top1": 1,
            "exact_move_correct": 1,
            "legal_r_precision": 1.0,
        }
        add_scores(metrics, score, exact_id=7, legal_count=4, geometry_count=16, pseudo_legal_count=8)
        summary = summarize(metrics, 125)
        oracle = summary["oracle_rule_baselines"]
        self.assertEqual(oracle["coverage"], 1.0)
        self.assertEqual(oracle["uniform_geometry_legal_accuracy"], 0.25)
        self.assertEqual(oracle["uniform_pseudo_legal_legal_accuracy"], 0.5)
        self.assertEqual(oracle["uniform_geometry_exact_accuracy"], 1.0 / 16.0)
        self.assertEqual(oracle["uniform_pseudo_legal_exact_accuracy"], 1.0 / 8.0)

    def test_eos_is_supervised_only_for_complete_games(self):
        from factorized_prompt import factorized_vocabulary_tokens
        from factorized_prompt_data import FactorizedPromptSequenceDataset

        vocab = {token: index for index, token in enumerate(factorized_vocabulary_tokens())}
        record = self.record()
        record["move_tokens"] = ["7g7f", "3c3d"]
        record["move_annotations"] = [
            {"eligible": True, "piece": "<P>", "source": "<SQ_7g>"},
            {"eligible": True, "piece": "<P>", "source": "<SQ_3c>"},
        ]
        record.pop("factorized_move_ids", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            complete = FactorizedPromptSequenceDataset(
                str(path), vocab, max_moves=2, randomize_each_epoch=False,
            )[0]
            truncated = FactorizedPromptSequenceDataset(
                str(path), vocab, max_moves=1, randomize_each_epoch=False,
            )[0]
        self.assertEqual(int(complete["input_ids"][-1]), vocab["<EOS>"])
        self.assertEqual(int(complete["eos_target_mask"].sum()), 1)
        self.assertNotEqual(int(truncated["input_ids"][-1]), vocab["<EOS>"])
        self.assertEqual(int(truncated["eos_target_mask"].sum()), 0)

    def test_tiny_llama_and_vanilla_evaluation(self):
        from evaluate_factorized_moves import main
        from factorized_prompt import factorized_vocabulary_tokens, write_factorized_vocabulary
        from models import ModelConfig, build_model

        record = self.record()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab_path = root / "vocab.json"
            write_factorized_vocabulary(vocab_path)
            vocab = {token: index for index, token in enumerate(factorized_vocabulary_tokens())}
            evaluation = root / "evaluation.jsonl"
            evaluation.write_text(json.dumps(record) + "\n", encoding="utf-8")
            for model_type in ("vanilla", "llama"):
                config = ModelConfig(vocab_size=len(vocab), max_seq_len=64, d_model=16, n_layers=1, n_heads=4, d_ff=32, dropout=0.0)
                model = build_model(model_type, config)
                checkpoint = root / (model_type + ".pt")
                torch.save({
                    "model_type": model_type, "config": config.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "new_prompt": {"move_encoding": "factorized_v3_no_eom", "terminal_encoding": "eos_on_complete_decisive_game_v1", "state_prompt_mode": "implicit_initial", "start_selection": "fixed_initial"},
                }, checkpoint)
                output = root / (model_type + ".json")
                argv = [
                    "evaluate_factorized_moves.py", "--checkpoint", str(checkpoint),
                    "--evaluation-jsonl", str(evaluation), "--vocab", str(vocab_path),
                    "--output", str(output), "--history-distances", "0",
                    "--primary-history-distances", "0", "--max-queries", "1",
                    "--batch-size", "1", "--device", "cpu", "--progress-every", "0",
                ]
                with patch.object(sys, "argv", argv):
                    main()
                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(payload["model_type"], model_type)
                self.assertEqual(payload["metrics"]["primary"]["queries"], 1)
                self.assertEqual(payload["metrics"]["primary"]["greedy_syntactic_rate"], 1.0)
                self.assertIn("canonical_perplexity", payload["metrics"]["primary"])
                self.assertIn("canonical_move_perplexity", payload["metrics"]["primary"])

    def test_action_probe_queries_use_correct_postfix_positions(self):
        from evaluate_factorized_action_probes import read_queries
        from factorized_prompt import factorized_vocabulary_tokens

        vocab = {token: index for index, token in enumerate(factorized_vocabulary_tokens())}
        record = {
            "game_id": "action-probe",
            "game_result": 1,
            "terminal_token": "<EOS>",
            "start_candidates": [{"start_ply": 0, "state_prompt_tokens": []}],
            "move_tokens": ["7g7f", "2b3c+", "P*5e"],
            "legal_drop_available_by_ply": [False, True, True],
            "promotion_choice_available_by_ply": [False, True, False],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            queries = read_queries(path, vocab, "implicit_initial", (0, 1, 2), 100, 512)

        self.assertEqual(queries["actual_destination_nonpromote"][0]["tokens"], ["<BOS>", "<MOVES>", "<SQ_7g>"])
        self.assertEqual(queries["actual_destination_promote"][0]["tokens"], ["<BOS>", "<MOVES>", "<SQ_7g>", "<SQ_7f>", "<SQ_2b>", "<PROMOTE>"])
        self.assertEqual(queries["actual_drop_destination"][0]["tokens"], ["<BOS>", "<MOVES>", "<SQ_7g>", "<SQ_7f>", "<SQ_2b>", "<PROMOTE>", "<SQ_3c>", "<DROP>", "<P>"])
        self.assertEqual(queries["actual_destination_promote"][0]["target"], 20)
        self.assertEqual([item["target"] for item in queries["actual_promote_optional"]], [1])
        self.assertEqual(queries["actual_drop_destination"][0]["target"], 40)
        self.assertEqual([item["target"] for item in queries["drop_available"]], [0, 1, 1])
        self.assertEqual([item["target"] for item in queries["terminal_next"]], [0, 1])
        self.assertEqual(queries["terminal_next"][1]["tokens"][-1], "<SQ_5e>")

    def test_action_probe_length_bucketing_keeps_labels_aligned(self):
        from evaluate_factorized_action_probes import extract_features

        class FakeModel:
            config = SimpleNamespace(d_model=1)

            def __call__(self, input_ids, **kwargs):
                # 最終入力tokenを特徴量とし，queryの順序を直接観測可能にする．
                hidden = input_ids.to(torch.float32).unsqueeze(-1)
                return SimpleNamespace(hidden_states=(hidden,))

        vocabulary = {"<PAD>": 0, "<BOS>": 1, "<MOVES>": 2, "<SQ_1a>": 3}
        queries = [
            {"tokens": ["<BOS>", "<MOVES>", "<SQ_1a>"], "target": 30, "recurrent_start": 2},
            {"tokens": ["<BOS>", "<MOVES>"], "target": 20, "recurrent_start": 2},
        ]
        features, labels = extract_features(
            FakeModel(), queries, vocabulary, ["layer_0"], torch.device("cpu"),
            batch_size=2, amp_dtype=None, pool_batches=8, progress=0, label="test",
        )
        self.assertEqual(labels.tolist(), [20, 30])
        self.assertEqual(features["layer_0"].flatten().tolist(), [2.0, 3.0])

    def test_optional_promotion_fallback_is_for_the_target_move(self):
        from evaluate_factorized_action_probes import _promotion_choice_available_by_ply

        record = {
            "evaluation_steps": [{
                "target_move": "7g7f",
                # 別の指手だけが成り／不成りを選べても，targetは任意成りではない．
                "legal_moves": ["7g7f", "2b3c", "2b3c+"],
            }]
        }
        self.assertEqual(_promotion_choice_available_by_ply(record, 1), [False])

    def test_game_split_overlap_is_rejected(self):
        from evaluate_factorized_action_probes import assert_disjoint_game_ids

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for split, game_id in (("train", "same"), ("validation", "valid"), ("evaluation", "same")):
                path = root / (split + ".jsonl")
                path.write_text(json.dumps({"game_id": game_id}) + "\n", encoding="utf-8")
                paths[split] = path
            with self.assertRaisesRegex(ValueError, "game_id overlap"):
                assert_disjoint_game_ids(paths)

    def test_move_query_reader_is_bounded_by_batch_size(self):
        from evaluate_factorized_moves import iter_query_batches
        from factorized_prompt import factorized_vocabulary_tokens

        vocab = {token: index for index, token in enumerate(factorized_vocabulary_tokens())}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.jsonl"
            path.write_text(
                "".join(json.dumps(dict(self.record(), game_id="g{}".format(index))) + "\n" for index in range(7)),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                evaluation_jsonl=str(path), history_distances=(0,), max_games=7,
                max_queries=5, candidates_per_game=1, start_selection="fixed_initial",
                state_prompt_mode="explicit", batch_size=2,
            )
            statistics = {"games": 0, "queries": 0}
            batches = list(iter_query_batches(args, vocab, 64, statistics))
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual(statistics["queries"], 5)
        self.assertLessEqual(max(map(len, batches)), args.batch_size)

    def test_ap_move_evaluation_keeps_oracle_piece_tokens_in_history(self):
        from evaluate_factorized_moves import iter_query_batches
        from factorized_prompt import factorized_vocabulary_tokens

        vocab = {token: index for index, token in enumerate(factorized_vocabulary_tokens())}
        record = self.record()
        record["move_tokens"] = ["7g7f", "3c3d"]
        record["move_annotations"] = [
            {"eligible": True, "piece": "<P>", "source": "<SQ_7g>"},
            {"eligible": True, "piece": "<P>", "source": "<SQ_3c>"},
        ]
        record["evaluation_steps"] = [
            {"ply": 0, "target_move": "7g7f", "legal_moves": ["7g7f"]},
            {"ply": 1, "target_move": "3c3d", "legal_moves": ["3c3d"]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                evaluation_jsonl=str(path), history_distances=(1,), max_games=1,
                max_queries=1, candidates_per_game=1, state_prompt_mode="implicit_initial",
                evaluation_annotation_mode="ap", batch_size=1,
                length_bucket_pool_batches=1,
            )
            query = list(iter_query_batches(args, vocab, 64, {"games": 0, "queries": 0}))[0][0]
        tokens = {value: token for token, value in vocab.items()}
        prefix = [tokens[value] for value in query["prefix_ids"]]
        self.assertEqual(
            prefix,
            ["<BOS>", "<MOVES>", "<P>", "<SQ_7g>", "<SQ_7f>", "<P>"],
        )
        self.assertEqual(query["unannotated_prefix_length"], 5)
        self.assertEqual(query["ap_annotation_id"], vocab["<P>"])

    def test_ap_summary_separates_piece_conditioned_and_chess_comparable_perplexity(self):
        from evaluate_factorized_moves import empty_total, add, summarize

        total = empty_total()
        add(total, {}, {
            "move_subtokens": 2,
            "move_nll": 0.4,
            "canonical_move_nll": 0.3,
            "grammar_normalized_move_nll": 0.2,
            "ap_mode_queries": 1,
            "ap_annotation_examples": 1,
            "ap_annotation_nll": 0.7,
            "ap_annotated_move_nll": 1.1,
        })
        result = summarize(total)
        self.assertAlmostEqual(result["canonical_move_perplexity"], math.exp(0.3))
        self.assertAlmostEqual(result["ap_annotated_move_perplexity"], math.exp(1.1))
        self.assertAlmostEqual(result["ap_piece_conditioned_move_perplexity"], math.exp(0.3))
        self.assertAlmostEqual(result["ap_canonical_move_perplexity"], math.exp(1.1))
        self.assertAlmostEqual(result["ap_annotation_cross_entropy"], 0.7)
        self.assertEqual(result["ap_annotation_examples"], 1)

    def test_batched_beam_matches_single_query_beam(self):
        from evaluate_factorized_moves import beam_batch_cached, beam_single_cached
        from factorized_prompt import factorized_vocabulary_tokens
        from models import ModelConfig, build_model

        vocabulary = {
            token: index for index, token in enumerate(factorized_vocabulary_tokens())
        }
        prefixes = [
            [vocabulary["<BOS>"], vocabulary["<MOVES>"]],
            [vocabulary["<BOS>"], vocabulary["<MOVES>"]],
        ]
        for model_type in ("vanilla", "llama"):
            torch.manual_seed(7)
            config = ModelConfig(
                vocab_size=len(vocabulary), max_seq_len=16, d_model=16,
                n_layers=1, n_heads=4, d_ff=32, dropout=0.0,
            )
            model = build_model(model_type, config).eval()
            expected = [
                beam_single_cached(model, prefix, vocabulary, torch.device("cpu"), 5)
                for prefix in prefixes
            ]
            actual = beam_batch_cached(
                model, prefixes, vocabulary, torch.device("cpu"),
                beam_size=5, micro_batch_size=2,
            )
            self.assertEqual(
                [[move for move, _ in values] for values in actual],
                [[move for move, _ in values] for values in expected],
            )
            for actual_query, expected_query in zip(actual, expected):
                self.assertEqual(len(actual_query), len(expected_query))
                for (_, actual_score), (_, expected_score) in zip(actual_query, expected_query):
                    self.assertAlmostEqual(actual_score, expected_score, places=5)

    def test_hand_transition_metrics_separates_capture_and_drop(self):
        from evaluate_factorized_hand_dynamics import hand_transition_metrics

        events = [
            {
                "event_type": "capture", "changed_slot": 0,
                "before_hands": [0] * 14, "after_hands": [1] + [0] * 13,
            },
            {
                "event_type": "drop", "changed_slot": 7,
                "before_hands": [0] * 7 + [2] + [0] * 6,
                "after_hands": [0] * 7 + [1] + [0] * 6,
            },
        ]
        before = torch.tensor([events[0]["before_hands"], events[1]["before_hands"]])
        after = torch.tensor([events[0]["after_hands"], events[1]["after_hands"]])
        metrics = hand_transition_metrics(before, after, events)
        self.assertEqual(metrics["capture"]["events"], 1)
        self.assertEqual(metrics["drop"]["events"], 1)
        self.assertEqual(metrics["all"]["changed_slot_delta_accuracy"], 1.0)
        self.assertEqual(metrics["all"]["full_hand_delta_exact_match"], 1.0)

        wrong_after = after.clone()
        wrong_after[1, 7] = 2
        wrong = hand_transition_metrics(before, wrong_after, events)
        self.assertEqual(wrong["capture"]["changed_slot_delta_accuracy"], 1.0)
        self.assertEqual(wrong["drop"]["changed_slot_delta_accuracy"], 0.0)

    def test_best_checkpoint_can_omit_optimizer_state(self):
        from factorized_prompt import factorized_vocabulary_tokens
        from models import ModelConfig, build_model
        from train_new_prompt import save_checkpoint

        config = ModelConfig(
            vocab_size=len(factorized_vocabulary_tokens()), max_seq_len=16,
            d_model=8, n_layers=1, n_heads=2, d_ff=16, dropout=0.0,
        )
        model = build_model("vanilla", config)
        optimizer = torch.optim.AdamW(model.parameters())
        args = SimpleNamespace(
            model_type="vanilla", model_size="small", move_encoding="factorized_v3_no_eom",
            state_prompt_mode="explicit", start_selection="fixed_initial",
            annotation_mode="vanilla", annotation_probability=0.0,
            hint_loss_weight=1.0, eos_loss_weight=1.0, max_hints=0, max_moves=1, seed=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, model, optimizer, args, 1, 1, 1.0, include_optimizer=False)
            payload = torch.load(path, map_location="cpu")
        self.assertNotIn("optimizer_state_dict", payload)
        self.assertEqual(
            payload["new_prompt"]["training_objective"],
            "factorized_action_mle_proportional_rap_v1",
        )


if __name__ == "__main__":
    unittest.main()
