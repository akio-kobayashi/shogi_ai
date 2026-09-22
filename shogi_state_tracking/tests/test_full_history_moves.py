"""全指手を対象とする教師強制評価の検証。

重点は2つである。既存の``evaluate_factorized_moves.py``と指標の定義が
一致していること（canonicalマスクと文法制約の集合）、および系列上の採点位置が
「その位置のlogitsが次のtokenを予測する」という対応になっていることである。
位置がずれても値は出てしまうため、テストで固定する。
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

try:
    import torch
except ImportError:
    torch = None


VOCAB_CANDIDATES = (
    MODULE_DIR / "factorized_v3_eos_data" / "vocab.json",
    MODULE_DIR / "student_data" / ".extracted" / "analysis_bundle" / "dataset" / "vocab.json",
)


def find_vocabulary():
    """語彙はデータディレクトリか，収集済みbundleのどちらかにある。"""
    for candidate in VOCAB_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None



@unittest.skipIf(torch is None, "PyTorch is not installed")
class SlotDefinitionTest(unittest.TestCase):
    """種別ごとの許容集合とマスクが既存実装と一致すること。"""

    @classmethod
    def setUpClass(cls):
        import evaluate_factorized_full_history_moves as full
        import evaluate_factorized_moves as sampled
        from data import load_vocabulary
        cls.full = full
        cls.sampled = sampled
        path = find_vocabulary()
        if path is None:
            raise unittest.SkipTest("vocab.json is not available")
        cls.vocabulary = load_vocabulary(path)

    def setUp(self):
        self.allowed, self.masked = self.full.slot_tables(self.vocabulary)
        self.drop = self.vocabulary[self.full.DROP_TOKEN]
        self.promote = self.vocabulary[self.full.PROMOTE_TOKEN]
        self.square = self.vocabulary["<SQ_7g>"]

    def current_for(self, slot):
        """既存実装が同じ判断をするときの``current``を返す。"""
        return {
            self.full.SLOT_START: [],
            self.full.SLOT_DROP_PIECE: [self.drop],
            self.full.SLOT_DROP_DEST: [self.drop, self.vocabulary["<P>"]],
            self.full.SLOT_DEST_OR_PROMOTE: [self.square],
            self.full.SLOT_PROMOTE_DEST: [self.square, self.promote],
        }[slot]

    def test_allowed_sets_match_the_sampled_evaluator(self):
        for slot in self.allowed:
            with self.subTest(slot=slot):
                expected = self.sampled.grammar_allowed(self.current_for(slot), self.vocabulary)
                self.assertEqual(sorted(self.allowed[slot]), sorted(expected))

    def test_canonical_mask_matches_the_sampled_evaluator(self):
        """マスク後のNLLが既存関数と一致すること。<DROP>直後だけ除外しない。"""
        torch.manual_seed(0)
        vector = torch.randn(len(self.vocabulary))
        for slot in self.allowed:
            with self.subTest(slot=slot):
                target = self.allowed[slot][0]
                expected = self.sampled.canonical_nll_for_component(
                    vector, target, self.current_for(slot), self.vocabulary)
                excluded = self.masked[slot]
                if excluded:
                    candidate = vector.clone()
                    candidate[torch.tensor(excluded, dtype=torch.long)] = -float("inf")
                else:
                    candidate = vector
                actual = -float(torch.log_softmax(candidate, dim=-1)[target])
                self.assertAlmostEqual(actual, expected, places=5)

    def test_drop_piece_slot_is_the_only_unmasked_one(self):
        unmasked = [slot for slot, value in self.masked.items() if not value]
        self.assertEqual(unmasked, [self.full.SLOT_DROP_PIECE])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SlotClassificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import evaluate_factorized_full_history_moves as full
        from data import load_vocabulary
        cls.full = full
        path = find_vocabulary()
        if path is None:
            raise unittest.SkipTest("vocab.json is not available")
        cls.vocabulary = load_vocabulary(path)

    def classify(self, tokens):
        ids = [self.vocabulary[token] for token in tokens]
        return self.full.classify_slots(ids, self.vocabulary)

    def test_normal_move(self):
        self.assertEqual(self.classify(["<SQ_7g>", "<SQ_7f>"]),
                         [self.full.SLOT_START, self.full.SLOT_DEST_OR_PROMOTE])

    def test_promotion(self):
        self.assertEqual(self.classify(["<SQ_2b>", "<PROMOTE>", "<SQ_3c>"]),
                         [self.full.SLOT_START, self.full.SLOT_DEST_OR_PROMOTE,
                          self.full.SLOT_PROMOTE_DEST])

    def test_drop(self):
        self.assertEqual(self.classify(["<DROP>", "<P>", "<SQ_5e>"]),
                         [self.full.SLOT_START, self.full.SLOT_DROP_PIECE,
                          self.full.SLOT_DROP_DEST])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ScoringPositionTest(unittest.TestCase):
    """採点位置が次token予測の対応になっていること。"""

    @classmethod
    def setUpClass(cls):
        import evaluate_factorized_full_history_moves as full
        from data import load_vocabulary
        cls.full = full
        path = find_vocabulary()
        if path is None:
            raise unittest.SkipTest("vocab.json is not available")
        cls.vocabulary = load_vocabulary(path)

    def build(self, moves, annotation_mode="vanilla", max_seq_len=2560):
        record = {
            "start_candidates": [{"start_ply": 0}],
            "move_tokens": [move for move, _ in moves],
            "move_annotations": [annotation for _, annotation in moves],
        }
        args = SimpleNamespace(state_prompt_mode="implicit_initial",
                               evaluation_annotation_mode=annotation_mode)
        return self.full.build_game(record, args, self.vocabulary, max_seq_len)

    def test_positions_predict_the_targets(self):
        moves = [("7g7f", {"piece": "PAWN", "eligible": False}),
                 ("3c3d", {"piece": "PAWN", "eligible": False}),
                 ("2b3c+", {"piece": "BISHOP", "eligible": False})]
        game = self.build(moves)
        ids = game["token_ids"]
        for move in game["moves"]:
            for position, target in zip(move["positions"], move["target_ids"]):
                # position のlogitsが position+1 のtokenを予測する。
                self.assertEqual(ids[position + 1], target,
                                 f"ply {move['ply']} position {position}")

    def test_first_scored_position_is_the_moves_token(self):
        game = self.build([("7g7f", {"piece": "PAWN", "eligible": False})])
        self.assertEqual(game["token_ids"][game["moves"][0]["positions"][0]],
                         self.vocabulary["<MOVES>"])

    def test_annotation_position_predicts_the_annotation(self):
        moves = [("7g7f", {"piece": "PAWN", "eligible": True})]
        game = self.build(moves, annotation_mode="ap")
        move = game["moves"][0]
        self.assertIsNotNone(move["annotation_position"])
        self.assertEqual(game["token_ids"][move["annotation_position"] + 1], move["annotation_id"])
        for position, target in zip(move["positions"], move["target_ids"]):
            self.assertEqual(game["token_ids"][position + 1], target)

    def test_every_move_of_the_game_is_scored(self):
        moves = [(usi, {"piece": "PAWN", "eligible": False})
                 for usi in ("7g7f", "3c3d", "2g2f", "8c8d")]
        game = self.build(moves)
        self.assertEqual([move["ply"] for move in game["moves"]], [0, 1, 2, 3])
        self.assertEqual(game["truncated_moves"], 0)

    def test_moves_beyond_the_sequence_limit_are_counted_not_dropped_silently(self):
        moves = [(usi, {"piece": "PAWN", "eligible": False})
                 for usi in ("7g7f", "3c3d", "2g2f", "8c8d")]
        game = self.build(moves, max_seq_len=6)
        self.assertLess(len(game["moves"]), 4)
        self.assertEqual(game["truncated_moves"] + len(game["moves"]), 4)
        self.assertEqual(game["total_plies"], 4)


class PlyBucketTest(unittest.TestCase):
    def test_buckets_are_contiguous_and_cover_every_ply(self):
        import importlib.util
        spec = importlib.util.find_spec("evaluate_factorized_full_history_moves")
        self.assertIsNotNone(spec)
        source = (MODULE_DIR / "evaluate_factorized_full_history_moves.py").read_text(encoding="utf-8")
        self.assertIn("PLY_BUCKETS", source)
        # torchなしでも境界の整合だけは確かめる。
        namespace: dict = {}
        for line in source.splitlines():
            if line.startswith("PLY_BUCKETS"):
                exec(line, namespace)
                break
        buckets = namespace["PLY_BUCKETS"]
        self.assertEqual(buckets[0][0], 1, "ply 0 is reported separately")
        for (_, high), (low, _) in zip(buckets, buckets[1:]):
            self.assertEqual(low, high + 1, "buckets must not overlap or leave gaps")


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MoveKindTest(unittest.TestCase):
    """指手の種類の判定。終盤で駒打ち比率が上がるため，手数と交絡する。"""

    @classmethod
    def setUpClass(cls):
        import evaluate_factorized_full_history_moves as full
        from data import load_vocabulary
        path = find_vocabulary()
        if path is None:
            raise unittest.SkipTest("vocab.json is not available")
        cls.full = full
        cls.vocabulary = load_vocabulary(path)

    def kind(self, tokens):
        return self.full.move_kind([self.vocabulary[token] for token in tokens], self.vocabulary)

    def test_normal(self):
        self.assertEqual(self.kind(["<SQ_7g>", "<SQ_7f>"]), "normal")

    def test_promotion(self):
        self.assertEqual(self.kind(["<SQ_2b>", "<PROMOTE>", "<SQ_3c>"]), "promotion")

    def test_drop(self):
        self.assertEqual(self.kind(["<DROP>", "<P>", "<SQ_5e>"]), "drop")

    def test_build_game_records_the_kind(self):
        record = {
            "start_candidates": [{"start_ply": 0}],
            "move_tokens": ["7g7f", "P*5e", "2b3c+"],
            "move_annotations": [{"piece": "PAWN", "eligible": False}] * 3,
        }
        args = SimpleNamespace(state_prompt_mode="implicit_initial",
                               evaluation_annotation_mode="vanilla")
        game = self.full.build_game(record, args, self.vocabulary, 2560)
        self.assertEqual([move["kind"] for move in game["moves"]],
                         ["normal", "drop", "promotion"])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DenominatorTest(unittest.TestCase):
    """分母の違う指標が，全指手で割られていないこと。"""

    @classmethod
    def setUpClass(cls):
        import evaluate_factorized_full_history_moves as full
        cls.full = full

    def base_total(self, **extra):
        total = {"queries": 10.0, "move_subtokens": 23.0, "move_nll": 10.0,
                 "canonical_move_nll": 10.0, "grammar_normalized_move_nll": 10.0,
                 "drop_moves": 2.0}
        total.update(extra)
        return total

    def test_drop_piece_uses_drop_moves_as_the_denominator(self):
        result = self.full.summarize(self.base_total(
            drop_piece_correct=1.0, drop_piece_correct_top5=2.0, drop_piece_applicable=2.0))
        self.assertAlmostEqual(result["drop_piece_top1"], 0.5)
        self.assertAlmostEqual(result["drop_piece_top5"], 1.0)
        self.assertEqual(result["drop_piece_examples"], 2)

    def test_raw_drop_piece_counters_do_not_appear_as_averages(self):
        result = self.full.summarize(self.base_total(
            drop_piece_correct=1.0, drop_piece_correct_top5=2.0, drop_piece_applicable=2.0))
        for key in ("drop_piece_correct", "drop_piece_correct_top5", "drop_piece_applicable"):
            self.assertNotIn(key, result)

    def test_promotion_uses_its_own_denominator(self):
        result = self.full.summarize(self.base_total(
            promotion_decision_correct=3.0, promotion_decision_applicable=4.0))
        self.assertAlmostEqual(result["promotion_decision_top1"], 0.75)
        self.assertEqual(result["promotion_decision_examples"], 4)

    def test_groups_without_drops_omit_the_drop_piece_metrics(self):
        result = self.full.summarize(self.base_total())
        self.assertNotIn("drop_piece_top1", result)
