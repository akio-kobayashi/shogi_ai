import csv
import datetime as dt
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import create_dataset

try:
    create_dataset.import_cshogi()
    HAS_CSHOGI = True
except RuntimeError:
    HAS_CSHOGI = False


def metadata_row(
    path,
    black,
    white,
    rating_b="3100",
    rating_w="3200",
    result="1",
    moves="100",
):
    return {
        "file_path": path,
        "kif_index": "0",
        "black_player": black,
        "white_player": white,
        "rating_b": rating_b,
        "rating_w": rating_w,
        "game_result": result,
        "total_moves": moves,
    }


class DatasetSplitTest(unittest.TestCase):
    def test_position_hash_ignores_sfen_move_number(self):
        first = "9/9/9/9/9/9/9/9/9 b - 1"
        second = "9/9/9/9/9/9/9/9/9 b - 99"
        self.assertEqual(
            create_dataset.normalize_position_sfen(first),
            create_dataset.normalize_position_sfen(second),
        )
        self.assertEqual(
            create_dataset.make_position_hash(first),
            create_dataset.make_position_hash(second),
        )

    def test_position_scope_annotation_is_per_ply(self):
        record = {"position_hashes": ["a", "b", "c"]}
        create_dataset.annotate_position_scopes(record, {"a", "c"})
        self.assertEqual(
            record["position_scope_by_ply"],
            ["seen_position", "unseen_position", "seen_position"],
        )
        self.assertEqual(record["trajectory_scope"], "mixed_position")

        strict_record = {"position_hashes": ["x", "y"]}
        create_dataset.annotate_position_scopes(strict_record, {"a"})
        self.assertEqual(strict_record["trajectory_scope"], "strict_unseen_position")

    def test_date_is_extracted_from_directory(self):
        row = metadata_row("/csa/2025/03/04/game.csa", "a", "b")
        self.assertEqual(create_dataset.extract_game_date(row), dt.date(2025, 3, 4))

    def test_filter_uses_both_ratings_and_excludes_draws(self):
        rows = [
            metadata_row("/csa/2022/01/01/a.csa", "a", "b"),
            metadata_row("/csa/2022/01/01/b.csa", "a", "b", rating_w="2999"),
            metadata_row("/csa/2022/01/01/c.csa", "a", "b", result="0"),
            metadata_row("/csa/2022/01/01/d.csa", "a", "b", moves="79"),
        ]
        selected, rejected = create_dataset.filter_metadata(
            rows,
            min_date=dt.date(2022, 1, 1),
            max_date=None,
            min_rating=3000,
            min_moves=80,
            include_draws=False,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(rejected["rating"], 1)
        self.assertEqual(rejected["draw"], 1)
        self.assertEqual(rejected["moves"], 1)

    def test_engine_open_mixed_closed_are_based_on_train_names(self):
        rows = [
            metadata_row("/csa/2023/01/01/a.csa", "seen_a", "seen_b"),
            metadata_row("/csa/2024/11/01/b.csa", "seen_a", "new_a"),
            metadata_row("/csa/2025/01/01/c.csa", "seen_a", "seen_b"),
            metadata_row("/csa/2025/01/02/d.csa", "seen_a", "new_a"),
            metadata_row("/csa/2025/01/03/e.csa", "new_a", "new_b"),
        ]
        selected, _ = create_dataset.filter_metadata(
            rows,
            min_date=dt.date(2022, 1, 1),
            max_date=None,
            min_rating=3000,
            min_moves=80,
            include_draws=False,
        )
        splits = create_dataset.assign_splits(
            selected,
            validation_from=dt.date(2024, 10, 1),
            evaluation_from=dt.date(2025, 1, 1),
        )
        scopes = [row["engine_scope"] for row in splits["evaluation"]]
        self.assertEqual(scopes, ["open", "mixed", "closed"])

    def test_manifest_round_trip(self):
        row = metadata_row("/csa/2023/01/01/a.csa", "a", "b")
        selected, _ = create_dataset.filter_metadata(
            [row],
            min_date=dt.date(2022, 1, 1),
            max_date=None,
            min_rating=3000,
            min_moves=80,
            include_draws=False,
        )
        splits = create_dataset.assign_splits(
            selected,
            validation_from=dt.date(2024, 10, 1),
            evaluation_from=dt.date(2025, 1, 1),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.csv"
            create_dataset.write_manifest(path, splits["train"])
            loaded = create_dataset.load_manifest(path)
        self.assertEqual(loaded[0]["game_id"], splits["train"][0]["game_id"])

    def test_evaluation_sampling_is_balanced_and_deterministic(self):
        rows = []
        for scope in ("open", "mixed", "closed"):
            for index in range(7):
                rows.append(
                    {
                        "game_id": "{}-{}".format(scope, index),
                        "game_date": "2025-01-{:02d}".format(index + 1),
                        "engine_scope": scope,
                    }
                )
        first, summary = create_dataset.deterministic_scope_sample(rows, 5, seed=42)
        second, _ = create_dataset.deterministic_scope_sample(
            list(reversed(rows)), 5, seed=42
        )
        self.assertEqual(
            [row["game_id"] for row in first],
            [row["game_id"] for row in second],
        )
        self.assertEqual(
            {scope: values["selected"] for scope, values in summary.items()},
            {"open": 2, "mixed": 2, "closed": 1},
        )


class StateEncodingTest(unittest.TestCase):
    @unittest.skipUnless(HAS_CSHOGI, "cshogi is not installed")
    def test_standard_position_is_exactly_96_tokens(self):
        cshogi = create_dataset.import_cshogi()
        tokens = create_dataset.encode_initial_state(cshogi.Board(), cshogi)
        self.assertEqual(len(tokens), 96)
        self.assertEqual(tokens[-1], "TURN_BLACK")
        self.assertEqual(tokens[0], "SQ_W_L")
        self.assertEqual(tokens[80], "SQ_B_L")
        self.assertEqual(tokens[81:95], ["HAND_0"] * 14)

    def test_vocabulary_has_unique_tokens(self):
        tokens = create_dataset.base_vocabulary() + ["7g7f", "3c3d"]
        self.assertEqual(len(tokens), len(set(tokens)))

    def test_fixed_move_vocabulary_covers_move_promotion_and_drop(self):
        moves = create_dataset.all_usi_move_tokens()
        self.assertEqual(len(moves), len(set(moves)))
        self.assertIn("7g7f", moves)
        self.assertIn("8h2b+", moves)
        self.assertIn("P*5e", moves)

    @unittest.skipUnless(HAS_CSHOGI, "cshogi is not installed")
    def test_csa_manifest_is_exported_without_intermediate_positions(self):
        csa = (
            "V2.2\n"
            "N+black\n"
            "N-white\n"
            "PI\n"
            "+\n"
            "+7776FU\n"
            "-3334FU\n"
            "%TORYO\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csa_path = root / "game.csa"
            csa_path.write_text(csa, encoding="utf-8")
            row = metadata_row(
                str(csa_path),
                "black",
                "white",
                result="2",
                moves="2",
            )
            selected, _ = create_dataset.filter_metadata(
                [row],
                min_date=dt.date(2022, 1, 1),
                max_date=None,
                min_rating=3000,
                min_moves=0,
                include_draws=False,
            )
            # 一時CSAパスに日付がないため、テスト用に日付を明示する。
            if not selected:
                row["game_date"] = "2023-01-01"
                selected, _ = create_dataset.filter_metadata(
                    [row],
                    min_date=dt.date(2022, 1, 1),
                    max_date=None,
                    min_rating=3000,
                    min_moves=0,
                    include_draws=False,
                )
            splits = create_dataset.assign_splits(
                selected,
                validation_from=dt.date(2024, 10, 1),
                evaluation_from=dt.date(2025, 1, 1),
            )
            manifest = root / "train.csv"
            output = root / "train.jsonl"
            errors = root / "errors.csv"
            create_dataset.write_manifest(manifest, splits["train"])
            summary = create_dataset.export_manifest(
                manifest,
                output,
                errors,
                prefix_from=None,
                prefix_to=None,
                strict=True,
                limit=None,
            )
            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary["written_games"], 1)
        self.assertEqual(record["move_tokens"], ["7g7f", "3c3d"])
        self.assertEqual(len(record["initial_state_tokens"]), 96)
        self.assertNotIn("positions", record)
        self.assertNotIn("state_tokens_by_ply", record)


if __name__ == "__main__":
    unittest.main()
