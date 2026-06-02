# -*- coding: utf-8 -*-
"""
build-h5 で生成した HDF5 ファイルを train / val にゲーム単位で分割するスクリプト。

Usage:
    python split_h5_dataset.py --input-h5 data.h5 --output-dir split --val-ratio 0.2 --seed 42
"""
import argparse
import sys
from pathlib import Path

try:
    import h5py
except ImportError:
    sys.exit("エラー: h5pyがインストールされていません。'pip install h5py' を実行してください。")


def split_h5(input_h5: str, output_dir: str, val_ratio: float, seed: int) -> None:
    input_path = Path(input_h5)
    if not input_path.exists():
        sys.exit(f"エラー: 入力ファイル '{input_path}' が見つかりません。")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ゲームグループ名を収集
    with h5py.File(input_path, "r") as f:
        game_names = sorted(
            name for name in f.keys()
            if isinstance(f[name], h5py.Group) and "positions" in f[name]
        )

    if not game_names:
        sys.exit("エラー: 入力HDF5に有効な game group が見つかりません。")

    # シード付きシャッフル → train / val 分割
    import random
    random.seed(seed)
    shuffled = list(game_names)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1.0 - val_ratio))
    # 片方が空になるのを避ける
    split_idx = max(1, min(split_idx, len(shuffled) - 1))

    train_names = shuffled[:split_idx]
    val_names = shuffled[split_idx:]

    # train.h5 出力
    train_path = out_dir / "train.h5"
    with h5py.File(input_path, "r") as f_src, h5py.File(train_path, "w") as f_dst:
        f_dst.attrs["split"] = "train"
        f_dst.attrs["num_games"] = len(train_names)
        f_dst.attrs["num_total_games"] = len(game_names)
        for name in train_names:
            f_src.copy(name, f_dst, name)
    print(f"train: {len(train_names)} games -> {train_path}")

    # val.h5 出力
    val_path = out_dir / "val.h5"
    with h5py.File(input_path, "r") as f_src, h5py.File(val_path, "w") as f_dst:
        f_dst.attrs["split"] = "val"
        f_dst.attrs["num_games"] = len(val_names)
        f_dst.attrs["num_total_games"] = len(game_names)
        for name in val_names:
            f_src.copy(name, f_dst, name)
    print(f"val:   {len(val_names)} games -> {val_path}")

    print(f"\n分割完了 ({len(game_names)} games total, seed={seed})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="build-h5 で生成した HDF5 ファイルを train / val にゲーム単位で分割する。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-h5", required=True, help="入力となる HDF5 ファイルのパス。")
    parser.add_argument("--output-dir", required=True, help="分割結果を出力するディレクトリ。train.h5, val.h5 が作成されます。")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="検証セットの比率 (0.0-1.0)。")
    parser.add_argument("--seed", type=int, default=42, help="シャッフルの乱数シード。同じ値で再現性あり。")
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        sys.exit("エラー: --val-ratio は 0.0 以上 1.0 未満の値を指定してください。")

    split_h5(args.input_h5, args.output_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
