# create_dataset.py

## 概要

`create_dataset.py` は、CSA形式の棋譜ファイルから、AIの学習データセットを生成するための多機能スクリプトです。
スクリプトは複数のサブコマンドで構成され、これらをパイプラインとして組み合わせることで、要件に応じた多様なデータセットを生成できます。

## ワークフロー

本スクリプトは、生成したいデータセットの種類に応じて、主に3つのワークフローをサポートします。

### A) `.bin`形式 (エンジン評価あり)
`extract` → `filter` → `count-sfen` (任意) → `evaluate` → `generate`

### B) `.bin`形式 (エンジン評価なし)
`extract` → `filter` → `count-sfen` (任意) → `label` → `generate`

### C) `.h5`形式 (高機能版)
`extract` → `filter` → `build-h5`

※ `count-sfen` を実行し、その後のコマンドで `--db-path` と `--max-sfen-count` を指定することで、局面の重複を抑えた高品質なデータセットを高速に作成できます。

---

## コマンド詳細

### `extract`
(中略)

### `filter`
(中略)

### `count-sfen`
全棋譜をスキャンしてSFENの出現頻度をカウントし、SQLite DBに保存します。重複局面の制限を行う場合に事前に実行します。
```bash
python src/create_dataset.py count-sfen --input-csv <フィルタ済みCSV> --db-path <DBパス>
```

### `label`
エンジンを使わず、対局結果のみから評価値を付与（ラベリング）します。
```bash
python src/create_dataset.py label --input-csv <フィルタ済みCSV> --output-csv <ラベル付きCSV> [DBオプション]
```
**DBオプション:**
*   `--db-path`: `count-sfen`で作成したDBを指定します。
*   `--max-sfen-count`: 同一局面の最大出力回数（デフォルト: 0=無制限）。指定すると特定の定跡への偏りを防げます。

### `evaluate`
フィルタリング済みCSVを元に、USIエンジンで各局面を評価し、評価値とSFENを含むCSVを生成します。
```bash
python src/create_dataset.py evaluate --input-csv <フィルタ済みCSV> --output-csv <評価値付きCSV> --engine-path <エンジンパス> [探索オプション] [DBオプション]
```
**DBオプション:**
*   `--db-path`: SFEN頻度と評価値キャッシュを管理するDBパス。
*   `--max-sfen-count`: 同一局面の最大出力回数。
*   **キャッシュ機能**: DBに同一の探索条件（SFEN, depth等）の評価値がある場合、エンジンの再探索をスキップして高速化します。

### `generate`
(中略)

### `build-h5`
フィルタリング済みCSVを元に、USIエンジンで詳細な評価を行い、階層的なHDF5データセット (`.h5`) を直接生成します。
局面ごとの統計的な特徴量（王手、駒取り等）も自動的に付与されます。
```bash
python src/create_dataset.py build-h5 --input-csv <フィルタ済みCSV> --output-h5 <出力H5ファイル> --engine-path <エンジンパス> [探索オプション]
```
**オプション:**
*   `--db-path`: 評価値のキャッシュ（再利用）にDBを使用します。

### `analyze_sfen_dist`
`count-sfen` で作成したDBを分析し、局面出現頻度の分布をヒストグラムとして出力します。
```bash
python src/analyze_sfen_dist.py --db-path <DBパス> --output-img <画像出力パス>
```
**出力内容:**
*   総ユニーク局面数、最大出現頻度などの統計情報。
*   出現頻度上位の局面（SFEN）リスト。
*   出現頻度および出力頻度のヒストグラム（PNG画像）。

### `analyze_app` (Gradio版 解析ツール)
Gradioを使用したWeb UI形式の棋譜分析ツールです。評価値グラフと盤面図（SVG）を連動させて、インタラクティブに悪手を分析できます。
```bash
# 起動方法
python src/analyze_app.py
```
**主な機能:**
*   **インタラクティブグラフ**: Plotlyを使用し、グラフ上の点を選択するとその局面の盤面が表示されます。
*   **盤面SVG表示**: `cshogi` を使用して、駒の動きがわかりやすい局面図を表示します。
*   **外部アクセス**: 起動時に表示される `Public URL` (https://xxx.gradio.app) を使うことで、外部のブラウザからもアクセス可能です。

**探索オプション:**
*   `evaluate`コマンドと同様のオプション（`--depth`, `--nodes`, `--movetime`, `--early-xxx`等）が使用可能です。
*   `--num-pv`: MultiPVで取得する候補手の数（デフォルト: 5）

---
## 設定ファイル (`wsl2/config.yaml`)
各コマンドのオプションは`wsl2/config.yaml`にまとめて記述することで、コマンドライン入力を簡略化できます。
```bash
python src/create_dataset.py -c wsl2/config.yaml <command>
```
