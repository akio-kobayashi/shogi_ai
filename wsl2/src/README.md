# create_dataset.py

## 概要

`create_dataset.py` は、CSA形式の棋譜ファイルから、AIの学習データセットを生成するための多機能スクリプトです。
スクリプトは複数のサブコマンドで構成され、これらをパイプラインとして組み合わせることで、要件に応じた多様なデータセットを生成できます。

## ワークフロー

本スクリプトは、生成したいデータセットの種類に応じて、主に3つのワークフローをサポートします。

### A) `.bin`形式 (エンジン評価あり)
`extract` → `filter` → `evaluate` → `generate`

### B) `.bin`形式 (エンジン評価なし)
`extract` → `filter` → `label` → `generate`

### C) `.h5`形式 (高機能版)
`extract` → `filter` → `build-h5`

---

## コマンド詳細

### `extract`
CSAファイル群から全棋譜のメタデータを抽出し、CSVファイルを生成します。
```bash
python src/create_dataset.py extract --csa-dir <棋譜ディレクトリ> --output-csv <出力CSVパス>
```

### `filter`
メタデータCSVをレーティングや手数などの条件でフィルタリングします。
```bash
python src/create_dataset.py filter --input-csv <入力CSV> --output-csv <出力CSV> [フィルタオプション]
```
**主なフィルタオプション:**
*   `--min-rating`: 最低レーティング
*   `--max-moves`: 最大手数
*   `--no-draws`: 引き分けの対局を除外します。
*   `--filter-by-rating-outcome`: レーティングが高い方が勝利した対局のみに絞り込みます（番狂わせを除外）。

### `label`
エンジンを使わず、対局結果のみから評価値を付与（ラベリング）します。
```bash
python src/create_dataset.py label --input-csv <フィルタ済みCSV> --output-csv <ラベル付きCSV>
```

### `evaluate`
フィルタリング済みCSVを元に、USIエンジンで各局面を評価し、評価値とSFENを含むCSVを生成します。
```bash
python src/create_dataset.py evaluate --input-csv <フィルタ済みCSV> --output-csv <評価値付きCSV> --engine-path <エンジンパス>
```

### `generate`
評価値付きCSVを元に、最終的な`.bin`形式の学習データセットを生成します。
```bash
python src/create_dataset.py generate --input-csv <評価値付きCSV> --output-dir <出力ディレクトリ>
```

### `build-h5`
フィルタリング済みCSVを元に、USIエンジンで詳細な評価を行い、階層的なHDF5データセット (`.h5`) を直接生成します。
```bash
python src/create_dataset.py build-h5 --input-csv <フィルタ済みCSV> --output-h5 <出力H5ファイル> --engine-path <エンジンパス>
```

---
## 設定ファイル (`wsl2/config.yaml`)
各コマンドのオプションは`wsl2/config.yaml`にまとめて記述することで、コマンドライン入力を簡略化できます。
```bash
python src/create_dataset.py -c wsl2/config.yaml <command>
```
