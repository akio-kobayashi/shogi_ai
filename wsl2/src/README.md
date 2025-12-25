# create_dataset.py

## 概要

`create_dataset.py` は、CSA (Computer Shogi Association) 形式の棋譜ファイルから、AIの学習データセットを生成するための多機能スクリプトです。
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
CSAファイル群から全棋譜のメタデータを抽出し、`metadata.csv`を生成します。
```bash
python src/create_dataset.py extract --csa-dir <棋譜ディレクトリ> --output-csv <出力CSVパス>
```

### `filter`
`metadata.csv`をレーティングや手数などの条件でフィルタリングし、`filtered.csv`を生成します。
```bash
python src/create_dataset.py filter --input-csv <入力CSV> --output-csv <出力CSV> [フィルタオプション]
```

### `label`
エンジンを使わず、対局結果のみから評価値を付与（ラベリング）します。出力は`evaluate`と同じ形式のCSVです。
```bash
python src/create_dataset.py label --input-csv <フィルタ済みCSV> --output-csv <ラベル付きCSV>
```

### `evaluate`
`filter`または`label`で生成されたCSVを元に、USIエンジンで各局面を評価し、評価値とSFENを含む`evaluated.csv`を生成します。
```bash
python src/create_dataset.py evaluate --input-csv <フィルタ済みCSV> --output-csv <評価値付きCSV> --engine-path <エンジンパス>
```

### `generate`
`evaluate`または`label`で生成された評価値付きCSVを元に、最終的な`.bin`形式の学習データセットを生成します。
```bash
python src/create_dataset.py generate --input-csv <評価値付きCSV> --output-dir <出力ディレクトリ>
```

### `build-h5`
`filter`で生成されたCSVを元に、USIエンジンでMultiPVを含む詳細な評価を行い、階層的なHDF5データセット (`.h5`) を直接生成します。
```bash
python src/create_dataset.py build-h5 --input-csv <フィルタ済みCSV> --output-h5 <出力H5ファイル> --engine-path <エンジンパス>
```

---
## 設定ファイル (`wsl2/config.yaml`)
各コマンドのオプションは`wsl2/config.yaml`にまとめて記述することで、コマンドライン入力を簡略化できます。
```bash
python src/create_dataset.py -c wsl2/config.yaml <command>
```