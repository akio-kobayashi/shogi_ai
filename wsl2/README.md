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

### D) 局面頻度集計（外部メモリ方式）
`extract` → `filter` → `count-sfen`

### E) ユニーク局面評価
`extract` → `filter` → `count-sfen` → `evaluate-sfen`

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
python src/create_dataset.py evaluate --input-csv <フィルタ済みCSV> --output-csv <評価値付きCSV> --engine-path <エンジンパス> [探索オプション]
```
`evaluate` と `evaluate-sfen` は共通して次の評価オプションを持ちます。
*   `--input-csv`: 入力CSVのパス
*   `--engine-path`: USIエンジン実行ファイルのパス
*   `--output-csv`: 評価結果CSVの出力先
*   `--db-path`: 評価値キャッシュ用SQLite DB
*   `--depth`: 探索深さ（デフォルト: 10）
*   `--nodes`: 探索ノード数
*   `--movetime`: 1局面あたりの思考時間（ミリ秒）
*   `--eval-workers`: 並列ワーカー数（2以上でプロセス並列）

`evaluate` 固有のオプション:
*   `--max-sfen-count`: 同一SFENの最大出力回数
*   `--early-depth`, `--early-nodes`, `--early-movetime`: 序盤用の探索パラメータ
*   `--early-ply-threshold`: 序盤用のパラメータを適用する最大手数（デフォルト: 0）
*   `--min-ply`, `--max-ply`: 評価対象とする手数の範囲
*   `--eval-mode`: `stream`（逐次評価）/`unique`（ユニーク局面評価後に展開）

**補足（制約）:**
*   `--eval-workers > 1` の場合は、`--db-path` と `--max-sfen-count` は使用できません。
*   `--eval-mode unique` は現在 `--eval-workers=1` のみ対応です。
*   `--eval-mode unique` では `--db-path` は使用できません。

### `evaluate-sfen`
`count-sfen` などで作成した SFEN 一覧 CSV を元に、各ユニーク局面へ評価値を付与します。
```bash
python src/create_dataset.py evaluate-sfen --input-csv <頻度CSV> --output-csv <評価値付きSFEN CSV> --engine-path <エンジンパス> [探索オプション]
```
**主なオプション:**
*   `evaluate` と共通の評価オプションを使用: 入力CSV、エンジン、出力先、探索条件、並列ワーカー数を指定可能

**注意:**
*   `--eval-workers > 1` の場合は `--db-path` は使用できません。

**入力要件:**
*   少なくとも `sfen` 列を含むCSVであること
*   `count-sfen` の出力をそのまま入力可能

### `count-sfen`
フィルタリング済みCSVを元に、SFENの局面頻度を外部メモリ方式で集計し、CSVを生成します（SQLite不要）。
```bash
python src/create_dataset.py count-sfen --input-csv <フィルタ済みCSV> --output-csv <頻度CSV>
```
**主なオプション:**
*   `--min-ply`, `--max-ply`: 集計対象とする手数の範囲
*   `--min-count`: 出力する最小出現回数（デフォルト: 1）
*   `--num-buckets`: 外部メモリ集計で使用するバケット数（デフォルト: 1024）
*   `--temp-dir`: 一時バケットファイルの出力先
*   `--keep-temp`: 集計後も一時ファイルを保持

**出力CSV列:**
*   `sfen`
*   `total_count`
*   `black_win_count`

### `generate`
評価値付きCSVを元に、最終的な`.bin`形式の学習データセットを生成します。
```bash
python src/create_dataset.py generate --input-csv <評価値付きCSV> --output-dir <出力ディレクトリ>
```
**主なオプション:**
*   `--val-split`: 検証データ比率
*   `--min-ply`, `--max-ply`: 生成対象とする手数の範囲
*   `--quiet-level`: 静止局面フィルタの強さ (`none` / `1` / `2` / `3`)
*   `--sfen-count-csv`: `count-sfen` が出力した頻度CSV
*   `--sfen-sampling-mode`: `none` / `fixed` / `sqrt` / `log10`
*   `--sfen-cutoff-value`: `fixed`方式の上限値
*   `--sfen-sampling-min-freq`: この頻度未満のSFENにはサンプリング上限を適用しない

**静止局面フィルタ (`--quiet-level`)**
*   `none`: フィルタなし（デフォルト）
*   `1`: 終局、入玉、反復、王手中の局面を除外
*   `2`: `1` に加えて、取り手・王手手・成り手・1手詰め筋がある局面を除外
*   `3`: `2` に加えて、自玉への直接利きがなく、玉の逃げ道が乏しすぎない局面に限定

**頻度サンプリングの仕様:**
*   しきい値未満を除外するフィルタではなく、同一SFENごとに確率サンプリングを行います。
*   サンプル後の期待頻度が以下に収束するように受理確率を決めます。
*   `fixed`: 期待頻度 = `sfen-cutoff-value`
*   `sqrt`: 期待頻度 = `sqrt(total_count)`
*   `log10`: 期待頻度 = `log10(total_count)`（最小1）
*   `total_count < sfen-sampling-min-freq` のSFENは、サンプリングせず全件残します。

### `build-h5`
フィルタリング済みCSVを元に、USIエンジンで詳細な評価を行い、階層的なHDF5データセット (`.h5`) を直接生成します。
```bash
python src/create_dataset.py build-h5 --input-csv <フィルタ済みCSV> --output-h5 <出力H5ファイル> --engine-path <エンジンパス> [探索オプション]
```
**探索オプション:**
*   `evaluate`コマンドと同様のオプション（`--depth`, `--nodes`, `--movetime`, `--early-xxx`等）が使用可能です。
*   `--num-pv`: MultiPVで取得する候補手の数（デフォルト: 5）

---
## 設定ファイル (`wsl2/config.yaml`)
各コマンドのオプションは`wsl2/config.yaml`にまとめて記述することで、コマンドライン入力を簡略化できます。
```bash
python src/create_dataset.py <command> --config wsl2/config.yaml
```

必要に応じて、`--config` の内容はその後ろのコマンドラインオプションで上書きできます。

```bash
python src/create_dataset.py evaluate --config wsl2/config.yaml --depth 12
```
