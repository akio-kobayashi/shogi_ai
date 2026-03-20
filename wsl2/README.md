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

### E) ユニーク局面評価（静止/非静止を分離）
`extract` → `filter` → `count-sfen` → `classify-sfen` → `evaluate-sfen` → `merge-eval-sfen` → `generate`

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
*   `classify-sfen` の `output-quiet-csv` / `output-tactical-csv` もそのまま入力可能

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

### `merge-extract`
複数の `extract` 出力CSVを1つにまとめます。日付ごと・ディレクトリごとに分割抽出したメタデータを後段の `filter` に渡す前段として使います。
```bash
python src/create_dataset.py merge-extract \
  --input-csvs extract_a.csv,extract_b.csv,extract_c.csv \
  --output-csv merged_extract.csv
```
**主なオプション:**
*   `--input-csvs`: マージ対象CSVのカンマ区切りリスト
*   `--output-csv`: マージ後CSVの出力先

### `plot-sfen-histogram`
`count-sfen` の出力CSVから `total_count` 分布のヒストグラム画像を生成します。
```bash
python src/create_dataset.py plot-sfen-histogram \
  --input-csv sfen_counts.csv \
  --output-png sfen_histogram.png \
  --log-x --log-y
```
**主なオプション:**
*   `--input-csv`: `count-sfen` の出力CSV
*   `--output-png`: 出力画像パス
*   `--count-column`: 可視化する列名。通常は `total_count`
*   `--bins`: ビン数
*   `--max-count`: この値を超える頻度を描画から除外
*   `--log-x`, `--log-y`: 各軸を対数表示

### `classify-sfen`
`count-sfen` の出力などを元に、SFEN を静止局面と非静止局面へ分類します。高コストな評価の前にデータを分けるための前処理です。
```bash
python src/create_dataset.py classify-sfen \
  --input-csv <頻度CSV> \
  --output-quiet-csv <静止局面CSV> \
  --output-tactical-csv <非静止局面CSV>
```
**主なオプション:**
*   `--quiet-level`: 静止局面判定の強さ (`1` / `2` / `3`)
*   `--output-quiet-csv`: 静止局面側の出力先
*   `--output-tactical-csv`: 非静止局面側の出力先

**使い方の意図:**
*   静止局面には shallow / 0手読み評価を付ける
*   非静止局面には静止探索付き、またはより重い評価を付ける

### `merge-eval-sfen`
複数の評価済み SFEN CSV を1つにまとめます。`classify-sfen` 後に quiet / tactical を別条件で評価した結果を、`generate` に渡す前段として使います。
```bash
python src/create_dataset.py merge-eval-sfen \
  --input-csvs <静止局面CSV>,<非静止局面CSV> \
  --output-csv <マージ後CSV>
```
**主なオプション:**
*   `--input-csvs`: マージ対象CSVのカンマ区切りリスト
*   `--output-csv`: マージ後CSVの出力先

### `corn-thresholds`
`generate` と同じ SFEN 頻度補正を前提に、`nnue-pytorch` の `CORN` 用閾値を構築します。
学習前の分割は、`.bin` 生成後に `nnue-pytorch/corn_thresholds.py` でやるより、このコマンドを source of truth にする方が自然です。
```bash
python src/create_dataset.py corn-thresholds \
  --input-csv <評価済みSFEN CSV> \
  --sfen-count-csv <頻度CSV> \
  --sfen-sampling-mode sqrt \
  --num-thresholds 7 \
  --score-scaling 361 \
  --teacher-temperature 1.0 \
  --corn-aux-weight 0.1
```
**主なオプション:**
*   `--input-csv`: 単一の評価済みCSVを入力する場合
*   `--input-csvs`: 分割評価した複数CSVをカンマ区切りでまとめて入力する場合
*   `--min-ply`, `--max-ply`: 閾値計算対象とする手数の範囲。入力CSVに `ply` 列がある場合のみ適用
*   `--sfen-count-csv`: `count-sfen` の出力CSV
*   `--sfen-sampling-mode`: `generate` と同じ頻度補正方式 (`none` / `fixed` / `sqrt` / `log10`)
*   `--sfen-cutoff-value`: `fixed` 方式の上限値
*   `--sfen-sampling-min-freq`: この頻度未満のSFENには補正を適用しない
*   `--num-thresholds`: 生成する閾値数。K個なら K+1 クラス
*   `--score-scaling`, `--teacher-temperature`: cp から teacher-logit へ変換する係数
*   `--corn-aux-weight`: 出力例に含める `nnue-pytorch` 側の補助損失重み

**出力内容:**
*   cp 空間の閾値
*   teacher-logit 空間へ変換した `corn_aux_thresholds`
*   `nnue-pytorch` の CLI / config に貼れる例

**注意:**
*   `evaluate-sfen` / `merge-eval-sfen` の出力には通常 `ply` 列がありません。その場合、`corn-thresholds` は ply フィルタを適用せず、入力された全局面の評価値分布を使います。

### `diff-sfen`
candidate 側の SFEN 一覧から、base 側に存在する SFEN を除外します。高レート側に出現しない low-only SFEN を作る用途を想定しています。
```bash
python src/create_dataset.py diff-sfen \
  --base-csv <高レート側SFEN CSV> \
  --candidate-csv <低レート側SFEN CSV> \
  --output-csv <low-only SFEN CSV>
```
**主なオプション:**
*   `--base-csv`: 差分の基準となるCSV
*   `--candidate-csv`: 差分抽出対象のCSV
*   `--output-csv`: 差分抽出後CSVの出力先

### `adjust-eval`
評価済み CSV の `eval_score_cp` を縮小・ゼロ寄せ・クリップします。low-only SFEN を学習データへ弱く加える用途を想定しています。
```bash
python src/create_dataset.py adjust-eval \
  --input-csv <評価済みCSV> \
  --output-csv <調整後CSV> \
  --mode scale --scale 0.5
```
**主なオプション:**
*   `--mode`: `scale` / `zero` / `clip`
*   `--scale`: `scale` / `clip` 時の係数
*   `--max-abs-cp`: `clip` 時の絶対値上限
*   出力には `source_eval_score_cp`, `eval_adjust_mode`, `eval_adjust_param` 列が追加されます

### `generate`
評価値付きCSVを元に、最終的な`.bin`形式の学習データセットを生成します。
```bash
python src/create_dataset.py generate --input-csv <評価値付きCSV> --output-dir <出力ディレクトリ>
```
**主なオプション:**
*   `--input-csv`: 単一の評価値付きCSVを直接入力する場合
*   `--input-csvs`: 分割評価した複数の評価値付きCSVをカンマ区切りでまとめて直接入力する場合
*   `--positions-csv`: `label` 出力などの局面展開済みCSVを入力し、内部で `eval-sfen` 結果を join する場合
*   `--positions-csvs`: 分割された局面展開済みCSVをカンマ区切りでまとめて join する場合
*   `--eval-sfen-csv`: `evaluate-sfen` で生成した評価済みSFEN CSV
*   `--eval-sfen-csvs`: 分割評価した複数の評価済みSFEN CSVをカンマ区切りで指定する場合
*   `--val-split`: 検証データ比率
*   `--min-ply`, `--max-ply`: 生成対象とする手数の範囲
*   `--quiet-level`: 静止局面フィルタの強さ (`none` / `1` / `2` / `3`)。後方互換用で、新フローでは `classify-sfen` を推奨
*   `--sfen-count-csv`: `count-sfen` が出力した頻度CSV
*   `--sfen-sampling-mode`: `none` / `fixed` / `sqrt` / `log10`
*   `--sfen-cutoff-value`: `fixed`方式の上限値
*   `--sfen-sampling-min-freq`: この頻度未満のSFENにはサンプリング上限を適用しない

**補足:**
*   `--input-csv` と `--input-csvs` は同時指定できません。
*   `--positions-csv` と `--positions-csvs` は同時指定できません。
*   `--eval-sfen-csv` と `--eval-sfen-csvs` は同時指定できません。
*   direct入力 (`--input-csv` / `--input-csvs`) と join入力 (`--positions-csv` / `--eval-sfen-csv` 系) は同時指定できません。
*   `--input-csvs` を使うと、複数の評価済みCSVを `generate` 内でヘッダ一致確認のうえ連結してから `.bin` を生成できます。
*   join入力では、局面CSV側の `game_result` と `ply` を保持したまま、`sfen` 一致で `eval_score_cp` を付与します。評価値が無い局面は読み飛ばします。

**join入力の例:**
```bash
python src/create_dataset.py generate \
  --positions-csv labeled_positions.csv \
  --eval-sfen-csv evaluated_sfen.csv \
  --output-dir out
```

**静止局面フィルタ (`--quiet-level`)**
*   `none`: フィルタなし（デフォルト）
*   `1`: 終局、入玉、反復、王手中の局面を除外
*   `2`: `1` に加えて、1手詰め筋がある局面を除外
*   `3`: `2` に加えて、SEE風に得な取り手、王手候補、成り筋、玉周辺の危険がある局面を除外

**`quiet-level=3` の SEE 実装範囲**
*   本実装はフル SEE ではなく、`_has_see_like_capture()` による軽量な SEE 風近似です。
*   各合法 capture について、取る駒の価値、着手後にそのマスへいる自駒の価値、相手の最小価値の取り返し駒、自分の最小価値の取り返し駒を比較します。
*   相手に取り返しがない capture は「有利な取り」として非静止扱いにします。
*   1回の recapture / re-recapture を仮定したときに材得が残る capture も「有利な取り」として非静止扱いにします。
*   盤上の利きは `cshogi.Board.attackers_to()` を使って評価しています。

**一般的な SEE のうち未実装のもの**
*   交換列を最後まで展開するフルの swap list / gain 配列計算はしていません。
*   2回目以降の再取り、X線利きの連鎖、長い交換列の収束判定はしていません。
*   王手回避や詰み、王手の連続、成りによる価値変化を SEE 本体には統合していません。
*   持ち駒打ちを含む将棋特有の複雑な交換列は SEE としては扱っていません。
*   そのため `quiet-level=3` は「YaneuraOu の qsearch / フル SEE 相当」ではなく、「qsearch が必要になりやすい局面を多めに落とす高コスト判定」です。

**頻度サンプリングの仕様:**
*   しきい値未満を除外するフィルタではなく、同一SFENごとに「目標出力回数」を決めます。
*   実際の採用は `目標出力回数 / 真の頻度(total_count)` を keep probability とする確率サンプリングです。
*   同じ評価済み行を複製して増やすことはしません。
*   `fixed`: 目標出力回数 = `sfen-cutoff-value`
*   `sqrt`: 目標出力回数 = `sqrt(total_count)`
*   `log10`: 目標出力回数 = `log10(total_count)`（最小1）
*   `total_count < sfen-sampling-min-freq` のSFENは、サンプリングせず全件残します。
*   `total_count >= sfen-sampling-min-freq` のSFENでも、目標出力回数は `sfen-sampling-min-freq` 未満には下げません。これにより、しきい値直上のSFENが直下のSFENより少なくなる逆転を防ぎます。

**頻度サンプリングの例:**
*   頻度調整なし
```bash
python src/create_dataset.py generate \
  --input-csv evaluated.csv \
  --output-dir out \
  --sfen-sampling-mode none
```
*   単純カット: 出現頻度100以上のSFENは最大100回に圧縮
```bash
python src/create_dataset.py generate \
  --input-csv evaluated.csv \
  --output-dir out \
  --sfen-count-csv sfen_counts.csv \
  --sfen-sampling-mode fixed \
  --sfen-cutoff-value 100 \
  --sfen-sampling-min-freq 100
```
*   `sqrt`: 出現頻度100以上のSFENを `sqrt(total_count)` 回まで圧縮
```bash
python src/create_dataset.py generate \
  --input-csv evaluated.csv \
  --output-dir out \
  --sfen-count-csv sfen_counts.csv \
  --sfen-sampling-mode sqrt \
  --sfen-sampling-min-freq 100
```
*   `log10`: 出現頻度100以上のSFENを `log10(total_count)` 回まで圧縮
```bash
python src/create_dataset.py generate \
  --input-csv evaluated.csv \
  --output-dir out \
  --sfen-count-csv sfen_counts.csv \
  --sfen-sampling-mode log10 \
  --sfen-sampling-min-freq 100
```

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
