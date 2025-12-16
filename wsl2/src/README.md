# create_dataset.py

## 概要

`create_dataset.py` は、CSA (Computer Shogi Association) 形式の棋譜ファイルから、[nodchip/nnue-pytorch](https://github.com/nodchip/nnue-pytorch) の学習データおよび検証データ (`.bin` 形式) を生成するためのスクリプトです。

このスクリプトは、以下の3つの独立したフェーズ（サブコマンド）で構成されており、それぞれを個別に実行することで、柔軟かつ堅牢なデータ生成パイプラインを構築できます。

1.  **`extract`**: CSAファイルから棋譜のメタデータ（ファイルパス、レーティング、勝敗など）を抽出し、CSVファイルに保存します。
2.  **`evaluate`**: 抽出されたメタデータCSVの各局面をUSI (Universal Shogi Interface) エンジンで評価し、評価値付きのCSVファイルを生成します。
3.  **`generate`**: メタデータCSV（または評価値付きメタデータCSV）を読み込み、フィルタリング、訓練/検証データへの分割を行い、最終的なバイナリ形式のデータセットを生成します。

## 使用方法

### 前提条件

*   Python 3.8+
*   必要なライブラリ: `cshogi`, `numpy`, `tqdm` (インストールされていない場合は `pip install -r requirements.txt` でインストールしてください)
*   USIエンジン (オプション: `evaluate` コマンドを使用する場合)

### コマンド構造

```bash
python src/create_dataset.py <command> [options]
```

利用可能なコマンドは `extract`, `evaluate`, `generate` です。

### 設定ファイル (`config.yaml`) による一括設定

多くのオプションを毎回コマンドラインで指定する代わりに、`config.yaml` という設定ファイルにまとめて記述することができます。

```bash
python src/create_dataset.py -c config.yaml <command> [options]
```

`--config` (または `-c`) オプションでYAMLファイルを指定すると、その内容がデフォルト設定として読み込まれます。
**コマンドラインで個別に指定したオプションは、YAMLファイルの設定よりも優先されます。**

#### `config.yaml` の例

以下は `generate` コマンドの全オプションを設定する例です。ファイルのキーはサブコマンド名に対応しています。

```yaml
generate:
  output_dir: "filtered_data"
  metadata_csv: "output_data/metadata.csv"
  evaluated_metadata_csv: "output_data/evaluated_metadata.csv"
  
  # フィルタリング設定
  min_rating: 2800
  max_rating: 4000
  max_rating_diff: 800
  min_moves: 50
  max_moves: 256
  allowed_results: "win,lose" # 勝ち負けが決した棋譜のみ
  filter_by_rating_outcome: true # レーティングが高い方が勝った棋譜のみ

  # データ生成設定
  win_value: 600
  min_ply: 40
  max_ply: 256
  
  # 実行制御
  val_split: 0.05
```


### 1. `extract` コマンド: メタデータの抽出

CSAファイルが格納されているディレクトリをスキャンし、各棋譜の基本情報をCSVファイルに抽出します。この処理は時間がかかる場合があるため、一度実行すれば、`metadata.csv` が存在する場合はスキップされます。

```bash
python src/create_dataset.py extract --csa-dir <CSAファイルルートディレクトリのパス> [オプション]
```

**オプション:**

*   `--csa-dir` (必須): CSAファイルが格納されているルートディレクトリのパス。
*   `--output-dir` (デフォルト: `output_data`): 生成されたメタデータCSVを保存するディレクトリ。
*   `--metadata-csv` (デフォルト: `<output-dir>/metadata.csv`): メタデータCSVの出力パス。

**例:**

```bash
# /path/to/your/csa_files からメタデータを抽出し、output_data/metadata.csv に保存
python src/create_dataset.py extract --csa-dir /path/to/your/csa_files --output-dir output_data
```

### 2. `evaluate` コマンド: USIエンジンによる局面評価

`extract` コマンドで生成されたメタデータCSVを読み込み、指定されたUSIエンジンで各局面の評価値を計算し、評価値付きの新しいCSVファイルとして出力します。

```bash
python src/create_dataset.py evaluate --metadata-csv <入力メタデータCSVのパス> --engine-path <USIエンジンの実行ファイルのパス> [オプション]
```

**オプション:**

*   `--metadata-csv` (必須): `extract` コマンドで生成された入力メタデータCSVのパス。
*   `--engine-path` (必須): 使用するUSIエンジンの実行ファイルのパス。
*   `--output-csv` (デフォルト: `<output-dir>/evaluated_metadata.csv`): 評価値付きメタデータCSVの出力パス。
*   `--depth` (デフォルト: `10`): USIエンジンの探索の深さ。
*   `--min-ply` (デフォルト: `20`): 評価を開始する最小手数。この手数未満の局面は評価されません。
*   `--max-ply` (デフォルト: `512`): 評価する最大手数。この手数を超える局面は評価されません。

**注意点 (Windows .exe エンジンと Linux 環境):**

*   **WSL2環境の場合**: WSL2上では、Linux環境からWindowsの `.exe` 形式の実行ファイルを直接呼び出すことが可能です。したがって、USIエンジンがWindowsの `.exe` であっても、WSL2のLinux環境からそのまま実行できます。Wineは不要です。
*   **純粋なLinux環境の場合**: 純粋なLinux環境でWindowsの `.exe` 形式の USI エンジンを使用する場合、[Wine](https://www.winehq.org/) がインストールされ、正しく設定されている必要があります。スクリプトは警告を表示しますが、Wine の自動起動は行いません。必要に応じて `sudo apt install wine` などで Wine をインストールしてください。

**例:**

```bash
# output_data/metadata.csv を元に、指定エンジンで局面を評価し、output_data/evaluated_metadata.csv に保存
python src/create_dataset.py evaluate --metadata-csv output_data/metadata.csv --engine-path /path/to/your/engine.exe --output-csv output_data/evaluated_metadata.csv --depth 12
```

### 3. `generate` コマンド: 学習データセットの生成

フィルタリング、訓練/検証データへの分割を行い、最終的なバイナリ形式のデータセット (`train.bin`, `val.bin`) を生成します。評価値は、`--evaluated-metadata-csv` が指定されていればそれを使用し、なければ棋譜の勝敗結果から生成されます。

```bash
python src/create_dataset.py generate [オプション]
```

**オプション:**

*   `--output-dir` (デフォルト: `output_data`): 生成されたデータセット (`.bin`) を保存するディレクトリ。
*   `--metadata-csv` (デフォルト: `<output-dir>/metadata.csv`): 入力となるメタデータCSVのパス。
*   `--evaluated-metadata-csv` (デフォルト: `<output-dir>/evaluated_metadata.csv`): 評価値付きメタデータCSVのパス。これを指定すると、棋譜の勝敗ではなく、このCSVに含まれる評価値が使用されます。
*   `--min-rating` (デフォルト: `3000`): 学習対象とする対局者の最低レーティング。
*   `--max-rating` (デフォルト: `9999`): 学習対象とする対局者の最大レーティング。
*   `--max-rating-diff` (デフォルト: `1000`): 学習対象とする対局者間のレーティング差の上限。
*   `--min-moves` (デフォルト: `0`): 学習対象とする棋譜の最小手数。
*   `--max-moves` (デフォルト: `999`): 学習対象とする棋譜の最大手数。
*   `--allowed-results` (デフォルト: `"win,lose,draw"`): 含める勝敗結果をカンマ区切りで指定 (`win`, `lose`, `draw`, `interrupt`)。
*   `--filter-by-rating-outcome` (フラグ): これを指定すると、レーティングが高い方が勝った棋譜のみを対象とします。
*   `--win-value` (デフォルト: `600`): 棋譜の勝敗から評価値を生成する場合の絶対値 (例: 先手勝ちなら `600`、後手勝ちなら `-600`)。
*   `--min-ply` (デフォルト: `20`): この手数未満の局面は学習データに含めません。
*   `--max-ply` (デフォルト: `512`): 安全のための手数の上限。この手数を超える局面は学習データに含めません。
*   `--val-split` (デフォルト: `0.1`): 検証データとして分割する割合 (0.0-1.0)。

**例:**

```bash
# 棋譜の勝敗結果を評価値として使用してデータセットを生成
python src/create_dataset.py generate --output-dir output_data --metadata-csv output_data/metadata.csv --min-rating 2500 --max-rating-diff 500 --val-split 0.05

# USIエンジンで評価した値を使用してデータセットを生成
python src/create_dataset.py generate --output-dir output_data --metadata-csv output_data/metadata.csv --evaluated-metadata-csv output_data/evaluated_metadata.csv --min-rating 2500 --max-rating-diff 500 --val-split 0.05
```
