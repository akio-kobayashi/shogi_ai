# create_dataset.py

## 概要

`create_dataset.py` は、CSA (Computer Shogi Association) 形式の棋譜ファイルから、[nodchip/nnue-pytorch](https://github.com/nodchip/nnue-pytorch) などで利用可能な学習データセットを生成するためのスクリプトです。

このスクリプトは、以下の4つの独立したフェーズ（サブコマンド）で構成されており、それぞれを個別に実行することで、柔軟かつ堅牢なデータ生成パイプラインを構築できます。

1.  **`extract`**: CSAファイルから棋譜のメタデータ（ファイルパス、レーティング、勝敗など）を抽出し、`metadata.csv`を生成します。
2.  **`filter`**: `metadata.csv`をレーティングや手数などの条件でフィルタリングし、対象を絞った`filtered.csv`を生成します。
3.  **`evaluate`**: `filtered.csv`に含まれる棋譜の各局面をUSIエンジンで評価し、評価値とSFEN文字列を含む`evaluated.csv`を生成します。
4.  **`generate`**: `evaluated.csv`を元に、最終的な学習データセット（`.bin`または`.h5`形式）を生成します。

## 使用方法

### 前提条件

*   Python 3.8+
*   必要なライブラリ: `cshogi`, `numpy`, `tqdm`, `h5py` など (`pip install -r requirements.txt` でインストールしてください)
*   USIエンジン (オプション: `evaluate` コマンドを使用する場合)

### コマンド構造

```bash
python src/create_dataset.py <command> [options]
```

利用可能なコマンドは `extract`, `filter`, `evaluate`, `generate` です。

### 設定ファイル (`wsl2/config.yaml`) による一括設定

多くのオプションを毎回コマンドラインで指定する代わりに、`wsl2/config.yaml` という設定ファイルにまとめて記述することができます。

```bash
python src/create_dataset.py -c wsl2/config.yaml <command> [options]
```

`--config` (または `-c`) オプションでYAMLファイルを指定すると、その内容がデフォルト設定として読み込まれます。
**コマンドラインで個別に指定したオプションは、YAMLファイルの設定よりも優先されます。**

#### `config.yaml` の例

```yaml
# create_dataset.py の設定ファイル
#
# 推奨ワークフロー:
# 1. extract: CSAファイルからメタデータを抽出 -> metadata.csv
# 2. filter: metadata.csv をフィルタリング -> filtered.csv
# 3. evaluate: filtered.csv の局面を評価 -> evaluated.csv (評価値・SFEN付き)
# 4. generate: evaluated.csv から学習データを生成 -> train.bin/h5, val.bin/h5

# 'extract' コマンドの設定
extract:
  csa_dir: "/path/to/your/csa_files"
  output_dir: "output_data"

# 'filter' コマンドの設定
filter:
  metadata_csv: "output_data/metadata.csv"
  output_csv: "output_data/filtered.csv"
  min_rating: 3000
  max_rating: 9999
  max_rating_diff: 1000
  min_moves: 40
  max_moves: 512
  allowed_results: "win,lose,draw"
  filter_by_rating_outcome: false

# 'evaluate' コマンドの設定
evaluate:
  metadata_csv: "output_data/filtered.csv"
  engine_path: "/path/to/your/usi_engine"
  output_csv: "output_data/evaluated.csv"
  depth: 12
  min_ply: 20
  max_ply: 512

# 'generate' コマンドの設定
generate:
  input_csv: "output_data/evaluated.csv"
  output_dir: "output_data"
  format: "bin"  # 出力形式: 'bin' または 'hdf5'
  val_split: 0.1
```

---

### 1. `extract` コマンド: メタデータの抽出

CSAファイル群をスキャンし、各棋譜の基本情報（レーティング、勝敗など）をCSVファイルに抽出します。

```bash
python src/create_dataset.py extract --csa-dir <CSAファイルルートディレクトリのパス>
```
*   **入力:** CSAファイル群
*   **出力:** `metadata.csv`

### 2. `filter` コマンド: 棋譜のフィルタリング

`metadata.csv`を読み込み、レーティングや手数などの条件に基づいて学習対象とする棋譜を絞り込みます。

```bash
python src/create_dataset.py filter --metadata-csv <入力CSV> --output-csv <出力CSV> [フィルタオプション]
```
*   **入力:** `metadata.csv`
*   **出力:** `filtered.csv`

### 3. `evaluate` コマンド: USIエンジンによる局面評価

`filtered.csv`で指定された棋譜の各局面をUSIエンジンで評価し、評価値と局面のSFEN文字列を含む新しいCSVファイルを生成します。

```bash
python src/create_dataset.py evaluate --metadata-csv <フィルタ済みCSV> --output-csv <出力CSV> --engine-path <エンジンパス>
```
*   **入力:** `filtered.csv`
*   **出力:** `evaluated.csv` (評価値とSFEN付き)

### 4. `generate` コマンド: 学習データセットの生成

`evaluated.csv`を元に、最終的なバイナリ形式のデータセットを生成します。出力形式として、従来の`.bin`形式と、メタデータを含むHDF5形式 (`.h5`) を選択できます。

```bash
python src/create_dataset.py generate --input-csv <評価値付きCSV> --output-dir <出力ディレクトリ> [--format <bin|hdf5>]
```
*   **入力:** `evaluated.csv`
*   **出力:** `train.bin`/`val.bin` または `train.h5`/`val.h5`

**オプション:**
*   `--format` (デフォルト: `bin`): 出力形式を`bin`または`hdf5`から選択します。

---

## ワークフローの例

1.  **メタデータ抽出**:
    ```bash
    python src/create_dataset.py -c wsl2/config.yaml extract
    ```

2.  **フィルタリング**:
    ```bash
    python src/create_dataset.py -c wsl2/config.yaml filter
    ```

3.  **局面評価**:
    ```bash
    python src/create_dataset.py -c wsl2/config.yaml evaluate
    ```

4.  **データセット生成**:
    *   (`.bin`形式で生成)
        ```bash
        python src/create_dataset.py -c wsl2/config.yaml generate --format bin
        ```
    *   (`.h5`形式で生成)
        ```bash
        python src/create_dataset.py -c wsl2/config.yaml generate --format hdf5
        ```