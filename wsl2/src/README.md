# create_dataset.py

## 概要

`create_dataset.py` は、CSA (Computer Shogi Association) 形式の棋譜ファイルから、[nodchip/nnue-pytorch](https://github.com/nodchip/nnue-pytorch) などで利用可能な学習データセットを生成するためのスクリプトです。

生成するデータセットの形式に応じて、2つのワークフローが用意されています。

*   **`.bin`形式 (従来形式):** `extract` → `filter` → `evaluate` → `generate`
*   **`.h5`形式 (拡張形式):** `extract` → `filter` → `build-h5`

拡張形式であるHDF5 (`.h5`) ファイルには、MultiPVによる複数の指し手候補や王手情報など、よりリッチなメタデータが階層的に格納されます。

## コマンドとワークフロー

### 設定ファイル (`wsl2/config.yaml`)

各コマンドの多くのオプションは、`wsl2/config.yaml`にまとめて記述することができます。
`-c` (または `--config`) オプションでこのファイルを指定することで、コマンド実行が容易になります。

```bash
python src/create_dataset.py -c wsl2/config.yaml <command>
```

---

### ワークフローA: `.bin`形式データセットの生成

#### ステップ1: `extract`
CSAファイル群から全棋譜のメタデータを抽出し、`metadata.csv`を生成します。
```bash
python src/create_dataset.py -c wsl2/config.yaml extract
```

#### ステップ2: `filter`
`metadata.csv`をレーティングや手数などの条件でフィルタリングし、`filtered.csv`を生成します。
```bash
python src/create_dataset.py -c wsl2/config.yaml filter
```

#### ステップ3: `evaluate`
`filtered.csv`に含まれる棋譜の各局面をUSIエンジンで評価し、評価値とSFEN文字列を含む`evaluated.csv`を生成します。
```bash
python src/create_dataset.py -c wsl2/config.yaml evaluate
```

#### ステップ4: `generate`
`evaluated.csv`を元に、最終的な学習データセット `train.bin` と `val.bin` を生成します。
```bash
python src/create_dataset.py -c wsl2/config.yaml generate
```

---

### ワークフローB: `.h5`形式データセットの生成

#### ステップ1: `extract`
(ワークフローAと共通)
```bash
python src/create_dataset.py -c wsl2/config.yaml extract
```

#### ステップ2: `filter`
(ワークフローAと共通)
```bash
python src/create_dataset.py -c wsl2/config.yaml filter
```

#### ステップ3: `build-h5`
`filtered.csv`を元に、USIエンジンでMultiPVを含む詳細な評価を行い、階層的なHDF5データセット (`.h5`) を直接生成します。
```bash
python src/create_dataset.py -c wsl2/config.yaml build-h5
```
このコマンドは、内部でエンジン評価とファイル生成を同時に行います。

### HDF5ファイルの構造

`build-h5`コマンドによって生成されるHDF5ファイルは、以下のような階層構造を持ちます。

*   各「対局」が1つの**グループ** (`/game_0`, `/game_1`, ...) として格納されます。
*   各対局グループの**属性 (Attributes)**に、プレイヤー名やレーティングなどの対局メタデータが保存されます。
*   各対局グループ内に、その対局の全局面の情報を格納した`positions`**データセット**が作成されます。
*   `positions`データセットの各行が1局面に対応し、その中には`PackedSfenValue`、王手フラグ、そして**可変長の指し手候補リスト**などが構造体として格納されます。