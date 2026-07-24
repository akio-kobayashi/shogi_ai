# 将棋状態追跡Transformer用データセット

CSA棋譜から、Transformerデコーダの潜在状態追跡を調べるためのデータセットを作成する。
モデルへ途中局面は与えず、各対局を次の系列として扱う。

```text
<BOS> [開始局面96トークン] <MOVES> [USI指し手1] ... [USI指し手N] <EOS>
```

開始局面は次の固定長表現である。

- 盤面81トークン：`1a, 1b, ..., 1i, 2a, ..., 9i` の順
- 持ち駒14トークン：先手・後手それぞれ `P, L, N, S, G, B, R`
- 手番1トークン

指し手はBPEで分割せず、USI形式の1手を1トークンとする。途中局面はJSONLへ
保存しない。線形プローブ用の正解局面は、実験時に開始SFENから指し手を再生して
生成する。

指手語彙はデータ中に観測された手だけから作らず、異なる2マス間の通常移動と成り、
7種類の駒打ちを列挙した固定USI語彙とする。この語彙には局面によって不可能な手も
含まれるが、任意局面の合法手を語彙外にしない。既存の観測指手だけの`vocab.json`や
CoT用特殊tokenを含まないschema version 2を使っている場合は、
`create_dataset.py export`を再実行してschema version 3の語彙を生成する。

## 既定の抽出条件

- 2022年1月1日以降
- 先手・後手ともレーティング3000以上
- 80手以上
- `game_result == 0` を除外
- 2022年1月～2024年9月：学習
- 2024年10月～12月：検証
- 2025年1月以降：評価
- 評価は合計5,000局（`open` 1,667、`mixed` 1,667、`closed` 1,666）

評価・検証データには、学習期間中の対局者名義との関係から次のラベルを付ける。

- `open`：両対局者名義が学習集合に存在
- `mixed`：一方だけが学習集合に存在
- `closed`：両対局者名義が学習集合に存在しない

これはエンジンファミリーではなく、metadataの対局者名義による分類である。
評価候補全体を使う場合は `--evaluation-games 0` を指定する。
抽出は `sampling_seed + game_id` のSHA-1値に基づくため、入力CSVの行順に依存せず
再現可能である。

## 実行方法

### uvによる環境構築

実行対象はLinux／WSL2である。Apple Siliconを含むmacOSは実験実行環境に含めない。
Python 3.10以上3.14未満を対象とし、cshogiなどの共通依存は`uv.lock`から、
PyTorchは選択したaccelerator用の公式indexからプロジェクト専用の`.venv`へ導入する。
ここで使用するcshogiはPyPI公開版ではなく、データ作成・評価に必要な変更を加えた
forkのcommit `c447085`へ固定している。

CPU、CUDA 13.0、ROCm 7.2のいずれかを明示して構築する。引数を省略した場合は、
大容量のNVIDIA/AMD packageを取得しないCPU版になる。

```bash
cd /path/to/shogi_ai/shogi_state_tracking
./setup_env.sh cpu
./setup_env.sh cuda
./setup_env.sh rocm
```

同じ`.venv`のままbackendを変更することはできない。CPU、CUDA、ROCmを切り替える
場合は`.venv`を削除してから再構築する。異なるCUDA/ROCm版が必要な場合は、
例えば次のように明示する。

```bash
CUDA_BACKEND=cu128 ./setup_env.sh cuda
ROCM_BACKEND=rocm6.4 ./setup_env.sh rocm
```

既定のPyTorchは2.13.0であり、`TORCH_VERSION`で上書きできる。GPU版は展開時に
大きな空き容量を必要とするため、setupはcache側とproject側の双方に15 GiB以上の
空きを要求する。cacheを別の大容量filesystemへ置く場合は次のように指定する。

```bash
UV_CACHE_DIR=/large-volume/uv-cache ./setup_env.sh cuda
```

旧設定でproject直下の`.uv-cache`が容量を消費している場合は、他のuv処理が動いて
いないことを確認してから`UV_CACHE_DIR=.uv-cache uv cache clean`で削除できる。

`setup_env.sh`は共通依存について`uv sync --frozen --inexact`を実行するため、
ロックファイルと`pyproject.toml`が一致しない場合は停止する。PyTorchについては
uv公式の`--torch-backend`を用い、選択したbackendだけを取得する。依存関係を
変更した場合に限り、開発者が`uv lock`で`uv.lock`を更新する。

環境構築後、各シェルスクリプトは既定で`.venv/bin/python`を使用する。別のPythonを
使う場合は`PYTHON_BIN=/path/to/python`で上書きできる。`requirements.txt`には
backend非依存のcshogiだけを記載しており、再現実験では`setup_env.sh`を使用する。

### データセット作成

まず、CSA本体を読まずにmetadataを抽出・分割できる。

```bash
python create_dataset.py split \
  --metadata-csv ../wsl2/metadata.csv \
  --output-dir data
```

次に、CSA本体が置かれた環境でJSONLへ変換する。

```bash
python create_dataset.py export --output-dir data
```

metadata中のパスと実際のCSAルートが異なる場合は、接頭辞を置換する。

```bash
python create_dataset.py export \
  --output-dir data \
  --path-prefix-from /path/on/source-machine/csa_raw \
  --path-prefix-to /mnt/data/csa_raw
```

`build`で両方を連続実行することもできる。

```bash
python create_dataset.py build \
  --metadata-csv ../wsl2/metadata.csv \
  --output-dir data
```

最初は `--limit 10` を付けて、各splitの先頭10局でCSA変換を確認する。
`--strict`を指定すると、CSAの欠落や不整合が1件でもあれば処理を停止する。

## 出力

```text
data/
  manifests/
    train.csv
    validation.csv
    validation_open.csv
    validation_mixed.csv
    validation_closed.csv
    evaluation_eligible.csv
    evaluation.csv
    evaluation_open.csv
    evaluation_mixed.csv
    evaluation_closed.csv
  datasets/
    train.jsonl
    validation.jsonl
    evaluation.jsonl
  errors/
    train.csv
    validation.csv
    evaluation.csv
  split_summary.json
  export_summary.json
  vocab.json
```

JSONLには対局者名やレートも分析用メタデータとして保存するが、モデル入力には
使用しない。`engine_scope`によるopen/closed評価にも用いる。

## JSONLレコード

```json
{
  "schema_version": 1,
  "game_id": "...",
  "split": "evaluation",
  "engine_scope": "closed",
  "game_date": "2025-01-01",
  "initial_sfen": "...",
  "initial_state_tokens": ["SQ_W_L", "... 96 tokens ...", "TURN_BLACK"],
  "move_tokens": ["7g7f", "3c3d"],
  "black_player": "...",
  "white_player": "...",
  "rating_b": 3200.0,
  "rating_w": 3100.0,
  "game_result": 1
}
```

`initial_sfen`は検証・再生用であり、その文字列を直接モデルへ入力しない。

## モデル

`models/`には、同一backbone設定を使う次の2モデルを実装している。

- `VanillaTransformer`：pre-norm causal decoder
- `T2MLRTransformer`：中央層へTemporal Middle-Layer Recurrenceを追加

```python
from models import ModelConfig, T2MLRConfig, parameter_matched_vanilla_config
from models import VanillaTransformer, T2MLRTransformer

vanilla = VanillaTransformer(
    ModelConfig(vocab_size=len(vocab), n_layers=8, d_model=256, n_heads=8)
)
t2mlr = T2MLRTransformer(
    T2MLRConfig(
        vocab_size=len(vocab),
        n_layers=8,
        d_model=256,
        n_heads=8,
        l_start=3,
        l_end=4,
        jacobi_depth=4,
    )
)

# parameter数を揃える主比較では、T²MLRの追加分をVanillaのFFN幅で補う。
vanilla_matched = VanillaTransformer(parameter_matched_vanilla_config(t2mlr.config))
```

両モデルの`forward()`は`DecoderOutput`を返す。`hidden_states`は全層の
線形プローブに、T²MLRの`recurrent_states`は再帰cacheそのもののプローブに使う。
T²MLRには、学習用のJacobi型parallel approximationと、評価用の
`exact_recurrence=True`逐次経路がある。

同じ`d_ff`を使う構造一致条件ではT²MLRに融合module分のparameterが増える。
主結果には`parameter_matched_vanilla_config()`によるparameter一致条件を用い、
同一backbone幅の比較は補助実験として報告する。

T²MLR移植の出典、公式実装との対応、簡略化点は
`models/t2mlr.py`と`THIRD_PARTY_NOTICES.md`に明記している。

## 線形プローブ

各指し手直後の単一の隠れ表現から、盤面81マス、持ち駒14種類の枚数、手番を
線形プローブで復号する。
ここで測るのは「境界表現から局面情報が線形復号可能か」であり、Transformerの
KV cache全体が明示的な盤面データ構造になっていると仮定しない。
状態差分棋譜対、同一局面・異履歴、activation patchingによる因果的検証は
将来実験として分離する。

学習済みcheckpointは、少なくとも次の形式で保存する。

```python
torch.save(
    {
        "model_type": "vanilla",  # または "t2mlr"
        "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
    },
    "checkpoints/model.pt",
)
```

線形プローブの学習と評価は次のように行う。

```bash
python evaluate_probes.py \
  --checkpoint checkpoints/model.pt \
  --vocab data/vocab.json \
  --train-jsonl data/datasets/train.jsonl \
  --validation-jsonl data/datasets/validation.jsonl \
  --evaluation-jsonl data/datasets/evaluation.jsonl \
  --output-dir results/probes
```

既定では最終層、T²MLRの再帰状態、現在指手のtoken embedding対照を評価する。全層を
評価する場合は`--sources layers,recurrent,token_embedding`を指定する。出力は
`probe_metrics.json`と`linear_probes.pt`であり、前者には盤面・持ち駒・手番の指標、
履歴長別・open/mixed/closed別指標、次手予測lossとtop-k accuracyが含まれる。また、
各次手予測位置についてtop-1合法手率、top-5内の合法手有無、合法手への確率質量、
合法手の語彙収録率をcshogiで計算する。`<EOS>`予測位置は合法手評価から除外する。

### シェルからの評価

評価条件をまとめた`scripts/run_probe_evaluation.sh`を用意している。checkpointだけを
指定すれば、データと語彙には上記の既定パスを使う。

```bash
cd /path/to/shogi_ai/shogi_state_tracking

CHECKPOINT=checkpoints/vanilla.pt \
  scripts/run_probe_evaluation.sh standard
```

利用可能なmodeは次のとおりである。

```text
standard              最終層、再帰状態、現在指手embedding
all-layers            全層、再帰状態、現在指手embedding
untrained             未学習backboneのstandard対照
all-layers-untrained  未学習backboneの全層対照
```

パスと主要条件は環境変数で上書きできる。

```bash
CHECKPOINT=checkpoints/t2mlr.pt \
SEED=20260725 \
DEVICE=cuda \
POSITIONS_PER_GAME=24 \
BATCH_SIZE=2048 \
  scripts/run_probe_evaluation.sh all-layers
```

`VOCAB_PATH`、`TRAIN_JSONL`、`VALIDATION_JSONL`、`EVALUATION_JSONL`、
`OUTPUT_DIR`、`PYTHON_BIN`も同様に指定できる。modeより後の引数は
`evaluate_probes.py`へそのまま渡される。

## 開始局面のランダム化

学習では、各対局について少なくとも40手の追跡区間が残る範囲に最大40個の開始候補を
等間隔に作り、epochごとに1個を選び直す。

```python
from data import RandomStartSequenceDataset

train_dataset = RandomStartSequenceDataset(
    "data/datasets/train.jsonl",
    token_to_id=vocab,
    candidate_count=40,
    min_suffix_moves=40,
    randomize_each_epoch=True,
)

for epoch in range(num_epochs):
    train_dataset.set_epoch(epoch)
    # train ...
```

実データや40局面分の状態を複製せず、選ばれた位置まで`initial_sfen`からcshogiで
再生する。選択は`seed, epoch, game_id`から決定するため再現可能である。

検証・評価では`randomize_each_epoch=False`とし、開始局面を固定する。開始位置への
偶然依存を調べる場合は`samples_per_game=3`などとして、1対局から複数の固定開始系列を
作る。モデルには`start_sfen`や`start_ply`を渡さず、96状態トークンと指し手だけを渡す。

## 学習

answer-onlyの棋譜次手予測は次のように実行する。

```bash
scripts/run_training.sh pretrain vanilla --match-t2mlr
scripts/run_training.sh pretrain t2mlr
```

checkpointにはモデル種別、設定、重み、stage、epoch、step、seedを保存する。
`best.pt`はvalidation loss最小、`last.pt`は最終epochの状態である。

## Synthetic reasoning trace

現在の状態追跡実験へ、backbone自身が生成した複数の読み筋を用いるCoT-like SFTを
追加している。基本比較はVanilla/T²MLRとanswer-only/CoTの2×2である。

```bash
scripts/run_2x2_experiment.sh --epochs 10 --batch-size 32
```

trace生成では合法手mask、ルールによる補正、エンジン探索を使わない。cshogiは
教師棋譜の再生と学習後の合法性評価だけに使う。CoT条件では自由生成の最終指手精度、
trace合法率、候補内正解recall、answer-trace整合率を測り、answer-only/CoTの両方へ
同じ線形probeを適用する。

特殊tokenを追加しているため、既存の`vocab.json`とcheckpointはそのまま流用せず、
`create_dataset.py export`から再生成・再学習すること。

現行実装では、T²MLRのbranch stateを`<SEP>`ごとにrootへ戻さない。
