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

CPU、CUDA 13.0、ROCmのいずれかを明示して構築する。ROCmは実行環境を判定し、
native LinuxではROCm 7.1、WSL2ではAMDが検証したROCm 7.2向けwheelを使用する。
引数を省略した場合は、大容量のNVIDIA/AMD packageを取得しないCPU版になる。

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

native LinuxのROCm条件では、PyTorch 2.13.0が要求する
`triton-rocm==3.7.1`をPyTorch公式のaggregate wheel indexから先に導入する。
別のTorch版を指定し、要求されるTriton版が異なる場合は
`ROCM_TRITON_VERSION`も同時に指定する。

```bash
TORCH_VERSION=2.12.1 \
ROCM_TRITON_VERSION=3.6.0 \
  ./setup_env.sh rocm
```

WSL2ではnative Linux用wheelを使用しない。native LinuxのROCm wheelは
`/sys/class/kfd/kfd/topology/nodes`を参照するが、WSL2のGPU経路は`/dev/dxg`を
使用するためである。WSL2ではAMD公式の検証済み構成であるPyTorch 2.9.1、
ROCm 7.2、Triton 3.5.1、NumPy 1.26.4を導入し、wheel内のHSA runtimeを除いて
`/opt/rocm/lib/libhsa-runtime64.so.1`を使用する。`/dev/dxg`またはこのruntimeが
存在しない場合、Windows側AMD WSL driver／WSL側ROCmの準備不足として停止する。

`setup_env.sh`は共通依存について`uv sync --frozen --inexact`を実行するため、
ロックファイルと`pyproject.toml`が一致しない場合は停止する。PyTorchについては
uv公式の`--torch-backend`を用い、選択したbackendだけを取得する。依存関係を
変更した場合に限り、開発者が`uv lock`で`uv.lock`を更新する。

setup末尾では、`cshogi`単体、`torch`単体、実プログラムと同じ
`torch`→`cshogi`順の同時import、accelerator runtimeの順に検証する。
`[verify 1/4]`から`[verify 3/4]`まで成功して`[verify 4/4]`でabortする場合は、
Python packageの解決ではなく、ホストのGPU、ROCm/CUDA driver、デバイス権限の
組合せを確認する。失敗した環境は`.torch-backend`へ確定記録されないため、
原因を修正した後に同じsetupコマンドを再実行できる。GPU版ではacceleratorが
利用可能でない場合もsetup失敗とする。

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
`probe_metrics.json`、`linear_probes.pt`、`probe_predictions.pt`であり、前者には盤面・持ち駒・手番の指標、
履歴長別・open/mixed/closed別指標、次手予測lossとtop-k accuracyが含まれる。また、
各次手予測位置についてtop-1合法手率、top-5内の合法手有無、合法手への確率質量、
合法手の語彙収録率をcshogiで計算する。`<EOS>`予測位置は合法手評価から除外する。
同時に`probe_predictions.pt`へ評価位置ごとの盤面・持ち駒・手番の正解と予測、盤面の
正解クラス確率、距離、対局IDを保存する。このファイルは可視化専用であり、モデルの
学習には使用しない。

### プローブ結果の可視化

`visualize_probes.py`は外部描画ライブラリを必要とせず、SVGを生成する。盤面上の数値は
各マスの復元精度である。`occupied-accuracy`では空マスを除いて集計する。

```bash
python visualize_probes.py aggregate \
  --predictions results/probes/vanilla.pt/standard/seed_20260724/probe_predictions.pt \
  --source final \
  --metric occupied-accuracy \
  --output results/figures/vanilla-occupied.svg
```

個別局面では、背景色が正解クラス確率、緑枠が正解、赤枠と矢印が誤予測を表す。

```bash
python visualize_probes.py position \
  --predictions results/probes/vanilla.pt/standard/seed_20260724/probe_predictions.pt \
  --source final \
  --index 0 \
  --output results/figures/vanilla-position-0.svg
```

VanillaとT²MLRなど、同じ評価順で作成した2つのアーティファクトの差分も表示できる。
正値は第1ファイルの方が復元精度が高いマスである。

```bash
python visualize_probes.py difference \
  --predictions results/probes/vanilla/probe_predictions.pt \
  --predictions-b results/probes/t2mlr/probe_predictions.pt \
  --source final \
  --output results/figures/vanilla-minus-t2mlr.svg
```

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

trace生成を短く試す場合は，生成条件を環境変数で指定できる．

```bash
MAX_GAMES=10 POSITIONS_PER_GAME=1 LINES=1 LINE_LENGTH=2 \
  scripts/run_2x2_experiment.sh --epochs 1 --max-steps 10
```

`run_2x2_experiment.sh`は，既定では学習用・検証用traceの生成とCoT学習までを行い，
時間のかかる評価用trace生成，推論評価，probe評価を実行しない．評価は学習完了後に
別プロセスとして実行する．

```bash
scripts/run_2x2_evaluation.sh
```

1モデルだけを評価する場合は，次を使う．

```bash
scripts/run_cot_evaluation.sh vanilla
```

従来どおり一連の処理を連続して実行する場合は，環境変数で有効化できる．

```bash
RUN_EVALUATION=1 scripts/run_2x2_experiment.sh --epochs 10 --batch-size 32
```

trace生成では合法手mask、ルールによる補正、エンジン探索を使わない。cshogiは
教師棋譜の再生と学習後の合法性評価だけに使う。CoT条件では自由生成の最終指手精度、
trace合法率、候補内正解recall、answer-trace整合率を測り、answer-only/CoTの両方へ
同じ線形probeを適用する。

特殊tokenを追加しているため、既存の`vocab.json`とcheckpointはそのまま流用せず、
`create_dataset.py export`から再生成・再学習すること。

現行実装では、T²MLRのbranch stateを`<SEP>`ごとにrootへ戻さない。

## 別計算機への実験環境移行

実験コード、`uv.lock`、生成済み`data/`を、チェックサム付きアーカイブへまとめられる。
`.venv`と`.uv-cache`はCUDA/ROCmやホストOSに依存するため、アーカイブには含めない。
移行先では、展開後に移行先のアクセラレータに合わせて環境を再構築する。

```bash
cd /path/to/shogi_ai/shogi_state_tracking
scripts/package_experiment.sh /mnt/transfer/shogi-state-tracking.tar.gz
```

結果やcheckpointも含める場合は、容量とデータの取り扱いを確認してから指定する。

```bash
scripts/package_experiment.sh \
  --include-results \
  --include-checkpoints \
  /mnt/transfer/shogi-state-tracking-with-artifacts.tar.gz
```

アーカイブには`MIGRATION_MANIFEST.json`が入り、コード・データ・成果物のSHA-256を
記録する。移行先では、既存の空でないディレクトリを上書きせずに展開・検証する。

```bash
scripts/restore_experiment.sh \
  /mnt/transfer/shogi-state-tracking.tar.gz \
  /work/shogi_state_tracking

cd /work/shogi_state_tracking
./setup_env.sh cpu       # 移行先がNVIDIAならcuda、AMDならrocm
```

生成データには対局者名や入力CSAのパスが含まれる場合がある。アーカイブを公開・共有
する前に、`data/`および`--include-results`、`--include-checkpoints`の内容を確認する。
入力CSAや`metadata.csv`は自動的には同梱しないため、再度CSAからexportする場合は、
移行先のパスに合わせて`--path-prefix-from`と`--path-prefix-to`を指定する。
