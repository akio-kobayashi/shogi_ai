# 将棋状態追跡Transformer用データセット

学生向けのColab実験は
[`notebooks/shogi_state_tracking_colab.ipynb`](notebooks/shogi_state_tracking_colab.ipynb)から開始できる．
任意CSA対局における飛・角・竜・馬・玉の推定確率を追う場合は，
[`notebooks/csa_piece_trajectory.ipynb`](notebooks/csa_piece_trajectory.ipynb)を使う．
実データがない場合はtoy棋譜を自動生成し，指手予測，線形probe，盤面SVG可視化までを
小規模に実行する．

ノートブックから再利用する処理は[`colab_utils.py`](colab_utils.py)にまとめている．
`prepare_dataset`，`train_small_model`，`evaluate_next_move`，
`run_probe_evaluation`，`render_probe_svg`を呼び出せば，Colab側でCLIの引数列を
組み立てなくても同じ実験を実行できる．

## 学習中の進捗とGPUメモリ

`train_model.py`は開始時，データ読み込み後，指定間隔ごとの学習step，検証開始，
checkpoint保存，終了時に進捗を標準出力へ出す．`scripts/run_training.sh`からは
Pythonの出力を非バッファで実行する．

ROCmの`HIPCachingAllocator`が出す`memory allocation failed with OOM`は，
内部の解放・再試行後に処理が継続する警告の場合がある．`RuntimeError`や
`torch.OutOfMemoryError`のtracebackと終了コードが出なければ，致命的なOOMとは限らない．
ただし，割り当て余力が少ない状態なので，警告を正常とはみなさず，次のように新しい実行を
小さくして確認する．既に起動しているプロセスのbatch sizeは途中では変わらない．

```bash
BATCH_SIZE=4 MAX_SEQ_LEN=320 AMP=auto PROGRESS_EVERY=1 \
  scripts/run_training.sh pretrain vanilla
```

small/base/largeを固定設定で比較する場合は次を使う．max_seq_lenは512固定で，
`MAX_SEQ_LEN=512`下でサイズ比較を一括実行できる。

```bash
scripts/compare_transformer_sizes_maxseq512.sh
```

指手・合法手評価，盤面・持ち駒・手番の全層probe，王手probe，集計まで一括で行う場合は次を使う。

```bash
scripts/run_transformer_size_compare_512.sh
```

学習後に全サイズの数値評価だけを再実行する場合は次を使う．

```bash
scripts/run_transformer_size_compare_512_evaluation.sh
```

このスクリプトは`RUN_TRAIN=0`を固定し，既存checkpointを使う．既定では数値評価を全て実行する．
`RUN_MOVE_EVALUATION=0`，`RUN_PROBES=0`，`RUN_CHECK_PROBES=0`で各評価を個別に省略できる．
指手・状態・王手の評価は，既定で開始ply `0,24,25,32,33` の実局面を96 tokenの
promptとして与える．偶数plyだけに偏らないよう，24/32に対応する25/33も含める．
各開始局面の後に40手以上残る対局だけをその条件の評価へ使う．開始条件を変更する場合は，
`EVALUATION_START_PLIES=0,40,41 EVALUATION_MIN_SUFFIX_MOVES=40`のように指定する．
王手probe用の均衡化データは既定で再作成するが，既存の`data/check_probe/`を使う場合は
`PREPARE_CHECK_PROBE_DATA=0`とする．

大駒・玉のヒートマップは自動選択しない．数値評価後に対局・手数・駒種を選び，
`visualize_major_piece_probe.py`を別途実行する．`RUN_VISUALIZATIONS=1`はこの判断を促すだけで，
恣意的な局面選択による図の自動生成は行わない．

`AMP=auto`はCUDA／ROCmで自動混合精度を有効にする。BF16対応GPUではBF16を使い，
対応していない場合はFP16と勾配スケーラを使う。無効化する場合は`AMP=off`とする。
ROCmでもPyTorch上のデバイス名は通常`cuda`であり，この設定で利用できる。ただし，
`torch.cuda.is_available()`や`torch.cuda.is_bf16_supported()`がランタイム異常で中断する
環境では，AMP以前にROCm／PyTorchの組合せを修正する必要がある。

`PROGRESS_EVERY`はシェルから渡す追加引数として，例えば
`scripts/run_training.sh pretrain vanilla --progress-every 1`のように指定できる．

学習用のランダム開始系列は，開始局面の固定99トークン（`<BOS>`，局面96，
`<MOVES>`，`<EOS>`）を除いた範囲で`--max-seq-len`までに切り詰める．したがって，
極端に長い対局のsuffixがbatch全体を長くすることはない．`start_ply`は保持されるため，
後段の局面進展別評価では元棋譜上の位置も利用できる．

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

マスク実験では，次の4種類の予約tokenを末尾へ追加したschema version 4を使う。

- `<MASK_MOVE>`：1手tokenの置換
- `<MASK_SQUARE>`：盤面81マスの状態tokenの置換
- `<MASK_HAND>`：持ち駒14要素の置換
- `<MASK_TURN>`：手番tokenの置換

通常の語彙はv3のtoken idを維持したまま，マスクtokenだけを末尾へ追加する。
ただし語彙サイズは変わるため，v3 checkpointへv4語彙をそのまま読み込むことはできない。
マスクtokenは予約されるだけであり，実際のマスク系列の生成とマスク復元損失は別実験で実装する。

## 既定の抽出条件

- 2022年1月1日以降
- 先手・後手ともレーティング3000以上
- 80手以上
- `game_result == 0` を除外
- 2022年1月～2024年9月：学習
- 2024年10月～12月：検証
- 2025年1月以降：評価
- 評価は合計5,000局（`player_scope`の`open` 1,667、`mixed` 1,667、`closed` 1,666）

評価・検証データには、学習期間中の対局者名義との関係から`player_scope`を付ける。

- `open`：両対局者名義が学習集合に存在
- `mixed`：一方だけが学習集合に存在
- `closed`：両対局者名義が学習集合に存在しない

これはエンジンファミリーではなく、metadataの対局者名義による分類である。
旧フィールド`engine_scope`は互換性のため残すが、今後は`player_scope`を使う。

これとは別に、CSAを再生した各局面について`position_scope`を付ける。

- `seen_position`：正規化したSFEN（盤面・持ち駒・手番）が学習集合に存在
- `unseen_position`：学習集合に同一局面が存在しない
- `strict_unseen_position`：入力系列中の全局面が`unseen_position`

局面の正規化ではSFENの手数フィールドを除外し、残りの盤面・手番・持ち駒をSHA-256で
ハッシュ化する。`position_scope_by_ply`は初期局面を0番目とする局面ごとのラベルであり、
ランダム開始位置から作るサンプルの`position_scope`と`trajectory_scope`は前処理時に決まる。
したがって、`player_scope`は対局者分布の重複、`position_scope`は局面の未見性を表し、
両者を混同しない。
なお，CSV manifestは対局単位なので，`position_scope`はexport前には
`pending_export`である．実際の局面単位の判定はJSONLの`position_scope_by_ply`を参照する．
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

まず、CSA本体を読まずにmetadataを抽出・時系列分割できる。ただし、局面の
`position_scope`はCSAの再生が必要なため、後続の`export`で付与される。

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

マスク実験用語彙を生成する場合は，次のように指定する。

```bash
python create_dataset.py export \
  --output-dir data \
  --include-mask-tokens
```

`build`でも同じ`--include-mask-tokens`を指定できる。指定しない場合は従来のv3語彙を生成する。

### 既存の学習済みモデルへ局面スコープを付ける

すでに別計算機で学習を実行している場合，学習をやり直す必要はない．学習済み
checkpointと，そのcheckpointの作成に使った`vocab.json`はそのまま保持する．
CSAを再読込せず，変換済みJSONLに含まれる`initial_sfen`と`move_tokens`だけを再生して，
局面スコープ付きJSONLを別ディレクトリへ作る．

```bash
DATA_DIR=/path/to/data \
OUTPUT_DIR=/path/to/data/scoped_datasets \
scripts/annotate_position_scopes.sh
```

この処理はモデルの入力トークン列を変更しない．したがって，評価時には元の語彙を指定したまま，
出力された`scoped_datasets/{train,validation,evaluation}.jsonl`を使う．

```bash
python evaluate_probes.py \
  --checkpoint results/training/vanilla/seed_20260724/best.pt \
  --vocab data/vocab.json \
  --train-jsonl data/scoped_datasets/train.jsonl \
  --validation-jsonl data/scoped_datasets/validation.jsonl \
  --evaluation-jsonl data/scoped_datasets/evaluation.jsonl \
  --output-dir results/probes_scoped
```

checkpointと語彙の組合せを変えるとtoken idが一致しない可能性があるため，評価時に新しい
`vocab.json`を再生成して置き換えてはならない．

### 系列長の決定

`metadata.csv`の最大対局長をそのまま`max_seq_len`にすると、少数の長い対局が
causal attentionのメモリを支配する。先に対局長の分位点を調べ、学習用対局の
95パーセンタイルを目安に系列長を決める。

```bash
python analyze_metadata_lengths.py \
  --metadata-csv ../wsl2/metadata.csv
```

このスクリプトは`create_dataset.py split`と同じ既定条件（2022年1月以降、両者
レート3000以上、80手以上、引き分けを除外）で、train／validation／evaluation
eligibleごとの`total_moves`と、固定局面prefix 99トークンを含む候補系列長の
coverageをJSONで出力する。`total_moves`はCSAの指し手数であり、千日手の有無は
metadataだけでは判定できない。

例えば、trainのp95が221手なら、

```text
99（固定prefix）+221（手数）=320トークン
```

となる。したがって現在の既定値`MAX_SEQ_LEN=320`は、trainの約95%を1つの
windowに収める値として説明できる。残りの長い対局はrandom-start windowingで
切り出す。validationやevaluationの完全対局を一つの系列に保持する必要はなく、
同じwindow長で評価する。全splitのp95を一つの値で覆うなら352になるが、学習時の
メモリを優先し、まずはtrain基準の320を採用する。

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
使用しない。`player_scope`（旧`engine_scope`）と、局面ごとの
`position_scope_by_ply`を評価の層別に用いる。

## JSONLレコード

```json
{
  "schema_version": 2,
  "game_id": "...",
  "split": "evaluation",
  "player_scope": "closed",
  "engine_scope": "closed",
  "position_scope_by_ply": ["unseen_position", "unseen_position", "..."],
  "trajectory_scope": "strict_unseen_position",
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
ここで確認したいのは、Transformerの中に将棋盤そのものが保存されているかどうかではない。
開始局面と正しい指し手履歴を入力したとき、指し手の区切り位置にある隠れ表現から、その時点の
盤面・持ち駒・手番を当てられるかを調べる。盤面・持ち駒・手番を当てられれば、それらの情報が
隠れ表現から読み出せる形で含まれていると考える。実際には、隠れ表現に単純な線形プローブを
取り付けて予測する。TransformerのKV cache全体が明示的な盤面データ構造になっているとは仮定しない。
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
`probe_metrics.json`、`probe_metrics_detail.json`、`linear_probes.pt`、`probe_predictions.pt`である。
`probe_metrics.json`はモデル間比較・発表資料向けの主要指標だけを含む小さなサマリーである。
履歴長別・`player_scope`（open/mixed/closed）別・`position_scope`別の全指標，および各probe epochの
履歴は`probe_metrics_detail.json`に保存する。
評価JSONLの各マス・各持ち駒slot・手番について，学習用probe集合の最頻値を常に出す
`positionwise_train_majority`ベースラインも計算する。各sourceの`evaluation_minus_majority`は，
このベースラインとの差であり，詳細ファイルには駒種別の`board_accuracy_by_class`差分も含まれる。
プローブ評価では，計算負荷の低い指手予測・合法手指標を既定で実行しない。
旧形式の一括出力が必要な場合だけ，`--include-language-model`を追加する。
同時に`probe_predictions.pt`へ評価位置ごとの盤面・持ち駒・手番の正解と予測、盤面の
正解クラス確率、距離、対局IDを保存する。このファイルは可視化専用であり、モデルの
学習には使用しない。
`probe_metrics_detail.json`では，対局者名義による層別は`strata`，未見局面による層別は
`position_strata`に保存する．

### 大駒・玉位置プローブ

大駒（飛車・角・龍・馬）と玉の位置には，別の専用headを増やさない。既存の盤面復元
線形probeが出力する81マス×29クラスの分布から，各クラスの位置復元精度を
`probe_metrics_detail.json`の`board_accuracy_by_class`で読む。これは，同じ特徴量・同じ
線形復号器で全駒種を比べるためである。評価時には`board_samples_by_class`も併記し，
希少な成駒の精度を過大解釈しない。

個別局面について飛車・角・龍・馬・玉の位置分布を盤面上に描く場合は，保存済みの
`linear_probes.pt`から必要な局面だけを再計算する。使用例は「大駒・玉の可視化」を参照する。

### 王手プローブ用データセットと評価

王手状態は少数なので，評価JSONLをそのまま使わず，cshogiで再生した正例・負例を同数にした
状態単位の集合を作る。各行には開始局面，当該plyまでの教師指手列，その直後の`in_check`ラベルを
保存する。これはモデル学習用データではなく，後段の抽象状態probe専用である。

```bash
scripts/create_check_probe_datasets.sh
```

既定では各splitを混ぜずに，モデルの最大系列長320に対応する221手以内の状態を使う。学習・検証・
評価集合からそれぞれ王手と非王手を最大20,000・5,000・10,000件ずつ抽出する。正例が少ない場合は，
利用可能な正例数に合わせて両クラスを同数へ縮小する。別のcheckpoint長や件数は環境変数で指定する。

```bash
MAX_PREFIX_MOVES=413 \
EVALUATION_SAMPLES_PER_CLASS=5000 \
scripts/create_check_probe_datasets.sh
```

単一splitだけを作る場合は，従来どおり次を用いる。

```bash
INPUT_JSONL=data/datasets/evaluation.jsonl \
OUTPUT_JSONL=data/check_probe/evaluation.jsonl \
scripts/create_check_probe_dataset.sh
```

作成後は，decoderを凍結したまま，末尾指手トークンの隠れ表現から「その手の直後に手番側の玉が
王手されているか」を二値線形分類する。`accuracy`だけでなく，均衡集合での`balanced_accuracy`，
`precision`，`recall`，`F1`を多数派ベースラインとの差とともに保存する。

```bash
CHECKPOINT=checkpoints/model.pt \
scripts/run_check_probe_evaluation.sh all-layers
```

出力は`results/check-probes/.../check_probe_metrics.json`と`check_linear_probes.pt`である。
この結果は，王手情報が線形に読み出せることを示す操作的な指標であり，モデルが明示的なルール表や
盤面変数を保持していることの証明ではない。

### 指手予測・合法手評価

指手予測loss，top-k，合法手率，合法手への確率質量は，プローブ学習を伴わない独立評価として
実行する．

```bash
CHECKPOINT=checkpoints/model.pt \
  scripts/run_move_evaluation.sh
```

結果は`results/moves/<checkpoint>/seed_<seed>/move_metrics.json`へ保存する．この評価も
元JSONLの開始局面から固定的に再生し，評価時のランダム開始を行わない．
ただし，これは正解棋譜履歴を毎回入力するteacher-forced評価である．したがって，
自己回帰ロールアウト，対局勝率，探索の代替性能を直接測るものではない．

学習・検証ではランダム開始系列を使うが，評価では元JSONLの開始局面（`start_ply=0`）から
固定的に再生する．したがって評価局面はseedによって変化しない．既定の
`POSITIONS_PER_GAME=16`は各対局から評価する局面数を抑えるための決定論的な抽出であり，
全局面を使う場合は`POSITIONS_PER_GAME=0`を指定する．

### 評価指標の読み方

評価値は，指手をどれだけ予測できたかという**出力評価**と，隠れ表現から局面情報を
どれだけ復号できたかという**状態評価**に分けて読む．両者は別の能力を測るため，一つの
数値にまとめない．

#### 1．指手予測の評価

指手を1トークンとして扱い，教師指手を `m`，モデルの分布を `p(m)` とする．
`<EOS>`は対局終了の制御トークンであり，指手の評価には含めない．`eos_cross_entropy`は
補助的に別集計する．

ここでの教師指手は棋譜に記録された手である．したがってtop-1率とtop-5内率は，エンジン最善手との
一致率や棋力ではなく，棋譜の指手分布に対する模倣精度を表す．

| 指標 | 計算 | 分かること |
|---|---|---|
| 指手予測loss（cross entropy） | `-1/N Σ log p(m_i)` | 教師指手へどれだけ確率を与えたか．低いほどよい |
| 指手予測top-1率 | 1位の指手が教師指手と一致した割合 | 最有力手を一手で当てる能力 |
| 指手予測top-5内率 | 上位5手に教師指手が含まれる割合 | 候補手として残せる能力 |
| 合法手top-1率 | 1位の指手が，その局面の合法手集合に含まれる割合 | 最有力出力がルール上合法か |
| 合法手top-5内率 | 上位5手のうち少なくとも1手が合法である割合 | 合法な候補を上位5手に含められるか |
| 合法手への確率質量 | `Σ p(a | s)`（`a`が合法手）の平均 | 合法手全体にどれだけ確率を配分したか |

ここで `L(s)` はcshogiが生成した局面 `s` の合法手集合である．合法手top-1率が高くても
良い手を選べるとは限らない．例えば，合法だが教師指手ではない手を選んでいる可能性がある．
一方，合法手への確率質量が高くtop-1率が低い場合は，合法手を広く候補にしているが，分布が
十分に集中していないと解釈できる．

合法手のUSI表現が語彙に含まれている割合は，`mean_legal_move_vocabulary_coverage`として
別に記録する．語彙外の合法手は，モデルが確率を与えられないため，合法手確率質量とは分けて
報告する．合法手集合やルール情報は学習には使わず，評価時の判定にだけ用いる．

#### 2．局面状態の復元評価

線形プローブが，凍結したTransformerの特徴から次の局面情報を復号する．これは「内部に明示的な
盤面データ構造がある」ことの証明ではなく，その情報が線形に読み出せるかの評価である．

| 指標 | 計算 | 分かること |
|---|---|---|
| 盤面81マスの復元精度 | 81マスの正解率の平均 | 各マスの駒種・所属・空マスを復元できるか |
| 盤面occupancy精度 | 81マスについて，空マス／駒ありを二値で判定 | 駒の有無だけを復元できたか |
| occupiedマスの駒精度 | 正解が駒ありのマスだけについて，駒種・所属まで一致した割合 | 空マスの多さを除き，駒の内容を復元できたか |
| 盤面完全一致率 | 81マスすべてが一致した割合 | 局面の盤上配置を完全に復元できた割合 |
| 持ち駒の復元精度 | 先後7種類ずつ，計14スロットの枚数正解率 | 持ち駒の種類と枚数を追跡できるか |
| 持ち駒完全一致率 | 14スロットすべてが一致した割合 | 持ち駒全体を完全に復元できた割合 |
| 持ち駒MAE | 14スロットの枚数誤差の平均 | 何枚ずれているか．0が完全一致 |
| 手番の復元精度 | 先手・後手の正解率 | 現在の手番を復元できるか |
| 局面完全一致率 | 盤面81マス，持ち駒14スロット，手番のすべてが一致した割合 | 局面全体を完全に復元できた割合 |

盤面81マスの復元精度が高くても，空マスを多く当てているだけの場合がある．そのため，通常は
`board_square_accuracy`，`board_occupancy_accuracy`，`board_piece_accuracy_on_occupied`を併記する．
旧名`board_occupied_accuracy`は後者と同じ値で，互換性のため残している．最も厳しい指標は
`full_state_exact_match`であり，部分的な正解は完全一致として数えない．

#### 3．結果の解釈順序

まず指手予測lossとtop-k率で，モデルが指手系列を学習できているかを確認する．次に合法手指標で，
出力が将棋の形式・ルールに適合しているかを確認する．最後に線形プローブの盤面・持ち駒・手番の
指標を見て，指手予測に必要な状態情報が隠れ表現から読み出せるかを調べる．

モデル間の比較では，`top-1`だけでなく`top-5`と合法手確率質量を併記し，状態復元では盤面・
持ち駒・手番を分けて報告する．これにより，「指手を当てられない」のか，「合法手を出せない」のか，
「局面情報を保持できていない」のかを区別できる．

#### 4．occupied指標とチェス研究の違い

`occupied`という名前は，二つの異なる指標と混同しやすい．本実験では次の三つを分ける．

```text
全マス正解率
  = 81マスの「空／駒種・所属」の一致率

盤面occupancy精度
  = 81マスの「空か／駒があるか」だけの一致率

occupiedマスの駒精度
  = 正解が駒ありのマスに限定した「駒種・所属」の一致率
```

例えば81マス中61マスが空いている局面で，モデルが全マスを空と予測すると，全マス正解率と
occupancy精度は約75.3%になるが，occupiedマスの駒精度は0%である．このため，空マスの多さに
よる見かけの高精度を避けるには，三つを併記する必要がある．

チェスの状態追跡研究 [Toshniwal et al.](https://arxiv.org/abs/2102.13249) は，64マス全体を
復元する課題ではない．棋譜の接頭辞とプロンプトから，特定駒の移動先を予測する．実際の移動先と
一致する`ExM accuracy`と，その駒にとって合法な移動先なら正解とする`LgM accuracy`を分け，
合法移動先の個数を `R` とした`R-Precision@R`も報告している．

したがって，本実験の指標との対応は次のようになる．

| 本実験 | チェス研究との関係 |
|---|---|
| 指手top-1率 | 実際の棋譜手を当てるExMに近いが，指手全体を直接評価する |
| 合法手top-1率 | LgM@1に近い |
| 合法手top-5内率 | 上位5候補に合法手が1つでもあるかというhit@5であり，R-Precisionではない |
| 合法手への確率質量 | 確率分布全体の指標であり，Toshniwalらの主指標とは異なる |
| 盤面81マス・持ち駒・手番 | チェスの移動先プローブを拡張した，将棋の状態復元指標 |

つまり，盤面復元指標と合法手指標は目的が異なる．前者は「局面情報を読み出せるか」，後者は
「合法な指手候補を出せるか」を測る．両者を同じ正解率として比較してはならない．

### プローブ結果の可視化

`visualize_probes.py`は外部描画ライブラリを必要とせず，SVGを生成する．盤面上の数値は
各マスの復元精度である．`occupied-accuracy`では，正解が駒ありのマスだけを対象に，駒種・所属まで
一致した割合を集計する．これは「駒があるか」だけの二値occupancy精度とは異なる．

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

大駒の状態追跡を調べる場合は，評価アーティファクトへ全クラス確率を保存しない。
代わりに`linear_probes.pt`，checkpoint，評価JSONLから指定局面だけを再計算し，飛車・角・龍・馬の
確率をcshogiで再生した正解盤面へ重ねる。

```bash
python visualize_major_piece_probe.py \
  --checkpoint results/model-comparison/t2mlr/best.pt \
  --vocab data/vocab.json \
  --evaluation-jsonl data/datasets/evaluation.jsonl \
  --probes results/probes/t2mlr-all-layers/linear_probes.pt \
  --game-id GAME_ID \
  --ply 42 \
  --piece black_R \
  --source layer_3 \
  --output results/figures/game-major-rook-ply42.svg
```

各マスの背景色と数値は`P(先手飛車がそのマスにある)`であり，緑枠はcshogiで再生した正解位置を表す。
`--piece`には飛車・角・龍・馬に加え，`black_K`と`white_K`も指定できる。同一対局の連続したplyへ
適用すれば，大駒の移動，捕獲，成り，駒打ち，および玉移動に対して隠れ表現がどのように更新されるかを追える。

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
シェル実行では`state_0`を主指標へ含めない。prompt読取りの確認が必要な場合は
`INCLUDE_INITIAL_STATE=1`を指定する。

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

answer-onlyの棋譜指手予測は次のように実行する。

```bash
scripts/run_training.sh pretrain vanilla --match-t2mlr
scripts/run_training.sh pretrain t2mlr
```

checkpointにはモデル種別、設定、重み、stage、epoch、step、seedを保存する。
`best.pt`はvalidation loss最小、`last.pt`は最終epochの状態である。
学習の既定値は最大50 epoch，validation lossが5 epoch連続で改善しなければ停止する
early stoppingである。改善判定の最小差は`1e-4`とする。`training_history.json`には
完了epoch数と`stop_reason`も保存される。確認用に短く実行する場合は，例えば次のように
上書きできる。

```bash
scripts/run_training.sh pretrain vanilla \
  --epochs 10 --early-stopping-patience 3
```

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

同一局面からの複数読み筋は既定でまとめて生成する．GPUメモリを抑える場合は
`LINE_BATCH_SIZE`で分割幅を指定できる．

```bash
LINE_BATCH_SIZE=1 scripts/run_cot_experiment.sh vanilla --epochs 1 --max-steps 10
```

`run_2x2_experiment.sh`は，既定では学習用・検証用traceの生成とCoT学習までを行い，
時間のかかる評価用trace生成，推論評価，probe評価を実行しない．評価は学習完了後に
別プロセスとして実行する．

実行ログは各traceの隣の`.log`と，実験全体の`run.log`に保存される．ログレベルと保存先は
環境変数で変更できる．

```bash
LOG_LEVEL=DEBUG LOG_FILE=results/debug.log scripts/run_2x2_experiment.sh
```

ログには設定値，モデル構成，対局ごとの処理時間，decision位置ごとのtrace数，累積速度，
最終summaryが記録される．

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
