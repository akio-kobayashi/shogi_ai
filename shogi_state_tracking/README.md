# 将棋の指手系列から形成されるTransformer内部状態の分析

平手初期局面からの指手系列だけを自己回帰Transformerへ与え，盤面，持ち駒，手番などの将棋状態が隠れ表現からどの程度読み出せるかを調べる実験パッケージである．ゲーム規則，合法手マスク，途中局面はモデルへ与えない．

現在の主実験は，通常移動へ駒種トークンを確率的に挿入するRAP（Randomly Annotated Piece type）の有無を比較する．盤面情報を直接教師として与えるのではなく，弱い駒種注釈が指手予測と内部状態の形成へ与える影響を評価する．

学習目的は`factorized_action_mle_proportional_rap_v1`である．従来の指手単位正規化を維持し，RAP tokenのNLL和を同じ分子へ加えることで，RAPの寄与を実際の挿入数に比例させる．`q=0`の損失は修正前と同一なのでRAPなしモデルは再利用できるが，旧RAPモデルは再学習する．新RAPモデルの出力には`-proportional-rap-v1`を付け，旧checkpointと混在させない．token列とdataset schemaは変わらないので，現行factorized_v3 datasetの再生成は不要である．

## 現行仕様

- 実験名：`factorized_v3`
- artifact schema：version 4
- 入力：`<BOS> <MOVES> 指手系列`
- 開始局面：平手初期局面を暗黙の前提とし，トークンとしては入力しない
- 指手表現：移動元，成りフラグ，移動先を分解した可変長表現
- 語彙：125トークン
- BPE：使用しない
- `<EOM>`：使用しない
- `<EOS>`：引き分けを除く完全棋譜の終端だけで学習する
- 途中切断：`max_moves`または`max_seq_len`で切断した系列へ`<EOS>`を付けない
- 学習時の規則利用：合法手マスク，局面遷移器，エンジン評価を使用しない
- 評価時の規則利用：正解局面と合法性を算出するためにcshogiを使用する

`factorized_v3`という名称は指手文法を表す．現行datasetとcheckpointの互換性はschema version 4で管理する．旧schemaのdatasetとcheckpointは再利用できないため，混在させないこと．

詳しい設計根拠は[NEW_EXPERIMENT_DESIGN.md](NEW_EXPERIMENT_DESIGN.md)，プローブの定義は[PROBE_DESIGN.md](PROBE_DESIGN.md)を参照する．

## 実験の順序

1. RAPなし／ありの主比較
2. RAP挿入率のablation
3. 開始局面を変更する発展実験

第1・第2段階では平手初期局面を暗黙に固定する．任意の開始局面を明示する第3段階は，別dataset・別manifestとして扱い，主実験へ混在させない．

## 環境構築

実験対象はLinux／WSL2であり，Apple Siliconを含むmacOSは学習・評価環境に含めない．使用するcshogiはPyPI版ではなく，変更を加えたforkのcommit `c44708550d2dcd569179846a1ac35ad889d5ebb9`である．

```bash
cd /path/to/shogi_ai/shogi_state_tracking

# いずれか一つを選ぶ
./setup_env.sh cpu
./setup_env.sh cuda
./setup_env.sh rocm
```

backendを変更する場合は，同じ`.venv`を使い回さず再構築する．異なるPyTorch backendやcache配置が必要な場合は，例えば次のように指定する．

```bash
CUDA_BACKEND=cu128 ./setup_env.sh cuda
ROCM_BACKEND=rocm6.4 ./setup_env.sh rocm
UV_CACHE_DIR=/large-volume/uv-cache ./setup_env.sh cuda
```

## データセットの作成

### metadata.csvから全段階を構築する

CSA本体へアクセスできる計算機で実行する．`metadata.csv`のCSAパスと実ファイルの配置が異なる場合は，`create_dataset.py build`用のパス置換オプションを末尾へ渡す．

```bash
./scripts/setup_factorized_v3_data.sh \
  metadata.csv \
  factorized_v3_eos_data
```

このスクリプトは次を順に行う．

1. `create_dataset.py build`によるCSAの抽出・JSONL化
2. 中間new-prompt datasetの生成
3. factorized datasetへの変換
4. dataset全件検査

既に中間new-prompt datasetがある場合は，そのディレクトリを第1引数に指定できる．

```bash
./scripts/setup_factorized_v3_data.sh \
  existing_new_prompt_data \
  factorized_v3_eos_data
```

### 検査だけを再実行する

```bash
.venv/bin/python validate_factorized_v3_dataset.py \
  --dataset-dir factorized_v3_eos_data
```

検査では，schema，語彙，指手文法，RAP教師，局面別annotation，終端教師，非引き分け条件を確認する．

## 既定の抽出条件

- 2022年1月1日以降
- 先手・後手ともレーティング3000以上
- 80手以上
- 引き分けを除外
- 2022年1月～2024年9月：学習
- 2024年10月～12月：検証
- 2025年1月以降：評価
- 評価集合：最大5,000局

datasetには対局者名義の重複を表す`player_scope`と，学習集合での局面既出性を表す`position_scope_by_ply`を保存する．両者は別の概念である．

- `player_scope`：`open`，`mixed`，`closed`
- `position_scope_by_ply`：`seen_position`，`unseen_position`
- `trajectory_scope`：入力系列全体が未見かどうかを含む系列単位の分類

また，各ply直前の`legal_drop_available_by_ply`と`promotion_choice_available_by_ply`を保存する．これらは学習時の合法手マスクではなく，後段の診断プローブ用ラベルである．

## 主実験

Linuxでは，OSを巻き込むメモリ枯渇を避けるため`MEMORY_MAX`を必須とする．主実験は，同じdataset，語彙，モデル設定，seedを用いてRAPなしとRAPありだけを比較する．既定はLlama型decoderのbaseサイズ，RAP挿入率0.15である．

```bash
MEMORY_MAX=100G MEMORY_HIGH=90G \
  ./scripts/run_factorized_main_experiment.sh \
  factorized_v3_eos_data \
  factorized_v3_eos_results
```

主な変更例：

```bash
MODEL_TYPE=llama \
MODEL_SIZE=small \
BATCH_SIZE=16 \
SEEDS=20260802 \
RAP_PROBABILITY=0.15 \
MEMORY_MAX=100G MEMORY_HIGH=90G \
  ./scripts/run_factorized_main_experiment.sh \
  factorized_v3_eos_data \
  factorized_v3_eos_results
```

`MODEL_TYPE`は`llama`または`vanilla`，`MODEL_SIZE`は`small`，`base`，`large`を指定できる．学習が`best.pt`を生成しなかった場合，評価を開始せず異常終了する．

## RAP挿入率のablation

主比較の後に，必要な挿入率だけを追加実行する．

```bash
RAP_RATES=0.0,0.05,0.15,0.30,1.0 \
MODEL_TYPE=llama MODEL_SIZE=base \
MEMORY_MAX=100G MEMORY_HIGH=90G \
  ./scripts/run_factorized_rap_ablation.sh \
  factorized_v3_eos_data \
  factorized_v3_eos_results
```

既に`best.pt`がある条件は再学習しない．再学習する場合は`FORCE_TRAIN=1`を明示する．

## 評価

学習済みcheckpointだけを評価する場合は，次のスクリプトを使う．

```bash
MEMORY_MAX=100G MEMORY_HIGH=90G \
  ./scripts/run_factorized_evaluation.sh \
  factorized_v3_eos_results/llama-base/implicit-initial/vanilla-p0.0/seed-20260802/best.pt \
  factorized_v3_eos_data \
  factorized_v3_eos_data/vocab.json \
  factorized_v3_eos_results/manual-evaluation \
  main
```

第5引数で評価範囲を選ぶ．

| stage | 内容 |
|---|---|
| `moves` | 指手予測と合法性 |
| `token` | RAP token probe |
| `probes` | 盤面・持ち駒・手番などの線形状態プローブ |
| `action-probes` | 指手構成要素の条件付き線形プローブ |
| `main` | `moves`，`token`，`probes`，終端プローブ |
| `all` | 上記すべて |

主な評価指標は次のとおりである．

- 指手cross entropy，raw／canonical／文法正規化perplexity，top-1／top-5
- top-1合法率，top-5内合法手率，合法手への確率質量
- 盤面81マスの正解率，occupied盤面精度，駒種別指標，盤面完全一致率
- 持ち駒のslot精度，非零持ち駒精度，MAE，完全一致率
- 手番，王手，駒打ち可能性
- 指手種別，移動元，移動先，成り選択，駒打ち駒種，駒打ち先
- 完全棋譜終端の線形復号精度

盤面・持ち駒・手番の主状態プローブでは，次手の最初のトークン直前にある`h_pre`と，その次手を指す直前の局面ラベルを対応させる．RAPなしの完全prefixを入力し，ply 0を主集計から除外する．prefix入力とfull-sequence入力の隠れ状態が一致するかも自動検査し，因果マスクによる未来情報遮断を確認する．

条件付き指手プローブは，内部状態の証拠ではなく，指手の各構成要素が層ごとにどの程度線形復号可能かを調べる補助分析として扱う．

## 出力

1条件の既定出力は次の構成になる．

```text
RESULTS_DIR/
  llama-base/
    implicit-initial/
      vanilla-p0.0/
        seed-20260802/
          best.pt
          last.pt
          training_history.json
          run_manifest.json
          train.log
          tensorboard/
          evaluation/
            move_metrics.json
            probes/
            terminal-probe/
      rap-p0.15/
        seed-20260802/
          ...
```

TensorBoardは次のように起動する．

```bash
./scripts/launch_tensorboard.sh factorized_v3_eos_results
```

## 分析結果の転送

すべての学習・評価が終了したことを確認した後，独立した後処理として，数値結果，
学習履歴，run manifest，ログ，dataset manifestを一つの
archiveへまとめられる．datasetのJSONLと`best.pt`／`last.pt`は含めないため，実験環境
そのものを移行するpackageより小さい．テキスト中の絶対pathとhome名も置換する．

```bash
./scripts/package_analysis_results.sh \
  factorized_v3_eos_results \
  factorized_v3_analysis.tar.gz \
  --dataset-dir factorized_v3_eos_data
```

この`factorized_v3_analysis.tar.gz`を転送すれば，RAP条件間の指手指標，状態probe，
指手probe，終端probe，学習曲線を分析できる．盤面ヒートマップを別計算機で再生成する
ためにprobe重みや予測tensorも必要な場合だけ，次を追加する．

```bash
./scripts/package_analysis_results.sh \
  factorized_v3_eos_results \
  factorized_v3_analysis_with_probes.tar.gz \
  --dataset-dir factorized_v3_eos_data \
  --include-probe-artifacts
```

TensorBoard eventを含める場合は`--include-tensorboard`，ログを除く場合は`--no-logs`を
指定する．archive内の`COLLECTION_MANIFEST.json`には，収録ファイルのSHA-256，実行code
のcommit，見つからなかった任意評価結果を記録する．

## モデル

現行主実験では次のdecoderを選択できる．

- `llama`：RMSNorm，RoPE，SwiGLUを用いるLlama型causal decoder
- `vanilla`：標準的なpre-norm causal Transformer decoder

どちらも同じtokenizer，dataset，学習損失，評価器を使用する．T²MLR，旧96トークン局面prompt，Chain of Moves／synthetic reasoning traceは過去・将来実験としてコードを残しているが，現行の主比較には含めない．

## 再現性と互換性

- datasetとcheckpointは`dataset_manifest.json`，語彙hash，schema versionで照合する．
- 現行の完全棋譜終端学習には`terminal_encoding = eos_on_complete_decisive_game_v1`が必要である．
- 旧datasetを新checkpointへ流用せず，仕様変更後はdatasetを最初から再作成する．
- checkpointの出力パスは`model_type`，`model_size`，入力条件，RAP率，seedを含む．
- 学習・検証・評価は対局単位で分離する．
- 実験中のdatasetやcheckpointを上書きせず，新しいrootディレクトリを用いる．

## 関連文書

- [NEW_EXPERIMENT_DESIGN.md](NEW_EXPERIMENT_DESIGN.md)：現行の語彙・指手文法・実験条件
- [PROBE_DESIGN.md](PROBE_DESIGN.md)：状態・指手・終端プローブの定義
- [PAPER_OUTLINE_JA.md](PAPER_OUTLINE_JA.md)：論文構成案
- [FUTURE_EXPERIMENTS.md](FUTURE_EXPERIMENTS.md)：因果介入，Chain of Moves，探索蒸留などの将来課題
- [COT_EXPERIMENT.md](COT_EXPERIMENT.md)：旧synthetic reasoning trace実験
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)：移植コードと第三者ライセンス

## 注意

本研究で線形プローブの精度が高いことは，盤面や持ち駒の情報が隠れ表現から線形に読み出せることを示す．それだけで，Transformerが明示的な盤面データ構造を保持していること，その表現を因果的に利用して指手を決めていること，あるいはゲーム規則を完全に獲得したことまでは示さない．
