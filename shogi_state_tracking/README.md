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

`MODEL_TYPE`は`llama`または`vanilla`，`MODEL_SIZE`は`small`，`base`，`large`，`reference`を指定できる．`reference`はToshniwal et al. (2021)のGPT-2 small型に深さとhidden sizeを合わせた12層・幅768・12 headの構成である．125語彙ではLlama型が約85.0M，Vanilla型が最大系列長2560の学習可能な位置埋め込みを含めて約87.0Mとなる．学習が`best.pt`を生成しなかった場合，評価を開始せず異常終了する．

先行研究対応の主実験は次のように実行する．`--model-size reference`は，主実験，RAP挿入率ablation，AP，個別baselineのすべてで同じ出力ラベル`llama-reference`へ伝播する．

```bash
MEMORY_MAX=100G MEMORY_HIGH=90G \
  ./scripts/run_factorized_main_experiment.sh \
  factorized_v3_eos_data factorized_v3_reference_results \
  --model-type llama --model-size reference

MEMORY_MAX=100G MEMORY_HIGH=90G \
  ./scripts/run_factorized_ap_experiment.sh \
  factorized_v3_eos_data factorized_v3_reference_results \
  --model-type llama --model-size reference
```

評価だけを再実行する場合は，サイズ名を別途指定する必要はない．checkpoint内の`model_config`からreference構成を復元する．

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

`q=1.0`はAP（Always Piece Type）条件として扱い，結果を`ap-p1.0-proportional-annotation-v1`へ保存する．APはRAPと異なり，学習時だけでなく評価時の履歴にも全通常移動の駒種を与えるoracle条件である．評価対象となる現在指手についても正解駒種をpromptとして与え，その後の指手subtokenを評価する．APだけを追加実行する場合は次を用いる．

```bash
MEMORY_MAX=100G MEMORY_HIGH=90G \
  ./scripts/run_factorized_ap_experiment.sh \
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

### 持ち駒遷移・駒打ちの独立評価

学習済みcheckpointに対し，既存の総合評価とは独立して線形状態プローブを学習し，
駒取りによる持ち駒の増加，駒打ちによる減少，および駒打ちの正当性を評価できる．

```bash
MEMORY_MAX=100G MEMORY_HIGH=90G \
  ./scripts/run_factorized_hand_evaluation.sh \
  factorized_v3_eos_results/llama-small/implicit-initial/vanilla-p0.0/seed-20260802/best.pt \
  factorized_v3_eos_data \
  factorized_v3_eos_data/vocab.json
```

出力先はcheckpointと同じrunにある`evaluation/hand-evaluation`である．このシェルは
`linear-probes/linear_probes.pt`を自分で作成してから，`hand_dynamics_metrics.json`を
出力する．第4引数を与えた場合だけ任意の出力先へ変更できる．既存probeを再利用する場合だけ，
`REUSE_LINEAR_PROBES=1`を指定する．主な指標は次のとおりである．

- 駒取り・駒打ち前後の持ち駒完全一致率
- 変化した持ち駒slotの前後正解率と増減正解率
- 関係しない13 slotを変更しなかった率
- 14 slot全体の差分完全一致率
- `<DROP>`を条件とした保有駒への確率質量
- 選択駒種を実際に持っている率
- 選択駒種を合法地点へ打つ確率質量とtop-1完全合法率

駒打ち行動の評価対象は，正解次手が駒打ちである局面に限定する．これによりAPでも，
現在指手の正解駒種tokenが入力へ先に現れることを避ける．cshogiは教師ラベルと合法性の
判定にだけ使用し，モデル入力や学習には使用しない．APは過去の通常移動へ駒種注釈を持つ
oracle条件なので，RAPなし・RAPとの公平な無注釈比較ではないことに注意する．

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
          hand-evaluation/
            hand_dynamics_metrics.json
            linear-probes/
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
指手probe，終端probe，持ち駒遷移・駒打ち評価，学習曲線を分析できる．
持ち駒評価は各runの`evaluation/hand-evaluation/hand_dynamics_metrics.json`から収集される．
収集元は`COLLECTION_MANIFEST.json`の`result_locations`で確認できる．
盤面ヒートマップを別計算機で再生成する
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
