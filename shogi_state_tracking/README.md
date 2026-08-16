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

### LishogiのPRO／LP棋譜を評価用に収集する

Lishogiには全PRO／LP利用者を列挙する公開APIがない．そのため，収集を次の二段階に分ける．

1. 上位利用者又は任意の起点から公開対局メタデータ上の対局者グラフを幅優先に走査する．発見した利用者の公開プロフィールを最大300名ずつ照合し，現在のタイトルが`PRO`又は`LP`である利用者を抽出する．この段階では指手を取得しない．実行シェルは公式API文書に掲載された`YokoyamaTomoki`と`Shogi_Harbour`も探索のbootstrapに使うが，この2名を完全一覧とは扱わない．
2. 確認済みPRO／LPを起点として棋譜を本取得する．本取得時にも起点利用者の現在のタイトルを再検証する．

一連の実行には次を使う．既定では2022年1月以降を対象とし，探索は中断後に同じコマンドで再開できる．

```bash
MAX_USERS_THIS_RUN=500 \
TARGET_GAMES=1000 \
./scripts/collect_lishogi_pro_lp.sh data/lishogi-pro-lp
```

既知の公開利用者を探索起点へ追加する場合は，1行1名のファイルを第2引数へ指定する．空行と`#`以後は無視される．

```text
# lishogi_users.txt
user_name_1
user_name_2
```

```bash
./scripts/collect_lishogi_pro_lp.sh \
  data/lishogi-pro-lp \
  lishogi_users.txt
```

探索範囲は`MAX_DISCOVERED_USERS`，1回に走査する人数は`MAX_USERS_THIS_RUN`，利用者当たりのメタデータ件数は`DISCOVERY_GAMES_PER_USER`で制限する．例えば，探索を段階的に広げるには同じ出力先に対して`MAX_USERS_THIS_RUN=2000`として再実行する．この方法は公開対局から到達できる利用者だけを対象とするため，PRO／LPの完全一覧を保証しない．`discovery/manifest.json`にもこの制約を記録する．

`MIN_RATING`，`MAX_RATING`，`UNTIL`を指定すると，本取得の抽出条件へも引き継がれる．

サーバー負荷を抑えた試験では，例えば`REQUEST_DELAY=1.0 MAX_USERS_THIS_RUN=10 DISCOVERY_GAMES_PER_USER=5 MAX_PROFILE_USERS_THIS_RUN=100 TARGET_GAMES=10 COLLECTION_GAMES_PER_USER=20`とする．プロフィールは既定で24時間のキャッシュを使う．即時再照合が必要な場合だけ`REFRESH_PROFILES=1`を指定する．

PRO／LPが少ない場合は，タイトル保持者だけに限定せず，現在の探索結果から確認済み非BOT利用者へ対象を広げられる．探索時には`discovery/non_bot_users.txt`も生成される．既存の`profile_cache.json`だけから一覧を作り直す場合は，ネットワークアクセスなしで次を実行する．

```bash
./.venv/bin/python build_non_bot_user_list.py \
  data/lishogi-pro-lp/discovery
```

その後，`USER_SCOPE=non-bot`を指定すると，PRO／LPのタイトル条件を外し，プロフィール確認済みの非BOT利用者を収集対象にする．

```bash
SKIP_DISCOVERY=1 \
USER_SCOPE=non-bot \
COLLECTION_OUTPUT_DIR=data/lishogi-non-bot/games \
APPEND_NEW_GAMES=1 \
REQUEST_DELAY=2.0 \
TARGET_GAMES=50 \
./scripts/collect_lishogi_pro_lp.sh data/lishogi-pro-lp-light
```

この一覧は「人間であることの完全な証明」ではなく，`BOT`タイトル，無効アカウント，プロフィール未照合のプレースホルダーを除いた「非BOT登録利用者」の近似である．PRO／LP棋譜は学習データへ混ぜず，外部評価用サブセットとして保持する．

収集済みの非BOT棋譜をfactorized_v3の評価形式へ変換する場合は，500局をそのまま`evaluation.jsonl`へ変換する．このスクリプトは学習・検証分割を作らず，評価データだけを生成する．

```bash
./.venv/bin/python build_lishogi_factorized_evaluation.py \
  --input-jsonl data/lishogi-non-bot/games/games.jsonl \
  --output-dir data/lishogi-non-bot-factorized-eval \
  --max-games 500 \
  --min-plies 80 \
  --overwrite
```

出力には`evaluation.jsonl`，`vocab.json`，`dataset_manifest.json`が含まれる．これは評価専用であり，`train.jsonl`や`validation.jsonl`の代わりに使用してはならない．指手評価は次のように直接実行する．

```bash
${PYTHON_BIN:-./.venv/bin/python} evaluate_factorized_moves.py \
  --checkpoint CHECKPOINT \
  --evaluation-jsonl data/lishogi-non-bot-factorized-eval/evaluation.jsonl \
  --vocab data/lishogi-non-bot-factorized-eval/vocab.json \
  --output RESULTS/lishogi-non-bot/move_metrics.json \
  --max-games 500
```

線形プローブを評価する場合は，プローブの学習には元の機械棋譜`train.jsonl`／`validation.jsonl`を使い，評価部分だけをこの非BOT`evaluation.jsonl`へ差し替える．非BOT棋譜をプローブ学習側へ混ぜない．

referenceモデルに対する評価専用シェルも用意している．指手評価は評価データだけを受け取り，線形プローブ評価は機械棋譜データと非BOT評価データを別々に受け取る．

```bash
MEMORY_MAX=100G MEMORY_HIGH=90G \
  ./scripts/run_reference_lishogi_move_evaluation.sh \
  factorized_v3_reference_results/llama-reference/implicit-initial/vanilla-p0.0/seed-20260802/best.pt \
  data/lishogi-non-bot-factorized-eval \
  factorized_v3_reference_results/llama-reference/implicit-initial/vanilla-p0.0/seed-20260802/evaluation/lishogi-non-bot/moves

MEMORY_MAX=100G MEMORY_HIGH=90G \
  ./scripts/run_reference_lishogi_linear_probe_evaluation.sh \
  factorized_v3_reference_results/llama-reference/implicit-initial/vanilla-p0.0/seed-20260802/best.pt \
  factorized_v3_eos_data \
  data/lishogi-non-bot-factorized-eval \
  factorized_v3_reference_results/llama-reference/implicit-initial/vanilla-p0.0/seed-20260802/evaluation/lishogi-non-bot/linear-probes
```

線形プローブシェルは，機械棋譜のtrain／validationでprobeを学習し，非BOT棋譜でのみ評価する．

本取得では，レーティング対象，標準将棋，リアルタイム，平手初期局面，80 ply以上，決着局に限定する．匿名対局，組込みAI，棋譜又は現在の公開プロフィールが`BOT`である対局者を除外する．API tokenを使う場合は`LISHOGI_TOKEN`環境変数へ設定する．

Python環境がOSのCA bundleを自動検出しない場合だけ，例えば`CA_FILE=/etc/ssl/cert.pem`を指定する．TLS検証を無効化するオプションは設けていない．

探索ディレクトリには，取得に必要な公開ユーザ名を含む`discovery_state.json`，`profile_cache.json`，`titled_users.txt`が置かれるため，公開用の分析パッケージには含めない．本取得の`games/games.jsonl`では利用者IDを固定salt付きSHA-256で仮名化し，公開タイトルだけを保存する．`KEEP_RAW=1`を指定した場合だけ公開ユーザ名を含むAPI応答も保存するため，通常は指定しない．`data/`はGit管理対象外である．

同一条件で既存出力へ再実行した場合，`games/manifest.json`を照合し，条件が一致すれば利用者ごとの最新棋譜時刻以降だけを取得する．条件を変更した場合は安全のため全再走査へ戻る．強制的に全範囲を取り直す場合は`FULL_RESCAN=1`を指定する．

既存`games.jsonl`へ今回分を追記する運用では，`APPEND_NEW_GAMES=1 NEW_GAMES_PER_RUN=100`を指定する．この場合，既存局をゲームIDで重複除去しながら，今回新たに採用できた局を最大100局追記する．追記モードでは`NEW_GAMES_PER_RUN`が優先され，未指定なら`TARGET_GAMES`を今回追記局数として扱う．通常モードの`TARGET_GAMES`は累積総数の上限である．

探索キューを進めず既存のタイトル一覧から棋譜だけを増分取得する場合は`SKIP_DISCOVERY=1`を指定する．探索だけ実行する場合は`DISCOVERY_ONLY=1`を指定する．

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
| `chess` | チェス先行研究に対応するStart／End token評価 |
| `probes` | 盤面・持ち駒・手番などの線形状態プローブ |
| `action-probes` | 指手構成要素の条件付き線形プローブ |
| `main` | `moves`，`token`，`probes`，終端プローブ |
| `all` | 上記すべて |

### チェス先行研究対応の比較評価

ToshniwalらのStart／End課題と比較する評価だけを，既存checkpointへ独立して
追加実行できる．再学習およびdataset再生成は不要である．

```bash
MEMORY_MAX=100G MEMORY_HIGH=90G \
  ./scripts/run_factorized_chess_protocol.sh \
  factorized_v3_eos_results/llama-small/implicit-initial/vanilla-p0.0/seed-20260802/best.pt \
  factorized_v3_eos_data \
  factorized_v3_eos_data/vocab.json
```

既定では，平手初期局面から51～100 plyの履歴を持ち，非歩の盤上駒を動かす
非成り指手のうち，その移動元に合法な成り分岐がない事例を全評価棋譜から
seed固定で1,000件抽出する．Start-Actual／Start-Otherおよび
End-Actual／End-Otherについて，チェス論文と同じExM，LgM accuracy，
LgM R-Precisionを算出する．順位は81座標だけに制限せず，全125語彙上で求める．
出力はcheckpoint配下の`evaluation/chess-protocol/chess_protocol_metrics.json`である．

同じJSONには，合法集合サイズの平均・中央値・25／75／90分位点と，
一様81マス，一様全語彙，合法集合内一様選択の偶然水準も記録する．さらに，
合法集合サイズ（1，2～3，4～7，8以上），駒種，大駒・小駒・玉の区分ごとに
LgM，ExMおよびR-Precisionを集計する．Start課題の合法集合は移動元集合，
End課題の合法集合は移動先集合であり，両者を同じ意味には解釈しない．
End課題にはさらに，正解駒種・移動元・手番だけを用いる幾何学的移動集合
`G`と，正解盤面から自駒衝突・飛び越し・強制成りを除いた疑似合法集合`P`を
評価器内で構成する．`L ⊆ P ⊆ G`を検査し，一様選択時のLgM基準
`E[|L|/|G|]`，`E[|L|/|P|]`とExM基準`E[1/|G|]`，`E[1/|P|]`を記録する．
これらはモデルに駒種や盤面を追加入力する条件ではなく，評価器だけが正解状態を
参照するoracle baselineである．

これは将棋への対応評価であり，盤面サイズと規則そのものはチェスと異なる．また，
VanillaモデルのStart課題では駒種tokenが分布外入力となるため，RAPとの公平な
比較にはEnd課題を用い，Start課題はRAPが獲得した状態追跡の診断として扱う．

主な評価指標は次のとおりである．

- 指手cross entropy，raw／canonical／文法正規化perplexity，top-1／top-5
- top-1合法率，top-5内合法手率，合法手への確率質量

`move_metrics.json`の`complete_move_evaluation`は，teacher forcingによる構成要素
評価から完全指手評価を分離する．通常移動は，正解局面の移動元にある駒種，
移動元，成り・不成り，移動先の組として定義し，駒打ちは`DROP`，駒種，打ち先の
組として定義する．モデルには正解構成要素を与えず，文法制約付きbeamから得た
完全USI指手について，本譜との完全一致とcshogi合法手集合への所属を判定する．
全体に加え，通常移動・成り・駒打ち別，駒種別，大駒・小駒・玉別に集計する．
beam幅は5であるため，top-1／top-5は全指手空間の厳密な大域順位ではない．

同じ`moves`段階で`distribution_baselines.json`も生成する．評価対象局面を正規化
SFEN（盤面・持ち駒・手番）で同定し，学習集合を再生して同一局面に続く指手を
集計する．局面別多数派指手のcoverageと一致率，学習中の局面出現回数，異なる
次手数，最頻指手率，指手エントロピーを報告する．さらに，履歴長，seen／unseen，
strict-unseenを含むtrajectory scope，学習局面頻度0，1，2～4，5～9，10回以上で
層別化する．評価集合内の局面別多数派率も分布の記述値として出力するが，これは
評価ラベル自身を使うため予測baselineとは扱わない．
分布baselineはモデルに依存しないため，dataset manifestと評価条件をキーとして
`DATASET_DIR/evaluation-cache/`へ保存し，同じデータセットを評価するVanilla，
RAP，APで共有する．再計算する場合は`FORCE_DISTRIBUTION_BASELINE=1`を指定する．
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

### 複数のresults rootを統合する場合

標準評価とLishogi評価を別々のresults rootへ出力した場合は，単一rootを指定する
`package_analysis_results.sh`ではなく，次の統合収集スクリプトを使う．vanilla／RAP／AP
の各条件について，standardと`lishogi-non-bot`の両方の
`move_metrics.json`・`probe_metrics.json`が揃っていなければ既定で停止する．

```bash
RESULTS_PARENT_1=/path/to/results-parent-on-the-execution-machine
RESULTS_PARENT_2=/another/path/containing/results
DATASET_DIR=/path/to/dataset

./scripts/collect_factorized_analysis.sh \
  /path/to/factorized_v3_analysis_reference_full.tar.gz \
  --scan-root "${RESULTS_PARENT_1}" \
  --scan-root "${RESULTS_PARENT_2}" \
  --dataset-dir "${DATASET_DIR}" \
  --force
```

Lishogiだけを確認用に収集する場合は，必須データセットを明示的に限定する．

```bash
./scripts/collect_factorized_analysis.sh \
  /path/to/factorized_v3_lishogi_only.tar.gz \
  --scan-root /path/to/lishogi-results-parent \
  --dataset-dir /path/to/lishogi-factorized-eval \
  --expected-dataset lishogi-non-bot \
  --allow-incomplete \
  --force
```

作成後は，必ずarchiveのcoverageを確認する．

```bash
tar -xOzf factorized_v3_analysis_reference_full.tar.gz \
  analysis_bundle/COLLECTION_MANIFEST.json \
  | jq '{expected_matrix,observed_matrix,missing_matrix}'
```

`collect_factorized_analysis.py`は入力rootを読み取るだけで，元のresultsやcheckpointを
削除しない．checkpointとゲームJSONLもarchiveには含めない．指定したrootで標準評価が
見つからない場合，root名に`result`または`analysis`を含むときは親ディレクトリの兄弟
results rootを自動探索する．自動探索を無効にする場合は
`--no-auto-sibling-discovery`を指定する．

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
