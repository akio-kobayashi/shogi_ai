0. Pythonのライブラリをインストール
  - sudo apt update
  - sudo apt install python3-dev
1. WSL2ターミナルで以下を実行
  - ./setup_env.sh
2. セットアップが終了したら，以下のコマンドで仮想環境を有効化
  - source .venv/bin/activate

## Windows側のエンジンを利用する場合

WSL2上のスクリプトからWindows側の将棋エンジン（例: やねうら王）を呼び出す場合、以下のようなバッチファイルを作成して `--engine-path` に指定してください。

### バッチファイルの例 (`run_engine.bat`)
エンジンの実行ファイルと同じディレクトリに作成します。

```batch
@echo off
REM Linux側 evaluate から呼び出す用

cd /d "%~dp0"

yaneuraou.exe %*
```

### 実行例
```bash
python src/create_dataset.py evaluate \
    --input-csv metadata.csv \
    --output-csv evaluated.csv \
    --engine-path /mnt/c/shogi/yaneuraou/run_engine.bat
```