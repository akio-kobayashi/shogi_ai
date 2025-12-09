#!/bin/bash
# WSL2環境でuvを使用してPython仮想環境をセットアップするスクリプト

# エラーが発生した場合はスクリプトを即座に停止
set -e

# --- 1. uvのインストール ---
# Astralの公式インストールスクリプトを使用してuvをインストール
echo "🚀 Installing uv..."
if ! command -v uv &> /dev/null
then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # インストール後、PATHを有効にするためにシェル設定を再読み込み
    source "$HOME/.cargo/env"
    echo "✅ uv has been installed."
else
    echo "✅ uv is already installed."
fi


# --- 2. 仮想環境の作成 ---
VENV_DIR=".venv"
echo "🐍 Creating virtual environment in '$VENV_DIR'..."
if [ ! -d "$VENV_DIR" ]; then
    uv venv
    echo "✅ Virtual environment created."
else
    echo "✅ Virtual environment already exists."
fi


# --- 3. 依存関係のインストール ---
echo "📦 Installing dependencies from requirements.txt..."
# 仮想環境を有効化してからインストール
source "$VENV_DIR/bin/activate"
uv pip install -r requirements.txt
deactivate


# --- 完了 ---
echo "🎉 Setup complete!"
echo "To activate the virtual environment, run the following command:"
echo "source $VENV_DIR/bin/activate"
