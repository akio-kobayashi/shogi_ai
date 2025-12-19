#!/usr/bin/env bash
set -euo pipefail

# Floodgate CSA を日付範囲でダウンロードするスクリプト
# 使い方:
#   export DATA_ROOT=/path/to/data
#   ./download_floodgate_csa.sh 2010-01-01 2010-01-31

# ===== OS 判定 =====
OS="$(uname -s)"

date_add() {
  local base="$1"   # "YYYY-MM-DD"
  local delta="$2"  # いまのところ "+1d" だけ対応

  if [[ "$OS" == "Linux" ]]; then
    case "$delta" in
      +1d) date -d "$base +1 day" +%Y-%m-%d ;;
      -1d) date -d "$base -1 day" +%Y-%m-%d ;;
      *)   echo "Unsupported delta: $delta" >&2; exit 1 ;;
    esac

  elif [[ "$OS" == "Darwin" ]]; then
    case "$delta" in
      +1d) date -j -v+1d -f "%Y-%m-%d" "$base" +%Y-%m-%d ;;
      -1d) date -j -v-1d -f "%Y-%m-%d" "$base" +%Y-%m-%d ;;
      *)   echo "Unsupported delta: $delta" >&2; exit 1 ;;
    esac

  else
    echo "Unsupported OS: $OS" >&2
    exit 1
  fi
}

# ===== DATA_ROOT チェック =====
if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "ERROR: DATA_ROOT が未定義です．" >&2
  echo "例:  DATA_ROOT=/mnt/data/shogi $0 2010-01-01 2010-01-31" >&2
  exit 1
fi

# ===== 引数チェック =====
if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 START_DATE END_DATE" >&2
  echo "例:    $0 2010-01-01 2010-01-31" >&2
  exit 1
fi

START_DATE="$1"   # "YYYY-MM-DD"
END_DATE="$2"

BASE_URL="http://wdoor.c.u-tokyo.ac.jp/shogi/LATEST"
OUT_DIR="${DATA_ROOT}/floodgate/csa_raw"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

current="${START_DATE}"

# ===== 日付ループ =====
while :; do
  # current は "YYYY-MM-DD" 固定なので文字列だけで分解する（date は使わない）
  yyyy="${current%%-*}"
  rest="${current#*-}"
  mm="${rest%%-*}"
  dd="${current##*-}"

  url="${BASE_URL}/${yyyy}/${mm}/${dd}/"
  echo "=== 取得中 ${current} ==="
  echo "URL: ${url}"

  # NOP (サーバ負荷配慮)
  sleep 2

  wget -r -np -nH --cut-dirs=2 -A "*.csa" -nc --wait=1 --random-wait --limit-rate=500k "${url}" || {
    echo "[WARN] No CSA found for ${current}"
  }

  # 終了日のチェック（文字列比較でOK）
  if [[ "${current}" == "${END_DATE}" ]]; then
    break
  fi

  # 次の日へ（ここだけ date_add を使う）
  current="$(date_add "${current}" "+1d")"
done

echo "データ収集終了. CSAを以下のディレクトリに保存: ${OUT_DIR}"
