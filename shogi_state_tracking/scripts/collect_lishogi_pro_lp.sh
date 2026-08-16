#!/usr/bin/env bash
# 公開対局メタデータから利用者を発見し，確認済み利用者の棋譜だけを本取得する．
# USER_SCOPE=titled（既定）はPRO／LP，USER_SCOPE=non-botは現在の探索結果に
# 含まれる確認済み非BOT利用者を対象にする．
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-${SCRIPT_DIR}/data/lishogi-pro-lp}"
SEED_USERS_FILE="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
DISCOVERY_DIR="${OUTPUT_ROOT}/discovery"
GAMES_DIR="${COLLECTION_OUTPUT_DIR:-${OUTPUT_ROOT}/games}"

discovery_args=(
  --output-dir "${DISCOVERY_DIR}"
  # 公式API文書に掲載された公開タイトル保持アカウントをbootstrapに使う．
  # これらを全利用者一覧とは扱わず，実行時にプロフィールを再検証する．
  --seed-user YokoyamaTomoki
  --seed-user Shogi_Harbour
  --leaderboard-size "${LEADERBOARD_SIZE:-200}"
  --max-discovered-users "${MAX_DISCOVERED_USERS:-10000}"
  --max-users-this-run "${MAX_USERS_THIS_RUN:-500}"
  --max-profile-users-this-run "${MAX_PROFILE_USERS_THIS_RUN:-3000}"
  --max-games-per-user "${DISCOVERY_GAMES_PER_USER:-50}"
  --since "${SINCE:-2022-01-01T00:00:00Z}"
  --request-delay "${REQUEST_DELAY:-0.10}"
)
if [[ -n "${SEED_USERS_FILE}" ]]; then
  [[ -f "${SEED_USERS_FILE}" ]] || { echo "seed user list does not exist: ${SEED_USERS_FILE}" >&2; exit 2; }
  discovery_args+=(--seed-users-file "${SEED_USERS_FILE}")
fi
if [[ "${REFRESH_PROFILES:-0}" == 1 ]]; then
  discovery_args+=(--refresh-profiles)
fi
discovery_args+=(--profile-cache-ttl-hours "${PROFILE_CACHE_TTL_HOURS:-24}")
if [[ -n "${CA_FILE:-}" ]]; then
  discovery_args+=(--ca-file "${CA_FILE}")
fi
if [[ -n "${UNTIL:-}" ]]; then
  discovery_args+=(--until "${UNTIL}")
fi
if [[ "${OVERWRITE_DISCOVERY:-0}" == 1 ]]; then
  discovery_args+=(--overwrite)
fi

if [[ "${SKIP_DISCOVERY:-0}" != 1 ]]; then
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/discover_lishogi_titled_users.py" "${discovery_args[@]}"
else
  echo "discovery skipped: using existing discovery cache in ${DISCOVERY_DIR}" >&2
fi

if [[ "${DISCOVERY_ONLY:-0}" == 1 ]]; then
  echo "discovery-only run complete: ${DISCOVERY_DIR}/manifest.json"
  exit 0
fi

USER_SCOPE="${USER_SCOPE:-titled}"
case "${USER_SCOPE}" in
  titled)
    USERS_FILE="${DISCOVERY_DIR}/titled_users.txt"
    [[ -s "${USERS_FILE}" ]] || {
      echo "No verified PRO/LP users were found in this bounded crawl." >&2
      echo "Resume with a larger MAX_USERS_THIS_RUN or provide a seed-user file as argument 2." >&2
      exit 3
    }
    REQUIRE_TITLE_FILTER=1
    ;;
  non-bot|non_bot)
    USERS_FILE="${DISCOVERY_DIR}/non_bot_users.txt"
    [[ -s "${USERS_FILE}" ]] || {
      echo "No verified non-BOT users were found in the discovery cache." >&2
      echo "Run discovery first, then resume with USER_SCOPE=non-bot." >&2
      exit 3
    }
    REQUIRE_TITLE_FILTER=0
    ;;
  *)
    echo "USER_SCOPE must be titled or non-bot, got: ${USER_SCOPE}" >&2
    exit 2
    ;;
esac

collection_args=(
  --users-file "${USERS_FILE}"
  --output-dir "${GAMES_DIR}"
  --max-games-per-user "${COLLECTION_GAMES_PER_USER:-1000}"
  --min-plies "${MIN_PLIES:-80}"
  --since "${SINCE:-2022-01-01T00:00:00Z}"
  --request-delay "${REQUEST_DELAY:-0.10}"
)
if [[ "${REQUIRE_TITLE_FILTER}" == 1 ]]; then
  collection_args+=(--required-user-title PRO --required-user-title LP)
fi
if [[ "${APPEND_NEW_GAMES:-0}" == 1 ]]; then
  # 追記モードではTARGET_GAMESを指定した既存の実行例も自然に解釈できる
  # ようにする．NEW_GAMES_PER_RUNが明示された場合はこちらを優先する．
  collection_args+=(--target-new-games "${NEW_GAMES_PER_RUN:-${TARGET_GAMES:-100}}")
else
  collection_args+=(--target-games "${TARGET_GAMES:-1000}")
fi
if [[ "${INCLUDE_DRAWS:-0}" == 1 ]]; then
  collection_args+=(--include-draws)
fi
if [[ "${KEEP_RAW:-0}" == 1 ]]; then
  collection_args+=(--keep-raw)
fi
if [[ "${OVERWRITE_GAMES:-0}" == 1 ]]; then
  collection_args+=(--overwrite)
fi
if [[ "${FULL_RESCAN:-0}" == 1 ]]; then
  collection_args+=(--full-rescan)
fi
if [[ -n "${CA_FILE:-}" ]]; then
  collection_args+=(--ca-file "${CA_FILE}")
fi
if [[ -n "${MIN_RATING:-}" ]]; then
  collection_args+=(--min-rating "${MIN_RATING}")
fi
if [[ -n "${MAX_RATING:-}" ]]; then
  collection_args+=(--max-rating "${MAX_RATING}")
fi
if [[ -n "${UNTIL:-}" ]]; then
  collection_args+=(--until "${UNTIL}")
fi

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/collect_lishogi_games.py" "${collection_args[@]}"

echo "discovery manifest: ${DISCOVERY_DIR}/manifest.json"
echo "user scope: ${USER_SCOPE}"
echo "verified user list: ${USERS_FILE}"
echo "collected games: ${GAMES_DIR}/games.jsonl"
