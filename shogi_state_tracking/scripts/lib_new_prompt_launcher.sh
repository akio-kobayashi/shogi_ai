#!/usr/bin/env bash
# 新prompt学習launcherが共有するモデルサイズ・追加引数の解決処理。
# このファイルは実行せず，run_new_prompt*.shからsourceして使う。

new_prompt_extract_launcher_args() {
  NEW_PROMPT_CLI_MODEL_SIZE=""
  NEW_PROMPT_CLI_NUM_WORKERS=""
  NEW_PROMPT_EXTRA_ARGS=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-size)
        [[ $# -ge 2 ]] || { echo "--model-size requires small, base, or large" >&2; return 2; }
        new_prompt_set_cli_model_size "$2" || return 2
        shift 2
        ;;
      --model-size=*)
        new_prompt_set_cli_model_size "${1#--model-size=}" || return 2
        shift
        ;;
      --num-workers)
        [[ $# -ge 2 ]] || { echo "--num-workers requires a non-negative integer" >&2; return 2; }
        new_prompt_set_cli_num_workers "$2" || return 2
        shift 2
        ;;
      --num-workers=*)
        new_prompt_set_cli_num_workers "${1#--num-workers=}" || return 2
        shift
        ;;
      --model-type|--output-dir)
        echo "$1 is controlled by the launcher and must not be passed as an extra option" >&2
        return 2
        ;;
      --model-type=*|--output-dir=*)
        echo "${1%%=*} is controlled by the launcher and must not be passed as an extra option" >&2
        return 2
        ;;
      *)
        NEW_PROMPT_EXTRA_ARGS+=("$1")
        shift
        ;;
    esac
  done
  return 0
}


new_prompt_set_cli_model_size() {
  local candidate="$1"
  case "${candidate}" in
    small|base|large) ;;
    *) echo "--model-size must be small, base, or large: ${candidate}" >&2; return 2 ;;
  esac
  if [[ -n "${NEW_PROMPT_CLI_MODEL_SIZE}" && "${NEW_PROMPT_CLI_MODEL_SIZE}" != "${candidate}" ]]; then
    echo "--model-size was specified more than once with conflicting values" >&2
    return 2
  fi
  NEW_PROMPT_CLI_MODEL_SIZE="${candidate}"
}


new_prompt_set_cli_num_workers() {
  local candidate="$1"
  case "${candidate}" in
    ''|*[!0-9]*) echo "--num-workers must be a non-negative integer: ${candidate}" >&2; return 2 ;;
  esac
  if [[ -n "${NEW_PROMPT_CLI_NUM_WORKERS}" && "${NEW_PROMPT_CLI_NUM_WORKERS}" != "${candidate}" ]]; then
    echo "--num-workers was specified more than once with conflicting values" >&2
    return 2
  fi
  NEW_PROMPT_CLI_NUM_WORKERS="${candidate}"
}


new_prompt_resolve_single_model_size() {
  local default_size="$1"
  NEW_PROMPT_MODEL_SIZE="${NEW_PROMPT_CLI_MODEL_SIZE:-${SCALE_SIZES:-${MODEL_SIZE:-${default_size}}}}"
  case "${NEW_PROMPT_MODEL_SIZE}" in
    small|base|large) ;;
    *)
      echo "model size must be small, base, or large; use --model-size, SCALE_SIZES, or MODEL_SIZE" >&2
      return 2
      ;;
  esac
}


new_prompt_resolve_model_sizes() {
  local default_sizes="$1"
  local selected="${NEW_PROMPT_CLI_MODEL_SIZE:-${SCALE_SIZES:-${default_sizes}}}"
  IFS=',' read -r -a NEW_PROMPT_MODEL_SIZES <<< "${selected}"
  for new_prompt_size in "${NEW_PROMPT_MODEL_SIZES[@]}"; do
    case "${new_prompt_size}" in
      small|base|large) ;;
      *)
        echo "model sizes must contain only small, base, or large: ${new_prompt_size}" >&2
        return 2
        ;;
    esac
  done
}
