#!/usr/bin/env python3
"""評価成果物が，現在の評価器の版で作られたかを判定する。

各評価器は `EVALUATOR_VERSION` を持ち，成果物の `provenance.evaluator_version` に記録する。
版番号は，指標の定義や計算方法が変わり，既存の成果物と意味が変わるときだけ上げる。

ファイルの有無だけで作り直しを判断すると，評価器を変えた後も古い成果物が残る。
逆に git の commit で判断すると，値が変わらない変更（名前の変更など）でも
全成果物を作り直すことになる。版番号はその中間で，作り直しが必要な変更だけを拾う。

版番号を導入する前の成果物は記録を持たないので，版1とみなす。

    python artifact_versions.py check <成果物のパス>
        現在の版なら終了コード0，古ければ1，書き手が未登録なら2，読めなければ3
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT = Path(__file__).resolve().parent
# 版番号を記録する前の成果物の版。
UNVERSIONED = 1

# 成果物のファイル名 -> それを書く評価器。同じ名前を複数の場所で使う場合も書き手は同じ。
WRITERS: dict[str, str] = {
    "move_metrics.json": "evaluate_factorized_moves.py",
    "distribution_baselines.json": "evaluate_factorized_distribution_baselines.py",
    "full_history_move_metrics.json": "evaluate_factorized_full_history_moves.py",
    "token_probe_metrics.json": "evaluate_factorized_token_probe.py",
    "probe_metrics.json": "evaluate_new_prompt_probes.py",
    "action_probe_metrics.json": "evaluate_factorized_action_probes.py",
    "chess_protocol_metrics.json": "evaluate_factorized_chess_protocol.py",
    "hand_dynamics_metrics.json": "evaluate_factorized_hand_dynamics.py",
    "policy_relevance_metrics.json": "evaluate_factorized_policy_relevance.py",
    "confidence_trajectory.json": "evaluate_factorized_drop_relevance.py",
    "attention_metrics.json": "evaluate_factorized_drop_attention.py",
    "action_condition_attention_ablation.json": "evaluate_factorized_drop_attention.py",
    "action_condition_metrics.json": "evaluate_factorized_action_condition.py",
    "action_condition_robustness.json": "evaluate_factorized_action_condition_robustness.py",
}

VERSION_PATTERN = re.compile(r"^EVALUATOR_VERSION\s*=\s*(\d+)\s*$", re.M)


def current_version(script: str) -> int:
    """評価器のソースから版番号を読む。importすると torch が要るので読むだけにする。"""
    match = VERSION_PATTERN.search((PROJECT / script).read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"{script} does not define EVALUATOR_VERSION")
    return int(match.group(1))


def recorded_version(payload: Mapping[str, Any]) -> int:
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping) and isinstance(provenance.get("evaluator_version"), int):
        return int(provenance["evaluator_version"])
    return UNVERSIONED


def status(path: Path) -> tuple[str, str]:
    """(current|stale|unregistered|unreadable, 説明) を返す。"""
    writer = WRITERS.get(path.name)
    if writer is None:
        return "unregistered", f"no writer is registered for {path.name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return "unreadable", f"cannot read {path}: {error}"
    have, want = recorded_version(payload), current_version(writer)
    if have < want:
        return "stale", f"{path.name} is evaluator version {have}; {writer} is now version {want}"
    return "current", f"{path.name} is evaluator version {have}"


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] != "check":
        print(__doc__, file=sys.stderr)
        return 2
    state, message = status(Path(argv[2]))
    print(message, file=sys.stderr)
    return {"current": 0, "stale": 1, "unregistered": 2, "unreadable": 3}[state]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
