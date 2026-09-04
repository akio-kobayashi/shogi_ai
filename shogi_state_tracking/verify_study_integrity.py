#!/usr/bin/env python3
"""Validate a factorized-v3 study before aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


CONDITIONS = (
    "vanilla-p0.0",
    "rap-p0.15-proportional-rap-v1",
    "rap-p0.25-proportional-rap-v1",
    "ap-p1.0-proportional-annotation-v1",
)
PRIMARY_CONDITIONS = CONDITIONS[:3]
DEFAULT_SEEDS = ("20260802", "20260803", "20260804")


@dataclass
class Finding:
    check: str
    status: str
    run: str | None
    detail: str
    path: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="factorized-v3 study integrity gate")
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--report")
    parser.add_argument("--allow-missing", default="", help="comma-separated check names")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seeds", default=",".join(DEFAULT_SEEDS))
    parser.add_argument("--target-epochs", type=int, default=50)
    parser.add_argument("--causal-threshold", type=float, default=1e-4)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def results_root(study_root: Path) -> Path:
    nested = study_root / "results"
    return nested if nested.is_dir() else study_root


def expected_runs(conditions: Iterable[str], seeds: Iterable[str]) -> Iterable[tuple[str, str]]:
    for condition in conditions:
        selected = tuple(seeds) if condition in PRIMARY_CONDITIONS else (tuple(seeds)[0],)
        for seed in selected:
            yield condition, seed


def locate_run(root: Path, condition: str, seed: str) -> Path:
    direct = root / "llama-reference" / "implicit-initial" / condition / f"seed-{seed}"
    if direct.is_dir():
        return direct
    matches = list(root.glob(f"**/llama-reference/implicit-initial/{condition}/seed-{seed}"))
    return matches[0] if len(matches) == 1 else direct


def artifact_contract(condition: str) -> dict[str, tuple[Path, ...]]:
    common = {
        "moves": (Path("move_metrics.json"),),
        "distribution-baselines": (Path("distribution_baselines.json"),),
        "lishogi-moves": (Path("lishogi-non-bot/moves/move_metrics.json"),),
        "probes": (Path("probes/probe_metrics.json"), Path("probes/linear_probes.pt")),
        "lishogi-probes": (Path("lishogi-non-bot/linear-probes/probe_metrics.json"),),
        "token-probe": (Path("token_probe_metrics.json"),),
        "chess-protocol": (Path("chess-protocol/chess_protocol_metrics.json"),),
        "terminal-probe": (Path("terminal-probe/action_probe_metrics.json"),),
        "hand-dynamics": (Path("hand-evaluation/hand_dynamics_metrics.json"),),
        "policy-relevance": (Path("policy-relevance/policy_relevance_metrics.json"),),
    }
    if condition in PRIMARY_CONDITIONS:
        common["action-condition"] = (
            Path("action-condition/primary/action_condition_metrics.json"),
            Path("action-condition/primary/action_condition_robustness.json"),
            Path("action-condition/primary/action_condition_attention_ablation.json"),
        )
        common["drop-relevance"] = (
            Path("drop-relevance/confidence_trajectory.json"),
            Path("drop-relevance/attention_metrics.json"),
        )
    else:
        common["action-condition"] = tuple(
            Path(f"action-condition/{mode}/{name}")
            for mode in ("oracle-native", "sensitivity-no-annotation")
            for name in ("action_condition_metrics.json", "action_condition_robustness.json")
        )
    return common


def checkpoint_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload.get("epoch", -1))
    except Exception:
        return None


def provenance_commit(payload: dict[str, Any]) -> str | None:
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        value = provenance.get("git_commit")
        return value if isinstance(value, str) and value else None
    value = payload.get("git_commit")
    return value if isinstance(value, str) and value else None


def main() -> int:
    args = parse_args()
    study = Path(args.study_root).expanduser().resolve()
    root = results_root(study)
    conditions, seeds = csv_values(args.conditions), csv_values(args.seeds)
    allowed = set(csv_values(args.allow_missing))
    findings: list[Finding] = []
    dataset_signatures: dict[str, tuple[str | None, str | None]] = {}

    def add(check: str, ok: bool, detail: str, run: str | None = None, path: Path | None = None) -> None:
        status = "pass" if ok else ("allowed" if check in allowed else "fail")
        findings.append(Finding(check, status, run, detail, str(path) if path else None))

    for condition, seed in expected_runs(conditions, seeds):
        label = f"{condition}/seed-{seed}"
        run_dir = locate_run(root, condition, seed)
        if not run_dir.is_dir():
            add("checkpoint", False, "run directory is missing", label, run_dir)
            continue

        checkpoint = run_dir / "last.pt"
        epoch = checkpoint_epoch(checkpoint)
        add("epoch", epoch == args.target_epochs,
            f"epoch={epoch}, expected={args.target_epochs}", label, checkpoint)

        manifest_path = run_dir / "run_manifest.json"
        try:
            manifest = load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            add("run-manifest", False, str(error), label, manifest_path)
            manifest = {}
        dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
        schema = dataset.get("schema_version")
        add("schema-version", schema == 4, f"dataset schema_version={schema}, expected=4", label, manifest_path)
        manifest_hash = None
        manifest_value = dataset.get("manifest")
        if isinstance(manifest_value, str):
            candidate = Path(manifest_value)
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            if candidate.is_file():
                manifest_hash = sha256_file(candidate)
        vocab_hash = dataset.get("vocab_sha256") if isinstance(dataset.get("vocab_sha256"), str) else None
        dataset_signatures[label] = (manifest_hash, vocab_hash)
        add("dataset-hash", bool(manifest_hash and vocab_hash),
            f"manifest_sha256={manifest_hash}, vocab_sha256={vocab_hash}", label, manifest_path)

        evaluation = run_dir / "evaluation"
        for stage, relatives in artifact_contract(condition).items():
            for relative in relatives:
                artifact = evaluation / relative
                exists = artifact.is_file()
                add("artifacts", exists, f"{stage}: {'present' if exists else 'missing'}", label, artifact)
                if exists and artifact.suffix == ".json":
                    try:
                        payload = load_json(artifact)
                        commit = provenance_commit(payload)
                        add("artifact-commit", bool(commit), f"git_commit={commit}", label, artifact)
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        add("artifact-json", False, str(error), label, artifact)

        probe_path = evaluation / "probes/probe_metrics.json"
        if probe_path.is_file():
            probe = load_json(probe_path)
            add("game-splits", probe.get("game_splits_disjoint") is True,
                f"game_splits_disjoint={probe.get('game_splits_disjoint')}", label, probe_path)
            alignment = probe.get("causal_prefix_full_alignment")
            maxima = []
            passed = True
            if isinstance(alignment, dict):
                for split in alignment.values():
                    if not isinstance(split, dict):
                        passed = False
                        continue
                    passed = passed and split.get("passed") is True
                    values = split.get("max_abs_diff", {})
                    if isinstance(values, dict):
                        maxima.extend(float(value) for value in values.values())
            else:
                passed = False
            maximum = max(maxima, default=float("inf"))
            add("causal-mask", passed and maximum <= args.causal_threshold,
                f"max_abs_diff={maximum:g}, threshold={args.causal_threshold:g}", label, probe_path)

        action_paths = [path for paths in artifact_contract(condition).values() for path in paths
                        if path.name == "action_condition_attention_ablation.json"]
        for relative in action_paths:
            path = evaluation / relative
            if path.is_file():
                payload = load_json(path)
                records = payload.get("ablation_contrast_records")
                present = isinstance(records, (list, dict)) and bool(records)
                add("example-records", present, "ablation_contrast_records present" if present else
                    "ablation_contrast_records missing or empty", label, path)

        stage_log = evaluation / "stage_log.json"
        add("stage-log", stage_log.is_file(), "present" if stage_log.is_file() else "missing", label, stage_log)

    complete_signatures = {value for value in dataset_signatures.values() if all(value)}
    add("dataset-consistency", len(complete_signatures) == 1,
        f"distinct complete dataset signatures={len(complete_signatures)}")

    failed = [finding for finding in findings if finding.status == "fail"]
    report = {
        "format_version": 1,
        "study_root": str(study),
        "target_epochs": args.target_epochs,
        "causal_threshold": args.causal_threshold,
        "summary": {
            "passed": not failed,
            "checks": len(findings),
            "passed_checks": sum(item.status == "pass" for item in findings),
            "allowed_failures": sum(item.status == "allowed" for item in findings),
            "failures": len(failed),
        },
        "findings": [asdict(finding) for finding in findings],
    }
    output = Path(args.report).expanduser().resolve() if args.report else study / "integrity_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"] | {"report": str(output)}, ensure_ascii=False))
    for finding in failed:
        print(f"FAIL [{finding.check}] {finding.run or 'study'}: {finding.detail} ({finding.path or '-'})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
