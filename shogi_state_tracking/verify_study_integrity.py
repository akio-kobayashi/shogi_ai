#!/usr/bin/env python3
"""Validate a factorized-v3 study before aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
DATASET_DIR_PLACEHOLDER = "<DATASET_DIR>"


@dataclass
class Finding:
    check: str
    status: str
    run: str | None
    detail: str
    path: str | None = None
    stage: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="factorized-v3 study integrity gate")
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--report")
    parser.add_argument(
        "--allow-missing",
        default="",
        help="comma-separated CHECK, CHECK:STAGE, CHECK:RUN or CHECK:RUN:STAGE selectors",
    )
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seeds", default=",".join(DEFAULT_SEEDS))
    parser.add_argument(
        "--oracle-seed",
        help="seed used for the AP oracle condition (default: first entry of --seeds)",
    )
    parser.add_argument(
        "--dataset-dir",
        help="dataset directory used to resolve run_manifest dataset paths",
    )
    parser.add_argument("--target-epochs", type=int, default=50)
    parser.add_argument("--causal-threshold", type=float, default=1e-4)
    parser.add_argument(
        "--check-torch-provenance",
        action="store_true",
        help="also verify provenance inside .pt artifacts (requires torch)",
    )
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


def expected_runs(
    conditions: Iterable[str], seeds: Iterable[str], oracle_seed: str | None = None
) -> Iterable[tuple[str, str]]:
    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("at least one seed is required")
    oracle = oracle_seed or seed_values[0]
    for condition in conditions:
        selected = seed_values if condition in PRIMARY_CONDITIONS else (oracle,)
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


def history_epoch(path: Path) -> int | None:
    """Read the last completed epoch from training_history.json.

    Epoch numbering stays continuous across a fixed-epoch continuation, so the
    maximum entry is the completed epoch count.
    """
    if not path.is_file():
        return None
    try:
        history = load_json(path).get("history")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(history, list) or not history:
        return None
    epochs = [entry.get("epoch") for entry in history if isinstance(entry, dict)]
    values = [int(value) for value in epochs if isinstance(value, int)]
    return max(values) if values else None


def checkpoint_epoch(path: Path) -> tuple[int | None, str | None]:
    """Return (epoch, error) read from a checkpoint. Only used as a fallback."""
    if not path.is_file():
        return None, "checkpoint is missing"
    try:
        import torch
    except ImportError as error:
        return None, f"torch unavailable: {error}"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - torch raises a wide range of errors
        return None, f"checkpoint unreadable: {error}"
    value = payload.get("epoch") if isinstance(payload, dict) else None
    return (int(value), None) if isinstance(value, int) else (None, "checkpoint has no epoch")


def resolve_manifest(value: Any, bases: Iterable[Path]) -> Path | None:
    """Resolve a run_manifest dataset path against a list of candidate bases."""
    if not isinstance(value, str) or not value:
        return None
    candidates: list[Path] = []
    base_list = [base for base in bases if base is not None]
    if DATASET_DIR_PLACEHOLDER in value:
        for base in base_list:
            candidates.append(Path(value.replace(DATASET_DIR_PLACEHOLDER, str(base))))
    else:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.extend(base / candidate for base in base_list)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def provenance_commit(payload: dict[str, Any]) -> str | None:
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        value = provenance.get("git_commit")
        return value if isinstance(value, str) and value else None
    value = payload.get("git_commit")
    return value if isinstance(value, str) and value else None


def torch_payload_commit(path: Path) -> tuple[str | None, str | None]:
    """Return (commit, error) for a .pt artifact. Only used with --check-torch-provenance."""
    try:
        import torch
    except ImportError as error:
        return None, f"torch unavailable: {error}"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - torch raises a wide range of errors
        return None, f"artifact unreadable: {error}"
    if not isinstance(payload, dict):
        return None, "artifact is not a dict"
    return provenance_commit(payload), None


def working_tree_status(repository: Path) -> tuple[bool | None, str]:
    """Return (clean, detail) for the repository holding this script."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"git unavailable: {error}"
    if completed.returncode != 0:
        return None, f"git exited {completed.returncode}: {completed.stderr.strip()}"
    dirty = [line for line in completed.stdout.splitlines() if line.strip()]
    if not dirty:
        return True, "working tree is clean"
    preview = ", ".join(line[3:] for line in dirty[:5])
    suffix = ", ..." if len(dirty) > 5 else ""
    return False, f"{len(dirty)} modified or untracked entries: {preview}{suffix}"


def selector_matches(check: str, run: str | None, stage: str | None, allowed: set[str]) -> bool:
    keys = {check}
    if stage:
        keys.add(f"{check}:{stage}")
    if run:
        keys.add(f"{check}:{run}")
        if stage:
            keys.add(f"{check}:{run}:{stage}")
    return bool(keys & allowed)


def main() -> int:
    args = parse_args()
    study = Path(args.study_root).expanduser().resolve()
    root = results_root(study)
    conditions, seeds = csv_values(args.conditions), csv_values(args.seeds)
    if not seeds:
        print("at least one seed is required")
        return 2
    allowed = set(csv_values(args.allow_missing))
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else None
    findings: list[Finding] = []
    dataset_signatures: dict[str, tuple[str | None, str | None]] = {}

    def add(
        check: str,
        ok: bool,
        detail: str,
        run: str | None = None,
        path: Path | None = None,
        stage: str | None = None,
    ) -> None:
        if ok:
            status = "pass"
        elif selector_matches(check, run, stage, allowed):
            status = "allowed"
        else:
            status = "fail"
        findings.append(Finding(check, status, run, detail, str(path) if path else None, stage))

    def read_json(path: Path, run: str | None, stage: str | None = None) -> dict[str, Any] | None:
        """Load JSON, recording a finding instead of raising on malformed input."""
        try:
            return load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            add("artifact-json", False, str(error), run, path, stage)
            return None

    for condition, seed in expected_runs(conditions, seeds, args.oracle_seed):
        label = f"{condition}/seed-{seed}"
        run_dir = locate_run(root, condition, seed)
        if not run_dir.is_dir():
            add("checkpoint", False, "run directory is missing", label, run_dir)
            continue

        history_path = run_dir / "training_history.json"
        checkpoint = run_dir / "last.pt"
        epoch = history_epoch(history_path)
        source: Path = history_path
        error: str | None = None
        if epoch is None:
            epoch, error = checkpoint_epoch(checkpoint)
            source = checkpoint
        detail = f"epoch={epoch}, expected={args.target_epochs}, source={source.name}"
        if error:
            detail = f"{detail} ({error})"
        add("epoch", epoch == args.target_epochs, detail, label, source)

        manifest_path = run_dir / "run_manifest.json"
        manifest = read_json(manifest_path, label, "run-manifest")
        if manifest is None:
            add("run-manifest", False, "run_manifest.json is unreadable", label, manifest_path)
            manifest = {}
        dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
        schema = dataset.get("schema_version")
        add("schema-version", schema == 4, f"dataset schema_version={schema}, expected=4", label, manifest_path)

        resolved_manifest = resolve_manifest(
            dataset.get("manifest"), (dataset_dir, Path.cwd(), study, run_dir)
        )
        manifest_hash = sha256_file(resolved_manifest) if resolved_manifest else None
        vocab_hash = dataset.get("vocab_sha256") if isinstance(dataset.get("vocab_sha256"), str) else None
        dataset_signatures[label] = (manifest_hash, vocab_hash)
        hash_detail = f"manifest_sha256={manifest_hash}, vocab_sha256={vocab_hash}"
        if resolved_manifest is None:
            hash_detail = (
                f"{hash_detail}; could not resolve {dataset.get('manifest')!r}"
                " (pass --dataset-dir)"
            )
        add("dataset-hash", bool(manifest_hash and vocab_hash), hash_detail, label, manifest_path)

        evaluation = run_dir / "evaluation"
        # Each artifact is parsed once here and reused by the checks below.
        payloads: dict[Path, dict[str, Any]] = {}
        for stage, relatives in artifact_contract(condition).items():
            for relative in relatives:
                artifact = evaluation / relative
                exists = artifact.is_file()
                add(
                    "artifacts",
                    exists,
                    f"{stage} ({relative.as_posix()}): {'present' if exists else 'missing'}",
                    label,
                    artifact,
                    stage,
                )
                if exists and artifact.suffix == ".json":
                    payload = read_json(artifact, label, stage)
                    if payload is not None:
                        payloads[artifact] = payload
                        commit = provenance_commit(payload)
                        add("artifact-commit", bool(commit), f"git_commit={commit}", label, artifact, stage)
                elif exists and artifact.suffix == ".pt" and args.check_torch_provenance:
                    commit, error = torch_payload_commit(artifact)
                    detail = f"git_commit={commit}" if error is None else error
                    add("artifact-commit", bool(commit), detail, label, artifact, stage)

        probe_path = evaluation / "probes/probe_metrics.json"
        if probe_path.is_file():
            probe = payloads.get(probe_path)
            if probe is not None:
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
                payload = payloads.get(path)
                if payload is not None:
                    records = payload.get("ablation_contrast_records")
                    present = isinstance(records, (list, dict)) and bool(records)
                    add("example-records", present, "ablation_contrast_records present" if present else
                        "ablation_contrast_records missing or empty", label, path, "action-condition")

        stage_log = evaluation / "stage_log.json"
        add("stage-log", stage_log.is_file(), "present" if stage_log.is_file() else "missing", label, stage_log)

    incomplete = sorted(label for label, value in dataset_signatures.items() if not all(value))
    distinct = {value for value in dataset_signatures.values() if all(value)}
    consistent = bool(dataset_signatures) and not incomplete and len(distinct) == 1
    consistency_detail = (
        f"runs={len(dataset_signatures)}, unresolved={len(incomplete)}, distinct signatures={len(distinct)}"
    )
    if incomplete:
        consistency_detail = f"{consistency_detail}; unresolved: {', '.join(incomplete)}"
    add("dataset-consistency", consistent, consistency_detail)

    clean, tree_detail = working_tree_status(Path(__file__).resolve().parent)
    add("working-tree", clean is True, tree_detail)

    failed = [finding for finding in findings if finding.status == "fail"]
    report = {
        "format_version": 2,
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
