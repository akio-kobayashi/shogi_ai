#!/usr/bin/env python3
"""複数の実験results rootからfactorized-v3の分析結果を統合する．

従来の各シェルは，標準評価とLishogi評価を異なるresults rootへ出力することが
ある．このスクリプトはそれらを読み取り専用で走査し，重複を検査したうえで一つの
分析archiveへまとめる．checkpointとゲームJSONLは収集しない．

既定では，vanilla／RAP／APとstandard／lishogi-non-botの組み合わせについて，
move_metrics.jsonとprobe_metrics.jsonが揃わなければ失敗する．部分的な収集を
許す場合だけ--allow-incompleteを指定する．
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
from typing import Iterable, Mapping


CONDITIONS = (
    "vanilla-p0.0",
    "rap-p0.15-proportional-rap-v1",
    "ap-p1.0-proportional-annotation-v1",
)
REQUIRED_DATASETS = ("standard", "lishogi-non-bot")
REQUIRED_RESULT_TYPES = ("move_metrics.json", "probe_metrics.json")
TRACKED_RESULT_NAMES = {
    "run_manifest.json",
    "training_history.json",
    "move_metrics.json",
    "token_probe_metrics.json",
    "probe_metrics.json",
    "action_probe_metrics.json",
    "hand_dynamics_metrics.json",
    "chess_protocol_metrics.json",
    "policy_relevance_metrics.json",
    "confidence_trajectory.json",
    "attention_metrics.json",
}
EXCLUDED_NAMES = {"best.pt", "last.pt"}
PROBE_ARTIFACTS = {"linear_probes.pt", "action_probes.pt", "probe_predictions.pt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="複数のfactorized-v3 results rootから分析archiveを作成する"
    )
    parser.add_argument("output", help="出力する.tar.gz")
    parser.add_argument(
        "results_roots",
        nargs="*",
        help="収集対象のresults root．親ディレクトリを指定すれば配下を再帰走査する",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=[],
        help="収集対象の親ディレクトリ（複数指定可）．results_rootsの別名",
    )
    parser.add_argument("--dataset-dir", help="dataset_manifest.jsonとvocab.jsonを収集するdataset root")
    parser.add_argument(
        "--expected-condition",
        action="append",
        dest="expected_conditions",
        choices=CONDITIONS,
        help="必須条件を限定する（複数指定可）．省略時は3条件",
    )
    parser.add_argument(
        "--expected-dataset",
        action="append",
        dest="expected_datasets",
        choices=REQUIRED_DATASETS,
        help="必須データセットを限定する（複数指定可）．省略時はstandardとlishogi-non-bot",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="必須条件の欠落があってもarchiveを作成する（manifestに欠落を記録）",
    )
    parser.add_argument("--include-probe-artifacts", action="store_true", help="linear_probes.pt等も含める")
    parser.add_argument("--no-logs", action="store_true", help="*.logを含めない")
    parser.add_argument(
        "--no-auto-sibling-discovery",
        action="store_true",
        help="指定rootの親にある別results rootを自動探索しない",
    )
    parser.add_argument("--force", action="store_true", help="既存archiveを置き換える")
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_value(project: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sanitize_text(text: str, replacements: Iterable[tuple[str, str]]) -> str:
    for source, destination in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        text = text.replace(source, destination)
    text = re.sub(r"/home/[^/\s\"']+", "<HOME>", text)
    text = re.sub(r"/Users/[^/\s\"']+", "<HOME>", text)
    return text


def selected(path: Path, include_probe_artifacts: bool, include_logs: bool) -> bool:
    if not path.is_file() or path.is_symlink() or path.name in EXCLUDED_NAMES:
        return False
    if any(part in {".venv", ".uv-cache", "__pycache__", "checkpoints"} for part in path.parts):
        return False
    # 親ディレクトリを指定しても，ゲームデータや設定JSONを誤って収集しない．
    if path.suffix == ".json":
        return path.name in TRACKED_RESULT_NAMES or path.name in {
            "artifact_verification.json",
            "dataset_manifest.json",
            "vocab.json",
            "split_summary.json",
            "export_summary.json",
        }
    if path.suffix == ".svg" and "drop-relevance" in path.parts:
        return True
    if include_logs and path.suffix == ".log":
        return True
    return include_probe_artifacts and path.name in PROBE_ARTIFACTS


def condition_for(parts: tuple[str, ...]) -> str | None:
    for condition in CONDITIONS:
        if condition in parts:
            return condition
    # 古いrap-p0.15の出力も，明示的に収集できるようにする．
    if "rap-p0.15" in parts:
        return "rap-p0.15"
    return None


def dataset_for(parts: tuple[str, ...]) -> str | None:
    if "lishogi-non-bot" in parts:
        return "lishogi-non-bot"
    if "evaluation" in parts:
        return "standard"
    return None


def result_type_for(parts: tuple[str, ...]) -> str | None:
    name = parts[-1]
    return name if name in TRACKED_RESULT_NAMES else None


def coverage_for(relative: PurePosixPath) -> tuple[str, str, str] | None:
    parts = tuple(relative.parts)
    condition = condition_for(parts)
    dataset = dataset_for(parts)
    result_type = result_type_for(parts)
    if condition is None or dataset is None or result_type is None:
        return None
    return condition, dataset, result_type


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


def required_matrix(
    observed: Mapping[tuple[str, str], set[str]],
    conditions: Iterable[str],
    datasets: Iterable[str],
) -> list[dict[str, object]]:
    missing: list[dict[str, object]] = []
    for condition in conditions:
        for dataset in datasets:
            found = observed.get((condition, dataset), set())
            absent = sorted(set(REQUIRED_RESULT_TYPES) - found)
            if absent:
                missing.append({"condition": condition, "dataset": dataset, "missing": absent})
    return missing


def _contains_result_metrics(path: Path) -> bool:
    """results rootらしいディレクトリかを軽量に判定する．"""
    for candidate in path.rglob("*"):
        if candidate.is_file() and candidate.name in REQUIRED_RESULT_TYPES:
            return True
    return False


def discover_sibling_roots(roots: Iterable[Path]) -> list[Path]:
    """指定されたresultsディレクトリの兄弟results rootを探索する．

    標準結果とLishogi結果が，例えば
    ``factorized_v3_eos_results`` と ``factorized_v3_reference_results`` のように
   分かれている場合，片方だけを指定しても取りこぼさないための補助機能である．
    無関係な親ディレクトリを再帰走査しないよう，名前にresult(s)またはanalysisを
    含むrootだけを対象とする．
    """
    known = {root.resolve() for root in roots}
    discovered: set[Path] = set()
    for root in roots:
        if not any(token in root.name.lower() for token in ("result", "analysis")):
            continue
        try:
            siblings = list(root.parent.iterdir())
        except OSError:
            continue
        for sibling in siblings:
            if not sibling.is_dir() or sibling.resolve() in known or sibling.resolve() in discovered:
                continue
            if any(token in sibling.name.lower() for token in ("result", "analysis")) and _contains_result_metrics(sibling):
                discovered.add(sibling.resolve())
    return sorted(discovered)


def scan_results(
    roots: Iterable[Path],
    args: argparse.Namespace,
    replacements: Iterable[tuple[str, str]],
) -> tuple[dict[str, bytes], dict[str, list[str]], dict[tuple[str, str], set[str]], list[str]]:
    entries: dict[str, bytes] = {}
    source_files: dict[str, list[str]] = {}
    observed: dict[tuple[str, str], set[str]] = {}
    duplicate_paths: list[str] = []
    for root in roots:
        for source in sorted(root.rglob("*")):
            if not selected(source, args.include_probe_artifacts, not args.no_logs):
                continue
            relative = source.relative_to(root)
            destination = "analysis_bundle/results/{}".format(relative.as_posix())
            if source.suffix in {".json", ".log"}:
                data = sanitize_text(
                    source.read_text(encoding="utf-8", errors="replace"), replacements
                ).encode("utf-8")
            else:
                data = source.read_bytes()
            if destination in entries:
                duplicate_paths.append(destination)
                if entries[destination] != data:
                    raise ValueError(
                        "conflicting result files were found at {} from multiple roots".format(destination)
                    )
                continue
            entries[destination] = data
            source_files.setdefault(destination, []).append(str(source))
            covered = coverage_for(relative)
            if covered is not None:
                condition, data_name, result_type = covered
                observed.setdefault((condition, data_name), set()).add(result_type)
    return entries, source_files, observed, duplicate_paths


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    root_values = list(args.results_roots) + list(args.scan_root)
    if not root_values:
        raise SystemExit("results rootを1つ以上指定してください（位置引数または--scan-root）")
    roots = [Path(value).expanduser().resolve() for value in root_values]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"results root does not exist: {root}")
    dataset = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else None
    if dataset is not None and not dataset.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset}")
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"archive already exists (use --force): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    expected_conditions = tuple(args.expected_conditions or CONDITIONS)
    expected_datasets = tuple(args.expected_datasets or REQUIRED_DATASETS)

    def make_replacements(current_roots: Iterable[Path]) -> list[tuple[str, str]]:
        values = [(str(project), "<PROJECT_DIR>"), (str(Path.home()), "<HOME>")]
        values.extend((str(root), "<RESULTS_ROOT>") for root in current_roots)
        if dataset is not None:
            values.append((str(dataset), "<DATASET_DIR>"))
        return values

    replacements = make_replacements(roots)
    entries, source_files, observed, duplicate_paths = scan_results(roots, args, replacements)
    auto_discovered_roots: list[Path] = []
    missing_matrix = required_matrix(observed, expected_conditions, expected_datasets)

    # 標準とLishogiを兄弟rootへ出力する従来シェルに対応する．片方のrootだけが
    # 指定されても，親ディレクトリ直下のresults siblingを自動的に追加する．
    if missing_matrix and not args.no_auto_sibling_discovery:
        auto_discovered_roots = discover_sibling_roots(roots)
        if auto_discovered_roots:
            roots = roots + auto_discovered_roots
            replacements = make_replacements(roots)
            entries, source_files, observed, duplicate_paths = scan_results(roots, args, replacements)
            missing_matrix = required_matrix(observed, expected_conditions, expected_datasets)

    if dataset is not None:
        for name in ("dataset_manifest.json", "vocab.json", "split_summary.json", "export_summary.json"):
            source = dataset / name
            if source.is_file():
                destination = "analysis_bundle/dataset/{}".format(name)
                entries[destination] = sanitize_text(
                    source.read_text(encoding="utf-8", errors="replace"), replacements
                ).encode("utf-8")

    if not entries:
        raise ValueError("no analysis files were found under the supplied results roots")

    if missing_matrix and not args.allow_incomplete:
        message = json.dumps(
            {
                "error": "required result matrix is incomplete",
                "missing": missing_matrix,
                "observed": {
                    f"{condition}/{dataset_name}": sorted(values)
                    for (condition, dataset_name), values in sorted(observed.items())
                },
                "scanned_roots": [str(root) for root in roots],
                "auto_discovered_siblings": [str(root) for root in auto_discovered_roots],
                "discovered_metric_paths": sorted(
                    name
                    for name in entries
                    if name.endswith(("move_metrics.json", "probe_metrics.json"))
                ),
                "hint": "標準評価とLishogiの親results rootを全て指定するか，部分収集なら--allow-incompleteを指定してください",
            },
            ensure_ascii=False,
            indent=2,
        )
        raise SystemExit(message)

    manifest = {
        "format_version": 2,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": git_value(project, "rev-parse", "HEAD"),
        "source_dirty": bool(git_value(project, "status", "--short")),
        "input_results_roots": ["<RESULTS_ROOT>" for _ in roots],
        "auto_discovered_sibling_roots": len(auto_discovered_roots),
        "dataset_root": "<DATASET_DIR>" if dataset is not None else None,
        "options": {
            "include_probe_artifacts": bool(args.include_probe_artifacts),
            "include_logs": not args.no_logs,
            "allow_incomplete": bool(args.allow_incomplete),
        },
        "safety": {
            "model_checkpoints_included": False,
            "dataset_jsonl_included": False,
            "absolute_paths_sanitized_in_text_files": True,
        },
        "expected_matrix": {
            "conditions": list(expected_conditions),
            "datasets": list(expected_datasets),
            "required_result_types": list(REQUIRED_RESULT_TYPES),
        },
        "missing_matrix": missing_matrix,
        "observed_matrix": {
            f"{condition}/{dataset_name}": sorted(values)
            for (condition, dataset_name), values in sorted(observed.items())
        },
        "duplicate_identical_paths": sorted(duplicate_paths),
        "files": [],
    }
    for name, data in sorted(entries.items()):
        manifest["files"].append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": sha256(data),
                "source_count": len(source_files.get(name, [])),
            }
        )

    readme = """# Factorized-v3 analysis bundle

This archive was assembled from one or more result roots by
`collect_factorized_analysis.py`.  Standard and Lishogi evaluation roots are
checked as a matrix before packaging.  Model checkpoints and game JSONL files
are intentionally excluded.  See `COLLECTION_MANIFEST.json` for coverage and
missing-result diagnostics.
"""
    entries["analysis_bundle/COLLECTION_MANIFEST.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    entries["analysis_bundle/README.md"] = readme.encode("utf-8")

    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    for name, data in sorted(entries.items()):
                        add_bytes(archive, name, data)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        json.dumps(
            {
                "event": "factorized_analysis_package_created",
                "output": str(output),
                "files": len(entries),
                "bytes": output.stat().st_size,
                "missing_matrix": missing_matrix,
                "observed_matrix": manifest["observed_matrix"],
                "duplicate_identical_paths": len(duplicate_paths),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
