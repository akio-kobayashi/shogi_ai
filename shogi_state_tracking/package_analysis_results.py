#!/usr/bin/env python3
"""実験結果を，数値分析用の小さな転送archiveへまとめる．

dataset本体とモデルcheckpointは既定で含めない．JSONとlogに現れる実行計算機の
絶対pathはplaceholderへ置換する．
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
from typing import Dict, Iterable, List, Tuple


TEXT_SUFFIXES = {".json", ".log", ".txt", ".md"}
PROBE_ARTIFACTS = {"linear_probes.pt", "action_probes.pt", "probe_predictions.pt"}
EXCLUDED_NAMES = {"best.pt", "last.pt"}
EXPECTED_RESULT_NAMES = {
    "run_manifest.json",
    "training_history.json",
    "move_metrics.json",
    "token_probe_metrics.json",
    "probe_metrics.json",
    "action_probe_metrics.json",
    "hand_dynamics_metrics.json",
}
OPTIONAL_RESULT_NAMES = {
    "chess_protocol_metrics.json",
    "distribution_baselines.json",
    "policy_relevance_metrics.json",
}
TRACKED_RESULT_NAMES = EXPECTED_RESULT_NAMES | OPTIONAL_RESULT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析用の実験結果archiveを作成する")
    parser.add_argument("results_dir", help="収集する実験結果root")
    parser.add_argument("output", help="出力する.tar.gz")
    parser.add_argument("--dataset-dir", help="dataset_manifest.jsonとvocab.jsonだけを収集する")
    parser.add_argument("--include-probe-artifacts", action="store_true",
                        help="線形probe重み・予測.ptも含める（モデルcheckpointは含めない）")
    parser.add_argument("--include-tensorboard", action="store_true", help="TensorBoard event fileも含める")
    parser.add_argument("--no-logs", action="store_true", help="*.logを含めない")
    parser.add_argument("--force", action="store_true", help="既存archiveを置き換える")
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_value(project: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def replacements(project: Path, results: Path, dataset: Path | None) -> List[Tuple[str, str]]:
    values = [(str(results), "<RESULTS_DIR>"), (str(project), "<PROJECT_DIR>")]
    if dataset is not None:
        values.insert(1, (str(dataset), "<DATASET_DIR>"))
    home = str(Path.home())
    if home not in {value for value, _ in values}:
        values.append((home, "<HOME>"))
    # 長いpathを先に置換し，部分一致によるplaceholderの崩れを防ぐ．
    return sorted(values, key=lambda pair: len(pair[0]), reverse=True)


def sanitize_text(text: str, path_replacements: Iterable[Tuple[str, str]]) -> str:
    for source, destination in path_replacements:
        text = text.replace(source, destination)
    # 異なる計算機で作られたartifact内のhome pathも匿名化する．
    text = re.sub(r"/home/[^/\s\"']+", "<HOME>", text)
    text = re.sub(r"/Users/[^/\s\"']+", "<HOME>", text)
    return text


def selected_result_files(root: Path, args: argparse.Namespace) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name in EXCLUDED_NAMES:
            continue
        relative = path.relative_to(root)
        if any(part in {".venv", ".uv-cache", "__pycache__"} for part in relative.parts):
            continue
        if path.suffix == ".json" or (path.suffix == ".log" and not args.no_logs):
            yield path
        elif args.include_tensorboard and path.name.startswith("events.out.tfevents"):
            yield path
        elif args.include_probe_artifacts and path.name in PROBE_ARTIFACTS:
            yield path


def read_for_archive(path: Path, path_replacements: Iterable[Tuple[str, str]]) -> Tuple[bytes, bool]:
    if path.suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        return sanitize_text(text, path_replacements).encode("utf-8"), True
    return path.read_bytes(), False


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    results = Path(args.results_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    dataset = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else None
    if not results.is_dir():
        raise FileNotFoundError("results directory does not exist: {}".format(results))
    if dataset is not None and not dataset.is_dir():
        raise FileNotFoundError("dataset directory does not exist: {}".format(dataset))
    if output.exists() and not args.force:
        raise FileExistsError("archive already exists (use --force): {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)

    replace = replacements(project, results, dataset)
    entries: Dict[str, bytes] = {}
    sanitized_files: List[str] = []
    result_names = set()
    result_locations: Dict[str, List[str]] = {}
    for source in selected_result_files(results, args):
        relative = source.relative_to(results)
        destination = "analysis_bundle/results/{}".format(relative.as_posix())
        data, sanitized = read_for_archive(source, replace)
        entries[destination] = data
        result_names.add(source.name)
        if source.name in TRACKED_RESULT_NAMES:
            result_locations.setdefault(source.name, []).append(relative.as_posix())
        if sanitized:
            sanitized_files.append(destination)

    if dataset is not None:
        for name in ("dataset_manifest.json", "vocab.json", "split_summary.json", "export_summary.json"):
            source = dataset / name
            if source.is_file():
                data, sanitized = read_for_archive(source, replace)
                destination = "analysis_bundle/dataset/{}".format(name)
                entries[destination] = data
                if sanitized:
                    sanitized_files.append(destination)

    if not entries:
        raise ValueError("no analysis files were found under {}".format(results))

    git_status = git_value(project, "status", "--short")
    manifest = {
        "format_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": git_value(project, "rev-parse", "HEAD"),
        "source_dirty": bool(git_status and git_status != "unknown"),
        "results_root": "<RESULTS_DIR>",
        "dataset_root": "<DATASET_DIR>" if dataset is not None else None,
        "options": {
            "include_probe_artifacts": bool(args.include_probe_artifacts),
            "include_tensorboard": bool(args.include_tensorboard),
            "include_logs": not args.no_logs,
        },
        "safety": {
            "model_checkpoints_included": False,
            "dataset_jsonl_included": False,
            "absolute_paths_sanitized_in_text_files": True,
        },
        "present_result_types": sorted(result_names & TRACKED_RESULT_NAMES),
        "missing_common_result_types": sorted(EXPECTED_RESULT_NAMES - result_names),
        # 条件ディレクトリ外へ独立出力した評価も，どこから収集したか確認できる．
        "result_locations": {
            name: sorted(paths) for name, paths in sorted(result_locations.items())
        },
        "files": [],
    }
    for name, data in sorted(entries.items()):
        manifest["files"].append({
            "path": name,
            "bytes": len(data),
            "sha256": sha256(data),
            "text_sanitized": name in sanitized_files,
        })

    readme = """# Analysis bundle

This archive contains numerical results, run manifests and logs for analysis.
It intentionally excludes game JSONL files and model checkpoints.  Absolute paths in
text files are replaced with `<PROJECT_DIR>`, `<RESULTS_DIR>`, `<DATASET_DIR>` or `<HOME>`.
See `COLLECTION_MANIFEST.json` for checksums and missing optional result types.
"""
    manifest_data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    entries["analysis_bundle/COLLECTION_MANIFEST.json"] = manifest_data
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

    print(json.dumps({
        "event": "analysis_package_created",
        "output": str(output),
        "files": len(entries),
        "bytes": output.stat().st_size,
        "present_result_types": manifest["present_result_types"],
        "missing_common_result_types": manifest["missing_common_result_types"],
        "result_locations": manifest["result_locations"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
