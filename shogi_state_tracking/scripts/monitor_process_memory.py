#!/usr/bin/env python3
"""Linux上で対象プロセス木とcgroupのメモリを時系列記録する。"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def proc_status(pid: int):
    result = {}
    try:
        for line in Path("/proc/{}/status".format(pid)).read_text().splitlines():
            if line.startswith(("PPid:", "VmRSS:", "VmSwap:")):
                key, value = line.split(":", 1)
                result[key] = int(value.strip().split()[0])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return result


def process_tree(root: int):
    statuses = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            value = proc_status(int(entry.name))
            if value is not None:
                statuses[int(entry.name)] = value
    selected = {root}
    changed = True
    while changed:
        changed = False
        for pid, value in statuses.items():
            if pid not in selected and value.get("PPid") in selected:
                selected.add(pid)
                changed = True
    return {pid: statuses[pid] for pid in selected if pid in statuses}


def read_number(path: Path):
    try:
        value = path.read_text().strip()
        return value if value == "max" else int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def cgroup_directory(pid: int):
    try:
        for line in Path("/proc/{}/cgroup".format(pid)).read_text().splitlines():
            fields = line.split(":", 2)
            if len(fields) == 3 and fields[0] == "0":
                return Path("/sys/fs/cgroup") / fields[2].lstrip("/")
    except (FileNotFoundError, PermissionError):
        pass
    return None


def mem_available_kib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError):
        pass
    return None


def sample(pid: int, started: float):
    processes = process_tree(pid)
    cgroup = cgroup_directory(pid)
    result = {
        "elapsed_sec": round(time.monotonic() - started, 3),
        "root_pid": pid,
        "processes": len(processes),
        "tree_rss_kib": sum(value.get("VmRSS", 0) for value in processes.values()),
        "tree_swap_kib": sum(value.get("VmSwap", 0) for value in processes.values()),
        "mem_available_kib": mem_available_kib(),
    }
    if cgroup is not None:
        result.update({
            "cgroup": str(cgroup),
            "cgroup_memory_current": read_number(cgroup / "memory.current"),
            "cgroup_memory_peak": read_number(cgroup / "memory.peak"),
            "cgroup_swap_current": read_number(cgroup / "memory.swap.current"),
        })
    return result, bool(processes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output.open("a", encoding="utf-8", buffering=1) as handle:
        while True:
            value, alive = sample(args.pid, started)
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            if not alive:
                break
            time.sleep(max(0.2, args.interval))


if __name__ == "__main__":
    main()
