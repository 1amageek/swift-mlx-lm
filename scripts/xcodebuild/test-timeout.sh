#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <timeout-seconds> -- <command...>" >&2
  exit 64
fi

timeout_seconds="$1"
shift

if [ "$1" != "--" ]; then
  echo "usage: $0 <timeout-seconds> -- <command...>" >&2
  exit 64
fi
shift

python3 - "$timeout_seconds" "$@" <<'PY'
import collections
import os
import signal
import subprocess
import sys
import time

timeout = int(sys.argv[1])
command = sys.argv[2:]

process = subprocess.Popen(command, start_new_session=True)
deadline = time.monotonic() + timeout
peak_rss_kb = 0
peak_tree = []

def sample_process_tree(root_pid: int):
    try:
        ps_output = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,rss=,comm="],
            text=True,
        )
    except Exception:
        return 0, []

    children_by_parent = collections.defaultdict(list)
    entries = {}
    for raw_line in ps_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid = int(parts[0])
        ppid = int(parts[1])
        rss_kb = int(parts[2])
        command_name = parts[3]
        entries[pid] = (ppid, rss_kb, command_name)
        children_by_parent[ppid].append(pid)

    stack = [root_pid]
    visited = set()
    tree = []
    total_rss_kb = 0
    while stack:
        pid = stack.pop()
        if pid in visited:
            continue
        visited.add(pid)
        entry = entries.get(pid)
        if entry is None:
            continue
        ppid, rss_kb, command_name = entry
        total_rss_kb += rss_kb
        tree.append((pid, ppid, rss_kb, command_name))
        stack.extend(children_by_parent.get(pid, []))
    return total_rss_kb, sorted(tree)

try:
    while True:
        return_code = process.poll()
        current_rss_kb, current_tree = sample_process_tree(process.pid)
        if current_rss_kb > peak_rss_kb:
            peak_rss_kb = current_rss_kb
            peak_tree = current_tree
        if return_code is not None:
            if peak_rss_kb > 0:
                print(
                    f"[xcodebuild-timeout] peak-rss={peak_rss_kb / 1024:.1f} MiB",
                    file=sys.stderr,
                )
            raise SystemExit(return_code)
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(command, timeout)
        time.sleep(0.2)
except subprocess.TimeoutExpired:
    if peak_rss_kb > 0:
        print(
            f"[xcodebuild-timeout] timeout after {timeout}s, peak-rss={peak_rss_kb / 1024:.1f} MiB",
            file=sys.stderr,
        )
        for pid, ppid, rss_kb, command_name in peak_tree:
            print(
                f"[xcodebuild-timeout] pid={pid} ppid={ppid} rss={rss_kb / 1024:.1f} MiB cmd={command_name}",
                file=sys.stderr,
            )
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    raise SystemExit(124)
PY
