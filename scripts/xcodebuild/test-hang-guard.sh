#!/bin/bash
set -euo pipefail

repeats=1
timeout_seconds=30
lock_dir=".test-artifacts/xcodebuild-hang-guard.lock"
artifacts_root=".test-artifacts/hang-guard"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repeats)
      repeats="$2"
      shift 2
      ;;
    --timeout)
      timeout_seconds="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "usage: $0 [--repeats N] [--timeout SECONDS] -- <command...>" >&2
      exit 64
      ;;
  esac
done

if [ "$#" -eq 0 ]; then
  echo "usage: $0 [--repeats N] [--timeout SECONDS] -- <command...>" >&2
  exit 64
fi

if ! mkdir -p "$(dirname "$lock_dir")" "$(dirname "$artifacts_root")" 2>/dev/null; then
  :
fi

if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "Another hang-guard run is active." >&2
  exit 3
fi
trap 'rmdir "$lock_dir"' EXIT

timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="${artifacts_root}/${timestamp}"
mkdir -p "$run_dir"

run_index=1
while [ "$run_index" -le "$repeats" ]; do
  log_file="${run_dir}/run-${run_index}.log"
  diag_file="${run_dir}/run-${run_index}.diag.txt"
  echo "[hang-guard] run ${run_index}/${repeats}" | tee "$log_file"
  if ! scripts/xcodebuild/test-timeout.sh "$timeout_seconds" -- "$@" >>"$log_file" 2>&1; then
    {
      echo "command: $*"
      echo
      echo "ps:"
      ps aux | grep -E 'xcodebuild|xctest|swift' | grep -v grep || true
    } >"$diag_file"
    echo "FAILED: see ${log_file} and ${diag_file}" >&2
    exit 1
  fi
  run_index=$((run_index + 1))
done

echo "OK: completed ${repeats} run(s) without timeout"
