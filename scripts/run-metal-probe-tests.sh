#!/bin/bash
set -euo pipefail

destination="platform=macOS,arch=arm64"
build_timeout=120
test_timeout=60
derived_data_path=""
artifacts_root="${PWD}/.test-artifacts/metal-probes"
custom_suites=()
skip_build=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --destination)
      destination="$2"
      shift 2
      ;;
    --timeout)
      test_timeout="$2"
      shift 2
      ;;
    --build-timeout)
      build_timeout="$2"
      shift 2
      ;;
    --derived-data-path)
      derived_data_path="$2"
      shift 2
      ;;
    --suite)
      custom_suites+=("$2")
      shift 2
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    --help|-h)
      cat <<'EOF'
usage: scripts/run-metal-probe-tests.sh [options]

options:
  --destination <xcodebuild-destination>
  --timeout <seconds>
  --build-timeout <seconds>
  --derived-data-path <path>
  --suite <Target/Suite>
  --skip-build
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 64
      ;;
  esac
done

mkdir -p "$artifacts_root"
timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="${artifacts_root}/${timestamp}"
mkdir -p "$run_dir"

if [ -z "$derived_data_path" ]; then
  derived_data_path="${run_dir}/DerivedData"
fi

if [ "${#custom_suites[@]}" -gt 0 ]; then
  suites=("${custom_suites[@]}")
else
  suites=(
    "MetalCompilerTests/OptimizerEntryContractTests"
    "MetalCompilerTests/OptimizerAttentionProbeTests"
  )
fi

common_args=(
  -scheme swift-lm-Package
  -destination "$destination"
  CODE_SIGNING_ALLOWED=NO
  COMPILER_INDEX_STORE_ENABLE=NO
  -parallel-testing-enabled NO
  -jobs 1
  -quiet
  OTHER_SWIFT_FLAGS='$(inherited) -DENABLE_METAL_PROBES'
)

if [ -n "$derived_data_path" ]; then
  mkdir -p "$(dirname "$derived_data_path")"
  common_args+=(-derivedDataPath "$derived_data_path")
fi

if [ "$skip_build" -ne 1 ]; then
  echo "[metal-probes] build-for-testing"
  scripts/xcodebuild-test-timeout.sh "$build_timeout" -- \
    xcodebuild build-for-testing "${common_args[@]}" \
    | tee "${run_dir}/build.log"
fi

xctestrun_path="$(find "${derived_data_path}/Build/Products" -maxdepth 1 -name '*.xctestrun' | head -n 1)"
if [ -z "$xctestrun_path" ]; then
  echo "No .xctestrun found under ${derived_data_path}/Build/Products" >&2
  exit 66
fi

for suite in "${suites[@]}"; do
  suite_slug="${suite//\//-}"
  echo "[metal-probes] ${suite}"
  test_args=(
    -xctestrun "$xctestrun_path"
    -destination "$destination"
    CODE_SIGNING_ALLOWED=NO
    COMPILER_INDEX_STORE_ENABLE=NO
    -parallel-testing-enabled NO
    -jobs 1
    -quiet
  )
  if [ -n "$derived_data_path" ]; then
    test_args+=(-derivedDataPath "$derived_data_path")
  fi
  scripts/xcodebuild-test-timeout.sh "$test_timeout" -- \
    xcodebuild test-without-building "${test_args[@]}" -only-testing:"${suite}" \
    | tee "${run_dir}/${suite_slug}.log"
done

echo "[metal-probes] logs: ${run_dir}"
