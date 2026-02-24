#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_all.sh — Run all fast-umap examples (no figure generation)
#
# Examples verify that the code compiles and runs correctly.
# To generate benchmark figures, use ./bench.sh instead.
#
# Usage:
#   ./run_all.sh                 # run all examples
#   ./run_all.sh --skip-mnist    # skip MNIST (requires download)
#   ./run_all.sh --help
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BOLD=$'\033[1m'; CYAN=$'\033[1;36m'; GREEN=$'\033[1;32m'
YELLOW=$'\033[1;33m'; RED=$'\033[1;31m'; DIM=$'\033[2m'
RESET=$'\033[0m'

SKIP_MNIST=false

for arg in "$@"; do
  case "$arg" in
    --skip-mnist) SKIP_MNIST=true ;;
    --help|-h)
      cat <<'EOF'
Usage: ./run_all.sh [OPTIONS]

Runs all examples to verify correctness. No figures are generated.
For benchmarks and figure generation, use ./bench.sh instead.

Options:
  --skip-mnist    Skip MNIST example (downloads ~11 MB on first run)
  --help          Show this message
EOF
      exit 0
      ;;
    *)
      echo "${RED}Unknown option: $arg${RESET}" >&2
      echo "Run with --help for usage." >&2
      exit 1
      ;;
  esac
done

# ── Banner ────────────────────────────────────────────────────────────────────

echo ""
echo "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo "${BOLD}${CYAN}║          fast-umap — run all examples                   ║${RESET}"
echo "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

if ! command -v cargo &>/dev/null; then
  echo "${RED}Error: cargo not found. Install Rust from https://rustup.rs${RESET}" >&2
  exit 1
fi

echo "  Rust    : $(rustc --version)"
echo "  Cargo   : $(cargo --version)"
echo "  OS      : $(uname -s) $(uname -m)"
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────

PASS=0
FAIL=0
SKIP=0
TIMES=()

run_step() {
  local label="$1"; shift
  echo "${BOLD}━━━  ${label}  ━━━${RESET}"
  echo "${DIM}  \$ $*${RESET}"
  local start=$SECONDS
  if "$@"; then
    local elapsed=$(( SECONDS - start ))
    TIMES+=("${label}|${elapsed}")
    echo "${GREEN}  ✓ ${label} (${elapsed}s)${RESET}"
    PASS=$(( PASS + 1 ))
  else
    local elapsed=$(( SECONDS - start ))
    TIMES+=("${label}|${elapsed}")
    echo "${RED}  ✗ ${label} failed (${elapsed}s)${RESET}"
    FAIL=$(( FAIL + 1 ))
  fi
  echo ""
}

skip_step() {
  echo "${YELLOW}  ⊘ $1 — skipped${RESET}"
  SKIP=$(( SKIP + 1 ))
  echo ""
}

# ── Build ─────────────────────────────────────────────────────────────────────

run_step "Build (release)" \
  cargo build --release --examples

# ── Examples ──────────────────────────────────────────────────────────────────

run_step "Example: simple" \
  cargo run --release --example simple

run_step "Example: advanced" \
  cargo run --release --example advanced

if $SKIP_MNIST; then
  skip_step "Example: mnist"
else
  run_step "Example: mnist" \
    cargo run --release --example mnist
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "${BOLD}${CYAN}Summary${RESET}"
echo ""
printf "  ${BOLD}%-30s %8s${RESET}\n" "Step" "Time"
printf "  %-30s %8s\n" "──────────────────────────────" "────────"
for entry in "${TIMES[@]}"; do
  label="${entry%%|*}"
  secs="${entry##*|}"
  printf "  %-30s %7ds\n" "$label" "$secs"
done
echo ""
echo "  ${GREEN}Passed: ${PASS}${RESET}  ${RED}Failed: ${FAIL}${RESET}  ${YELLOW}Skipped: ${SKIP}${RESET}"
echo ""
echo "  To generate benchmark figures: ${BOLD}./bench.sh${RESET}"
echo ""

if (( FAIL > 0 )); then exit 1; fi
