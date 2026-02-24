#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# bench.sh — fast-umap full benchmark suite
#
# Runs all benchmarks and generates all figures:
#
#   1. Hardware micro-benchmarks   → benches/results/*.{md,svg}
#   2. Crate comparison            → figures/crate_comparison.{json,svg,md}
#   3. MNIST benchmark             → figures/mnist.png, figures/losses_model.png
#   4. Criterion suite (optional)  → target/criterion/
#
# Usage:
#   ./bench.sh                 # all benchmarks (no Criterion)
#   ./bench.sh --criterion     # + Criterion statistical suite
#   ./bench.sh --skip-mnist    # skip MNIST (saves ~70s)
#   ./bench.sh --only <name>   # run only one: hardware | comparison | mnist | criterion
#   ./bench.sh --help
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BOLD=$'\033[1m'; CYAN=$'\033[1;36m'; GREEN=$'\033[1;32m'
YELLOW=$'\033[1;33m'; RED=$'\033[1;31m'; DIM=$'\033[2m'
RESET=$'\033[0m'

RUN_HARDWARE=true
RUN_COMPARISON=true
RUN_MNIST=true
RUN_CRITERION=false

for arg in "$@"; do
  case "$arg" in
    --criterion)   RUN_CRITERION=true ;;
    --skip-mnist)  RUN_MNIST=false ;;
    --only)        RUN_HARDWARE=false; RUN_COMPARISON=false; RUN_MNIST=false; RUN_CRITERION=false ;;
    hardware)      RUN_HARDWARE=true ;;
    comparison)    RUN_COMPARISON=true ;;
    mnist)         RUN_MNIST=true ;;
    criterion)     RUN_CRITERION=true ;;
    --help|-h)
      cat <<'EOF'
Usage: ./bench.sh [OPTIONS]

Runs all benchmarks and generates all figures.

Options:
  --criterion     Include Criterion statistical micro-benchmarks
  --skip-mnist    Skip the MNIST benchmark (~70s)
  --only <names>  Run only named benchmarks (space-separated):
                    hardware | comparison | mnist | criterion
  --help          Show this message

Benchmarks:
  hardware      CPU + GPU micro-benchmarks (bench_report binary)
                → benches/results/cpu_*.{md,svg}, gpu_*.{md,svg}, comparison_*.{md,svg}

  comparison    fast-umap vs umap-rs on 6 dataset sizes
                → figures/crate_comparison.{json,svg,md}

  mnist         UMAP on 10K MNIST digits (downloads ~11 MB on first run)
                → figures/mnist.png, figures/losses_model.png

  criterion     Criterion statistical harness
                → target/criterion/ (HTML reports)

Examples:
  ./bench.sh                          # all benchmarks
  ./bench.sh --skip-mnist             # skip MNIST
  ./bench.sh --only comparison        # just crate comparison
  ./bench.sh --only mnist comparison  # MNIST + comparison
  ./bench.sh --criterion              # all + Criterion
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
echo "${BOLD}${CYAN}║          fast-umap — benchmark suite                    ║${RESET}"
echo "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Prerequisites ─────────────────────────────────────────────────────────────

if ! command -v cargo &>/dev/null; then
  echo "${RED}Error: cargo not found. Install Rust from https://rustup.rs${RESET}" >&2
  exit 1
fi

echo "  Rust    : $(rustc --version)"
echo "  Cargo   : $(cargo --version)"
echo "  OS      : $(uname -s) $(uname -m)"
echo ""

mkdir -p figures

# ── Helper ────────────────────────────────────────────────────────────────────

PASS=0
FAIL=0
SKIP=0
TIMES=()

run_step() {
  local label="$1"
  shift
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
  local label="$1"
  echo "${YELLOW}  ⊘ ${label} — skipped${RESET}"
  SKIP=$(( SKIP + 1 ))
  echo ""
}

# ── 1. Build everything in release mode ───────────────────────────────────────

run_step "Build (release)" \
  cargo build --release --examples --bins

# ── 2. Hardware micro-benchmarks (CPU + GPU) ─────────────────────────────────

if $RUN_HARDWARE; then
  run_step "Hardware micro-benchmarks" \
    cargo run --release --bin bench_report
else
  skip_step "Hardware micro-benchmarks"
fi

# ── 3. Crate comparison (fast-umap vs umap-rs) ──────────────────────────────

if $RUN_COMPARISON; then
  run_step "Crate comparison (fast-umap vs umap-rs)" \
    cargo run --release --example crate_comparison
else
  skip_step "Crate comparison"
fi

# ── 4. MNIST benchmark ──────────────────────────────────────────────────────

if $RUN_MNIST; then
  run_step "MNIST benchmark" \
    cargo run --release --example bench_mnist
else
  skip_step "MNIST benchmark"
fi

# ── 5. Criterion micro-benchmarks (optional) ─────────────────────────────────

if $RUN_CRITERION; then
  run_step "Criterion benchmarks" \
    cargo bench --bench umap_bench
else
  skip_step "Criterion benchmarks (pass --criterion to enable)"
fi

# ── Copy hardware results into figures/ ──────────────────────────────────────

if [ -d benches/results ]; then
  echo "${DIM}  Copying benches/results/* → figures/${RESET}"
  cp -f benches/results/*.svg figures/ 2>/dev/null || true
  cp -f benches/results/*.md  figures/ 2>/dev/null || true
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo "${BOLD}${CYAN}║                        Summary                          ║${RESET}"
echo "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# Timing table
printf "  ${BOLD}%-45s %8s${RESET}\n" "Step" "Time"
printf "  %-45s %8s\n" "─────────────────────────────────────────────" "────────"
for entry in "${TIMES[@]}"; do
  label="${entry%%|*}"
  secs="${entry##*|}"
  if (( secs >= 60 )); then
    mins=$(( secs / 60 ))
    rem=$(( secs % 60 ))
    printf "  %-45s %4dm %02ds\n" "$label" "$mins" "$rem"
  else
    printf "  %-45s %7ds\n" "$label" "$secs"
  fi
done
echo ""

# Status
total=$(( PASS + FAIL + SKIP ))
echo "  ${GREEN}Passed: ${PASS}${RESET}  ${RED}Failed: ${FAIL}${RESET}  ${YELLOW}Skipped: ${SKIP}${RESET}  Total: ${total}"
echo ""

# Generated figures
echo "${BOLD}  Generated figures:${RESET}"
echo ""
for f in \
  figures/crate_comparison.svg \
  figures/crate_comparison.md \
  figures/crate_comparison.json \
  figures/mnist.png \
  figures/losses_model.png \
  benches/results/cpu_*.svg \
  benches/results/gpu_*.svg \
  benches/results/comparison_*.svg \
; do
  if [ -f "$f" ] 2>/dev/null; then
    size=$(ls -lh "$f" | awk '{print $5}')
    printf "    %-50s %s\n" "$f" "$size"
  fi
done
echo ""

# Open hints
if [[ "$(uname)" == "Darwin" ]]; then
  echo "  Open charts:"
  echo "    open figures/crate_comparison.svg"
  echo "    open figures/mnist.png"
  echo "    open benches/results/comparison_*.svg"
  if $RUN_CRITERION; then
    echo "    open target/criterion/report/index.html"
  fi
  echo ""
fi

if (( FAIL > 0 )); then
  exit 1
fi
