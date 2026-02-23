#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# bench.sh — fast-umap benchmark runner (macOS / Linux)
#
# Runs two independent benchmark tools and saves hardware-tagged result files:
#
#   1. bench_report  (custom binary, saves Markdown + SVG to benches/results/)
#   2. cargo bench   (Criterion statistical harness, optional, --criterion flag)
#
# Usage:
#   ./bench.sh                 # hardware report only
#   ./bench.sh --criterion     # hardware report + full Criterion suite
#   ./bench.sh --gpu-only      # re-run only GPU report (CPU already done)
#   ./bench.sh --help
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BOLD=$'\033[1m'; CYAN=$'\033[1;36m'; GREEN=$'\033[1;32m'
YELLOW=$'\033[1;33m'; RESET=$'\033[0m'

RUN_CRITERION=false
HELP=false

for arg in "$@"; do
  case "$arg" in
    --criterion) RUN_CRITERION=true ;;
    --help|-h)   HELP=true ;;
  esac
done

if $HELP; then
  cat <<'EOF'
Usage: ./bench.sh [OPTIONS]

Options:
  --criterion    Also run the Criterion statistical benchmark suite.
                 Results go to target/criterion/ (HTML reports).
  --help         Show this message.

Output files (always written):
  benches/results/cpu_<cpu_name>.md        Markdown table
  benches/results/cpu_<cpu_name>.svg       SVG bar chart
  benches/results/gpu_<gpu_name>.md        (written only when a GPU is found)
  benches/results/gpu_<gpu_name>.svg
  benches/results/benchmark_results.md     Copy of the latest CPU run
  benches/results/benchmark_results.svg

Prerequisites on macOS:
  • Rust stable  — https://rustup.rs
  • Xcode CLT    — xcode-select --install   (provides Metal for GPU benchmarks)
EOF
  exit 0
fi

echo ""
echo "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo "${BOLD}${CYAN}║        fast-umap benchmark suite                        ║${RESET}"
echo "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── prerequisites check ───────────────────────────────────────────────────────
if ! command -v cargo &>/dev/null; then
  echo "Error: cargo not found. Install Rust from https://rustup.rs" >&2
  exit 1
fi

RUST_VER=$(rustc --version)
echo "  Rust   : $RUST_VER"

if [[ "$(uname)" == "Darwin" ]]; then
  MACOS_VER=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
  echo "  macOS  : $MACOS_VER"

  # Check for Xcode CLT (Metal framework lives here)
  if ! xcode-select -p &>/dev/null; then
    echo ""
    echo "${YELLOW}Warning: Xcode Command Line Tools not found.${RESET}"
    echo "  GPU benchmarks (Metal/WGPU) may be unavailable."
    echo "  Install with:  xcode-select --install"
    echo ""
  fi
fi

# ── build ─────────────────────────────────────────────────────────────────────
echo ""
echo "${BOLD}Building bench_report (release)…${RESET}"
cargo build --release --bin bench_report 2>&1 | \
  grep -E "^   Compiling|^    Finished|^error" || true
echo ""

# ── hardware report ───────────────────────────────────────────────────────────
echo "${BOLD}Running hardware benchmark report…${RESET}"
echo ""
cargo run --release --bin bench_report
echo ""

# ── optional Criterion suite ─────────────────────────────────────────────────
if $RUN_CRITERION; then
  echo "${BOLD}Running Criterion statistical benchmarks…${RESET}"
  echo "(This takes a few minutes — outputs go to target/criterion/)"
  echo ""
  cargo bench --bench umap_bench
  echo ""
  echo "${GREEN}Criterion HTML report:${RESET}"
  echo "  open target/criterion/report/index.html"
  echo ""
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo "${BOLD}${GREEN}Results saved to benches/results/:${RESET}"
ls -1 benches/results/
echo ""

if [[ "$(uname)" == "Darwin" ]]; then
  echo "Open charts in your browser:"
  echo "  open benches/results/benchmark_results.svg"
  echo "  open benches/results/benchmark_results.md"
fi
