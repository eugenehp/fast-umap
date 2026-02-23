#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# generate_images.sh — regenerate all example output images
#
# Runs the three examples that produce PNG files committed to the repo root.
# Requires a WGPU-compatible GPU (Metal on macOS, Vulkan on Linux/Windows).
#
# Usage:
#   ./generate_images.sh              # run all three examples
#   ./generate_images.sh simple       # run only the simple example
#   ./generate_images.sh advanced     # run only the advanced example
#   ./generate_images.sh mnist        # run only the mnist example
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BOLD=$'\033[1m'; GREEN=$'\033[1;32m'; YELLOW=$'\033[1;33m'; RESET=$'\033[0m'

RUN_SIMPLE=true
RUN_ADVANCED=true
RUN_MNIST=true

if [[ $# -gt 0 ]]; then
    RUN_SIMPLE=false; RUN_ADVANCED=false; RUN_MNIST=false
    for arg in "$@"; do
        case "$arg" in
            simple)   RUN_SIMPLE=true ;;
            advanced) RUN_ADVANCED=true ;;
            mnist)    RUN_MNIST=true ;;
            *) echo "Unknown example: $arg  (valid: simple | advanced | mnist)" >&2; exit 1 ;;
        esac
    done
fi

echo ""
echo "${BOLD}fast-umap — example image generator${RESET}"
echo ""

# ── prerequisites check ───────────────────────────────────────────────────────
if ! command -v cargo &>/dev/null; then
    echo "Error: cargo not found — install Rust from https://rustup.rs" >&2; exit 1
fi

if [[ "$(uname)" == "Darwin" ]] && ! xcode-select -p &>/dev/null; then
    echo "${YELLOW}Warning: Xcode CLT not found — Metal GPU may be unavailable.${RESET}"
    echo "  Install with: xcode-select --install"
fi

# ── build release once ────────────────────────────────────────────────────────
echo "${BOLD}Building examples (release)…${RESET}"
cargo build --release --examples 2>&1 | grep -E "^   Compiling|^    Finished|^error" || true
echo ""

run_example() {
    local name=$1; shift
    local outputs=("$@")
    echo "${BOLD}Running '$name' example…${RESET}"
    cargo run --release --example "$name"
    echo ""
    for f in "${outputs[@]}"; do
        if [[ -f "$f" ]]; then
            echo "  ${GREEN}✓${RESET}  $f"
        else
            echo "  ${YELLOW}?${RESET}  $f  (not found — check example output)"
        fi
    done
    echo ""
}

# ── run examples ─────────────────────────────────────────────────────────────
$RUN_SIMPLE   && run_example simple   "figures/plot.png" "figures/losses_model.png"
$RUN_ADVANCED && run_example advanced "figures/plot.png" "figures/losses_advanced.png"
$RUN_MNIST    && run_example mnist    "figures/mnist.png" "figures/losses_mnist.png"

echo "${BOLD}${GREEN}Done.${RESET}"
echo "Images are in figures/:"
for f in figures/plot.png figures/losses_model.png figures/losses_advanced.png \
          figures/mnist.png figures/losses_mnist.png; do
    [[ -f "$f" ]] && echo "  $f"
done
echo ""
