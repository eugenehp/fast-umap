#!/bin/bash

echo "=== COMPREHENSIVE FALLBACK VALIDATION ==="
echo

echo "1. Testing normal GPU operation (should succeed):"
cargo run --release --features gpu,cpu --example fallback_demo | grep -A 2 "GPU backend succeeded"
echo "Exit code: $?"
echo

echo "2. Testing forced GPU failure (should fallback to CPU):"
cargo run --release --features gpu,cpu --example forced_fallback_demo | grep -A 3 "Fallback mechanism"
echo "Exit code: $?"
echo

echo "3. Testing CPU-only compilation:"
cargo run --release --features cpu --example fallback_demo 2>&1 | head -3
echo "Exit code: $?"
echo

echo "4. Testing GPU-only compilation:"
cargo run --release --features gpu --example fallback_demo | head -3
echo "Exit code: $?"
echo

echo "=== ALL FALLBACK TESTS COMPLETED ==="
