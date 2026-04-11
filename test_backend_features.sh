#!/bin/bash

echo "=== COMPREHENSIVE BACKEND FEATURE VALIDATION ==="
echo

echo "1. Testing GPU backend (default features):"
cargo run --release --example backend_choice gpu | head -3
echo "Exit code: $?"
echo

echo "2. Testing CPU backend (with cpu feature):"
cargo run --release --features cpu --example backend_choice cpu | head -3
echo "Exit code: $?"
echo

echo "3. Testing feature demo (default):"
cargo run --release --example feature_demo | grep "GPU backend" | head -1
echo "Exit code: $?"
echo

echo "4. Testing feature demo (CPU-only):"
cargo run --release --features cpu --no-default-features --example feature_demo | grep "CPU backend" | head -1
echo "Exit code: $?"
echo

echo "5. Testing CPU training demo:"
cargo run --release --features cpu --example cpu_training_demo | head -5
echo "Exit code: $?"
echo

echo "6. Testing error handling:"
cargo run --release --example backend_choice invalid 2>&1 | head -2
echo "Exit code: $?"
echo

echo "=== ALL BACKEND FEATURE TESTS COMPLETED ==="
