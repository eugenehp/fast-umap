#!/bin/bash

echo "=== Testing Backend Choice Functionality ==="
echo

echo "1. Testing GPU backend:"
cargo run --release --example backend_choice gpu
echo "GPU test exit code: $?"
echo

echo "2. Testing CPU backend:"
cargo run --release --example backend_choice cpu
echo "CPU test exit code: $?"
echo

echo "3. Testing invalid backend (should fail gracefully):"
cargo run --release --example backend_choice invalid 2>&1 | head -5
echo "Invalid backend test exit code: $?"
echo

echo "4. Testing default backend (should be GPU):"
cargo run --release --example backend_choice
echo "Default test exit code: $?"
echo

echo "5. Testing CPU functionality in existing tests:"
cargo test --lib --quiet
echo "Library tests exit code: $?"
echo

echo "=== All backend choice tests completed ==="
