#!/bin/bash

echo "=== Testing Feature-Based Compilation ==="
echo

echo "1. Testing default features (gpu + verbose):"
cargo check --example feature_demo
echo "Exit code: $?"
echo

echo "2. Testing CPU-only features:"
cargo check --example feature_demo --features cpu --no-default-features
echo "Exit code: $?"
echo

echo "3. Testing minimal features (no backends):"
cargo check --lib --no-default-features
echo "Exit code: $?"
echo

echo "4. Testing all features:"
cargo check --example feature_demo --features all
echo "Exit code: $?"
echo

echo "5. Testing GPU backend explicitly:"
cargo check --example backend_choice --features gpu
echo "Exit code: $?"
echo

echo "6. Testing CPU backend explicitly:"
cargo check --example backend_choice --features cpu
echo "Exit code: $?"
echo

echo "=== Feature Compilation Tests Complete ==="
