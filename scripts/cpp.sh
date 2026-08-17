#!/usr/bin/env bash
set -e

# Detect default profile if not present
conan profile detect --force 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../cpp"

conan install . --output-folder=build --build=missing
cmake --preset conan-default
cmake --build --preset conan-release