#!/usr/bin/env bash
# Check that an assembly file contains no memory allocations

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <assembly-file.s>"
  exit 1
fi

asm_file="$1"

if [ ! -f "$asm_file" ]; then
  echo "ERROR: File not found: $asm_file"
  exit 1
fi

# Patterns that indicate memory allocations
ALLOC_PATTERNS="__rust_alloc|__rust_realloc|__rust_dealloc|malloc|calloc|realloc|alloc::alloc"

if grep -qiE "$ALLOC_PATTERNS" "$asm_file"; then
  echo "ERROR: Assembly file contains memory allocations:"
  grep -iE "$ALLOC_PATTERNS" "$asm_file"
  exit 1
fi

echo "✓ No memory allocations found in $asm_file"

