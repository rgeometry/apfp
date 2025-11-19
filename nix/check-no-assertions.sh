#!/usr/bin/env bash
# Check that an assembly file contains no assertions (panic, assert, etc.)

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

# Patterns that indicate assertions (panic, assert, etc.)
ASSERT_PATTERNS="panic|assert|__rust_start_panic|rust_begin_unwind"

if grep -qiE "$ASSERT_PATTERNS" "$asm_file"; then
  echo "ERROR: Assembly file contains assertions:"
  grep -iE "$ASSERT_PATTERNS" "$asm_file"
  exit 1
fi

echo "✓ No assertions found in $asm_file"

