#!/usr/bin/env bash
# Generate a table showing check results for assembly functions

set -euo pipefail

# Colors for output
RED_BG='\033[41m'
GREEN_BG='\033[42m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Unicode symbols
SEP_VERT="│"
SEP_HORIZ="─"
SEP_CROSS="┼"
SEP_TOP_LEFT="┌"
SEP_TOP_RIGHT="┐"
SEP_BOTTOM_LEFT="└"
SEP_BOTTOM_RIGHT="┘"
SEP_TOP_T="┬"
SEP_BOTTOM_T="┴"
SEP_LEFT_T="├"
SEP_RIGHT_T="┤"

# Check if assembly output directory is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <assembly-output-dir> <check-script-dir> [function-names...]"
  echo "  assembly-output-dir: Directory containing .s files"
  echo "  check-script-dir: Directory containing check scripts"
  echo "  function-names: Optional list of function names for proper filename matching"
  exit 1
fi

ASM_DIR="$1"
CHECK_DIR="$2"
shift 2 || true
FUNCTION_NAMES=("$@")

if [ ! -d "$ASM_DIR" ]; then
  echo "ERROR: Assembly directory not found: $ASM_DIR"
  exit 1
fi

# Find all .s files
asm_files=("$ASM_DIR"/*.s)
if [ ! -f "${asm_files[0]}" ]; then
  echo "ERROR: No .s files found in $ASM_DIR"
  exit 1
fi

# Define check scripts (easy to extend)
declare -a CHECK_SCRIPTS=(
  "check-no-assertions.sh:No Assertions"
  "check-no-allocations.sh:No Allocations"
)

# Map function names to filenames
declare -A FUNC_TO_FILE
declare -a FUNCTIONS=()

if [ ${#FUNCTION_NAMES[@]} -gt 0 ]; then
  # Use provided function names and map to filenames
  for func in "${FUNCTION_NAMES[@]}"; do
    # Convert function name to filename: apfp::analysis::ast_static::orient2d_fast -> apfp_analysis_ast_static_orient2d_fast
    filename="${func//::/_}"
    if [ -f "$ASM_DIR/$filename.s" ]; then
      FUNCTIONS+=("$func")
      FUNC_TO_FILE["$func"]="$filename.s"
    fi
  done
else
  # Fallback: extract from filenames (may lose underscore information)
  for asm_file in "${asm_files[@]}"; do
    if [ -f "$asm_file" ]; then
      filename=$(basename "$asm_file" .s)
      # Convert filename back to function name (replace _ with ::)
      func_name="${filename//_/::}"
      FUNCTIONS+=("$func_name")
      FUNC_TO_FILE["$func_name"]="$(basename "$asm_file")"
    fi
  done
fi

# Sort functions for consistent output
mapfile -t FUNCTIONS < <(printf '%s\n' "${FUNCTIONS[@]}" | sort)

# Group functions by module prefix (everything before the last ::)
declare -A MODULE_GROUPS
for func in "${FUNCTIONS[@]}"; do
  # Extract module prefix (everything before last ::)
  if [[ "$func" == *"::"* ]]; then
    module="${func%::*}"
    func_name="${func##*::}"
    if [ -z "${MODULE_GROUPS[$module]:-}" ]; then
      MODULE_GROUPS["$module"]=""
    fi
    MODULE_GROUPS["$module"]="${MODULE_GROUPS[$module]}$func_name"$'\n'
  else
    # No module prefix, use as-is
    module=""
    MODULE_GROUPS[""]="${MODULE_GROUPS[""]:-}$func"$'\n'
  fi
done

# Run checks and collect results
declare -A RESULTS

for func in "${FUNCTIONS[@]}"; do
  filename="${FUNC_TO_FILE["$func"]}"
  asm_file="$ASM_DIR/$filename"
  
  if [ ! -f "$asm_file" ]; then
    continue
  fi
  
  check_idx=0
  for check_entry in "${CHECK_SCRIPTS[@]}"; do
    IFS=':' read -r script_name check_label <<< "$check_entry"
    check_script="$CHECK_DIR/$script_name"
    
    if [ ! -f "$check_script" ]; then
      # Try without directory prefix if script not found
      check_script="$script_name"
    fi
    
    # Run the check script
    if "$check_script" "$asm_file" >/dev/null 2>&1; then
      RESULTS["$func:$check_idx"]="PASS"
    else
      RESULTS["$func:$check_idx"]="FAIL"
    fi
    
    ((check_idx++)) || true
  done
done

# Calculate column widths
max_func_len=0
max_module_len=0
for func in "${FUNCTIONS[@]}"; do
  # For display, use just the function name (last segment)
  if [[ "$func" == *"::"* ]]; then
    display_name="${func##*::}"
    module="${func%::*}"
    # Account for indentation (2 spaces) for function names
    func_len=$((${#display_name} + 2))
    module_len=${#module}
    if [ $func_len -gt $max_func_len ]; then
      max_func_len=$func_len
    fi
    if [ "$module_len" -gt "$max_module_len" ]; then
      max_module_len=$module_len
    fi
  else
    display_name="$func"
    func_len=$((${#display_name} + 2))
    if [ $func_len -gt $max_func_len ]; then
      max_func_len=$func_len
    fi
  fi
done

# Use the maximum of module length and function length (with indentation)
# Minimum width for function column
func_col_width=$((max_func_len > max_module_len ? max_func_len : max_module_len))
func_col_width=$((func_col_width > 15 ? func_col_width + 2 : 17))

# Width for each check column
check_col_width=18

# Print table header
echo ""
echo -e "${BOLD}${BLUE}Assembly Check Results${NC}"
echo ""

# Top border
printf '%s' "${SEP_TOP_LEFT}"
printf "%*s" $func_col_width | tr ' ' "${SEP_HORIZ}"
for check_entry in "${CHECK_SCRIPTS[@]}"; do
  IFS=':' read -r script_name check_label <<< "$check_entry"
  printf '%s' "${SEP_TOP_T}"
  printf "%*s" $check_col_width | tr ' ' "${SEP_HORIZ}"
done
printf '%s\n' "${SEP_TOP_RIGHT}"

# Header row
printf '%s%s%-*s%s' "${SEP_VERT}" "${BOLD}" $func_col_width "Function" "${NC}"
for check_entry in "${CHECK_SCRIPTS[@]}"; do
  IFS=':' read -r script_name check_label <<< "$check_entry"
  printf '%s%s%-*s%s' "${SEP_VERT}" "${BOLD}" $check_col_width "$check_label" "${NC}"
done
printf '%s\n' "${SEP_VERT}"

# Separator
printf '%s' "${SEP_LEFT_T}"
printf "%*s" $func_col_width | tr ' ' "${SEP_HORIZ}"
for check_entry in "${CHECK_SCRIPTS[@]}"; do
  printf '%s' "${SEP_CROSS}"
  printf "%*s" $check_col_width | tr ' ' "${SEP_HORIZ}"
done
printf '%s\n' "${SEP_RIGHT_T}"

# Data rows - grouped by module
for module in $(printf '%s\n' "${!MODULE_GROUPS[@]}" | sort); do
  mapfile -t funcs_in_module < <(printf '%s' "${MODULE_GROUPS[$module]}" | grep -v '^$' | sort)
  
  # Print module header row if there's a module prefix
  if [ -n "$module" ]; then
    printf '%s%s%s%-*s%s' "${SEP_VERT}" "${BOLD}" "${BLUE}" $func_col_width "$module::" "${NC}"
    for check_entry in "${CHECK_SCRIPTS[@]}"; do
      printf '%s%*s' "${SEP_VERT}" $check_col_width ""
    done
    printf '%s\n' "${SEP_VERT}"
  fi
  
  # Print function rows
  for func in "${funcs_in_module[@]}"; do
    # Get full function name
    if [ -n "$module" ]; then
      full_func="$module::$func"
    else
      full_func="$func"
    fi
    
    # Display just the function name
    display_name="$func"
    
    printf '%s  %-*s' "${SEP_VERT}" $((func_col_width - 2)) "$display_name"
    
    check_idx=0
    for check_entry in "${CHECK_SCRIPTS[@]}"; do
      IFS=':' read -r script_name check_label <<< "$check_entry"
      result="${RESULTS["$full_func:$check_idx"]:-UNKNOWN}"
      
      if [ "$result" = "PASS" ]; then
        # Green background for entire cell - fixed width
        printf '%s%s%*s%s' "${SEP_VERT}" "${GREEN_BG}" $check_col_width "" "${NC}"
      else
        # Red background for entire cell - fixed width
        printf '%s%s%*s%s' "${SEP_VERT}" "${RED_BG}" $check_col_width "" "${NC}"
      fi
      
      ((check_idx++)) || true
    done
    
    printf '%s\n' "${SEP_VERT}"
  done
done

# Bottom border
printf '%s' "${SEP_BOTTOM_LEFT}"
printf "%*s" $func_col_width | tr ' ' "${SEP_HORIZ}"
for check_entry in "${CHECK_SCRIPTS[@]}"; do
  printf '%s' "${SEP_BOTTOM_T}"
  printf "%*s" $check_col_width | tr ' ' "${SEP_HORIZ}"
done
printf '%s\n' "${SEP_BOTTOM_RIGHT}"

echo ""
