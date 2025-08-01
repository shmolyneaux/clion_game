#!/bin/bash
# Process cargo-llvm-lines output to remove percentages for cleaner diffs

# Read from stdin or file
input="${1:-/dev/stdin}"

# Process the output:
# - Keep the header lines as-is
# - For data lines, remove the percentage information in parentheses
awk '
NR <= 3 { print; next }  # Print first 3 lines (header) as-is
{
    # Remove percentage info: "26810 (37.8%, 37.8%)" becomes "26810"
    gsub(/ \([0-9.]+%, [0-9.]+%\)/, "", $0)
    print
}
' "$input"