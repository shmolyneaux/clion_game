#!/bin/bash
# Process cargo-bloat output to remove percentages for cleaner diffs

# Read from stdin or file
input="${1:-/dev/stdin}"

# Process the output:
# - Keep the header line as-is
# - For data lines, remove the percentage columns
# - Keep the .text section size line but remove percentages
awk '
NR == 1 { print "  File     Size Crate"; next }  # Simplified header
/^[[:space:]]+[0-9.]+%/ {  # Lines starting with percentage
    # Extract size and crate name, skip the two percentage columns
    size = $3
    # Everything after the 3rd field is the crate name
    crate = ""
    for (i = 4; i <= NF; i++) {
        if (i == 4) crate = $i
        else crate = crate " " $i
    }
    printf "%6s %s\n", size, crate
    next
}
/\.text section size/ {  # Special handling for the summary line
    # Find the size value (e.g., 885.8KiB)
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^[0-9.]+[KMGT]iB$/) {
            print $i " .text section size"
            break
        }
    }
    next
}
{ print }  # Print other lines as-is
' "$input"