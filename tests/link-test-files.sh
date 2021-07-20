#!/bin/bash
# Replacement for cmake -E create_symlink that is much faster.
#
# $1 is the source test directory
# $2 is the build test directory

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# We have to avoid using loops or other such things to work with weird file
# names (e.g., paths that include spaces)
find "$INPUT_DIR" \( -name '*.input' -o -name '*.output' \) -exec ln -f -s {} "$OUTPUT_DIR" \;
