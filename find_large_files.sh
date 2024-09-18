#!/bin/bash

# This script finds all files larger than 100 MB in the specified directory and its subdirectories.

# Check if a directory was provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

DIRECTORY="$1"

# Verify that the provided argument is a directory
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: '$DIRECTORY' is not a directory."
    exit 1
fi

# Use the find command to locate files larger than 100 MB
find "$DIRECTORY" -type f -size +100M -exec ls -lh {} \; | awk '{ print $9 ": " $5 }'
