#!/bin/bash

# Name of the output file
OUTPUT_FILE="lfs_files_sizes.txt"

# Empty the output file if it exists
> "$OUTPUT_FILE"

# Ensure all LFS files are present locally
git lfs pull

# Loop through each file tracked by Git LFS
git lfs ls-files -n | while read -r FILE; do
    if [ -f "$FILE" ]; then
        # Get the file size in bytes
        FILE_SIZE_BYTES=$(stat -c%s "$FILE")
        # Convert bytes to megabytes with 2 decimal places
        FILE_SIZE_MB=$(echo "scale=2; $FILE_SIZE_BYTES/1048576" | bc)
        # Write the filename and size to the output file
        printf "%-50s %10s MB\n" "$FILE" "$FILE_SIZE_MB" >> "$OUTPUT_FILE"
    else
        echo "File not found: $FILE" >&2
    fi
done

echo "File sizes have been written to $OUTPUT_FILE"
