#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <headers-amd64-folder> <headers-common-folder> <output-folder>"
    exit 1
fi

# Assign input arguments to variables
headers_amd64="$1"
headers_common="$2"
output_folder="$3"

# Check if the input folders exist
if [ ! -d "$headers_amd64" ]; then
    echo "Error: Folder '$headers_amd64' does not exist."
    exit 1
fi

if [ ! -d "$headers_common" ]; then
    echo "Error: Folder '$headers_common' does not exist."
    exit 1
fi

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Function to recursively merge directories
merge_directories() {
    local src="$1"
    local dest="$2"

    # Create the destination directory if it doesn't exist
    mkdir -p "$dest"

    # Iterate over all items in the source directory
    for item in "$src"/*; do
        local base_item="$(basename "$item")"

        if [ -d "$item" ]; then
            # If the item is a directory, recurse into it
            merge_directories "$item" "$dest/$base_item"
        elif [ -f "$item" ]; then
            # If the item is a file, copy it to the destination
            if [ -f "$dest/$base_item" ]; then
                echo "Warning: File '$dest/$base_item' already exists. Overwriting."
            fi
            cp -f "$item" "$dest/$base_item"
        fi
    done
}

merge_directories "$headers_amd64" "$output_folder"

merge_directories "$headers_common" "$output_folder"

echo "Merge completed successfully. Output folder: $output_folder"