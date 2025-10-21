#!/bin/bash

# Script to extract all RobotWin2 dataset ZIP files
# This will extract all ZIP files in the dataset directory structure

echo "Starting RobotWin2 dataset extraction..."
echo "This may take a while as there are many ZIP files to extract..."

cd /home/jose-barreiros/development/H_RDT/robotwin2_dataset/dataset

# Counter for progress tracking
total_files=$(find . -name "*.zip" | wc -l)
current=0

echo "Total ZIP files to extract: $total_files"

# Extract all ZIP files
find . -name "*.zip" | while read zipfile; do
    current=$((current + 1))
    echo "[$current/$total_files] Extracting: $zipfile"
    
    # Extract to the same directory as the ZIP file
    (cd "$(dirname "$zipfile")" && unzip -q "$(basename "$zipfile")")
    
    # Optional: Remove the ZIP file after extraction to save space
    # rm "$zipfile"
done

echo "Dataset extraction completed!"
echo "Checking extracted structure..."

# Check a few tasks to verify the structure
echo "Sample extracted structure:"
find . -name "*.hdf5" | head -10

echo "Total HDF5 files extracted: $(find . -name "*.hdf5" | wc -l)"
echo "Tasks with extracted data: $(find . -maxdepth 1 -type d | grep -v "^\.$" | wc -l)"


