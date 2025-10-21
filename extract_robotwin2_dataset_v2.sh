#!/bin/bash

# Improved script to extract all RobotWin2 dataset ZIP files
# This version handles file conflicts automatically

echo "Starting RobotWin2 dataset extraction (v2)..."
echo "This may take a while as there are many ZIP files to extract..."

cd /home/jose-barreiros/development/H_RDT/robotwin2_dataset/dataset

# Counter for progress tracking
total_files=$(find . -name "*.zip" | wc -l)
current=0

echo "Total ZIP files to extract: $total_files"

# Extract all ZIP files with automatic overwrite
find . -name "*.zip" | while read zipfile; do
    current=$((current + 1))
    echo "[$current/$total_files] Extracting: $zipfile"
    
    # Extract to the same directory as the ZIP file, overwriting existing files
    (cd "$(dirname "$zipfile")" && unzip -o -q "$(basename "$zipfile")")
    
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

# Check if we have the expected structure
echo ""
echo "Verifying dataset structure..."
echo "Checking if HDF5 files are accessible..."

# Test a few HDF5 files to make sure they're valid
test_files=$(find . -name "*.hdf5" | head -3)
for hdf5_file in $test_files; do
    if python3 -c "import h5py; f=h5py.File('$hdf5_file', 'r'); print(f'✅ $hdf5_file: {len(list(f.keys()))} groups'); f.close()" 2>/dev/null; then
        echo "✅ $hdf5_file is accessible"
    else
        echo "❌ $hdf5_file has issues"
    fi
done

echo ""
echo "✅ Dataset extraction and verification completed!"


