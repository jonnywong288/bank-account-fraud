#!/bin/bash

# Variables
EDA_NOTEBOOK="data_exploration.ipynb"          
OUTPUT_HTML="EDA_Report.html"
OUTPUT_DIR="output"

# Check if the notebook file exists
if [ ! -f "$EDA_NOTEBOOK" ]; then
    echo "Error: EDA notebook '$EDA_NOTEBOOK' not found."
    exit 1
fi

# Check if Jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "Error: Jupyter is not installed or not in PATH."
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 1: Execute the Jupyter Notebook
echo "Running the EDA notebook and generating the HTML report..."
export PYTHONWARNINGS="ignore:internal generated filenames are not absolute"
jupyter nbconvert --execute "$EDA_NOTEBOOK" --to html --output "$OUTPUT_HTML"

# Make sure notebook was executed
if [ $? -ne 0 ]; then
    echo "Error: Failed to execute the notebook or generate the HTML report."
    exit 1
fi

# Move the HTML report to the output folder
mv "$OUTPUT_HTML" "$OUTPUT_DIR/$OUTPUT_HTML"

# Step 3: Notify the user about the result
if [ $? -eq 0 ]; then
    echo "EDA report successfully generated: $OUTPUT_DIR/$OUTPUT_HTML"
else
    echo "Error: Failed to generate the EDA report."
    exit 1
fi



