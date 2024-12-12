#!/bin/bash

# Variables
EDA_NOTEBOOK="data_exploration.ipynb"          
OUTPUT_HTML="EDA_Report.html"
OUTPUT_DIR="output"

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 1: Execute the Jupyter Notebook
echo "Running the EDA notebook and generating the HTML report..."
export PYTHONWARNINGS="ignore:internal generated filenames are not absolute"
jupyter nbconvert --execute "$EDA_NOTEBOOK" --to html --output "$OUTPUT_HTML"

# Step 2: Move the HTML report to the output folder
mv "$OUTPUT_HTML" "$OUTPUT_DIR/$OUTPUT_HTML"

# Step 3: Notify the user about the result
if [ $? -eq 0 ]; then
    echo "EDA report successfully generated: $OUTPUT_DIR/$OUTPUT_HTML"
else
    echo "Error: Failed to generate the EDA report."
    exit 1
fi



