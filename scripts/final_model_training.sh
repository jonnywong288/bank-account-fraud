#!/bin/bash

# Determine script's directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"


NOTEBOOK="$ROOT_DIR/final_model_training.ipynb"
PY_SCRIPT="final_model_training.py"
OUTPUT_DIR="$ROOT_DIR/output"
LOG_FILE="$OUTPUT_DIR/training_log.txt"

# Check if the notebook exists
if [ ! -f "$NOTEBOOK" ]; then
    echo "Error: Notebook not found at $NOTEBOOK"
    exit 1
fi

# Check if Jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "Error: Jupyter is not installed or not in PATH."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Convert notebook to Python script
echo "Converting notebook to Python script..."
export PYTHONWARNINGS="ignore:internal generated filenames are not absolute"
jupyter nbconvert --to script "$NOTEBOOK"

# Check if the script was created
if [ ! -f "$ROOT_DIR/$PY_SCRIPT" ]; then
    echo "Error: Failed to convert notebook to Python script."
    exit 1
fi

# Run the script in the background using nohup
echo "Starting model training in the background..."
nohup python "$ROOT_DIR/$PY_SCRIPT" > "$LOG_FILE" 2>&1 &

# Notify user
if [ $? -eq 0 ]; then
    echo "Training started successfully."
    echo "Logs: $LOG_FILE"
else
    echo "Error: Failed to start training script."
    exit 1
fi
