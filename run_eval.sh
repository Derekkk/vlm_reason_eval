#!/bin/bash

# Evaluation script for Qwen VLM models
# Usage: bash run_eval.sh [options]

# Don't exit on error, continue with next model
set +e

# Default parameters
# Model list - add your models here
MODEL_PATHS=(
    "/data3/huzhe/workspace/model_cards/Qwen2.5-VL-7B-Instruct"
    "/data3/huzhe/workspace/evaluations/models/qwen25_vl_7b_guru_mixed_dapo_run2_step110"
    "/data3/huzhe/workspace/evaluations/models/qwen25_vl_7b_guru_mixed_grpo_run4_step150"
    # Add more models here
    # "/path/to/model3"
    # "/path/to/model4"
)

DATASETS=("mathvista" "mathverse" "mathvision" "hallusionbench" "emma-math" "emma-chem" "mmmu-pro-vision" "emma-physics" "mmmu-pro-10" "mmmu-pro-4")
N=8  # Number of answers to generate per question
DATA_SUBSET=100  # Number of data samples to evaluate on
LOG_DIR="./logs"
OUTPUT_DIR="./outputs"

# Function to get is_reason value for a model
# Models that should use is_reason=False
get_is_reason() {
    local model_path="$1"
    case "$model_path" in
        "/data3/huzhe/workspace/model_cards/Qwen2.5-VL-7B-Instruct")
            echo "False"
            ;;
        *)
            echo "True"
            ;;
    esac
}

# Set environment variables
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,3,4}  # Use all GPUs by default
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Convert datasets array to space-separated string
DATASETS_STR=$(IFS=' '; echo "${DATASETS[*]}")

# Print configuration
echo "=========================================="
echo "Batch Evaluation Configuration"
echo "=========================================="
echo "Number of models: ${#MODEL_PATHS[@]}"
echo "Datasets: $DATASETS_STR"
echo "N (answers per question): $N"
echo "Data Subset: $DATA_SUBSET"
echo "Log Directory: $LOG_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Loop through each model
TOTAL_MODELS=${#MODEL_PATHS[@]}
CURRENT_MODEL=0

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    CURRENT_MODEL=$((CURRENT_MODEL + 1))
    
    # Check if model path exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Warning: Model path does not exist: $MODEL_PATH"
        echo "Skipping to next model..."
        echo ""
        continue
    fi
    
    # Generate log file name with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    MODEL_NAME=$(basename "$MODEL_PATH")
    LOG_FILE="$LOG_DIR/log-${MODEL_NAME}-${TIMESTAMP}"
    
    # Get is_reason value for this model
    IS_REASON=$(get_is_reason "$MODEL_PATH")
    
    echo "=========================================="
    echo "[$CURRENT_MODEL/$TOTAL_MODELS] Processing model: $MODEL_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "is_reason: $IS_REASON"
    echo "Log File: $LOG_FILE"
    echo "=========================================="
    echo ""
    
    # Run evaluation for this model
    python3 main.py \
        --model_path "$MODEL_PATH" \
        --dataset $DATASETS_STR \
        --n "$N" \
        --data_subset "$DATA_SUBSET" \
        --is_reason "$IS_REASON" \
        2>&1 | tee "$LOG_FILE"
    
    # Check exit status
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Model $MODEL_NAME completed successfully!"
        echo "  Log file: $LOG_FILE"
        echo ""
    else
        echo ""
        echo "✗ Model $MODEL_NAME failed with exit code: $EXIT_CODE"
        echo "  Check log file: $LOG_FILE"
        echo "  Continuing with next model..."
        echo ""
    fi
    
    # Add a separator between models
    echo "----------------------------------------"
    echo ""
done

# Final summary
echo "=========================================="
echo "Batch Evaluation Summary"
echo "=========================================="
echo "Total models processed: $TOTAL_MODELS"
echo "All logs saved to: $LOG_DIR"
echo "All outputs saved to: $OUTPUT_DIR"
echo "=========================================="



