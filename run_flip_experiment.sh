#!/bin/bash
# Convenience script to download a problem and run flip_chunks_and_evaluate

PROBLEM_ID=${1:-"problem_1591"}
OUTPUT_FILE=${2:-"flip_chunk_results.json"}

echo "=========================================="
echo "Running flip chunks experiment"
echo "=========================================="
echo "Problem ID: $PROBLEM_ID"
echo "Output file: $OUTPUT_FILE"
echo ""

cd /root/mats-2

# The dataset will be loaded automatically by flip_chunks_and_evaluate.py
# It uses streaming mode so it will download on-demand

python3 flip_chunks_and_evaluate.py "$PROBLEM_ID" "$OUTPUT_FILE"

