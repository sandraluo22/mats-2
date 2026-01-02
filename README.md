# Math Rollouts Dataset - Flip Chunks Experiment

This repository contains tools to work with the math-rollouts dataset and run experiments that flip chunks in chain-of-thought reasoning to evaluate model performance.

## Overview

The main experiment (`flip_chunks_and_evaluate.py`) takes a problem from the dataset, introduces errors into individual reasoning chunks, and evaluates how the model performs with these corrupted chunks.

## Requirements

- Python 3.11+
- GPU with at least 16GB VRAM (for the 8B model)
- Dependencies: `pip install -r requirements.txt`
  - datasets >= 2.14.0
  - huggingface-hub >= 0.16.0
  - transformers >= 4.30.0
  - torch >= 2.0.0
  - accelerate >= 0.20.0
  - matplotlib >= 3.5.0
  - numpy >= 1.21.0

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Flip Chunks Experiment

The easiest way to run the experiment is with a known problem ID:

```bash
python3 flip_chunks_and_evaluate.py problem_1591
```

Or use the convenience script:

```bash
./run_flip_experiment.sh problem_1591
```

This will:
1. Download the problem data using streaming mode (~9 minutes for first run)
2. Load the problem and its chunks
3. Run the flip chunks experiment (2-5 minutes per chunk)
4. Save results to `flip_chunk_results.json`

### 3. Custom Output File

Specify a custom output file:

```bash
python3 flip_chunks_and_evaluate.py problem_1591 my_results.json
```

## Finding Problem IDs

### Get the First Problem ID

To quickly find the first problem ID in the dataset:

```bash
python3 get_first_problem_id.py
```

This will output something like:
```
✓ Found first problem ID: problem_1591
```

### Download Specific Problems

To download a single problem (saves problem IDs to `downloaded_problems.json`):

```bash
python3 download_1_problem.py
```

To download the first 10 problems:

```bash
python3 download_10_problems.py
```

**Note:** Downloading 10 problems takes significantly longer (~90 minutes with streaming mode) compared to downloading 1 problem (~9 minutes).

## Dataset Loading

The dataset loading uses **streaming mode** by default to avoid generating large Arrow cache files that require significant disk space.

### Loading Options

The `load_llama8b_dataset()` function supports several modes:

**Load a specific problem:**
```python
from load_dataset import load_llama8b_dataset
dataset = load_llama8b_dataset(problem_ids=['problem_1591'])
```

**Load first N problems:**
```python
dataset = load_llama8b_dataset(max_problems=10)
```

**Load all problems (not recommended - very slow):**
```python
dataset = load_llama8b_dataset()
```

### Streaming Mode vs Non-Streaming

- **Streaming mode (default)**: Processes examples one-by-one without generating Arrow cache files
  - ✅ Saves disk space
  - ✅ Works with limited storage
  - ❌ Slower (requires iterating through dataset)
  - ❌ Can't use `len()` on streaming datasets

- **Non-streaming mode**: Generates Arrow cache files for faster access
  - ✅ Much faster after initial cache generation
  - ✅ Can use `len()` and indexing
  - ❌ Requires significant disk space (~several GB)
  - ❌ Initial processing takes time

To disable streaming (if you have disk space):

```python
dataset = load_llama8b_dataset(problem_ids=['problem_1591'], use_streaming=False)
```

## Experiment Details

### What the Flip Chunks Experiment Does

For each chunk in the original chain-of-thought:

1. **Flip the chunk**: Generate an incorrect version with arithmetic/logical errors
2. **Solve with flipped chunk**: Run the model with all original chunks except the current one replaced with the flipped version
3. **Track metrics**: 
   - Final answer
   - Correctness (matches ground truth)
   - Sentence count
   - Token count
   - Uncertainty word occurrences
   - Full chain of thought

### Output Format

Results are saved as JSON with this structure:

```json
{
  "problem_id": "problem_1591",
  "problem": "Problem text...",
  "ground_truth_answer": "42",
  "num_chunks": 5,
  "original_chunks": ["chunk1", "chunk2", ...],
  "experiments": [
    {
      "flipped_chunk_idx": 0,
      "flipped_chunk": "Incorrect version...",
      "original_chunk": "Original version...",
      "final_answer": "43",
      "is_correct": false,
      "sentence_count": 15,
      "token_count": 234,
      "uncertainty_word_count": 3,
      "uncertainty_occurrences": ["maybe:1", "perhaps:2"],
      "full_cot": "Full solution text...",
      "input_token_count": 1500
    },
    ...
  ]
}
```

## Time Estimates

| Operation | Time |
|-----------|------|
| Get first problem ID | ~1 second |
| Download 1 problem (streaming) | ~9 minutes |
| Download 10 problems (streaming) | ~90 minutes |
| Flip chunks experiment (per chunk) | 2-5 minutes |
| Full experiment (5 chunks) | 10-25 minutes |

## File Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── load_dataset.py                # Dataset loading utilities (streaming mode)
├── flip_chunks_and_evaluate.py   # Main experiment script
├── download_10_problems.py        # Download first 10 problems
├── download_1_problem.py          # Download single problem
├── get_first_problem_id.py        # Quick script to find first problem ID
├── run_flip_experiment.sh         # Convenience script to run experiment
├── check_dataset_info.py          # Explore dataset structure
├── plot_kl_vs_accuracy.py         # Plot KL divergence vs accuracy
└── list_problems.py               # List all problems in dataset
```

## Troubleshooting

### Disk Space Issues

If you run out of disk space:
- The code uses streaming mode by default, which should help
- Clean up old cache: `rm -rf ~/.cache/huggingface`
- The streaming mode avoids generating large Arrow files

### Network Timeouts

The code includes retry logic with exponential backoff. If downloads fail:
- Check your internet connection
- Try again later (HuggingFace servers may be busy)
- The timeout is set to 5 minutes per request

### GPU Memory Issues

If you get CUDA out-of-memory errors:
- The model requires ~16GB VRAM
- Close other GPU processes
- Consider using a smaller model or CPU (much slower)

### Problem Not Found

If a problem ID isn't found:
- Verify the problem exists using `get_first_problem_id.py`
- Check that you're using the correct format: `problem_1591` (not `1591`)
- The problem might not be in the llama-8b portion of the dataset

## Examples

### Example 1: Quick Test with First Problem

```bash
# Get a problem ID
python3 get_first_problem_id.py

# Run experiment with that problem
python3 flip_chunks_and_evaluate.py problem_1591
```

### Example 2: Download and Use Specific Problem

```bash
# Download problem_1591
python3 -c "from load_dataset import load_llama8b_dataset; ds = load_llama8b_dataset(problem_ids=['problem_1591']); print(f'Loaded {len(ds)} examples')"

# Run experiment
python3 flip_chunks_and_evaluate.py problem_1591
```

### Example 3: Load Dataset in Python

```python
from load_dataset import load_llama8b_dataset

# Load specific problem
dataset = load_llama8b_dataset(problem_ids=['problem_1591'])

# Iterate through examples
for example in dataset:
    path = example.get("path", "")
    content = example.get("content", "")
    print(f"Path: {path}")
    print(f"Content length: {len(content)}")
```

## Notes

- **Streaming mode**: The dataset loading uses streaming mode by default. This is slower but requires less disk space.
- **Problem IDs**: Problem IDs follow the pattern `problem_NNNN` (e.g., `problem_1591`, `problem_330`)
- **Dataset**: This works with the `uzaymacar/math-rollouts` dataset, specifically the llama-8b portion
- **Model**: Uses `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` by default

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the dataset documentation: https://huggingface.co/datasets/uzaymacar/math-rollouts
3. Check that all dependencies are installed correctly

