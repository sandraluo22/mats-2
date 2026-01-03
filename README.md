# Math Rollouts Dataset - Intervention Experiments

This repository contains functions to work with the [math-rollouts dataset](https://huggingface.co/datasets/uzaymacar/math-rollouts) and run experiments that intervene on chain-of-thought reasoning (flipping chunks or ablating uncertainty words) to evaluate model performance and analyze reasoning patterns.

## Overview

The project includes two main intervention experiments:
1. **Flip Chunks Experiment**: Introduces errors into individual reasoning chunks and evaluates model performance
2. **Uncertainty Ablation Experiment**: Removes uncertainty words from chunks and evaluates the impact

Both experiments generate extensive statistics, visualizations, and analysis outputs to understand how different parts of the reasoning chain affect model accuracy and behavior.

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
  - scipy >= 1.10.0
  - scikit-learn >= 1.3.0

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Experiments

**Flip Chunks Experiment:**
```bash
python3 flip_chunks_and_evaluate.py problem_1591
```

**Uncertainty Ablation Experiment:**
```bash
python3 ablate_uncertainty_and_evaluate.py problem_1591
```

Both experiments save results to JSON files (`flip_chunk_results.json` and `ablate_uncertainty_results.json`).

### 3. Calculate Statistics

Calculate baseline statistics for normal rollouts:
```bash
python3 calculate_chunk_statistics.py problem_1591
```

This generates `chunk_statistics.json` with mean and standard deviation for:
- Sentence count
- Token count
- Uncertainty word count
- Uncertainty occurrences
- Accuracy

### 4. Generate Visualizations

All visualization scripts are in the `plot/` directory:

**Individual chunk statistics:**
```bash
python3 plot/plot_chunk_statistics.py
```

**Aggregated statistics by function tag:**
```bash
python3 plot/plot_aggregated_statistics.py
```

**Sentence-level visualization:**
```bash
python3 plot/plot_sentence_level_visualization.py
```

**Flip type statistics:**
```bash
python3 plot/plot_flip_type_statistics.py
```

**Control vs intervention accuracy:**
```bash
python3 plot/plot_control_vs_intervention_accuracy.py
```

### 5. Generate Analysis Tables

**Accuracy tables:**
```bash
python3 plot/generate_accuracy_tables.py
```

**Statistical analysis of accuracy tables:**
```bash
python3 plot/analyze_accuracy_tables.py
```

## Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── load_dataset.py                    # Dataset loading utilities
├── flip_chunks_and_evaluate.py        # Flip chunks experiment
├── ablate_uncertainty_and_evaluate.py # Uncertainty ablation experiment
├── calculate_chunk_statistics.py      # Calculate baseline statistics
│
├── plot/                              # Visualization and analysis scripts
│   ├── plot_chunk_statistics.py       # Individual chunk box plots
│   ├── plot_aggregated_statistics.py  # Aggregated stats by function tag
│   ├── plot_sentence_level_visualization.py  # Sentence-level heatmaps
│   ├── plot_flip_type_statistics.py   # Statistics by flip type
│   ├── plot_control_vs_intervention_accuracy.py  # Control vs intervention plot
│   ├── plot_anchor_statistics.py      # Anchor vs non-anchor analysis
│   ├── plot_repeated_final_answers.py # Analysis of padding patterns
│   │
│   ├── generate_accuracy_tables.py    # Generate accuracy comparison tables
│   ├── generate_case_study.py         # Select representative chunks
│   ├── generate_anchor_case_study.py  # Case study for anchor chunks
│   ├── generate_anchor_accuracy_table.py  # Anchor accuracy tables
│   ├── generate_multi_percentile_anchors.py  # Multi-percentile anchor analysis
│   │
│   ├── find_anchors.py                # Identify anchor chunks (top 10% KL)
│   ├── find_repeated_final_answers.py # Find chunks with padding patterns
│   │
│   ├── analyze_accuracy_tables.py     # Chi-square and log-likelihood tests
│   ├── analyze_uncertainty_gaps.py    # Analyze gaps between uncertainty words
│   ├── test_distributions.py          # Energy distance and MMD tests
│   ├── test_bimodality.py             # Hartigan's Dip bimodality tests
│   │
│   ├── calculate_anchor_distribution_by_function_tag.py
│   ├── calculate_anchor_distribution_by_flip_type.py
│   ├── calculate_non_anchor_accuracies.py
│   ├── calculate_chunk_similarity.py
│   │
│   ├── plot_anchor_accuracy_comparison.py  # Accuracy across percentiles
│   ├── update_case_study_statistics.py     # Update case studies with stats
│   └── generate_flip_type_accuracy_table.py
│
├── visualizations/                    # Generated visualizations and analysis
│   ├── chunk_statistics_plots/        # All plot outputs
│   │   ├── individual_chunk_plots/    # Per-chunk box plots
│   │   ├── ablation_plots/            # Ablation-specific plots
│   │   ├── anchors/                   # Anchor analysis plots
│   │   └── control_vs_intervention_accuracy.png
│   │
│   └── analysis/                      # Analysis outputs
│       ├── flip_accuracy_table.txt    # Accuracy comparison tables
│       ├── ablate_accuracy_table.txt
│       ├── flip_accuracy_analysis.txt # Statistical tests
│       ├── ablate_accuracy_analysis.txt
│       ├── case_study.json            # Representative chunks
│       ├── anchor_case_study.json     # Anchor chunk case studies
│       ├── anchors.json               # Anchor chunk indices (10%)
│       ├── anchors/                   # Multi-percentile anchor analysis
│       ├── distribution_tests.json     # Distribution comparison tests
│       ├── bimodality_tests.json      # Bimodality tests
│       ├── uncertainty_gaps_analysis.json
│       └── repeated_final_answers.json
│
├── flip_chunk_results.json            # Flip experiment results (large, gitignored)
├── ablate_uncertainty_results.json    # Ablate experiment results (large, gitignored)
├── chunk_statistics.json              # Baseline statistics
├── flip_types.txt                     # Chunk indices by flip type
└── chunks_to_flip.txt                 # Chunks used in experiments
```

## Data Sources

**Note**: The `flipped_chunks` (used in `chunks_to_flip.txt`) and `flip_types` (categorized in `flip_types.txt`) were manually imported and analyzed. These classifications were created through manual inspection of the chunks and their error types, and are used throughout the analysis and visualization scripts.

## Experiment Details

### Flip Chunks Experiment

For each chunk in the original chain-of-thought:
1. **Flip the chunk**: Generate an incorrect version with arithmetic/logical errors
2. **Solve with flipped chunk**: Run the model with all original chunks except the current one replaced with the flipped version
3. **Track metrics**: Final answer, correctness, sentence count, token count, uncertainty word occurrences, full chain of thought

### Uncertainty Ablation Experiment

For each chunk containing uncertainty words:
1. **Remove uncertainty words**: Strip out words like "maybe", "perhaps", "might", etc.
2. **Solve with ablated chunk**: Run the model with the uncertainty words removed
3. **Track metrics**: Same as flip experiment

## Analysis Features

### Anchor Chunks

Anchor chunks are identified based on `resampling_importance_kl` scores:
- Log-normalized KL divergence scores
- Top 10% (or other percentiles) selected as anchors
- Analysis includes anchor vs non-anchor comparisons

**Find anchors:**
```bash
python3 plot/find_anchors.py
```

**Generate anchor analysis:**
```bash
python3 plot/generate_multi_percentile_anchors.py
```

### Case Studies

Representative chunks are selected for detailed analysis:
- Above average, average, and below average for each function tag category
- Includes full chain of thought for each case
- Statistics and metadata included

**Generate case study:**
```bash
python3 plot/generate_case_study.py
```

### Statistical Tests

**Distribution Tests:**
- Energy Distance
- Maximum Mean Discrepancy (MMD)

**Bimodality Tests:**
- Hartigan's Dip statistic
- Resampling-based p-value calculation

**Accuracy Analysis:**
- Chi-square goodness of fit
- Log-likelihood comparison (honest, flipped, ignores probabilities)

## Visualization Types

1. **Individual Chunk Statistics**: Box plots for each chunk showing control distribution with flip/ablate points overlaid
2. **Aggregated Statistics**: Box plots grouped by function tag or flip type
3. **Sentence-Level Visualization**: Heatmap showing sentence-by-sentence breakdown with function tags and uncertainty words
4. **Control vs Intervention Accuracy**: Scatter plot binned by deciles showing relationship between control-time accuracy and post-intervention correctness
5. **Anchor Analysis**: Comparisons between anchor and non-anchor chunks across different percentiles

## Output Formats

### Experiment Results JSON

```json
{
  "problem_id": "problem_1591",
  "ground_truth_answer": "42",
  "experiments": [
    {
      "flipped_chunk_idx": 0,
      "final_answer": "43",
      "is_correct": false,
      "total_sentence_count": 150,
      "total_token_count": 2340,
      "total_uncertainty_word_count": 12,
      "full_cot": "..."
    }
  ]
}
```

### Statistics JSON

```json
{
  "chunk_0": {
    "avg_sentence_count": 12.5,
    "std_sentence_count": 2.3,
    "avg_token_count": 234.1,
    "std_token_count": 45.2,
    ...
  }
}
```

## Dataset Loading

The dataset loading uses **streaming mode** by default to avoid generating large Arrow cache files.

**Load a specific problem:**
```python
from load_dataset import load_llama8b_dataset
dataset = load_llama8b_dataset(problem_ids=['problem_1591'])
```

**Load first N problems:**
```python
dataset = load_llama8b_dataset(max_problems=10)
```

## Notes

- **Streaming mode**: Default dataset loading uses streaming mode (slower but saves disk space)
- **Problem IDs**: Follow pattern `problem_NNNN` (e.g., `problem_1591`)
- **Dataset**: Works with [`uzaymacar/math-rollouts`](https://huggingface.co/datasets/uzaymacar/math-rollouts) dataset, specifically the llama-8b portion
- **Model**: Uses `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` by default
- **Large files**: JSON results files and dataset directories are gitignored due to size

## Troubleshooting

### Disk Space Issues
- Code uses streaming mode by default
- Clean up cache: `rm -rf ~/.cache/huggingface`
- Large JSON files are gitignored

### GPU Memory Issues
- Model requires ~16GB VRAM
- Close other GPU processes
- Consider using CPU (much slower)

### Missing Dependencies
```bash
pip install scipy scikit-learn
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review dataset documentation: https://huggingface.co/datasets/uzaymacar/math-rollouts
3. Verify all dependencies are installed correctly
