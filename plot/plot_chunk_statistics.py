#!/usr/bin/env python3
"""
Create box and whisker plots for chunk statistics with overlay points from
flip_chunk_results.json and ablate_uncertainty_results.json.

For each chunk:
- Creates box plots from 100 normal rollouts (from dataset)
- Overlays dots from flip_chunk_results.json and ablate_uncertainty_results.json
- Plots 3 statistics side by side: sentence count, uncertainty word count, uncertainty occurrences
- One graph per chunk
"""

import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_from_disk
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

# Get project root directory (parent of plot folder)
PROJECT_ROOT = Path(__file__).parent.parent

# Uncertainty indicators (same as in other scripts)
UNCERTAINTY_WORDS = [
    "wait", "alternatively", "perhaps", "reconsider", "double-check", 
    "unlikely", "maybe", "might", "could be", "possibly", "doubt",
    "uncertain", "unsure", "hmm", "actually", "let me think"
]

def count_sentences(text: str) -> int:
    """Count sentences in text (simple heuristic: split by periods)."""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return len(sentences)

def count_uncertainty_words(text: str) -> Tuple[int, int]:
    """
    Count uncertainty words in text (case insensitive).
    
    Returns:
        Tuple of (total_count, number_of_unique_uncertainty_word_types)
    """
    text_lower = text.lower()
    total_count = 0
    unique_types = set()
    
    for word in UNCERTAINTY_WORDS:
        count = text_lower.count(word.lower())
        if count > 0:
            total_count += count
            unique_types.add(word)
    
    return total_count, len(unique_types)

def load_chunk_rollouts(dataset, chunk_idx: int) -> List[Dict]:
    """
    Load rollouts for a specific chunk from the dataset.
    
    Args:
        dataset: The loaded dataset
        chunk_idx: The chunk index (0-based)
    
    Returns:
        List of rollout dictionaries with rollout text
    """
    path_pattern = f'chunk_{chunk_idx}/solutions.json'
    examples = [ex for ex in dataset if path_pattern in ex.get('path', '')]
    
    if not examples:
        return []
    
    content = examples[0].get('content', '')
    if not content:
        return []
    
    try:
        solutions = json.loads(content)
        if isinstance(solutions, list):
            return solutions
        elif isinstance(solutions, dict) and 'solutions' in solutions:
            return solutions['solutions']
        else:
            return []
    except json.JSONDecodeError:
        return []

def load_original_chunks(dataset) -> Dict[int, str]:
    """
    Load original chunks from chunks_labeled.json, using actual chunk_idx from data.
    
    Returns:
        Dictionary mapping chunk_idx to chunk text
    """
    chunks_dict = {}
    
    # Find chunks_labeled.json
    chunks_labeled = None
    for ex in dataset:
        path = ex.get('path', '')
        if 'chunks_labeled.json' in path and 'incorrect_base_solution' in path and 'forced_answer' not in path:
            try:
                chunks_labeled = json.loads(ex.get('content', '[]'))
                break
            except json.JSONDecodeError:
                continue
    
    if chunks_labeled:
        # Use actual chunk_idx from data (don't assume sequential)
        for chunk_data in chunks_labeled:
            chunk_idx = chunk_data.get('chunk_idx')
            if chunk_idx is not None:
                chunk_text = chunk_data.get('chunk', '')
                if chunk_text:
                    chunks_dict[chunk_idx] = chunk_text
    
    return chunks_dict

def extract_statistics_from_rollouts(rollouts: List[Dict], chunk_idx: int, original_chunks_dict: Dict[int, str]) -> Dict[str, List[float]]:
    """
    Extract statistics from rollouts for box plot, including total counts.
    Uses actual chunk_idx from data to properly align chunks.
    
    Args:
        rollouts: List of rollout dictionaries
        chunk_idx: Current chunk index
        original_chunks_dict: Dictionary mapping chunk_idx to chunk text
    
    Returns:
        Dictionary with lists of values for each statistic (both rollout-only and total)
    """
    sentence_counts = []
    uncertainty_word_counts = []
    uncertainty_occurrence_counts = []
    
    # Total counts (include all previous chunks + current rollout)
    total_sentence_counts = []
    total_uncertainty_word_counts = []
    total_uncertainty_occurrence_counts = []
    
    # Get text from all chunks before current chunk (using actual chunk_idx)
    sorted_original_indices = sorted(original_chunks_dict.keys())
    previous_chunks_list = [original_chunks_dict[idx] for idx in sorted_original_indices if idx < chunk_idx]
    previous_chunks_text = " ".join(previous_chunks_list) if previous_chunks_list else ""
    
    for rollout_data in rollouts:
        rollout_text = rollout_data.get('rollout', '')
        if not rollout_text:
            continue
        
        # Count sentences in rollout only
        sentence_count = count_sentences(rollout_text)
        sentence_counts.append(sentence_count)
        
        # Count uncertainty words in rollout only
        uncertainty_count, occurrence_count = count_uncertainty_words(rollout_text)
        uncertainty_word_counts.append(uncertainty_count)
        uncertainty_occurrence_counts.append(occurrence_count)
        
        # Calculate total counts (previous chunks + current rollout)
        full_cot_text = previous_chunks_text + " " + rollout_text if previous_chunks_text else rollout_text
        total_sentence_count = count_sentences(full_cot_text)
        total_uncertainty_count, total_occurrence_count = count_uncertainty_words(full_cot_text)
        
        total_sentence_counts.append(total_sentence_count)
        total_uncertainty_word_counts.append(total_uncertainty_count)
        total_uncertainty_occurrence_counts.append(total_occurrence_count)
    
    return {
        'sentence_count': sentence_counts,
        'uncertainty_word_count': uncertainty_word_counts,
        'uncertainty_occurrences': uncertainty_occurrence_counts,
        'total_sentence_count': total_sentence_counts,
        'total_uncertainty_word_count': total_uncertainty_word_counts,
        'total_uncertainty_occurrences': total_uncertainty_occurrence_counts
    }

def load_experiment_results(flip_file: str, ablate_file: str) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """
    Load experiment results from JSON files and calculate total counts from full_cot.
    
    Returns:
        Tuple of (flip_results_dict, ablate_results_dict)
        Each dict maps chunk_idx to experiment data with total counts
    """
    flip_results = {}
    ablate_results = {}
    
    # Load flip results
    if os.path.exists(flip_file):
        with open(flip_file, 'r') as f:
            flip_data = json.load(f)
            for exp in flip_data.get('experiments', []):
                chunk_idx = exp.get('flipped_chunk_idx')
                if chunk_idx is not None:
                    # Get full_cot text (includes previous chunks + generated continuation)
                    full_cot = exp.get('full_cot', '')
                    
                    # Calculate total counts from full_cot
                    total_sentence_count = count_sentences(full_cot) if full_cot else 0
                    total_uncertainty_count, total_occurrence_count = count_uncertainty_words(full_cot) if full_cot else (0, 0)
                    
                    flip_results[chunk_idx] = {
                        'sentence_count': exp.get('sentence_count', 0),
                        'uncertainty_word_count': exp.get('uncertainty_word_count', 0),
                        'uncertainty_occurrences': len(exp.get('uncertainty_occurrences', [])),
                        'total_sentence_count': total_sentence_count,
                        'total_uncertainty_word_count': total_uncertainty_count,
                        'total_uncertainty_occurrences': total_occurrence_count
                    }
    
    # Load ablate results
    if os.path.exists(ablate_file):
        with open(ablate_file, 'r') as f:
            ablate_data = json.load(f)
            for exp in ablate_data.get('experiments', []):
                chunk_idx = exp.get('ablated_chunk_idx')
                if chunk_idx is not None:
                    # Get full_cot text (includes previous chunks + generated continuation)
                    full_cot = exp.get('full_cot', '')
                    
                    # Calculate total counts from full_cot
                    total_sentence_count = count_sentences(full_cot) if full_cot else 0
                    total_uncertainty_count, total_occurrence_count = count_uncertainty_words(full_cot) if full_cot else (0, 0)
                    
                    ablate_results[chunk_idx] = {
                        'sentence_count': exp.get('sentence_count', 0),
                        'uncertainty_word_count': exp.get('uncertainty_word_count', 0),
                        'uncertainty_occurrences': len(exp.get('uncertainty_occurrences', [])),
                        'total_sentence_count': total_sentence_count,
                        'total_uncertainty_word_count': total_uncertainty_count,
                        'total_uncertainty_occurrences': total_occurrence_count
                    }
    
    return flip_results, ablate_results

def plot_chunk_statistics(
    chunk_idx: int,
    rollout_stats: Dict[str, List[float]],
    flip_data: Optional[Dict],
    ablate_data: Optional[Dict],
    output_dir: str = "visualizations/chunk_statistics_plots",
    has_ablation: bool = False
):
    """
    Create box and whisker plot for a single chunk.
    
    Args:
        chunk_idx: Chunk index
        rollout_stats: Dictionary with lists of values for each statistic
        flip_data: Optional flip experiment data for this chunk
        ablate_data: Optional ablate experiment data for this chunk
        output_dir: Base directory to save plots
        has_ablation: Whether this chunk has ablation data (determines subfolder)
    """
    # Determine output directory based on whether chunk has ablation data
    if has_ablation:
        plot_dir = os.path.join(output_dir, "ablation_plots")
    else:
        plot_dir = os.path.join(output_dir, "individual_chunk_plots")
    
    # Create output directory (and parent directories if needed)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Chunk {chunk_idx} Statistics', fontsize=14, fontweight='bold')
    
    # Statistics to plot (using total counts)
    stats = [
        ('total_sentence_count', 'Total Sentence Count', 'Count'),
        ('total_uncertainty_word_count', 'Total Uncertainty Word Count', 'Count'),
        ('total_uncertainty_occurrences', 'Total Uncertainty Occurrences', 'Count')
    ]
    
    for idx, (stat_key, stat_name, ylabel) in enumerate(stats):
        ax = axes[idx]
        
        # Get data for this statistic
        data = rollout_stats.get(stat_key, [])
        
        if not data:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(stat_name)
            continue
        
        # Create box plot
        bp = ax.boxplot([data], tick_labels=[''], patch_artist=True, 
                        showmeans=True, meanline=True)
        
        # Style the box plot
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        # Overlay flip data point if available
        if flip_data and stat_key in flip_data:
            flip_value = flip_data[stat_key]
            ax.scatter([1], [flip_value], color='red', s=100, 
                      marker='o', zorder=5, label='Flip Chunk', edgecolors='black', linewidths=1.5)
        
        # Overlay ablate data point if available
        if ablate_data and stat_key in ablate_data:
            ablate_value = ablate_data[stat_key]
            ax.scatter([1], [ablate_value], color='green', s=100, 
                      marker='s', zorder=5, label='Ablate Uncertainty', edgecolors='black', linewidths=1.5)
        
        # Set labels and title
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(stat_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks([1])
        ax.set_xticklabels(['Normal Rollouts'])
        
        # Add legend if there are overlay points
        if (flip_data and stat_key in flip_data) or (ablate_data and stat_key in ablate_data):
            ax.legend(loc='upper right', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(plot_dir, f'chunk_{chunk_idx}_statistics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    folder_name = "ablation_plots" if has_ablation else "individual_chunk_plots"
    print(f"  ✓ Saved plot for chunk {chunk_idx} to {folder_name}/chunk_{chunk_idx}_statistics.png")

def main():
    """Main function to generate plots for all chunks."""
    import sys
    
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    output_dir = str(PROJECT_ROOT / "visualizations/chunk_statistics_plots")
    
    print("="*80)
    print("CHUNK STATISTICS PLOTTER")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    try:
        dataset = load_from_disk(dataset_path)
        print(f"✓ Loaded dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Load original chunks for calculating totals
    print(f"\nLoading original chunks...")
    original_chunks_dict = load_original_chunks(dataset)
    print(f"✓ Loaded {len(original_chunks_dict)} original chunks (indices: {min(original_chunks_dict.keys())} to {max(original_chunks_dict.keys())})")
    
    # Load experiment results
    print(f"\nLoading experiment results...")
    flip_results, ablate_results = load_experiment_results(flip_file, ablate_file)
    print(f"✓ Loaded {len(flip_results)} flip experiments")
    print(f"✓ Loaded {len(ablate_results)} ablate experiments")
    
    # Get all chunk indices from dataset
    print(f"\nFinding all chunks...")
    chunk_indices = set()
    for ex in dataset:
        path = ex.get('path', '')
        match = re.search(r'chunk_(\d+)/solutions\.json', path)
        if match:
            chunk_indices.add(int(match.group(1)))
    
    chunk_indices = sorted(chunk_indices)
    print(f"✓ Found {len(chunk_indices)} chunks (indices: {min(chunk_indices)} to {max(chunk_indices)})")
    
    # Create plots for each chunk
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    chunks_plotted = 0
    chunks_with_data = 0
    
    for chunk_idx in chunk_indices:
        print(f"\nProcessing chunk {chunk_idx}...")
        
        # Load rollouts for this chunk
        rollouts = load_chunk_rollouts(dataset, chunk_idx)
        
        if not rollouts:
            print(f"  Warning: No rollouts found for chunk {chunk_idx}")
            continue
        
        chunks_with_data += 1
        print(f"  Found {len(rollouts)} rollouts")
        
        # Extract statistics from rollouts (including totals)
        rollout_stats = extract_statistics_from_rollouts(rollouts, chunk_idx, original_chunks_dict)
        
        # Get experiment data for this chunk
        flip_data = flip_results.get(chunk_idx)
        ablate_data = ablate_results.get(chunk_idx)
        
        has_ablation = ablate_data is not None
        
        if flip_data:
            print(f"  Found flip experiment data")
        if ablate_data:
            print(f"  Found ablate experiment data (will save to ablation_plots/)")
        
        # Create plot
        plot_chunk_statistics(chunk_idx, rollout_stats, flip_data, ablate_data, output_dir, has_ablation)
        chunks_plotted += 1
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)
    
    # Count ablation plots
    ablation_count = sum(1 for chunk_idx in chunk_indices if ablate_results.get(chunk_idx) is not None)
    
    print(f"\nSummary:")
    print(f"  - Processed {chunks_with_data} chunks with data")
    print(f"  - Generated {chunks_plotted} individual chunk plots")
    print(f"  - Individual chunk plots ({chunks_plotted - ablation_count} chunks) saved to: {output_dir}/individual_chunk_plots/")
    print(f"  - Ablation plots ({ablation_count} chunks) saved to: {output_dir}/ablation_plots/")
    print(f"\nNote: To generate aggregated plot, run: python3 plot_aggregated_statistics.py")
    print(f"\nLegend:")
    print(f"  - Blue box plots: Distribution from 100 normal rollouts")
    print(f"  - Red circles: Flip chunk experiment results")
    print(f"  - Green squares: Ablate uncertainty experiment results")

if __name__ == "__main__":
    main()

