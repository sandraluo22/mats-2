#!/usr/bin/env python3
"""
Create aggregated box plots and visualizations grouped by flip type instead of function tag.

Creates:
- Box plots grouped by flip type category
- Heatmap showing flip types for chunks
- Bar charts for uncertainty word count by flip type
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

# Get project root directory (parent of plot folder)
PROJECT_ROOT = Path(__file__).parent.parent

# Uncertainty indicators
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

def load_flip_types(flip_types_file: str) -> Dict[int, str]:
    """
    Load flip types from flip_types.txt.
    
    Returns:
        Dictionary mapping chunk_idx to flip_type
    """
    flip_type_map = {}
    
    if not os.path.exists(flip_types_file):
        print(f"Warning: {flip_types_file} not found")
        return flip_type_map
    
    with open(flip_types_file, 'r') as f:
        content = f.read()
    
    # Parse each line
    for line in content.strip().split('\n'):
        if not line.strip() or ':' not in line:
            continue
        
        # Extract type name and chunk indices
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        
        type_name = parts[0].strip()
        chunk_indices_str = parts[1].strip()
        
        # Parse chunk indices (comma-separated, may have spaces)
        chunk_indices = []
        for idx_str in chunk_indices_str.split(','):
            idx_str = idx_str.strip()
            if idx_str:
                try:
                    chunk_idx = int(idx_str)
                    chunk_indices.append(chunk_idx)
                    flip_type_map[chunk_idx] = type_name
                except ValueError:
                    continue
    
    return flip_type_map

def load_chunk_rollouts(dataset, chunk_idx: int) -> List[Dict]:
    """Load rollouts for a specific chunk from the dataset."""
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
    """Load original chunks from chunks_labeled.json."""
    chunks_dict = {}
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
        for chunk_data in chunks_labeled:
            chunk_idx = chunk_data.get('chunk_idx')
            chunk_text = chunk_data.get('chunk', '')
            if chunk_idx is not None:
                chunks_dict[chunk_idx] = chunk_text
    
    return chunks_dict

def extract_statistics_from_rollouts(rollouts: List[Dict], chunk_idx: int, 
                                     original_chunks_dict: Dict[int, str], 
                                     tokenizer: AutoTokenizer) -> Dict[str, List[float]]:
    """Extract statistics from rollouts, including total counts."""
    sentence_counts = []
    token_counts = []
    uncertainty_word_counts = []
    
    total_sentence_counts = []
    total_token_counts = []
    total_uncertainty_word_counts = []
    
    # Get text from all chunks before current chunk
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
        
        # Count tokens in rollout only
        token_count = len(tokenizer.encode(rollout_text))
        token_counts.append(token_count)
        
        # Count uncertainty words in rollout only
        uncertainty_count, _ = count_uncertainty_words(rollout_text)
        uncertainty_word_counts.append(uncertainty_count)
        
        # Calculate total counts (previous chunks + current rollout)
        full_cot_text = previous_chunks_text + " " + rollout_text if previous_chunks_text else rollout_text
        total_sentence_count = count_sentences(full_cot_text)
        total_token_count = len(tokenizer.encode(full_cot_text))
        total_uncertainty_count, _ = count_uncertainty_words(full_cot_text)
        
        total_sentence_counts.append(total_sentence_count)
        total_token_counts.append(total_token_count)
        total_uncertainty_word_counts.append(total_uncertainty_count)
    
    return {
        'sentence_count': sentence_counts,
        'token_count': token_counts,
        'uncertainty_word_count': uncertainty_word_counts,
        'total_sentence_count': total_sentence_counts,
        'total_token_count': total_token_counts,
        'total_uncertainty_word_count': total_uncertainty_word_counts
    }

def load_experiment_results(flip_file: str, ablate_file: str, tokenizer: AutoTokenizer) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """Load experiment results from JSON files and calculate total counts."""
    flip_results = {}
    ablate_results = {}
    
    # Load flip results
    if os.path.exists(flip_file):
        with open(flip_file, 'r') as f:
            flip_data = json.load(f)
            for exp in flip_data.get('experiments', []):
                chunk_idx = exp.get('flipped_chunk_idx')
                if chunk_idx is not None:
                    full_cot = exp.get('full_cot', '')
                    
                    # Calculate total counts from full_cot
                    total_sentence_count = count_sentences(full_cot) if full_cot else 0
                    total_token_count = len(tokenizer.encode(full_cot)) if full_cot else 0
                    total_uncertainty_count, total_occurrence_count = count_uncertainty_words(full_cot) if full_cot else (0, 0)
                    
                    flip_results[chunk_idx] = {
                        'sentence_count': exp.get('sentence_count', 0),
                        'token_count': exp.get('token_count', 0),
                        'uncertainty_word_count': exp.get('uncertainty_word_count', 0),
                        'total_sentence_count': total_sentence_count,
                        'total_token_count': total_token_count,
                        'total_uncertainty_word_count': total_uncertainty_count
                    }
    
    # Load ablate results
    if os.path.exists(ablate_file):
        with open(ablate_file, 'r') as f:
            ablate_data = json.load(f)
            for exp in ablate_data.get('experiments', []):
                chunk_idx = exp.get('ablated_chunk_idx')
                if chunk_idx is not None:
                    full_cot = exp.get('full_cot', '')
                    
                    # Calculate total counts from full_cot
                    total_sentence_count = count_sentences(full_cot) if full_cot else 0
                    total_token_count = len(tokenizer.encode(full_cot)) if full_cot else 0
                    total_uncertainty_count, total_occurrence_count = count_uncertainty_words(full_cot) if full_cot else (0, 0)
                    
                    ablate_results[chunk_idx] = {
                        'sentence_count': exp.get('sentence_count', 0),
                        'token_count': exp.get('token_count', 0),
                        'uncertainty_word_count': exp.get('uncertainty_word_count', 0),
                        'total_sentence_count': total_sentence_count,
                        'total_token_count': total_token_count,
                        'total_uncertainty_word_count': total_uncertainty_count
                    }
    
    return flip_results, ablate_results

def plot_aggregated_statistics_by_flip_type(
    all_rollout_stats_by_chunk: Dict[int, Dict[str, List[float]]],
    all_flip_data: Dict[int, Dict],
    all_ablate_data: Dict[int, Dict],
    chunk_flip_types: Dict[int, str],
    output_dir: str = "visualizations/chunk_statistics_plots",
    remove_outliers: bool = False,
    outlier_percentile: float = 0.75
):
    """
    Create sorted box plots grouped by flip type category with overlay points.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define color map for flip types
    flip_type_colors = {
        'Number error (in process)': '#1f77b4',  # blue
        'Number error (in step\'s intermediate output)': '#ff7f0e',  # orange
        'Incorrect fact retrieval': '#2ca02c',  # green
        'Operation flip': '#d62728',  # red
        'Logic flip': '#9467bd',  # purple
        'Trivial logical flip': '#8c564b',  # brown
        'unknown': '#bcbd22'  # yellow-green
    }
    
    # Group rollout data by flip type
    rollout_data_by_type = {}
    for chunk_idx, stats_dict in all_rollout_stats_by_chunk.items():
        flip_type = chunk_flip_types.get(chunk_idx, 'unknown')
        
        if flip_type not in rollout_data_by_type:
            rollout_data_by_type[flip_type] = {
                'total_sentence_count': [],
                'total_token_count': [],
                'total_uncertainty_word_count': []
            }
        
        for stat_key in rollout_data_by_type[flip_type].keys():
            if stat_key in stats_dict:
                rollout_data_by_type[flip_type][stat_key].extend(stats_dict[stat_key])
    
    # Get all types that have data, sorted
    all_types = sorted([t for t in rollout_data_by_type.keys() if any(rollout_data_by_type[t].values())])
    
    if not all_types:
        print("  Warning: No data found for any flip types")
        return
    
    # Statistics to plot
    stats = [
        ('total_sentence_count', 'Total Sentence Count', 'Count'),
        ('total_token_count', 'Total Token Count', 'Count'),
        ('total_uncertainty_word_count', 'Total Uncertainty Word Count', 'Count')
    ]
    
    # Create figure with 3 subplots (one per statistic)
    title_suffix = " (No Outliers)" if remove_outliers else ""
    fig, axes = plt.subplots(1, 3, figsize=(max(28, len(all_types) * 3), 8), sharey=False)
    fig.suptitle(f'Aggregated Statistics by Flip Type{title_suffix}', fontsize=18, fontweight='bold')
    # Add more spacing between subplots to prevent axis overlap
    plt.subplots_adjust(wspace=0.5, left=0.08, right=0.95)
    
    for idx, (stat_key, stat_name, ylabel) in enumerate(stats):
        ax = axes[idx]
        
        # Prepare data for box plots: one list per type
        box_data = []
        box_labels = []
        
        # Collect flip and ablate data points grouped by type
        flip_data_by_type = {t: [] for t in all_types}
        ablate_data_by_type = {t: [] for t in all_types}
        
        for chunk_idx, flip_data in all_flip_data.items():
            if stat_key in flip_data:
                flip_type = chunk_flip_types.get(chunk_idx, 'unknown')
                if flip_type in flip_data_by_type:
                    flip_data_by_type[flip_type].append(flip_data[stat_key])
        
        for chunk_idx, ablate_data in all_ablate_data.items():
            if stat_key in ablate_data:
                # For ablate, we need to get the flip type of the chunk
                flip_type = chunk_flip_types.get(chunk_idx, 'unknown')
                if flip_type in ablate_data_by_type:
                    ablate_data_by_type[flip_type].append(ablate_data[stat_key])
        
        # Build box plot data for each type
        for flip_type in all_types:
            type_data = rollout_data_by_type[flip_type].get(stat_key, [])
            
            # Remove outliers if requested (only for total_sentence_count)
            if remove_outliers and stat_key == 'total_sentence_count' and type_data:
                percentile_threshold = np.percentile(type_data, outlier_percentile * 100)
                type_data = [x for x in type_data if x <= percentile_threshold]
            
            if type_data:  # Only add if there's data
                box_data.append(type_data)
                # Shorten long type names for labels
                label = flip_type.replace('Number error (in step\'s intermediate output)', 'Num error (step output)')
                label = label.replace('Number error (in process)', 'Num error (process)')
                label = label.replace('Incorrect fact retrieval', 'Fact retrieval error')
                label = label.replace('Trivial logical flip', 'Trivial logic flip')
                box_labels.append(label)
        
        if not box_data:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(stat_name, fontsize=14, fontweight='bold')
            continue
        
        # Create box plots for each type
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        # Style the box plots with type colors
        for i, (patch, flip_type) in enumerate(zip(bp['boxes'], all_types[:len(box_data)])):
            color = flip_type_colors.get(flip_type, 'lightblue')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Overlay flip data points (circles) - one per type box plot
        for type_idx, flip_type in enumerate(all_types[:len(box_data)]):
            if flip_type in flip_data_by_type and flip_data_by_type[flip_type]:
                x_pos = type_idx + 1  # Box plot positions start at 1
                np.random.seed(42 + type_idx)  # Different seed per type for jitter
                x_positions = np.random.normal(x_pos, 0.05, len(flip_data_by_type[flip_type]))
                color = flip_type_colors.get(flip_type, '#000000')
                
                ax.scatter(x_positions, flip_data_by_type[flip_type], color=color, s=60, 
                          marker='o', zorder=5, 
                          edgecolors='black', linewidths=1, alpha=0.8)
        
        # Overlay ablate data points (squares with bold edges) - one per type box plot
        for type_idx, flip_type in enumerate(all_types[:len(box_data)]):
            if flip_type in ablate_data_by_type and ablate_data_by_type[flip_type]:
                x_pos = type_idx + 1  # Box plot positions start at 1
                np.random.seed(43 + type_idx)  # Different seed per type for jitter
                x_positions = np.random.normal(x_pos, 0.05, len(ablate_data_by_type[flip_type]))
                color = flip_type_colors.get(flip_type, '#000000')
                
                ax.scatter(x_positions, ablate_data_by_type[flip_type], color=color, s=60, 
                          marker='s', zorder=5, 
                          edgecolors='black', linewidths=2.5, alpha=0.8)
        
        # Set labels and title
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(stat_name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        # Ensure x-axis labels don't overlap - add more padding
        ax.tick_params(axis='x', pad=15)
        # Adjust x-axis limits to give more room for labels
        ax.set_xlim(0.5, len(box_data) + 0.5)
        
        # Add two separate legends if there are overlay points
        has_flip = any(flip_data_by_type.values())
        has_ablate = any(ablate_data_by_type.values())
        
        if has_flip or has_ablate:
            from matplotlib.lines import Line2D
            
            # First legend: Shapes (Flip Chunk vs Ablate Uncertainty) - on the left
            shape_legend_elements = []
            if has_flip:
                shape_legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                                     markerfacecolor='gray', markeredgecolor='black', 
                                                     markersize=10, label='Flip Chunk', linewidth=1))
            if has_ablate:
                shape_legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                                    markerfacecolor='gray', markeredgecolor='black', 
                                                    markersize=10, label='Ablate Uncertainty', linewidth=2.5))
            
            if shape_legend_elements:
                shape_legend = ax.legend(handles=shape_legend_elements, loc='upper left', 
                                        fontsize=9, framealpha=0.9, title='Experiment Type')
                shape_legend.get_title().set_fontsize(9)
                ax.add_artist(shape_legend)  # Keep this legend when adding the second one
            
            # Second legend: Flip type colors - on the right
            overlay_types = set()
            for flip_type in all_types:
                if (flip_type in flip_data_by_type and flip_data_by_type[flip_type]) or \
                   (flip_type in ablate_data_by_type and ablate_data_by_type[flip_type]):
                    overlay_types.add(flip_type)
            
            if overlay_types:
                color_legend_elements = []
                for flip_type in sorted(overlay_types):
                    color = flip_type_colors.get(flip_type, '#000000')
                    label = flip_type.replace('Number error (in step\'s intermediate output)', 'Num error (step)')
                    label = label.replace('Number error (in process)', 'Num error (process)')
                    label = label.replace('Incorrect fact retrieval', 'Fact error')
                    label = label.replace('Trivial logical flip', 'Trivial logic')
                    color_legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                                         markerfacecolor=color, markeredgecolor='black', 
                                                         markersize=8, label=label))
                
                if color_legend_elements:
                    color_legend = ax.legend(handles=color_legend_elements, loc='upper right', 
                                            fontsize=9, framealpha=0.9, title='Flip Type', ncol=1)
                    color_legend.get_title().set_fontsize(9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if remove_outliers:
        filename = 'aggregated_statistics_by_flip_type_no_outliers.png'
    else:
        filename = 'aggregated_statistics_by_flip_type.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved aggregated plot to {output_path}")

def plot_uncertainty_bar_chart_by_flip_type(
    all_rollout_stats_by_chunk: Dict[int, Dict[str, List[float]]],
    all_flip_data: Dict[int, Dict],
    all_ablate_data: Dict[int, Dict],
    chunk_flip_types: Dict[int, str],
    output_dir: str = "visualizations/chunk_statistics_plots"
):
    """
    Create a bar chart showing uncertainty word count by flip type.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define color map for flip types
    flip_type_colors = {
        'Number error (in process)': '#1f77b4',  # blue
        'Number error (in step\'s intermediate output)': '#ff7f0e',  # orange
        'Incorrect fact retrieval': '#2ca02c',  # green
        'Operation flip': '#d62728',  # red
        'Logic flip': '#9467bd',  # purple
        'Trivial logical flip': '#8c564b',  # brown
        'unknown': '#bcbd22'  # yellow-green
    }
    
    # Group data by flip type
    stat_key = 'total_uncertainty_word_count'
    
    # Calculate average uncertainty word count for each flip type from rollouts
    type_rollout_means = {}
    type_rollout_data = {}
    
    for chunk_idx, stats_dict in all_rollout_stats_by_chunk.items():
        flip_type = chunk_flip_types.get(chunk_idx, 'unknown')
        if stat_key in stats_dict and stats_dict[stat_key]:
            if flip_type not in type_rollout_data:
                type_rollout_data[flip_type] = []
            type_rollout_data[flip_type].extend(stats_dict[stat_key])
    
    for flip_type, data in type_rollout_data.items():
        type_rollout_means[flip_type] = np.mean(data) if data else 0
    
    # Get flip and ablate values by type
    flip_values_by_type = {}
    ablate_values_by_type = {}
    
    for chunk_idx, flip_data in all_flip_data.items():
        if stat_key in flip_data:
            flip_type = chunk_flip_types.get(chunk_idx, 'unknown')
            if flip_type not in flip_values_by_type:
                flip_values_by_type[flip_type] = []
            flip_values_by_type[flip_type].append(flip_data[stat_key])
    
    for chunk_idx, ablate_data in all_ablate_data.items():
        if stat_key in ablate_data:
            flip_type = chunk_flip_types.get(chunk_idx, 'unknown')
            if flip_type not in ablate_values_by_type:
                ablate_values_by_type[flip_type] = []
            ablate_values_by_type[flip_type].append(ablate_data[stat_key])
    
    # Get all types that have data
    all_types = sorted(set(list(type_rollout_means.keys()) + 
                          list(flip_values_by_type.keys()) + 
                          list(ablate_values_by_type.keys())))
    
    if not all_types:
        print("  Warning: No data found for any flip types")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(all_types) * 1.5), 8))
    
    # Prepare data for bars
    x_pos = np.arange(len(all_types))
    bar_heights = [type_rollout_means.get(t, 0) for t in all_types]
    bar_colors = [flip_type_colors.get(t, '#bcbd22') for t in all_types]
    
    # Create bars
    bars = ax.bar(x_pos, bar_heights, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Overlay flip data points
    for type_idx, flip_type in enumerate(all_types):
        if flip_type in flip_values_by_type and flip_values_by_type[flip_type]:
            x_positions = np.random.normal(type_idx, 0.1, len(flip_values_by_type[flip_type]))
            color = flip_type_colors.get(flip_type, '#000000')
            ax.scatter(x_positions, flip_values_by_type[flip_type], color=color, s=60,
                      marker='o', zorder=5, edgecolors='black', linewidths=1, alpha=0.8)
    
    # Overlay ablate data points
    for type_idx, flip_type in enumerate(all_types):
        if flip_type in ablate_values_by_type and ablate_values_by_type[flip_type]:
            x_positions = np.random.normal(type_idx, 0.1, len(ablate_values_by_type[flip_type]))
            color = flip_type_colors.get(flip_type, '#000000')
            ax.scatter(x_positions, ablate_values_by_type[flip_type], color=color, s=60,
                      marker='s', zorder=5, edgecolors='black', linewidths=2.5, alpha=0.8)
    
    # Set labels
    ax.set_xlabel('Flip Type', fontsize=13)
    ax.set_ylabel('Total Uncertainty Word Count', fontsize=13)
    ax.set_title('Uncertainty Word Count by Flip Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    
    # Shorten labels for readability
    labels = [t.replace('Number error (in step\'s intermediate output)', 'Num error (step)')
              .replace('Number error (in process)', 'Num error (process)')
              .replace('Incorrect fact retrieval', 'Fact error')
              .replace('Trivial logical flip', 'Trivial logic') for t in all_types]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legends
    from matplotlib.lines import Line2D
    
    shape_legend_elements = []
    if any(flip_values_by_type.values()):
        shape_legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor='gray', markeredgecolor='black', 
                                           markersize=10, label='Flip Chunk', linewidth=1))
    if any(ablate_values_by_type.values()):
        shape_legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                           markerfacecolor='gray', markeredgecolor='black', 
                                           markersize=10, label='Ablate Uncertainty', linewidth=2.5))
    
    if shape_legend_elements:
        shape_legend = ax.legend(handles=shape_legend_elements, loc='upper left', 
                                fontsize=9, framealpha=0.9, title='Experiment Type')
        shape_legend.get_title().set_fontsize(9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'uncertainty_word_count_by_flip_type.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved uncertainty bar chart to {output_path}")

def plot_flip_type_heatmap(
    chunk_flip_types: Dict[int, str],
    output_dir: str = "visualizations/chunk_statistics_plots"
):
    """
    Create a heatmap showing flip types for all chunks.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define color map for flip types
    flip_type_colors = {
        'Number error (in process)': '#1f77b4',  # blue
        'Number error (in step\'s intermediate output)': '#ff7f0e',  # orange
        'Incorrect fact retrieval': '#2ca02c',  # green
        'Operation flip': '#d62728',  # red
        'Logic flip': '#9467bd',  # purple
        'Trivial logical flip': '#8c564b',  # brown
        'unknown': '#bcbd22'  # yellow-green
    }
    
    # Get all chunk indices (0-151)
    all_chunk_indices = sorted(chunk_flip_types.keys())
    if not all_chunk_indices:
        # If no flip types loaded, create empty heatmap
        all_chunk_indices = list(range(152))
    
    # Create heatmap data: one row per chunk
    num_chunks = len(all_chunk_indices)
    heatmap_data = np.zeros((num_chunks, 1))  # Single column heatmap
    
    # Map flip types to numeric values for coloring
    type_to_value = {}
    unique_types = sorted(set(chunk_flip_types.values()))
    for i, flip_type in enumerate(unique_types):
        type_to_value[flip_type] = i + 1
    
    # Fill heatmap with type values
    for i, chunk_idx in enumerate(all_chunk_indices):
        flip_type = chunk_flip_types.get(chunk_idx, 'unknown')
        heatmap_data[i, 0] = type_to_value.get(flip_type, 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(2, max(8, num_chunks * 0.1)))
    
    # Create custom colormap from flip type colors
    colors_list = [flip_type_colors.get(t, '#bcbd22') for t in unique_types]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors_list)
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, vmin=0.5, vmax=len(unique_types) + 0.5)
    
    # Set y-axis labels to chunk indices
    ax.set_yticks(range(num_chunks))
    ax.set_yticklabels([f'Chunk {idx}' for idx in all_chunk_indices], fontsize=8)
    ax.set_xticks([])
    ax.set_xlabel('Flip Type', fontsize=12)
    ax.set_title('Flip Types by Chunk', fontsize=14, fontweight='bold')
    
    # Add colorbar with type labels
    cbar = plt.colorbar(im, ax=ax, ticks=range(1, len(unique_types) + 1))
    cbar.set_ticklabels([t.replace('Number error (in step\'s intermediate output)', 'Num error (step)')
                         .replace('Number error (in process)', 'Num error (process)')
                         .replace('Incorrect fact retrieval', 'Fact error')
                         .replace('Trivial logical flip', 'Trivial logic') for t in unique_types])
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'flip_type_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved flip type heatmap to {output_path}")

def main():
    """Main function to generate flip type statistics plots."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    flip_types_file = str(PROJECT_ROOT / "flip_types.txt")
    output_dir = str(PROJECT_ROOT / "visualizations/chunk_statistics_plots")
    
    print("="*80)
    print("FLIP TYPE STATISTICS PLOTTER")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    try:
        dataset = load_from_disk(dataset_path)
        print(f"✓ Loaded dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Load flip types
    print(f"\nLoading flip types from: {flip_types_file}")
    chunk_flip_types = load_flip_types(flip_types_file)
    print(f"✓ Loaded flip types for {len(chunk_flip_types)} chunks")
    
    # Show flip type distribution
    from collections import Counter
    type_counts = Counter(chunk_flip_types.values())
    print("\nFlip type distribution:")
    for flip_type, count in sorted(type_counts.items()):
        print(f"  {flip_type}: {count} chunks")
    
    # Load experiment results
    print(f"\nLoading experiment results...")
    flip_results, ablate_results = load_experiment_results(flip_file, ablate_file, tokenizer)
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
    
    # Aggregate statistics across all chunks
    print("\n" + "="*80)
    print("AGGREGATING STATISTICS")
    print("="*80)
    
    # Load original chunks for calculating totals
    print("  Loading original chunks...")
    original_chunks_dict = load_original_chunks(dataset)
    print(f"  ✓ Loaded {len(original_chunks_dict)} original chunks")
    
    # Store stats by chunk
    all_rollout_stats_by_chunk = {}
    
    chunks_processed = 0
    for chunk_idx in chunk_indices:
        # Load rollouts for this chunk
        rollouts = load_chunk_rollouts(dataset, chunk_idx)
        
        if not rollouts:
            continue
        
        chunks_processed += 1
        if chunks_processed % 10 == 0:
            print(f"  Processed {chunks_processed}/{len(chunk_indices)} chunks...", end='\r')
        
        # Extract statistics from rollouts (including totals)
        rollout_stats = extract_statistics_from_rollouts(rollouts, chunk_idx, original_chunks_dict, tokenizer)
        
        # Store stats for this chunk
        all_rollout_stats_by_chunk[chunk_idx] = rollout_stats
    
    print(f"\n  ✓ Processed {chunks_processed} chunks")
    total_data_points = sum(sum(len(v) for v in stats.values()) for stats in all_rollout_stats_by_chunk.values())
    print(f"  ✓ Collected {total_data_points} data points across all chunks")
    
    # Create heatmap
    print("\n" + "="*80)
    print("GENERATING FLIP TYPE HEATMAP")
    print("="*80)
    plot_flip_type_heatmap(chunk_flip_types, output_dir)
    
    # Create aggregated plot (with outliers)
    print("\n" + "="*80)
    print("GENERATING AGGREGATED PLOT BY FLIP TYPE")
    print("="*80)
    plot_aggregated_statistics_by_flip_type(all_rollout_stats_by_chunk, flip_results, ablate_results, chunk_flip_types, output_dir, remove_outliers=False)
    
    # Create aggregated plot (without outliers for sentence count)
    print("\n" + "="*80)
    print("GENERATING AGGREGATED PLOT BY FLIP TYPE (NO OUTLIERS)")
    print("="*80)
    plot_aggregated_statistics_by_flip_type(all_rollout_stats_by_chunk, flip_results, ablate_results, chunk_flip_types, output_dir, remove_outliers=True, outlier_percentile=0.25)
    
    # Create uncertainty bar chart
    print("\n" + "="*80)
    print("GENERATING UNCERTAINTY BAR CHART BY FLIP TYPE")
    print("="*80)
    plot_uncertainty_bar_chart_by_flip_type(all_rollout_stats_by_chunk, flip_results, ablate_results, chunk_flip_types, output_dir)
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - Processed {chunks_processed} chunks with data")
    print(f"  - Flip type heatmap saved to: {output_dir}/flip_type_heatmap.png")
    print(f"  - Aggregated plot saved to: {output_dir}/aggregated_statistics_by_flip_type.png")
    print(f"  - Aggregated plot (no outliers) saved to: {output_dir}/aggregated_statistics_by_flip_type_no_outliers.png")
    print(f"  - Uncertainty bar chart saved to: {output_dir}/uncertainty_word_count_by_flip_type.png")
    print(f"\nLegend:")
    print(f"  - Box plots: Distribution from all normal rollouts across all chunks")
    print(f"  - Circles: Flip chunk experiment results")
    print(f"  - Squares: Ablate uncertainty experiment results")

if __name__ == "__main__":
    main()

