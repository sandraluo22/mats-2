#!/usr/bin/env python3
"""
Create aggregated box and whisker plots for all chunks with overlay points from
flip_chunk_results.json and ablate_uncertainty_results.json.

Creates one aggregated plot showing:
- Box plots from all normal rollouts across all chunks
- All flip chunk experiment results as overlay points
- All ablate uncertainty experiment results as overlay points
- 3 statistics side by side: sentence count, uncertainty word count, uncertainty occurrences
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

def extract_statistics_from_rollouts(rollouts: List[Dict], chunk_idx: int, original_chunks_dict: Dict[int, str], tokenizer: AutoTokenizer) -> Dict[str, List[float]]:
    """
    Extract statistics from rollouts for box plot, including total counts.
    Uses actual chunk_idx from data to properly align chunks.
    
    Args:
        rollouts: List of rollout dictionaries
        chunk_idx: Current chunk index
        original_chunks_dict: Dictionary mapping chunk_idx to chunk text
        tokenizer: Tokenizer for counting tokens
    
    Returns:
        Dictionary with lists of values for each statistic (both rollout-only and total)
    """
    sentence_counts = []
    token_counts = []
    uncertainty_word_counts = []
    
    # Total counts (include all previous chunks + current rollout)
    total_sentence_counts = []
    total_token_counts = []
    total_uncertainty_word_counts = []
    
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

def load_chunk_function_tags(dataset) -> Dict[int, List[str]]:
    """
    Load function tags for each chunk from chunks_labeled.json.
    
    Returns:
        Dictionary mapping chunk_idx to list of function_tags
    """
    chunk_tags = {}
    
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
        for chunk_data in chunks_labeled:
            chunk_idx = chunk_data.get('chunk_idx')
            function_tags = chunk_data.get('function_tags', [])
            if chunk_idx is not None:
                chunk_tags[chunk_idx] = function_tags if function_tags else ['unknown']
    
    return chunk_tags

def load_experiment_results(flip_file: str, ablate_file: str, tokenizer: AutoTokenizer) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """
    Load experiment results from JSON files and calculate total counts from full_cot.
    
    Args:
        flip_file: Path to flip_chunk_results.json
        ablate_file: Path to ablate_uncertainty_results.json
        tokenizer: Tokenizer for counting tokens
    
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
                    # Get full_cot text (includes previous chunks + generated continuation)
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

def plot_aggregated_statistics(
    all_rollout_stats_by_chunk: Dict[int, Dict[str, List[float]]],
    all_flip_data: Dict[int, Dict],
    all_ablate_data: Dict[int, Dict],
    chunk_function_tags: Dict[int, List[str]],
    output_dir: str = "visualizations/chunk_statistics_plots",
    remove_outliers: bool = False,
    outlier_percentile: float = 0.75
):
    """
    Create sorted box plots grouped by function tag category with overlay points.
    
    Args:
        all_rollout_stats_by_chunk: Dictionary mapping chunk_idx to dict of statistics lists
        all_flip_data: Dictionary mapping chunk_idx to flip experiment data
        all_ablate_data: Dictionary mapping chunk_idx to ablate experiment data
        chunk_function_tags: Dictionary mapping chunk_idx to list of function_tags
        output_dir: Directory to save plots
        remove_outliers: Whether to remove outliers (keep bottom percentile)
        outlier_percentile: Percentile threshold (e.g., 0.25 means keep bottom 25%, remove top 75%)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define color map for function tags (shared across all plots)
    function_tag_colors = {
        'problem_setup': '#1f77b4',  # blue
        'plan_generation': '#ff7f0e',  # orange
        'fact_retrieval': '#2ca02c',  # green
        'active_computation': '#d62728',  # red
        'self_checking': '#9467bd',  # purple
        'uncertainty_management': '#8c564b',  # brown
        'result_consolidation': '#e377c2',  # pink
        'final_answer_emission': '#7f7f7f',  # gray
        'unknown': '#bcbd22'  # yellow-green
    }
    
    # Group rollout data by function tag
    rollout_data_by_tag = {}
    for chunk_idx, stats_dict in all_rollout_stats_by_chunk.items():
        tags = chunk_function_tags.get(chunk_idx, ['unknown'])
        tag = tags[0] if tags else 'unknown'
        
        if tag not in rollout_data_by_tag:
            rollout_data_by_tag[tag] = {
                'total_sentence_count': [],
                'total_token_count': [],
                'total_uncertainty_word_count': []
            }
        
        for stat_key in rollout_data_by_tag[tag].keys():
            if stat_key in stats_dict:
                rollout_data_by_tag[tag][stat_key].extend(stats_dict[stat_key])
    
    # Get all tags that have data, sorted
    all_tags = sorted([tag for tag in rollout_data_by_tag.keys() if any(rollout_data_by_tag[tag].values())])
    
    if not all_tags:
        print("  Warning: No data found for any function tags")
        return
    
    # Statistics to plot (using total counts)
    stats = [
        ('total_sentence_count', 'Total Sentence Count', 'Count'),
        ('total_token_count', 'Total Token Count', 'Count'),
        ('total_uncertainty_word_count', 'Total Uncertainty Word Count', 'Count')
    ]
    
    # Create figure with 3 subplots (one per statistic)
    # Each subplot will have multiple box plots (one per function tag category)
    title_suffix = " (No Outliers)" if remove_outliers else ""
    fig, axes = plt.subplots(1, 3, figsize=(max(20, len(all_tags) * 2), 8), sharey=False)
    fig.suptitle(f'Aggregated Statistics by Function Tag{title_suffix}', fontsize=18, fontweight='bold')
    
    for idx, (stat_key, stat_name, ylabel) in enumerate(stats):
        ax = axes[idx]
        
        # Prepare data for box plots: one list per tag
        box_data = []
        box_labels = []
        
        # Collect flip and ablate data points grouped by tag
        flip_data_by_tag = {tag: [] for tag in all_tags}
        ablate_data_by_tag = {tag: [] for tag in all_tags}
        
        for chunk_idx, flip_data in all_flip_data.items():
            if stat_key in flip_data:
                tags = chunk_function_tags.get(chunk_idx, ['unknown'])
                tag = tags[0] if tags else 'unknown'
                if tag in flip_data_by_tag:
                    flip_data_by_tag[tag].append(flip_data[stat_key])
        
        for chunk_idx, ablate_data in all_ablate_data.items():
            if stat_key in ablate_data:
                tags = chunk_function_tags.get(chunk_idx, ['unknown'])
                tag = tags[0] if tags else 'unknown'
                if tag in ablate_data_by_tag:
                    ablate_data_by_tag[tag].append(ablate_data[stat_key])
        
        # Build box plot data for each tag
        for tag in all_tags:
            tag_data = rollout_data_by_tag[tag].get(stat_key, [])
            
            # Remove outliers if requested (only for total_sentence_count)
            if remove_outliers and stat_key == 'total_sentence_count' and tag_data:
                percentile_threshold = np.percentile(tag_data, outlier_percentile * 100)
                tag_data = [x for x in tag_data if x <= percentile_threshold]
            
            if tag_data:  # Only add if there's data
                box_data.append(tag_data)
                box_labels.append(tag.replace('_', ' ').title())
        
        if not box_data:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(stat_name, fontsize=14, fontweight='bold')
            continue
        
        # Create box plots for each tag
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        # Style the box plots with tag colors
        for i, (patch, tag) in enumerate(zip(bp['boxes'], all_tags[:len(box_data)])):
            color = function_tag_colors.get(tag, 'lightblue')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Overlay flip data points (circles) - one per tag box plot
        for tag_idx, tag in enumerate(all_tags[:len(box_data)]):
            if tag in flip_data_by_tag and flip_data_by_tag[tag]:
                x_pos = tag_idx + 1  # Box plot positions start at 1
                np.random.seed(42 + tag_idx)  # Different seed per tag for jitter
                x_positions = np.random.normal(x_pos, 0.05, len(flip_data_by_tag[tag]))
                color = function_tag_colors.get(tag, '#000000')
                
                ax.scatter(x_positions, flip_data_by_tag[tag], color=color, s=60, 
                          marker='o', zorder=5, 
                          edgecolors='black', linewidths=1, alpha=0.8)
        
        # Overlay ablate data points (squares with bold edges) - one per tag box plot
        for tag_idx, tag in enumerate(all_tags[:len(box_data)]):
            if tag in ablate_data_by_tag and ablate_data_by_tag[tag]:
                x_pos = tag_idx + 1  # Box plot positions start at 1
                np.random.seed(43 + tag_idx)  # Different seed per tag for jitter
                x_positions = np.random.normal(x_pos, 0.05, len(ablate_data_by_tag[tag]))
                color = function_tag_colors.get(tag, '#000000')
                
                ax.scatter(x_positions, ablate_data_by_tag[tag], color=color, s=60, 
                          marker='s', zorder=5, 
                          edgecolors='black', linewidths=2.5, alpha=0.8)
        
        # Set labels and title
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(stat_name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        # Add two separate legends if there are overlay points
        has_flip = any(flip_data_by_tag.values())
        has_ablate = any(ablate_data_by_tag.values())
        
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
            
            # Second legend: Function tag colors - on the right
            overlay_tags = set()
            for tag in all_tags:
                if (tag in flip_data_by_tag and flip_data_by_tag[tag]) or \
                   (tag in ablate_data_by_tag and ablate_data_by_tag[tag]):
                    overlay_tags.add(tag)
            
            if overlay_tags:
                color_legend_elements = []
                for tag in sorted(overlay_tags):
                    color = function_tag_colors.get(tag, '#000000')
                    color_legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                                         markerfacecolor=color, markeredgecolor='black', 
                                                         markersize=8, label=tag.replace('_', ' ').title()))
                
                if color_legend_elements:
                    color_legend = ax.legend(handles=color_legend_elements, loc='upper right', 
                                            fontsize=9, framealpha=0.9, title='Function Tag', ncol=1)
                    color_legend.get_title().set_fontsize(9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if remove_outliers:
        filename = 'aggregated_statistics_no_outliers.png'
    else:
        filename = 'aggregated_statistics.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved aggregated plot to {output_path}")

def plot_split_computation_statistics(
    all_rollout_stats_by_chunk: Dict[int, Dict[str, List[float]]],
    all_flip_data: Dict[int, Dict],
    all_ablate_data: Dict[int, Dict],
    chunk_function_tags: Dict[int, List[str]],
    output_dir: str = "visualizations/chunk_statistics_plots",
    remove_outliers: bool = False,
    outlier_percentile: float = 0.75
):
    """
    Create aggregated box plots with active_computation split into (process) and (result).
    
    Args:
        all_rollout_stats_by_chunk: Dictionary mapping chunk_idx to dict of statistics lists
        all_flip_data: Dictionary mapping chunk_idx to flip experiment data
        all_ablate_data: Dictionary mapping chunk_idx to ablate experiment data
        chunk_function_tags: Dictionary mapping chunk_idx to list of function_tags
        output_dir: Directory to save plots
        remove_outliers: Whether to remove outliers (keep bottom percentile)
        outlier_percentile: Percentile threshold (e.g., 0.25 means keep bottom 25%, remove top 75%)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define process chunk indices
    process_chunk_indices = {45, 49, 56, 59, 74, 76, 78, 80, 82, 89, 92, 96, 98, 102, 118, 120, 122, 134, 136, 138, 140, 142, 144}
    
    # Define color map for function tags (shared across all plots)
    function_tag_colors = {
        'problem_setup': '#1f77b4',  # blue
        'plan_generation': '#ff7f0e',  # orange
        'fact_retrieval': '#2ca02c',  # green
        'active_computation': '#d62728',  # red
        'active_computation (process)': '#ff4444',  # lighter red
        'active_computation (result)': '#cc0000',  # darker red
        'self_checking': '#9467bd',  # purple
        'uncertainty_management': '#8c564b',  # brown
        'result_consolidation': '#e377c2',  # pink
        'final_answer_emission': '#7f7f7f',  # gray
        'unknown': '#bcbd22'  # yellow-green
    }
    
    # Group rollout data by function tag, splitting active_computation
    rollout_data_by_tag = {}
    for chunk_idx, stats_dict in all_rollout_stats_by_chunk.items():
        tags = chunk_function_tags.get(chunk_idx, ['unknown'])
        tag = tags[0] if tags else 'unknown'
        
        # Split active_computation into (process) and (result)
        if tag == 'active_computation':
            if chunk_idx in process_chunk_indices:
                tag = 'active_computation (process)'
            else:
                tag = 'active_computation (result)'
        
        if tag not in rollout_data_by_tag:
            rollout_data_by_tag[tag] = {
                'total_sentence_count': [],
                'total_token_count': [],
                'total_uncertainty_word_count': []
            }
        
        for stat_key in rollout_data_by_tag[tag].keys():
            if stat_key in stats_dict:
                rollout_data_by_tag[tag][stat_key].extend(stats_dict[stat_key])
    
    # Get all tags that have data, sorted
    all_tags = sorted([tag for tag in rollout_data_by_tag.keys() if any(rollout_data_by_tag[tag].values())])
    
    if not all_tags:
        print("  Warning: No data found for any function tags")
        return
    
    # Statistics to plot (using total counts)
    stats = [
        ('total_sentence_count', 'Total Sentence Count', 'Count'),
        ('total_token_count', 'Total Token Count', 'Count'),
        ('total_uncertainty_word_count', 'Total Uncertainty Word Count', 'Count')
    ]
    
    # Create figure with 3 subplots (one per statistic)
    # Each subplot will have multiple box plots (one per function tag category)
    title_suffix = " (No Outliers)" if remove_outliers else ""
    fig, axes = plt.subplots(1, 3, figsize=(max(20, len(all_tags) * 2), 8), sharey=False)
    fig.suptitle(f'Aggregated Statistics by Function Tag (Split Computation){title_suffix}', fontsize=18, fontweight='bold')
    
    for idx, (stat_key, stat_name, ylabel) in enumerate(stats):
        ax = axes[idx]
        
        # Prepare data for box plots: one list per tag
        box_data = []
        box_labels = []
        
        # Collect flip and ablate data points grouped by tag
        flip_data_by_tag = {tag: [] for tag in all_tags}
        ablate_data_by_tag = {tag: [] for tag in all_tags}
        
        for chunk_idx, flip_data in all_flip_data.items():
            if stat_key in flip_data:
                tags = chunk_function_tags.get(chunk_idx, ['unknown'])
                tag = tags[0] if tags else 'unknown'
                
                # Split active_computation
                if tag == 'active_computation':
                    if chunk_idx in process_chunk_indices:
                        tag = 'active_computation (process)'
                    else:
                        tag = 'active_computation (result)'
                
                if tag in flip_data_by_tag:
                    flip_data_by_tag[tag].append(flip_data[stat_key])
        
        for chunk_idx, ablate_data in all_ablate_data.items():
            if stat_key in ablate_data:
                tags = chunk_function_tags.get(chunk_idx, ['unknown'])
                tag = tags[0] if tags else 'unknown'
                
                # Split active_computation
                if tag == 'active_computation':
                    if chunk_idx in process_chunk_indices:
                        tag = 'active_computation (process)'
                    else:
                        tag = 'active_computation (result)'
                
                if tag in ablate_data_by_tag:
                    ablate_data_by_tag[tag].append(ablate_data[stat_key])
        
        # Build box plot data for each tag
        for tag in all_tags:
            tag_data = rollout_data_by_tag[tag].get(stat_key, [])
            
            # Remove outliers if requested (only for total_sentence_count)
            if remove_outliers and stat_key == 'total_sentence_count' and tag_data:
                percentile_threshold = np.percentile(tag_data, outlier_percentile * 100)
                tag_data = [x for x in tag_data if x <= percentile_threshold]
            
            if tag_data:  # Only add if there's data
                box_data.append(tag_data)
                box_labels.append(tag.replace('_', ' ').title())
        
        if not box_data:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(stat_name, fontsize=14, fontweight='bold')
            continue
        
        # Create box plots for each tag
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        # Style the box plots with tag colors
        for i, (patch, tag) in enumerate(zip(bp['boxes'], all_tags[:len(box_data)])):
            color = function_tag_colors.get(tag, 'lightblue')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Overlay flip data points (circles) - one per tag box plot
        for tag_idx, tag in enumerate(all_tags[:len(box_data)]):
            if tag in flip_data_by_tag and flip_data_by_tag[tag]:
                x_pos = tag_idx + 1  # Box plot positions start at 1
                np.random.seed(42 + tag_idx)  # Different seed per tag for jitter
                x_positions = np.random.normal(x_pos, 0.05, len(flip_data_by_tag[tag]))
                color = function_tag_colors.get(tag, '#000000')
                
                ax.scatter(x_positions, flip_data_by_tag[tag], color=color, s=60, 
                          marker='o', zorder=5, 
                          edgecolors='black', linewidths=1, alpha=0.8)
        
        # Overlay ablate data points (squares with bold edges) - one per tag box plot
        for tag_idx, tag in enumerate(all_tags[:len(box_data)]):
            if tag in ablate_data_by_tag and ablate_data_by_tag[tag]:
                x_pos = tag_idx + 1  # Box plot positions start at 1
                np.random.seed(43 + tag_idx)  # Different seed per tag for jitter
                x_positions = np.random.normal(x_pos, 0.05, len(ablate_data_by_tag[tag]))
                color = function_tag_colors.get(tag, '#000000')
                
                ax.scatter(x_positions, ablate_data_by_tag[tag], color=color, s=60, 
                          marker='s', zorder=5, 
                          edgecolors='black', linewidths=2.5, alpha=0.8)
        
        # Set labels and title
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(stat_name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        # Add two separate legends if there are overlay points
        has_flip = any(flip_data_by_tag.values())
        has_ablate = any(ablate_data_by_tag.values())
        
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
            
            # Second legend: Function tag colors - on the right
            overlay_tags = set()
            for tag in all_tags:
                if (tag in flip_data_by_tag and flip_data_by_tag[tag]) or \
                   (tag in ablate_data_by_tag and ablate_data_by_tag[tag]):
                    overlay_tags.add(tag)
            
            if overlay_tags:
                color_legend_elements = []
                for tag in sorted(overlay_tags):
                    color = function_tag_colors.get(tag, '#000000')
                    color_legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                                         markerfacecolor=color, markeredgecolor='black', 
                                                         markersize=8, label=tag.replace('_', ' ').title()))
                
                if color_legend_elements:
                    color_legend = ax.legend(handles=color_legend_elements, loc='upper right', 
                                            fontsize=9, framealpha=0.9, title='Function Tag', ncol=1)
                    color_legend.get_title().set_fontsize(9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if remove_outliers:
        filename = 'split_computation_aggregated_statistics_no_outliers.png'
    else:
        filename = 'split_computation_aggregated_statistics.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved split computation aggregated plot to {output_path}")

def main():
    """Main function to generate aggregated plot."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    output_dir = str(PROJECT_ROOT / "visualizations/chunk_statistics_plots")
    
    print("="*80)
    print("AGGREGATED STATISTICS PLOTTER")
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
    
    # Load experiment results
    print(f"\nLoading experiment results...")
    flip_results, ablate_results = load_experiment_results(flip_file, ablate_file, tokenizer)
    print(f"✓ Loaded {len(flip_results)} flip experiments")
    print(f"✓ Loaded {len(ablate_results)} ablate experiments")
    
    # Load chunk function tags
    print(f"\nLoading chunk function tags...")
    chunk_function_tags = load_chunk_function_tags(dataset)
    print(f"✓ Loaded function tags for {len(chunk_function_tags)} chunks")
    
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
    print(f"  ✓ Loaded {len(original_chunks_dict)} original chunks (indices: {min(original_chunks_dict.keys())} to {max(original_chunks_dict.keys())})")
    
    # Store stats by chunk (to preserve chunk_idx for grouping by function tag)
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
    
    # Create aggregated plot (with outliers)
    print("\n" + "="*80)
    print("GENERATING AGGREGATED PLOT")
    print("="*80)
    plot_aggregated_statistics(all_rollout_stats_by_chunk, flip_results, ablate_results, chunk_function_tags, output_dir, remove_outliers=False)
    
    # Create aggregated plot (without outliers for sentence count)
    print("\n" + "="*80)
    print("GENERATING AGGREGATED PLOT (NO OUTLIERS)")
    print("="*80)
    plot_aggregated_statistics(all_rollout_stats_by_chunk, flip_results, ablate_results, chunk_function_tags, output_dir, remove_outliers=True, outlier_percentile=0.25)
    
    # Create split computation aggregated plot (with outliers)
    print("\n" + "="*80)
    print("GENERATING SPLIT COMPUTATION AGGREGATED PLOT")
    print("="*80)
    plot_split_computation_statistics(all_rollout_stats_by_chunk, flip_results, ablate_results, chunk_function_tags, output_dir, remove_outliers=False)
    
    # Create split computation aggregated plot (without outliers for sentence count)
    print("\n" + "="*80)
    print("GENERATING SPLIT COMPUTATION AGGREGATED PLOT (NO OUTLIERS)")
    print("="*80)
    plot_split_computation_statistics(all_rollout_stats_by_chunk, flip_results, ablate_results, chunk_function_tags, output_dir, remove_outliers=True, outlier_percentile=0.25)
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - Processed {chunks_processed} chunks with data")
    print(f"  - Aggregated plot saved to: {output_dir}/aggregated_statistics.png")
    print(f"  - Aggregated plot (no outliers) saved to: {output_dir}/aggregated_statistics_no_outliers.png")
    print(f"  - Split computation aggregated plot saved to: {output_dir}/split_computation_aggregated_statistics.png")
    print(f"  - Split computation aggregated plot (no outliers) saved to: {output_dir}/split_computation_aggregated_statistics_no_outliers.png")
    print(f"\nLegend:")
    print(f"  - Blue box plots: Distribution from all normal rollouts across all chunks")
    print(f"  - Red circles: Flip chunk experiment results")
    print(f"  - Green squares: Ablate uncertainty experiment results")

if __name__ == "__main__":
    main()

