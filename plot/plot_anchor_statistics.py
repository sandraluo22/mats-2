#!/usr/bin/env python3
"""
Create aggregated box plots comparing anchor vs non-anchor chunks,
with function tag color coding maintained.
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Set
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Uncertainty indicators
UNCERTAINTY_WORDS = [
    "wait", "alternatively", "perhaps", "reconsider", "double-check", 
    "unlikely", "maybe", "might", "could be", "possibly", "doubt",
    "uncertain", "unsure", "hmm", "actually", "let me think"
]

def count_sentences(text: str) -> int:
    """Count sentences in text."""
    if not text:
        return 0
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return len(sentences)

def count_uncertainty_words(text: str) -> int:
    """Count uncertainty words in text."""
    if not text:
        return 0
    text_lower = text.lower()
    total_count = 0
    for word in UNCERTAINTY_WORDS:
        total_count += text_lower.count(word.lower())
    return total_count

def load_chunk_rollouts(dataset, chunk_idx: int) -> List[Dict]:
    """Load rollouts for a specific chunk."""
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
    """Load original chunks."""
    chunks_dict = {}
    for ex in dataset:
        path = ex.get('path', '')
        if 'chunks_labeled.json' in path and 'incorrect_base_solution' in path and 'forced_answer' not in path:
            try:
                chunks_labeled = json.loads(ex.get('content', '[]'))
                for chunk_data in chunks_labeled:
                    chunk_idx = chunk_data.get('chunk_idx')
                    chunk_text = chunk_data.get('chunk', '')
                    if chunk_idx is not None:
                        chunks_dict[chunk_idx] = chunk_text
                break
            except json.JSONDecodeError:
                continue
    return chunks_dict

def load_chunk_function_tags(dataset) -> Dict[int, List[str]]:
    """Load function tags for each chunk."""
    chunk_tags = {}
    for ex in dataset:
        path = ex.get('path', '')
        if 'chunks_labeled.json' in path and 'incorrect_base_solution' in path and 'forced_answer' not in path:
            try:
                chunks_labeled = json.loads(ex.get('content', '[]'))
                for chunk_data in chunks_labeled:
                    chunk_idx = chunk_data.get('chunk_idx')
                    function_tags = chunk_data.get('function_tags', [])
                    if chunk_idx is not None:
                        chunk_tags[chunk_idx] = function_tags if function_tags else ['unknown']
                break
            except json.JSONDecodeError:
                continue
    return chunk_tags

def load_anchors(anchors_file: str) -> Set[int]:
    """Load anchor chunk indices."""
    anchor_set = set()
    if not Path(anchors_file).exists():
        return anchor_set
    with open(anchors_file, 'r') as f:
        data = json.load(f)
        for anchor in data.get('anchors', []):
            chunk_idx = anchor.get('chunk_idx')
            if chunk_idx is not None:
                anchor_set.add(chunk_idx)
    return anchor_set

def extract_statistics_from_rollouts(rollouts: List[Dict], chunk_idx: int, 
                                     original_chunks_dict: Dict[int, str], 
                                     tokenizer) -> Dict[str, List[float]]:
    """Extract statistics from rollouts."""
    stats = {
        'total_sentence_count': [],
        'total_token_count': [],
        'total_uncertainty_word_count': []
    }
    sorted_original_indices = sorted(original_chunks_dict.keys())
    previous_chunks_list = [original_chunks_dict[idx] for idx in sorted_original_indices if idx < chunk_idx]
    previous_chunks_text = " ".join(previous_chunks_list) if previous_chunks_list else ""
    
    for rollout_data in rollouts:
        rollout_text = rollout_data.get('rollout', '')
        if not rollout_text:
            continue
        total_text = previous_chunks_text + " " + rollout_text if previous_chunks_text else rollout_text
        stats['total_sentence_count'].append(float(count_sentences(total_text)))
        try:
            stats['total_token_count'].append(float(len(tokenizer.encode(total_text, add_special_tokens=False))))
        except Exception:
            stats['total_token_count'].append(0.0)
        stats['total_uncertainty_word_count'].append(float(count_uncertainty_words(total_text)))
    
    return stats

def load_experiment_results(flip_file: str, ablate_file: str, tokenizer) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """Load experiment results."""
    flip_results = {}
    ablate_results = {}
    
    if Path(flip_file).exists():
        with open(flip_file, 'r') as f:
            flip_data = json.load(f)
            for exp in flip_data.get('experiments', []):
                chunk_idx = exp.get('flipped_chunk_idx')
                if chunk_idx is not None:
                    full_cot = exp.get('full_cot', '')
                    if full_cot:
                        flip_results[chunk_idx] = {
                            'total_sentence_count': float(count_sentences(full_cot)),
                            'total_token_count': float(len(tokenizer.encode(full_cot, add_special_tokens=False))),
                            'total_uncertainty_word_count': float(count_uncertainty_words(full_cot)),
                            'full_cot': full_cot
                        }
    
    if Path(ablate_file).exists():
        with open(ablate_file, 'r') as f:
            ablate_data = json.load(f)
            for exp in ablate_data.get('experiments', []):
                chunk_idx = exp.get('ablated_chunk_idx')
                if chunk_idx is not None:
                    full_cot = exp.get('full_cot', '')
                    if full_cot:
                        ablate_results[chunk_idx] = {
                            'total_sentence_count': float(count_sentences(full_cot)),
                            'total_token_count': float(len(tokenizer.encode(full_cot, add_special_tokens=False))),
                            'total_uncertainty_word_count': float(count_uncertainty_words(full_cot)),
                            'full_cot': full_cot
                        }
    
    return flip_results, ablate_results

def plot_anchor_statistics(
    all_rollout_stats_by_chunk: Dict[int, Dict[str, List[float]]],
    all_flip_data: Dict[int, Dict],
    all_ablate_data: Dict[int, Dict],
    chunk_function_tags: Dict[int, List[str]],
    anchor_set: Set[int],
    output_dir: str = "visualizations/chunk_statistics_plots",
    remove_outliers: bool = False,
    outlier_percentile: float = 0.75
):
    """
    Create box plots comparing anchor vs non-anchor chunks, with function tag color coding.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define color map for function tags
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
    
    # Group rollout data by anchor status
    rollout_data_anchor = {stat: [] for stat in ['total_sentence_count', 'total_token_count', 'total_uncertainty_word_count']}
    rollout_data_non_anchor = {stat: [] for stat in ['total_sentence_count', 'total_token_count', 'total_uncertainty_word_count']}
    
    for chunk_idx, stats_dict in all_rollout_stats_by_chunk.items():
        is_anchor = chunk_idx in anchor_set
        for stat_key in rollout_data_anchor.keys():
            if stat_key in stats_dict:
                if is_anchor:
                    rollout_data_anchor[stat_key].extend(stats_dict[stat_key])
                else:
                    rollout_data_non_anchor[stat_key].extend(stats_dict[stat_key])
    
    # Statistics to plot
    stats = [
        ('total_sentence_count', 'Total Sentence Count', 'Count'),
        ('total_token_count', 'Total Token Count', 'Count'),
        ('total_uncertainty_word_count', 'Total Uncertainty Word Count', 'Count')
    ]
    
    # Create figure with 3 subplots
    title_suffix = " (No Outliers)" if remove_outliers else ""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=False)
    fig.suptitle(f'Anchor vs Non-Anchor Statistics{title_suffix}', fontsize=16, fontweight='bold')
    plt.subplots_adjust(wspace=0.3, left=0.08, right=0.95)
    
    for idx, (stat_key, stat_name, ylabel) in enumerate(stats):
        ax = axes[idx]
        
        # Prepare box plot data
        box_data = []
        box_labels = []
        
        # Anchor data
        anchor_data = rollout_data_anchor[stat_key].copy()
        if remove_outliers and stat_key == 'total_sentence_count' and anchor_data:
            percentile_threshold = np.percentile(anchor_data, outlier_percentile * 100)
            anchor_data = [x for x in anchor_data if x <= percentile_threshold]
        if anchor_data:
            box_data.append(anchor_data)
            box_labels.append('Anchor')
        
        # Non-anchor data
        non_anchor_data = rollout_data_non_anchor[stat_key].copy()
        if remove_outliers and stat_key == 'total_sentence_count' and non_anchor_data:
            percentile_threshold = np.percentile(non_anchor_data, outlier_percentile * 100)
            non_anchor_data = [x for x in non_anchor_data if x <= percentile_threshold]
        if non_anchor_data:
            box_data.append(non_anchor_data)
            box_labels.append('Non-Anchor')
        
        if not box_data:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(stat_name, fontsize=14, fontweight='bold')
            continue
        
        # Create box plots
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        # Style box plots (light gray for anchor/non-anchor distinction)
        for patch in bp['boxes']:
            patch.set_facecolor('#e0e0e0')
            patch.set_alpha(0.7)
        
        # Collect flip and ablate data points with function tag colors
        flip_data_points = []
        ablate_data_points = []
        flip_colors = []
        ablate_colors = []
        flip_x_positions = []
        ablate_x_positions = []
        
        # Process flip data
        for chunk_idx, flip_data in all_flip_data.items():
            if stat_key in flip_data:
                is_anchor = chunk_idx in anchor_set
                x_pos = 1 if is_anchor else 2  # 1 for anchor, 2 for non-anchor
                tags = chunk_function_tags.get(chunk_idx, ['unknown'])
                tag = tags[0] if tags else 'unknown'
                color = function_tag_colors.get(tag, '#000000')
                
                flip_data_points.append(flip_data[stat_key])
                flip_colors.append(color)
                flip_x_positions.append(x_pos)
        
        # Process ablate data
        for chunk_idx, ablate_data in all_ablate_data.items():
            if stat_key in ablate_data:
                is_anchor = chunk_idx in anchor_set
                x_pos = 1 if is_anchor else 2
                tags = chunk_function_tags.get(chunk_idx, ['unknown'])
                tag = tags[0] if tags else 'unknown'
                color = function_tag_colors.get(tag, '#000000')
                
                ablate_data_points.append(ablate_data[stat_key])
                ablate_colors.append(color)
                ablate_x_positions.append(x_pos)
        
        # Overlay flip data points (circles) - bold if anchor
        if flip_data_points:
            for i, (x_pos, y_val, color) in enumerate(zip(flip_x_positions, flip_data_points, flip_colors)):
                is_anchor_point = x_pos == 1
                np.random.seed(42 + i)
                x_jitter = np.random.normal(x_pos, 0.05)
                edgewidth = 2.5 if is_anchor_point else 1.0  # Bold edges for anchors
                ax.scatter(x_jitter, y_val, color=color, s=80 if is_anchor_point else 60, 
                          marker='o', zorder=5, 
                          edgecolors='black', linewidths=edgewidth, alpha=0.8)
        
        # Overlay ablate data points (squares) - bold if anchor
        if ablate_data_points:
            for i, (x_pos, y_val, color) in enumerate(zip(ablate_x_positions, ablate_data_points, ablate_colors)):
                is_anchor_point = x_pos == 1
                np.random.seed(43 + i)
                x_jitter = np.random.normal(x_pos, 0.05)
                edgewidth = 3.0 if is_anchor_point else 2.5  # Bold edges for anchors
                ax.scatter(x_jitter, y_val, color=color, s=80 if is_anchor_point else 60, 
                          marker='s', zorder=5, 
                          edgecolors='black', linewidths=edgewidth, alpha=0.8)
        
        # Set labels and title
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(stat_name, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=11)
    
    # Create legends
    # Legend 1: Experiment type (shapes) - left side
    from matplotlib.lines import Line2D
    legend1_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, markeredgecolor='black', markeredgewidth=1, label='Flip Chunk'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=10, markeredgecolor='black', markeredgewidth=2.5, label='Ablate Uncertainty')
    ]
    fig.legend(handles=legend1_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    # Legend 2: Function tags (colors) - right side
    legend2_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', linewidth=0.5) 
                       for tag, color in function_tag_colors.items()]
    legend2_labels = [tag.replace('_', ' ').title() for tag in function_tag_colors.keys()]
    fig.legend(legend2_elements, legend2_labels, loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
    
    # Save plot
    output_path = Path(output_dir) / f"anchor_statistics{title_suffix.replace(' ', '_').lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved anchor statistics plot to {output_path}")

def main():
    """Main function."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    anchors_file = str(PROJECT_ROOT / "visualizations/analysis/anchors.json")
    output_dir = str(PROJECT_ROOT / "visualizations/chunk_statistics_plots")
    
    print("="*80)
    print("ANCHOR STATISTICS PLOTTING")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ Loaded dataset")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    print("✓ Loaded tokenizer")
    
    # Load anchors
    print("Loading anchors...")
    anchor_set = load_anchors(anchors_file)
    print(f"✓ Loaded {len(anchor_set)} anchor chunks")
    
    # Load function tags and original chunks
    print("Loading function tags and original chunks...")
    chunk_function_tags = load_chunk_function_tags(dataset)
    original_chunks_dict = load_original_chunks(dataset)
    print(f"✓ Loaded function tags for {len(chunk_function_tags)} chunks")
    
    # Load experiment results
    print("Loading experiment results...")
    flip_results, ablate_results = load_experiment_results(flip_file, ablate_file, tokenizer)
    print(f"✓ Loaded {len(flip_results)} flip experiments")
    print(f"✓ Loaded {len(ablate_results)} ablate experiments")
    
    # Get all chunk indices
    chunk_indices = set()
    for ex in dataset:
        path = ex.get('path', '')
        match = re.search(r'chunk_(\d+)/solutions\.json', path)
        if match:
            chunk_indices.add(int(match.group(1)))
    chunk_indices = sorted(chunk_indices)
    
    # Aggregate control statistics
    print("\nAggregating control statistics...")
    all_rollout_stats_by_chunk = {}
    for chunk_idx in chunk_indices:
        rollouts = load_chunk_rollouts(dataset, chunk_idx)
        if rollouts:
            rollout_stats = extract_statistics_from_rollouts(rollouts, chunk_idx, original_chunks_dict, tokenizer)
            all_rollout_stats_by_chunk[chunk_idx] = rollout_stats
    
    print(f"✓ Collected statistics for {len(all_rollout_stats_by_chunk)} chunks")
    
    # Create plots
    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)
    
    plot_anchor_statistics(
        all_rollout_stats_by_chunk,
        flip_results,
        ablate_results,
        chunk_function_tags,
        anchor_set,
        output_dir,
        remove_outliers=False
    )
    
    plot_anchor_statistics(
        all_rollout_stats_by_chunk,
        flip_results,
        ablate_results,
        chunk_function_tags,
        anchor_set,
        output_dir,
        remove_outliers=True
    )
    
    print("\n" + "="*80)
    print("ANCHOR STATISTICS PLOTTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

