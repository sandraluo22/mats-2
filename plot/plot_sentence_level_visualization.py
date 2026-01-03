#!/usr/bin/env python3
"""
Create sentence-level visualization of chain of thought for flip and ablate experiments.

Shows:
- Left subplot: Flip chunk experiments
- Right subplot: Ablate uncertainty experiments
- Y-axis: Control (original COT) at bottom, then chunk experiments
- X-axis: Sentence numbers
- Color coding: Function tags for original chunks, uncertainty color for sentences with uncertainty words
"""

import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from datasets import load_from_disk
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

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    return sentences

def has_uncertainty_word(sentence: str) -> bool:
    """Check if sentence contains any uncertainty word."""
    sentence_lower = sentence.lower()
    for word in UNCERTAINTY_WORDS:
        if word.lower() in sentence_lower:
            return True
    return False

def load_original_chunks(dataset) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
    """
    Load original chunks from chunks_labeled.json, using actual chunk_idx from data.
    
    Returns:
        Tuple of (dict mapping chunk_idx to chunk text, dict mapping chunk_idx to function_tags)
    """
    chunks_dict = {}  # chunk_idx -> chunk_text
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
        # Use actual chunk_idx from data (don't assume sequential)
        for chunk_data in chunks_labeled:
            chunk_idx = chunk_data.get('chunk_idx')
            if chunk_idx is not None:
                chunk_text = chunk_data.get('chunk', '')
                function_tags = chunk_data.get('function_tags', [])
                
                if chunk_text:
                    chunks_dict[chunk_idx] = chunk_text
                    chunk_tags[chunk_idx] = function_tags if function_tags else ['unknown']
    
    return chunks_dict, chunk_tags

def get_control_sentences(chunks_dict: Dict[int, str], chunk_tags: Dict[int, List[str]]) -> Tuple[List[str], List[str], List[bool]]:
    """
    Get sentences from original chain of thought with their function tags and uncertainty flags.
    Uses actual chunk_idx from data.
    
    Returns:
        Tuple of (sentences, function_tags_per_sentence, has_uncertainty_per_sentence)
    """
    all_sentences = []
    all_tags = []
    all_uncertainty = []
    
    # Sort by chunk_idx to process in order
    sorted_chunk_indices = sorted(chunks_dict.keys())
    
    for chunk_idx in sorted_chunk_indices:
        chunk_text = chunks_dict[chunk_idx]
        sentences = split_into_sentences(chunk_text)
        tags = chunk_tags.get(chunk_idx, ['unknown'])
        tag = tags[0] if tags else 'unknown'
        
        for sentence in sentences:
            all_sentences.append(sentence)
            all_tags.append(tag)
            all_uncertainty.append(has_uncertainty_word(sentence))
    
    return all_sentences, all_tags, all_uncertainty

def process_experiment_sentences(full_cot: str, original_chunks_dict: Dict[int, str], chunk_tags: Dict[int, List[str]], 
                                 chunk_idx: int) -> Tuple[List[str], List[Optional[str]], List[bool]]:
    """
    Process sentences from an experiment's full_cot.
    Uses actual chunk_idx from data to properly align chunks.
    
    Note: full_cot starts from the flipped/ablated chunk, so we need to reconstruct
    the full chain by combining previous chunks + full_cot.
    
    Returns:
        Tuple of (sentences, function_tags_per_sentence, has_uncertainty_per_sentence)
    """
    # Get all chunk indices that exist in original_chunks_dict, sorted
    sorted_original_indices = sorted(original_chunks_dict.keys())
    
    # Get text from all original chunks before the experiment chunk
    # Include all chunks with chunk_idx < experiment chunk_idx
    previous_chunks_list = []
    for orig_chunk_idx in sorted_original_indices:
        if orig_chunk_idx < chunk_idx:
            previous_chunks_list.append(original_chunks_dict[orig_chunk_idx])
    
    previous_chunks_text = " ".join(previous_chunks_list) if previous_chunks_list else ""
    
    # Reconstruct full chain: previous chunks + full_cot (which starts from flipped/ablated chunk)
    if previous_chunks_text:
        full_chain = previous_chunks_text + " " + full_cot
    else:
        full_chain = full_cot
    
    sentences = split_into_sentences(full_chain)
    tags = []
    uncertainty_flags = []
    
    # Split previous chunks into sentences to know where original chunks end
    if previous_chunks_text:
        prev_sentences = split_into_sentences(previous_chunks_text)
        prev_sentence_count = len(prev_sentences)
    else:
        prev_sentence_count = 0
    
    # Process each sentence
    for i, sentence in enumerate(sentences):
        has_uncertainty = has_uncertainty_word(sentence)
        uncertainty_flags.append(has_uncertainty)
        
        # If sentence is from original chunks, use original tags
        if i < prev_sentence_count:
            # Find which chunk this sentence belongs to using actual chunk indices
            current_sentence_idx = 0
            tag = 'unknown'
            
            # Iterate through original chunks in order
            for orig_chunk_idx in sorted_original_indices:
                if orig_chunk_idx >= chunk_idx:
                    break  # Stop when we reach the experiment chunk
                
                chunk_text = original_chunks_dict[orig_chunk_idx]
                chunk_sentences = split_into_sentences(chunk_text)
                if current_sentence_idx <= i < current_sentence_idx + len(chunk_sentences):
                    orig_tags = chunk_tags.get(orig_chunk_idx, ['unknown'])
                    tag = orig_tags[0] if orig_tags else 'unknown'
                    break
                current_sentence_idx += len(chunk_sentences)
            tags.append(tag)
        else:
            # Model generated (from flip/ablate chunk onward) - no function tag
            tags.append(None)
    
    return sentences, tags, uncertainty_flags

def plot_sentence_level_visualization(
    flip_file: str,
    ablate_file: str,
    dataset,
    output_dir: str = "visualizations/chunk_statistics_plots"
):
    """
    Create sentence-level visualization for flip and ablate experiments.
    Creates two separate figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original chunks (using actual chunk_idx from data)
    print("Loading original chunks...")
    original_chunks_dict, chunk_tags = load_original_chunks(dataset)
    print(f"✓ Loaded {len(original_chunks_dict)} original chunks (indices: {min(original_chunks_dict.keys())} to {max(original_chunks_dict.keys())})")
    
    # Get control (original COT) sentences
    print("Processing control (original COT)...")
    control_sentences, control_tags, control_uncertainty = get_control_sentences(original_chunks_dict, chunk_tags)
    print(f"✓ Control has {len(control_sentences)} sentences")
    
    # Define colors
    function_tag_colors = {
        'problem_setup': '#1f77b4',  # blue
        'plan_generation': '#ff7f0e',  # orange
        'fact_retrieval': '#2ca02c',  # green
        'active_computation': '#d62728',  # red
        'self_checking': '#9467bd',  # purple
        'uncertainty_management': '#8c564b',  # brown
        'result_consolidation': '#e377c2',  # pink
        'final_answer_emission': '#7f7f7f',  # gray
        'unknown': '#bcbd22',  # yellow-green
        'uncertainty': '#ffd700',  # gold for uncertainty
        'generated': '#cccccc'  # gray for generated (no tag)
    }
    
    # Load and plot flip experiments
    print("\nLoading flip experiments...")
    flip_experiments = []
    if os.path.exists(flip_file):
        with open(flip_file, 'r') as f:
            flip_data = json.load(f)
            for exp in flip_data.get('experiments', []):
                chunk_idx = exp.get('flipped_chunk_idx')
                full_cot = exp.get('full_cot', '')
                if chunk_idx is not None and full_cot:
                    sentences, tags, uncertainty = process_experiment_sentences(
                        full_cot, original_chunks_dict, chunk_tags, chunk_idx
                    )
                    flip_experiments.append({
                        'chunk_idx': chunk_idx,
                        'sentences': sentences,
                        'tags': tags,
                        'uncertainty': uncertainty
                    })
    print(f"✓ Loaded {len(flip_experiments)} flip experiments")
    
    if flip_experiments:
        # Create figure for flip experiments with heatmap subplot
        fig1 = plt.figure(figsize=(18, max(8, len(flip_experiments) + 1) * 0.3))
        gs = fig1.add_gridspec(1, 2, width_ratios=[1, 10], hspace=0.3)
        ax_heatmap = fig1.add_subplot(gs[0, 0])
        ax_main = fig1.add_subplot(gs[0, 1])
        fig1.suptitle('Sentence-Level Chain of Thought: Flip Chunk Experiments', fontsize=16, fontweight='bold')
        
        plot_experiments(ax_main, control_sentences, control_tags, control_uncertainty, 
                         flip_experiments, function_tag_colors, chunk_tags, 'Flip Chunk Experiments')
        plot_chunk_heatmap(ax_heatmap, flip_experiments, chunk_tags, function_tag_colors)
        
        plt.tight_layout()
        output_path1 = os.path.join(output_dir, 'sentence_level_visualization_flip.png')
        plt.savefig(output_path1, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved flip chunk visualization to {output_path1}")
    
    # Load and plot ablate experiments
    print("\nLoading ablate experiments...")
    ablate_experiments = []
    if os.path.exists(ablate_file):
        with open(ablate_file, 'r') as f:
            ablate_data = json.load(f)
            for exp in ablate_data.get('experiments', []):
                chunk_idx = exp.get('ablated_chunk_idx')
                full_cot = exp.get('full_cot', '')
                if chunk_idx is not None and full_cot:
                    sentences, tags, uncertainty = process_experiment_sentences(
                        full_cot, original_chunks_dict, chunk_tags, chunk_idx
                    )
                    ablate_experiments.append({
                        'chunk_idx': chunk_idx,
                        'sentences': sentences,
                        'tags': tags,
                        'uncertainty': uncertainty
                    })
    print(f"✓ Loaded {len(ablate_experiments)} ablate experiments")
    
    if ablate_experiments:
        # Create figure for ablate experiments with heatmap subplot
        fig2 = plt.figure(figsize=(18, max(8, len(ablate_experiments) + 1) * 0.3))
        gs = fig2.add_gridspec(1, 2, width_ratios=[1, 10], hspace=0.3)
        ax_heatmap = fig2.add_subplot(gs[0, 0])
        ax_main = fig2.add_subplot(gs[0, 1])
        fig2.suptitle('Sentence-Level Chain of Thought: Ablate Uncertainty Experiments', fontsize=16, fontweight='bold')
        
        plot_experiments(ax_main, control_sentences, control_tags, control_uncertainty,
                         ablate_experiments, function_tag_colors, chunk_tags, 'Ablate Uncertainty Experiments')
        plot_chunk_heatmap(ax_heatmap, ablate_experiments, chunk_tags, function_tag_colors)
        
        plt.tight_layout()
        output_path2 = os.path.join(output_dir, 'sentence_level_visualization_ablate.png')
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved ablate uncertainty visualization to {output_path2}")

def plot_experiments(ax, control_sentences, control_tags, control_uncertainty, 
                     experiments, function_tag_colors, chunk_tags, title):
    """Plot experiments on a subplot."""
    max_sentences = max(len(control_sentences), 
                       max((len(exp['sentences']) for exp in experiments), default=0))
    
    # Sort experiments by chunk_idx
    experiments = sorted(experiments, key=lambda x: x['chunk_idx'])
    
    # Y-axis positions: control at bottom (0), then experiments (1, 2, ...)
    y_positions = {}
    y_positions['control'] = 0
    
    for i, exp in enumerate(experiments):
        y_positions[exp['chunk_idx']] = i + 1
    
    num_rows = len(experiments) + 1
    
    # Plot control at bottom
    plot_row(ax, control_sentences, control_tags, control_uncertainty, 
             y_positions['control'], max_sentences, function_tag_colors, 'Control (Original COT)', None)
    
    # Plot each experiment
    for exp in experiments:
        # Get function tag for this chunk
        exp_tags = chunk_tags.get(exp['chunk_idx'], ['unknown'])
        exp_tag = exp_tags[0] if exp_tags else 'unknown'
        label = f"Chunk {exp['chunk_idx']}"
        plot_row(ax, exp['sentences'], exp['tags'], exp['uncertainty'],
                 y_positions[exp['chunk_idx']], max_sentences, function_tag_colors, label, exp_tag)
    
    ax.set_xlabel('Sentence Number', fontsize=12)
    ax.set_ylabel('Experiment', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, max_sentences)
    ax.set_ylim(-0.5, num_rows - 0.5)
    
    # Set y-axis ticks and labels with colored squares
    y_ticks = [0] + [i + 1 for i in range(len(experiments))]
    y_labels = ['Control']
    
    # Create labels with colored squares for each chunk
    for exp in experiments:
        exp_tags = chunk_tags.get(exp['chunk_idx'], ['unknown'])
        exp_tag = exp_tags[0] if exp_tags else 'unknown'
        color = function_tag_colors.get(exp_tag, function_tag_colors.get('unknown', '#bcbd22'))
        # Use Unicode square character or create a colored patch
        y_labels.append(f"Chunk {exp['chunk_idx']}")
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Draw vertical red line where control ends
    control_end = len(control_sentences)
    ax.axvline(x=control_end, color='red', linewidth=2, linestyle='--', alpha=0.7, label='Control End')

def plot_chunk_heatmap(ax, experiments, chunk_tags, function_tag_colors):
    """Plot a heatmap showing chunk colors for all chunks 0-151 (152 total chunks).
    Chunk 0 is at bottom, chunk 151 is at top.
    """
    # Get all chunk indices from 0 to 151 (152 total chunks)
    all_chunk_indices = list(range(152))  # 0 to 151
    
    # Draw colored rectangles for each chunk
    # Position chunk 0 at bottom (y=0), chunk 151 at top (y=151)
    for chunk_idx in all_chunk_indices:
        exp_tags = chunk_tags.get(chunk_idx, ['unknown'])
        exp_tag = exp_tags[0] if exp_tags else 'unknown'
        color = function_tag_colors.get(exp_tag, function_tag_colors.get('unknown', '#bcbd22'))
        
        # Position: chunk 0 at y=0, chunk 151 at y=151 (no inversion)
        rect = mpatches.Rectangle((0, chunk_idx - 0.5), 1, 1, 
                                 facecolor=color, edgecolor='none')
        ax.add_patch(rect)
    
    # Set y-axis to show all chunks (0 at bottom, 151 at top)
    ax.set_ylim(-0.5, 151.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    # Show every 20th chunk for readability
    ax.set_yticks(range(0, 152, 20))
    ax.set_yticklabels([str(i) for i in range(0, 152, 20)])
    ax.set_ylabel('Chunk Index', fontsize=10)
    ax.set_title('All Chunks\nFunction Tags', fontsize=10, fontweight='bold')
    # No inversion - chunk 0 at bottom, 151 at top
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)

def plot_row(ax, sentences, tags, uncertainty_flags, y_pos, max_sentences, 
             function_tag_colors, label, chunk_tag=None):
    """Plot a single row (control or experiment) as sentence blocks."""
    for i, (sentence, tag, has_uncertainty) in enumerate(zip(sentences, tags, uncertainty_flags)):
        # Determine color
        if has_uncertainty:
            color = function_tag_colors.get('uncertainty', '#ffd700')
        elif tag:
            color = function_tag_colors.get(tag, function_tag_colors.get('unknown', '#bcbd22'))
        else:
            # Generated (no tag) - gray
            color = function_tag_colors.get('generated', '#cccccc')
        
        # Draw rectangle for sentence (sentence number i, width 1, height 0.8)
        rect = mpatches.Rectangle((i, y_pos - 0.4), 1, 0.8, 
                                 facecolor=color, edgecolor='none')
        ax.add_patch(rect)

def main():
    """Main function."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    output_dir = str(PROJECT_ROOT / "visualizations/chunk_statistics_plots")
    
    print("="*80)
    print("SENTENCE-LEVEL VISUALIZATION")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    try:
        dataset = load_from_disk(dataset_path)
        print(f"✓ Loaded dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Generate visualization
    plot_sentence_level_visualization(flip_file, ablate_file, dataset, output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

