#!/usr/bin/env python3
"""
Plot chunks with repeated final answers similar to aggregated statistics.
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Set
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

def load_repeated_final_answers(repeated_file: str) -> Dict[int, Dict]:
    """Load chunks with repeated final answers."""
    repeated_chunks = {}
    if not Path(repeated_file).exists():
        return repeated_chunks
    
    with open(repeated_file, 'r') as f:
        data = json.load(f)
        for chunk_data in data.get('chunks', []):
            chunk_idx = chunk_data.get('chunk_idx')
            if chunk_idx is not None:
                repeated_chunks[chunk_idx] = {
                    'total_sentence_count': chunk_data.get('total_sentence_count', 0),
                    'total_token_count': chunk_data.get('total_token_count', 0),
                    'total_uncertainty_word_count': chunk_data.get('total_uncertainty_word_count', 0),
                    'final_answer': chunk_data.get('final_answer', ''),
                    'is_correct': chunk_data.get('is_correct', False),
                    'experiment_type': chunk_data.get('experiment_type', ''),
                    'flip_type': chunk_data.get('flip_type', 'unknown')
                }
    
    return repeated_chunks

def plot_repeated_final_answers_statistics(
    all_rollout_stats_by_chunk: Dict[int, Dict[str, List[float]]],
    repeated_chunks_data: Dict[int, Dict],
    chunk_function_tags: Dict[int, List[str]],
    output_dir: str = "visualizations/chunk_statistics_plots"
):
    """
    Create box plots for chunks with repeated final answers, grouped by function tag.
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
    
    # Group rollout data by function tag (control distribution)
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
    
    # Statistics to plot
    stats = [
        ('total_sentence_count', 'Total Sentence Count', 'Count'),
        ('total_token_count', 'Total Token Count', 'Count'),
        ('total_uncertainty_word_count', 'Total Uncertainty Word Count', 'Count')
    ]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(max(20, len(all_tags) * 2), 8), sharey=False)
    fig.suptitle('Chunks with Repeated Final Answers vs Control', fontsize=18, fontweight='bold')
    plt.subplots_adjust(wspace=0.5, left=0.08, right=0.95)
    
    for idx, (stat_key, stat_name, ylabel) in enumerate(stats):
        ax = axes[idx]
        
        # Prepare data for box plots: one list per tag
        box_data = []
        box_labels = []
        
        # Collect repeated final answer data points grouped by tag
        repeated_data_by_tag = {tag: [] for tag in all_tags}
        
        for chunk_idx, repeated_data in repeated_chunks_data.items():
            if stat_key in repeated_data:
                tags = chunk_function_tags.get(chunk_idx, ['unknown'])
                tag = tags[0] if tags else 'unknown'
                if tag in repeated_data_by_tag:
                    repeated_data_by_tag[tag].append(repeated_data[stat_key])
        
        # Build box plot data for each tag (control distribution)
        for tag in all_tags:
            tag_data = rollout_data_by_tag[tag].get(stat_key, [])
            if tag_data:  # Only add if there's data
                box_data.append(tag_data)
                box_labels.append(tag.replace('_', ' ').title())
        
        if not box_data:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(stat_name, fontsize=14, fontweight='bold')
            continue
        
        # Create box plots for each tag (control distribution)
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        # Style the box plots with tag colors
        for i, (patch, tag) in enumerate(zip(bp['boxes'], all_tags[:len(box_data)])):
            color = function_tag_colors.get(tag, 'lightblue')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Overlay repeated final answer data points (red diamonds) - one per tag box plot
        for tag_idx, tag in enumerate(all_tags[:len(box_data)]):
            if tag in repeated_data_by_tag and repeated_data_by_tag[tag]:
                x_pos = tag_idx + 1  # Box plot positions start at 1
                np.random.seed(100 + tag_idx)  # Different seed per tag for jitter
                x_positions = np.random.normal(x_pos, 0.05, len(repeated_data_by_tag[tag]))
                color = function_tag_colors.get(tag, '#000000')
                
                ax.scatter(x_positions, repeated_data_by_tag[tag], color=color, s=100, 
                          marker='D', zorder=5, 
                          edgecolors='red', linewidths=2, alpha=0.8, label='Repeated Final Answer' if tag_idx == 0 else '')
        
        # Set labels and title
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(stat_name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='x', pad=15)
        ax.set_xlim(0.5, len(box_data) + 0.5)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', 
               markersize=12, markeredgecolor='red', markeredgewidth=2, label='Repeated Final Answer')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
    # Save plot
    output_path = Path(output_dir) / "repeated_final_answers_statistics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved repeated final answers plot to {output_path}")

def generate_accuracy_table(
    repeated_chunks_data: Dict[int, Dict],
    chunk_function_tags: Dict[int, List[str]],
    dataset,
    output_dir: str = "visualizations/analysis"
):
    """Generate accuracy table for chunks with repeated final answers."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load control accuracy (from normal rollouts)
    def load_control_accuracy(dataset):
        """Load control accuracy from base_solution.json."""
        for ex in dataset:
            path = ex.get('path', '')
            if 'base_solution.json' in path and 'incorrect_base_solution' in path and 'forced_answer' not in path:
                try:
                    base_solution = json.loads(ex.get('content', '{}'))
                    if isinstance(base_solution, list) and len(base_solution) > 0:
                        base_solution = base_solution[0]
                    final_answer = base_solution.get('final_answer', '')
                    is_correct = base_solution.get('is_correct', False)
                    return final_answer, is_correct
                except json.JSONDecodeError:
                    continue
        return None, None
    
    def load_chunk_rollout_accuracy(dataset, chunk_idx: int):
        """Get accuracy percentage for normal rollouts of a specific chunk."""
        rollouts = load_chunk_rollouts(dataset, chunk_idx)
        if not rollouts:
            return None
        
        correct_count = sum(1 for r in rollouts if r.get('is_correct', False))
        total_count = len(rollouts)
        return (correct_count / total_count * 100) if total_count > 0 else None
    
    control_final_answer, control_is_correct = load_control_accuracy(dataset)
    
    # Prepare table data
    table_rows = []
    
    for chunk_idx in sorted(repeated_chunks_data.keys()):
        chunk_data = repeated_chunks_data[chunk_idx]
        tags = chunk_function_tags.get(chunk_idx, ['unknown'])
        function_tag = tags[0] if tags else 'unknown'
        
        # Get control accuracy for this chunk
        control_accuracy = load_chunk_rollout_accuracy(dataset, chunk_idx)
        
        table_rows.append({
            'chunk_idx': chunk_idx,
            'function_tag': function_tag,
            'flip_type': chunk_data.get('flip_type', 'unknown'),
            'experiment_type': chunk_data.get('experiment_type', ''),
            'final_answer': chunk_data.get('final_answer', ''),
            'is_correct': chunk_data.get('is_correct', False),
            'control_correct_pct': control_accuracy
        })
    
    # Calculate overall control accuracy (average across all chunks)
    all_chunk_indices = set()
    for ex in dataset:
        path = ex.get('path', '')
        match = re.search(r'chunk_(\d+)/solutions\.json', path)
        if match:
            all_chunk_indices.add(int(match.group(1)))
    
    overall_control_accuracies = []
    for chunk_idx in all_chunk_indices:
        acc = load_chunk_rollout_accuracy(dataset, chunk_idx)
        if acc is not None:
            overall_control_accuracies.append(acc)
    
    overall_control_accuracy = np.mean(overall_control_accuracies) if overall_control_accuracies else None
    
    # Write table
    output_path = Path(output_dir) / "repeated_final_answers_accuracy_table.txt"
    
    with open(output_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("ACCURACY TABLE: CHUNKS WITH REPEATED FINAL ANSWERS\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Chunk IDX':<12} {'Function Tag':<25} {'Flip Type':<35} {'Experiment':<15} {'Final Answer':<15} {'Correct':<10} {'Control Correct %':<18}\n")
        f.write("-"*100 + "\n")
        
        for row in table_rows:
            f.write(f"{row['chunk_idx']:<12} {row['function_tag']:<25} {row['flip_type'][:34]:<35} {row['experiment_type']:<15} {str(row['final_answer']):<15} {str(row['is_correct']):<10} ")
            if row['control_correct_pct'] is not None:
                f.write(f"{row['control_correct_pct']:.2f}%\n")
            else:
                f.write("N/A\n")
        
        f.write("\n" + "-"*100 + "\n")
        f.write(f"{'Overall Control Accuracy':<50} ")
        if overall_control_accuracy is not None:
            f.write(f"{overall_control_accuracy:.2f}%\n")
        else:
            f.write("N/A\n")
        
        # Calculate statistics
        correct_count = sum(1 for row in table_rows if row['is_correct'])
        total_count = len(table_rows)
        accuracy_pct = (correct_count / total_count * 100) if total_count > 0 else 0
        
        f.write(f"{'Repeated Final Answers Accuracy':<50} {accuracy_pct:.2f}% ({correct_count}/{total_count})\n")
        f.write("="*100 + "\n")
    
    print(f"✓ Saved accuracy table to {output_path}")
    print(f"  - Total chunks with repeated final answers: {total_count}")
    print(f"  - Correct: {correct_count}, Incorrect: {total_count - correct_count}")
    print(f"  - Accuracy: {accuracy_pct:.2f}%")

def main():
    """Main function."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    repeated_file = str(PROJECT_ROOT / "visualizations/analysis/repeated_final_answers.json")
    output_dir = str(PROJECT_ROOT / "visualizations/chunk_statistics_plots")
    analysis_dir = str(PROJECT_ROOT / "visualizations/analysis")
    
    print("="*80)
    print("PLOTTING REPEATED FINAL ANSWERS STATISTICS")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ Loaded dataset")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    print("✓ Loaded tokenizer")
    
    # Load function tags and original chunks
    print("Loading function tags and original chunks...")
    chunk_function_tags = load_chunk_function_tags(dataset)
    original_chunks_dict = load_original_chunks(dataset)
    print(f"✓ Loaded function tags for {len(chunk_function_tags)} chunks")
    
    # Load repeated final answers data
    print("Loading repeated final answers data...")
    repeated_chunks_data = load_repeated_final_answers(repeated_file)
    print(f"✓ Loaded {len(repeated_chunks_data)} chunks with repeated final answers")
    
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
    
    plot_repeated_final_answers_statistics(
        all_rollout_stats_by_chunk,
        repeated_chunks_data,
        chunk_function_tags,
        output_dir
    )
    
    # Generate accuracy table
    print("\n" + "="*80)
    print("GENERATING ACCURACY TABLE")
    print("="*80)
    
    generate_accuracy_table(
        repeated_chunks_data,
        chunk_function_tags,
        dataset,
        analysis_dir
    )
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

