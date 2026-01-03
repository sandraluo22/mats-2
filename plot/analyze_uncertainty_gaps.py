#!/usr/bin/env python3
"""
Analyze gaps between uncertainty words in sentences.
Calculate average and standard deviation of sentences between uncertainty words.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
import statistics

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

def find_uncertainty_sentence_indices(text: str) -> List[int]:
    """
    Find sentence indices where uncertainty words appear.
    
    Returns:
        List of sentence indices (0-based) containing uncertainty words
    """
    if not text:
        return []
    
    # Split into sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    uncertainty_indices = []
    text_lower = text.lower()
    
    # Find all uncertainty word positions in the original text
    for word in UNCERTAINTY_WORDS:
        word_lower = word.lower()
        start = 0
        while True:
            pos = text_lower.find(word_lower, start)
            if pos == -1:
                break
            
            # Find which sentence this position belongs to
            # Count sentences up to this position
            text_before = text[:pos]
            sentence_count = len([s for s in text_before.split('.') if s.strip()])
            
            if sentence_count not in uncertainty_indices:
                uncertainty_indices.append(sentence_count)
            
            start = pos + 1
    
    return sorted(uncertainty_indices)

def calculate_uncertainty_gaps(uncertainty_indices: List[int], exclude_small: bool = False, min_gap: int = 20) -> List[int]:
    """
    Calculate gaps (in sentences) between consecutive uncertainty words.
    
    Args:
        uncertainty_indices: List of sentence indices where uncertainty words appear
        exclude_small: If True, exclude gaps smaller than min_gap
        min_gap: Minimum gap size to include (if exclude_small is True)
    
    Returns:
        List of gap sizes (in sentences)
    """
    if len(uncertainty_indices) < 2:
        return []
    
    gaps = []
    for i in range(len(uncertainty_indices) - 1):
        gap = uncertainty_indices[i + 1] - uncertainty_indices[i]
        if not exclude_small or gap >= min_gap:
            gaps.append(gap)
    
    return gaps

def analyze_uncertainty_gaps_for_text(text: str, exclude_small: bool = False) -> Dict:
    """
    Analyze uncertainty gaps for a single text.
    
    Returns:
        Dictionary with gap statistics
    """
    uncertainty_indices = find_uncertainty_sentence_indices(text)
    
    if len(uncertainty_indices) < 2:
        return {
            'num_uncertainty_words': len(uncertainty_indices),
            'num_gaps': 0,
            'mean_gap': None,
            'std_gap': None,
            'gaps': []
        }
    
    gaps = calculate_uncertainty_gaps(uncertainty_indices, exclude_small=exclude_small)
    
    if not gaps:
        return {
            'num_uncertainty_words': len(uncertainty_indices),
            'num_gaps': 0,
            'mean_gap': None,
            'std_gap': None,
            'gaps': []
        }
    
    mean_gap = statistics.mean(gaps)
    std_gap = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
    
    return {
        'num_uncertainty_words': len(uncertainty_indices),
        'num_gaps': len(gaps),
        'mean_gap': mean_gap,
        'std_gap': std_gap,
        'gaps': gaps
    }

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

def analyze_control_gaps(dataset, chunk_idx: int, original_chunks_dict: Dict[int, str]) -> Dict:
    """
    Analyze uncertainty gaps for control (normal rollouts) of a chunk.
    """
    rollouts = load_chunk_rollouts(dataset, chunk_idx)
    if not rollouts:
        return {'all_gaps': [], 'all_gaps_excluding_small': []}
    
    sorted_original_indices = sorted(original_chunks_dict.keys())
    previous_chunks_list = [original_chunks_dict[idx] for idx in sorted_original_indices if idx < chunk_idx]
    previous_chunks_text = " ".join(previous_chunks_list) if previous_chunks_list else ""
    
    all_gaps = []
    all_gaps_excluding_small = []
    
    for rollout_data in rollouts:
        rollout_text = rollout_data.get('rollout', '')
        if not rollout_text:
            continue
        
        total_text = previous_chunks_text + " " + rollout_text if previous_chunks_text else rollout_text
        
        # Analyze with all gaps
        stats_all = analyze_uncertainty_gaps_for_text(total_text, exclude_small=False)
        if stats_all['gaps']:
            all_gaps.extend(stats_all['gaps'])
        
        # Analyze excluding small gaps
        stats_excl = analyze_uncertainty_gaps_for_text(total_text, exclude_small=True)
        if stats_excl['gaps']:
            all_gaps_excluding_small.extend(stats_excl['gaps'])
    
    result = {
        'all_gaps': all_gaps,
        'all_gaps_excluding_small': all_gaps_excluding_small
    }
    
    if all_gaps:
        result['mean_gap'] = statistics.mean(all_gaps)
        result['std_gap'] = statistics.stdev(all_gaps) if len(all_gaps) > 1 else 0.0
    else:
        result['mean_gap'] = None
        result['std_gap'] = None
    
    if all_gaps_excluding_small:
        result['mean_gap_excluding_small'] = statistics.mean(all_gaps_excluding_small)
        result['std_gap_excluding_small'] = statistics.stdev(all_gaps_excluding_small) if len(all_gaps_excluding_small) > 1 else 0.0
    else:
        result['mean_gap_excluding_small'] = None
        result['std_gap_excluding_small'] = None
    
    return result

def analyze_experiment_gaps(experiment_file: str, original_chunks_dict: Dict[int, str], experiment_type: str) -> Dict[int, Dict]:
    """
    Analyze uncertainty gaps for flip or ablate experiments.
    
    Args:
        experiment_file: Path to flip_chunk_results.json or ablate_uncertainty_results.json
        original_chunks_dict: Dictionary of original chunks
        experiment_type: 'flip' or 'ablate'
    
    Returns:
        Dictionary mapping chunk_idx to gap statistics
    """
    results = {}
    
    if not Path(experiment_file).exists():
        return results
    
    with open(experiment_file, 'r') as f:
        data = json.load(f)
    
    sorted_original_indices = sorted(original_chunks_dict.keys())
    previous_chunks_list = [original_chunks_dict[idx] for idx in sorted_original_indices]
    previous_chunks_text = " ".join(previous_chunks_list) if previous_chunks_list else ""
    
    for exp in data.get('experiments', []):
        if experiment_type == 'flip':
            chunk_idx = exp.get('flipped_chunk_idx')
        else:  # ablate
            chunk_idx = exp.get('ablated_chunk_idx')
        
        if chunk_idx is None:
            continue
        
        full_cot = exp.get('full_cot', '')
        if not full_cot:
            continue
        
        # Analyze with all gaps
        stats_all = analyze_uncertainty_gaps_for_text(full_cot, exclude_small=False)
        
        # Analyze excluding small gaps
        stats_excl = analyze_uncertainty_gaps_for_text(full_cot, exclude_small=True)
        
        results[chunk_idx] = {
            'all_gaps': stats_all['gaps'],
            'all_gaps_excluding_small': stats_excl['gaps'],
            'mean_gap': stats_all['mean_gap'],
            'std_gap': stats_all['std_gap'],
            'mean_gap_excluding_small': stats_excl['mean_gap'],
            'std_gap_excluding_small': stats_excl['std_gap'],
            'num_uncertainty_words': stats_all['num_uncertainty_words']
        }
    
    return results

def main():
    """Main function to analyze uncertainty gaps."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    output_file = str(PROJECT_ROOT / "visualizations/analysis/uncertainty_gaps_analysis.json")
    
    print("="*80)
    print("ANALYZING UNCERTAINTY GAPS")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ Loaded dataset")
    
    # Load original chunks
    print("Loading original chunks...")
    original_chunks_dict = load_original_chunks(dataset)
    print(f"✓ Loaded {len(original_chunks_dict)} original chunks")
    
    # Get all chunk indices
    chunk_indices = set()
    for ex in dataset:
        path = ex.get('path', '')
        match = re.search(r'chunk_(\d+)/solutions\.json', path)
        if match:
            chunk_indices.add(int(match.group(1)))
    chunk_indices = sorted(chunk_indices)
    
    print(f"✓ Found {len(chunk_indices)} chunks with rollouts")
    
    # Analyze control gaps
    print("\n" + "="*80)
    print("ANALYZING CONTROL GAPS")
    print("="*80)
    
    control_results = {}
    for chunk_idx in chunk_indices:
        if chunk_idx % 20 == 0:
            print(f"  Processing chunk {chunk_idx}...")
        control_results[chunk_idx] = analyze_control_gaps(dataset, chunk_idx, original_chunks_dict)
    
    print(f"✓ Analyzed {len(control_results)} control chunks")
    
    # Analyze flip gaps
    print("\n" + "="*80)
    print("ANALYZING FLIP GAPS")
    print("="*80)
    
    flip_results = analyze_experiment_gaps(flip_file, original_chunks_dict, 'flip')
    print(f"✓ Analyzed {len(flip_results)} flip experiments")
    
    # Analyze ablate gaps
    print("\n" + "="*80)
    print("ANALYZING ABLATE GAPS")
    print("="*80)
    
    ablate_results = analyze_experiment_gaps(ablate_file, original_chunks_dict, 'ablate')
    print(f"✓ Analyzed {len(ablate_results)} ablate experiments")
    
    # Compile results
    output_data = {
        'control': {},
        'flip': {},
        'ablate': {}
    }
    
    # Format control results
    for chunk_idx, stats in control_results.items():
        output_data['control'][chunk_idx] = {
            'mean_gap': stats['mean_gap'],
            'std_gap': stats['std_gap'],
            'mean_gap_excluding_small': stats.get('mean_gap_excluding_small'),
            'std_gap_excluding_small': stats.get('std_gap_excluding_small'),
            'num_gaps': len(stats['all_gaps']),
            'num_gaps_excluding_small': len(stats['all_gaps_excluding_small'])
        }
    
    # Format flip results
    for chunk_idx, stats in flip_results.items():
        output_data['flip'][chunk_idx] = {
            'mean_gap': stats['mean_gap'],
            'std_gap': stats['std_gap'],
            'mean_gap_excluding_small': stats['mean_gap_excluding_small'],
            'std_gap_excluding_small': stats['std_gap_excluding_small'],
            'num_gaps': len(stats['all_gaps']),
            'num_gaps_excluding_small': len(stats['all_gaps_excluding_small']),
            'num_uncertainty_words': stats['num_uncertainty_words']
        }
    
    # Format ablate results
    for chunk_idx, stats in ablate_results.items():
        output_data['ablate'][chunk_idx] = {
            'mean_gap': stats['mean_gap'],
            'std_gap': stats['std_gap'],
            'mean_gap_excluding_small': stats['mean_gap_excluding_small'],
            'std_gap_excluding_small': stats['std_gap_excluding_small'],
            'num_gaps': len(stats['all_gaps']),
            'num_gaps_excluding_small': len(stats['all_gaps_excluding_small']),
            'num_uncertainty_words': stats['num_uncertainty_words']
        }
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    
    # Control summary
    control_means = [v['mean_gap'] for v in output_data['control'].values() if v['mean_gap'] is not None]
    control_stds = [v['std_gap'] for v in output_data['control'].values() if v['std_gap'] is not None]
    if control_means:
        print(f"Control (all gaps):")
        print(f"  Mean gap: {statistics.mean(control_means):.2f} sentences")
        print(f"  Std of gaps: {statistics.mean(control_stds):.2f} sentences")
    
    control_means_excl = [v['mean_gap_excluding_small'] for v in output_data['control'].values() if v.get('mean_gap_excluding_small') is not None]
    control_stds_excl = [v['std_gap_excluding_small'] for v in output_data['control'].values() if v.get('std_gap_excluding_small') is not None]
    if control_means_excl:
        print(f"Control (excluding gaps < 20):")
        print(f"  Mean gap: {statistics.mean(control_means_excl):.2f} sentences")
        print(f"  Std of gaps: {statistics.mean(control_stds_excl):.2f} sentences")
    
    # Flip summary
    flip_means = [v['mean_gap'] for v in output_data['flip'].values() if v['mean_gap'] is not None]
    flip_stds = [v['std_gap'] for v in output_data['flip'].values() if v['std_gap'] is not None]
    if flip_means:
        print(f"\nFlip (all gaps):")
        print(f"  Mean gap: {statistics.mean(flip_means):.2f} sentences")
        print(f"  Std of gaps: {statistics.mean(flip_stds):.2f} sentences")
    
    flip_means_excl = [v['mean_gap_excluding_small'] for v in output_data['flip'].values() if v.get('mean_gap_excluding_small') is not None]
    flip_stds_excl = [v['std_gap_excluding_small'] for v in output_data['flip'].values() if v.get('std_gap_excluding_small') is not None]
    if flip_means_excl:
        print(f"Flip (excluding gaps < 20):")
        print(f"  Mean gap: {statistics.mean(flip_means_excl):.2f} sentences")
        print(f"  Std of gaps: {statistics.mean(flip_stds_excl):.2f} sentences")
    
    # Ablate summary
    ablate_means = [v['mean_gap'] for v in output_data['ablate'].values() if v['mean_gap'] is not None]
    ablate_stds = [v['std_gap'] for v in output_data['ablate'].values() if v['std_gap'] is not None]
    if ablate_means:
        print(f"\nAblate (all gaps):")
        print(f"  Mean gap: {statistics.mean(ablate_means):.2f} sentences")
        print(f"  Std of gaps: {statistics.mean(ablate_stds):.2f} sentences")
    
    ablate_means_excl = [v['mean_gap_excluding_small'] for v in output_data['ablate'].values() if v.get('mean_gap_excluding_small') is not None]
    ablate_stds_excl = [v['std_gap_excluding_small'] for v in output_data['ablate'].values() if v.get('std_gap_excluding_small') is not None]
    if ablate_means_excl:
        print(f"Ablate (excluding gaps < 20):")
        print(f"  Mean gap: {statistics.mean(ablate_means_excl):.2f} sentences")
        print(f"  Std of gaps: {statistics.mean(ablate_stds_excl):.2f} sentences")

if __name__ == "__main__":
    main()

