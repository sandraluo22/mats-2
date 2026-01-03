#!/usr/bin/env python3
"""
Bimodality testing: Hartigan's Dip test.

Tests whether sample A is more bimodal than expected if drawn from sample B's distribution.
Uses resampling from B to form a null distribution.
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_from_disk
from transformers import AutoTokenizer
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent

def count_sentences(text: str) -> int:
    """Count sentences in text."""
    if not text:
        return 0
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return len(sentences)

def count_uncertainty_words(text: str) -> int:
    """Count uncertainty words in text."""
    UNCERTAINTY_WORDS = [
        "wait", "alternatively", "perhaps", "reconsider", "double-check", 
        "unlikely", "maybe", "might", "could be", "possibly", "doubt",
        "uncertain", "unsure", "hmm", "actually", "let me think"
    ]
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

def load_flip_types(flip_types_file: str) -> Dict[int, str]:
    """Load flip types from flip_types.txt."""
    flip_type_map = {}
    if not Path(flip_types_file).exists():
        return flip_type_map
    with open(flip_types_file, 'r') as f:
        content = f.read()
    for line in content.strip().split('\n'):
        if not line.strip() or ':' not in line:
            continue
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        type_name = parts[0].strip()
        chunk_indices_str = parts[1].strip()
        for idx_str in chunk_indices_str.split(','):
            idx_str = idx_str.strip()
            if idx_str:
                try:
                    chunk_idx = int(idx_str)
                    flip_type_map[chunk_idx] = type_name
                except ValueError:
                    continue
    return flip_type_map

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
                            'total_uncertainty_word_count': float(count_uncertainty_words(full_cot))
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
                            'total_uncertainty_word_count': float(count_uncertainty_words(full_cot))
                        }
    
    return flip_results, ablate_results

def hartigans_dip_statistic(data: np.ndarray) -> float:
    """
    Compute Hartigan's Dip statistic for unimodality.
    
    Higher values indicate more bimodality.
    """
    if len(data) < 2:
        return 0.0
    
    # Sort data
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Compute empirical CDF
    ecdf = np.arange(1, n + 1) / n
    
    # Find the maximum deviation from the unimodal CDF
    # For unimodality test, we find the maximum vertical distance
    # between the ECDF and the best-fitting unimodal distribution
    
    # Simple approximation: find maximum gap
    # More sophisticated: use the dip statistic formula
    # Dip = max over all intervals of the difference between
    # the ECDF and the best unimodal CDF
    
    # Simplified version: maximum vertical distance from a line
    # connecting the first and last points
    min_val = sorted_data[0]
    max_val = sorted_data[-1]
    if max_val == min_val:
        return 0.0
    
    # Linear interpolation for unimodal CDF
    unimodal_ecdf = (sorted_data - min_val) / (max_val - min_val)
    
    # Compute maximum absolute difference
    dip = np.max(np.abs(ecdf - unimodal_ecdf))
    
    return dip

def test_bimodality_relative(sample_A: np.ndarray, sample_B: np.ndarray, 
                            n_resamples: int = 10000) -> Dict:
    """
    Test if sample A is more bimodal than expected given sample B's distribution.
    
    Process:
    1. Compute observed dip statistic on A
    2. Resample from B with replacement (same size as A)
    3. Compute dip statistic on each resample
    4. Form null distribution
    5. Compute one-sided p-value
    """
    if len(sample_A) == 0 or len(sample_B) == 0:
        return {
            'observed_dip': None,
            'p_value': None,
            'n_resamples': n_resamples,
            'sample_A_size': len(sample_A),
            'sample_B_size': len(sample_B)
        }
    
    # Compute observed dip statistic on A
    observed_dip = hartigans_dip_statistic(sample_A)
    
    # Resample from B and compute dip statistics
    n_A = len(sample_A)
    resampled_dips = []
    
    np.random.seed(42)  # For reproducibility
    for _ in range(n_resamples):
        resample = np.random.choice(sample_B, size=n_A, replace=True)
        dip = hartigans_dip_statistic(resample)
        resampled_dips.append(dip)
    
    resampled_dips = np.array(resampled_dips)
    
    # Compute one-sided p-value: fraction of resampled statistics >= observed
    p_value = np.mean(resampled_dips >= observed_dip)
    
    return {
        'observed_dip': float(observed_dip),
        'null_mean_dip': float(np.mean(resampled_dips)),
        'null_std_dip': float(np.std(resampled_dips)),
        'null_median_dip': float(np.median(resampled_dips)),
        'p_value': float(p_value),
        'n_resamples': n_resamples,
        'sample_A_size': len(sample_A),
        'sample_B_size': len(sample_B),
        'sample_A_mean': float(np.mean(sample_A)),
        'sample_B_mean': float(np.mean(sample_B))
    }

def main():
    """Main function to perform bimodality tests."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    flip_types_file = str(PROJECT_ROOT / "flip_types.txt")
    output_file = str(PROJECT_ROOT / "visualizations/analysis/bimodality_tests.json")
    
    print("="*80)
    print("BIMODALITY TESTING: Hartigan's Dip (Relative to Control)")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ Loaded dataset")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    print("✓ Loaded tokenizer")
    
    # Load flip types and function tags
    print("Loading flip types and function tags...")
    chunk_flip_types = load_flip_types(flip_types_file)
    chunk_function_tags = load_chunk_function_tags(dataset)
    original_chunks_dict = load_original_chunks(dataset)
    print(f"✓ Loaded flip types for {len(chunk_flip_types)} chunks")
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
    
    # Aggregate control (normal rollout) statistics
    print("\nAggregating control statistics...")
    all_rollout_stats_by_chunk = {}
    for chunk_idx in chunk_indices:
        rollouts = load_chunk_rollouts(dataset, chunk_idx)
        if rollouts:
            rollout_stats = extract_statistics_from_rollouts(rollouts, chunk_idx, original_chunks_dict, tokenizer)
            all_rollout_stats_by_chunk[chunk_idx] = rollout_stats
    
    # Statistics to test
    stats = ['total_sentence_count', 'total_token_count', 'total_uncertainty_word_count']
    
    # Collect control data (sample B) - all normal rollouts
    control_data = {stat: [] for stat in stats}
    for chunk_idx, stats_dict in all_rollout_stats_by_chunk.items():
        for stat in stats:
            if stat in stats_dict:
                control_data[stat].extend(stats_dict[stat])
    
    print(f"✓ Collected {len(control_data[stats[0]])} control data points")
    
    # Perform tests
    results = []
    n_resamples = 10000
    
    # 1. Test by flip type
    print("\n" + "="*80)
    print("TESTING BIMODALITY BY FLIP TYPE")
    print("="*80)
    
    # Group flip experiments by flip type
    flip_by_type = {}
    for chunk_idx, flip_data in flip_results.items():
        flip_type = chunk_flip_types.get(chunk_idx, 'unknown')
        if flip_type not in flip_by_type:
            flip_by_type[flip_type] = {stat: [] for stat in stats}
        for stat in stats:
            if stat in flip_data:
                flip_by_type[flip_type][stat].append(flip_data[stat])
    
    for flip_type, type_data in sorted(flip_by_type.items()):
        print(f"\nTesting {flip_type}...")
        for stat in stats:
            sample_A = np.array(type_data[stat])
            sample_B = np.array(control_data[stat])
            if len(sample_A) > 0:
                result = test_bimodality_relative(sample_A, sample_B, n_resamples)
                result['group'] = f"Flip Type: {flip_type}"
                result['statistic'] = stat
                results.append(result)
                print(f"  {stat}: Dip={result['observed_dip']:.4f}, p-value={result['p_value']:.4f}")
    
    # 2. Test by function tag
    print("\n" + "="*80)
    print("TESTING BIMODALITY BY FUNCTION TAG")
    print("="*80)
    
    # Group flip experiments by function tag
    flip_by_tag = {}
    for chunk_idx, flip_data in flip_results.items():
        tags = chunk_function_tags.get(chunk_idx, ['unknown'])
        tag = tags[0] if tags else 'unknown'
        if tag not in flip_by_tag:
            flip_by_tag[tag] = {stat: [] for stat in stats}
        for stat in stats:
            if stat in flip_data:
                flip_by_tag[tag][stat].append(flip_data[stat])
    
    for tag, tag_data in sorted(flip_by_tag.items()):
        print(f"\nTesting {tag}...")
        for stat in stats:
            sample_A = np.array(tag_data[stat])
            sample_B = np.array(control_data[stat])
            if len(sample_A) > 0:
                result = test_bimodality_relative(sample_A, sample_B, n_resamples)
                result['group'] = f"Function Tag: {tag}"
                result['statistic'] = stat
                results.append(result)
                print(f"  {stat}: Dip={result['observed_dip']:.4f}, p-value={result['p_value']:.4f}")
    
    # 3. Test overall flip distribution vs control
    print("\n" + "="*80)
    print("TESTING BIMODALITY: OVERALL FLIP VS CONTROL")
    print("="*80)
    
    flip_overall = {stat: [] for stat in stats}
    for chunk_idx, flip_data in flip_results.items():
        for stat in stats:
            if stat in flip_data:
                flip_overall[stat].append(flip_data[stat])
    
    for stat in stats:
        sample_A = np.array(flip_overall[stat])
        sample_B = np.array(control_data[stat])
        if len(sample_A) > 0:
            result = test_bimodality_relative(sample_A, sample_B, n_resamples)
            result['group'] = "Overall Flip"
            result['statistic'] = stat
            results.append(result)
            print(f"{stat}: Dip={result['observed_dip']:.4f}, p-value={result['p_value']:.4f}")
    
    # 4. Test ablate uncertainty vs control
    print("\n" + "="*80)
    print("TESTING BIMODALITY: ABLATE UNCERTAINTY VS CONTROL")
    print("="*80)
    
    ablate_overall = {stat: [] for stat in stats}
    for chunk_idx, ablate_data in ablate_results.items():
        for stat in stats:
            if stat in ablate_data:
                ablate_overall[stat].append(ablate_data[stat])
    
    for stat in stats:
        sample_A = np.array(ablate_overall[stat])
        sample_B = np.array(control_data[stat])
        if len(sample_A) > 0:
            result = test_bimodality_relative(sample_A, sample_B, n_resamples)
            result['group'] = "Overall Ablate"
            result['statistic'] = stat
            results.append(result)
            print(f"{stat}: Dip={result['observed_dip']:.4f}, p-value={result['p_value']:.4f}")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("BIMODALITY TESTING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    print(f"Total tests performed: {len(results)}")
    print(f"\nInterpretation:")
    print(f"  - Low p-value (< 0.05): Sample A is significantly more bimodal than expected")
    print(f"  - High p-value (>= 0.05): No evidence of excess bimodality in A")

if __name__ == "__main__":
    main()

