#!/usr/bin/env python3
"""
Distribution testing: Energy Distance and Maximum Mean Discrepancy (MMD).

Tests whether flip/ablate experiment distributions differ from control distribution.
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_from_disk
from transformers import AutoTokenizer
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel

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

def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute energy distance between two samples.
    
    Energy distance: E(X,Y) = 2*E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]
    """
    X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else X
    Y = np.asarray(Y).reshape(-1, 1) if Y.ndim == 1 else Y
    
    n = len(X)
    m = len(Y)
    
    # Compute pairwise distances
    XX = cdist(X, X, metric='euclidean')
    YY = cdist(Y, Y, metric='euclidean')
    XY = cdist(X, Y, metric='euclidean')
    
    # Energy distance formula
    term1 = 2 * np.mean(XY)  # 2 * E[|X-Y|]
    term2 = np.mean(XX[np.triu_indices(n, k=1)]) if n > 1 else 0  # E[|X-X'|]
    term3 = np.mean(YY[np.triu_indices(m, k=1)]) if m > 1 else 0  # E[|Y-Y'|]
    
    energy_dist = term1 - term2 - term3
    return max(0, energy_dist)  # Energy distance is non-negative

def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) using RBF kernel.
    
    MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
    """
    X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else X
    Y = np.asarray(Y).reshape(-1, 1) if Y.ndim == 1 else Y
    
    n = len(X)
    m = len(Y)
    
    # Compute kernel matrices
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    
    # MMD^2
    mmd_squared = (np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY))
    return np.sqrt(max(0, mmd_squared))  # Return MMD (not squared)

def perform_distribution_tests(sample_A: np.ndarray, sample_B: np.ndarray, 
                               stat_name: str, group_name: str) -> Dict:
    """Perform Energy Distance and MMD tests."""
    if len(sample_A) == 0 or len(sample_B) == 0:
        return {
            'group': group_name,
            'statistic': stat_name,
            'energy_distance': None,
            'mmd': None,
            'sample_A_size': len(sample_A),
            'sample_B_size': len(sample_B)
        }
    
    # Compute test statistics
    energy_dist = energy_distance(sample_A, sample_B)
    mmd = mmd_rbf(sample_A, sample_B, gamma=1.0)
    
    return {
        'group': group_name,
        'statistic': stat_name,
        'energy_distance': float(energy_dist),
        'mmd': float(mmd),
        'sample_A_size': len(sample_A),
        'sample_B_size': len(sample_B),
        'sample_A_mean': float(np.mean(sample_A)),
        'sample_B_mean': float(np.mean(sample_B)),
        'sample_A_std': float(np.std(sample_A)),
        'sample_B_std': float(np.std(sample_B))
    }

def main():
    """Main function to perform distribution tests."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    flip_types_file = str(PROJECT_ROOT / "flip_types.txt")
    output_file = str(PROJECT_ROOT / "visualizations/analysis/distribution_tests.json")
    
    print("="*80)
    print("DISTRIBUTION TESTING: Energy Distance & MMD")
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
    
    # 1. Test by flip type
    print("\n" + "="*80)
    print("TESTING BY FLIP TYPE")
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
                result = perform_distribution_tests(sample_A, sample_B, stat, f"Flip Type: {flip_type}")
                results.append(result)
                print(f"  {stat}: Energy Distance={result['energy_distance']:.4f}, MMD={result['mmd']:.4f}")
    
    # 2. Test by function tag
    print("\n" + "="*80)
    print("TESTING BY FUNCTION TAG")
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
                result = perform_distribution_tests(sample_A, sample_B, stat, f"Function Tag: {tag}")
                results.append(result)
                print(f"  {stat}: Energy Distance={result['energy_distance']:.4f}, MMD={result['mmd']:.4f}")
    
    # 3. Test overall flip distribution vs control
    print("\n" + "="*80)
    print("TESTING OVERALL FLIP DISTRIBUTION VS CONTROL")
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
            result = perform_distribution_tests(sample_A, sample_B, stat, "Overall Flip")
            results.append(result)
            print(f"{stat}: Energy Distance={result['energy_distance']:.4f}, MMD={result['mmd']:.4f}")
    
    # 4. Test ablate uncertainty vs control
    print("\n" + "="*80)
    print("TESTING ABLATE UNCERTAINTY VS CONTROL")
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
            result = perform_distribution_tests(sample_A, sample_B, stat, "Overall Ablate")
            results.append(result)
            print(f"{stat}: Energy Distance={result['energy_distance']:.4f}, MMD={result['mmd']:.4f}")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("DISTRIBUTION TESTING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    print(f"Total tests performed: {len(results)}")

if __name__ == "__main__":
    main()

