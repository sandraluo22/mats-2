#!/usr/bin/env python3
"""
Find anchor chunks based on resampling_importance_kl scores.

Process:
1. Load resampling_importance_kl for each chunk from chunks_labeled.json
2. Log normalize the KL scores
3. Identify top 10% as anchors
4. Output chunk indices and both raw and normalized KL scores
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).parent.parent

def load_kl_scores(dataset) -> Dict[int, float]:
    """Load resampling_importance_kl scores for each chunk."""
    kl_scores = {}
    
    for ex in dataset:
        path = ex.get('path', '')
        if 'chunks_labeled.json' in path and 'incorrect_base_solution' in path and 'forced_answer' not in path:
            try:
                chunks_labeled = json.loads(ex.get('content', '[]'))
                for chunk_data in chunks_labeled:
                    chunk_idx = chunk_data.get('chunk_idx')
                    kl_score = chunk_data.get('resampling_importance_kl')
                    if chunk_idx is not None and kl_score is not None:
                        kl_scores[chunk_idx] = float(kl_score)
                break
            except json.JSONDecodeError as e:
                print(f"Error parsing chunks_labeled.json: {e}")
                continue
    
    return kl_scores

def log_normalize(values: np.ndarray) -> np.ndarray:
    """
    Log normalize values: log(x + epsilon) where epsilon is small to handle zeros.
    
    Then normalize to [0, 1] range.
    """
    epsilon = 1e-10
    # Add epsilon to handle zeros and negative values
    log_values = np.log(np.abs(values) + epsilon)
    
    # Normalize to [0, 1] range
    min_log = np.min(log_values)
    max_log = np.max(log_values)
    if max_log == min_log:
        return np.ones_like(log_values)  # All values are the same
    normalized = (log_values - min_log) / (max_log - min_log)
    
    return normalized

def find_anchors(kl_scores: Dict[int, float], top_percentile: float = 0.10) -> List[Dict]:
    """
    Find anchor chunks (top percentile by log-normalized KL scores).
    
    Returns list of dicts with chunk_idx, raw_kl, and normalized_kl.
    """
    if not kl_scores:
        return []
    
    # Convert to arrays
    chunk_indices = np.array(list(kl_scores.keys()))
    raw_values = np.array([kl_scores[idx] for idx in chunk_indices])
    
    # Log normalize
    normalized_values = log_normalize(raw_values)
    
    # Find top percentile - get exactly top N chunks
    n_top = max(1, int(np.ceil(len(normalized_values) * top_percentile)))
    
    # Get indices of top N by normalized score
    top_indices = np.argsort(-normalized_values)[:n_top]
    
    # Build results
    anchors = []
    for idx in top_indices:
        chunk_idx = int(chunk_indices[idx])
        anchors.append({
            'chunk_idx': chunk_idx,
            'raw_kl': float(raw_values[idx]),
            'normalized_kl': float(normalized_values[idx])
        })
    
    return anchors

def main():
    """Main function to find anchor chunks."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    output_file = str(PROJECT_ROOT / "visualizations/analysis/anchors.json")
    
    print("="*80)
    print("FINDING ANCHOR CHUNKS")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ Loaded dataset")
    
    # Load KL scores
    print("\nLoading resampling_importance_kl scores...")
    kl_scores = load_kl_scores(dataset)
    print(f"✓ Loaded KL scores for {len(kl_scores)} chunks")
    
    if not kl_scores:
        print("Error: No KL scores found!")
        return
    
    # Display summary statistics
    raw_values = np.array(list(kl_scores.values()))
    print(f"\nRaw KL Score Statistics:")
    print(f"  Min: {np.min(raw_values):.6f}")
    print(f"  Max: {np.max(raw_values):.6f}")
    print(f"  Mean: {np.mean(raw_values):.6f}")
    print(f"  Median: {np.median(raw_values):.6f}")
    print(f"  Std: {np.std(raw_values):.6f}")
    
    # Find anchors (top 10%)
    print("\nFinding top 10% anchor chunks...")
    anchors = find_anchors(kl_scores, top_percentile=0.10)
    
    print(f"\n✓ Found {len(anchors)} anchor chunks (top 10%)")
    print(f"\nAnchor Chunks:")
    print("-" * 80)
    print(f"{'Chunk IDX':<12} {'Raw KL':<15} {'Normalized KL':<15}")
    print("-" * 80)
    
    for anchor in anchors:
        print(f"{anchor['chunk_idx']:<12} {anchor['raw_kl']:<15.6f} {anchor['normalized_kl']:<15.6f}")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'total_chunks': len(kl_scores),
        'num_anchors': len(anchors),
        'top_percentile': 0.10,
        'anchors': anchors,
        'summary_statistics': {
            'raw_kl_min': float(np.min(raw_values)),
            'raw_kl_max': float(np.max(raw_values)),
            'raw_kl_mean': float(np.mean(raw_values)),
            'raw_kl_median': float(np.median(raw_values)),
            'raw_kl_std': float(np.std(raw_values))
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("ANCHOR FINDING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    print(f"\nAnchor chunk indices: {[a['chunk_idx'] for a in anchors]}")

if __name__ == "__main__":
    main()

