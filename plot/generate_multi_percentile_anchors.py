#!/usr/bin/env python3
"""
Generate anchor statistics, analysis, and accuracy tables for multiple percentile thresholds.
Creates outputs for 5%, 15%, 20%, and 25% anchor thresholds.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set
from datasets import load_from_disk
import subprocess
import sys

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
    """Log normalize values: log(x + epsilon) where epsilon is small to handle zeros."""
    epsilon = 1e-10
    log_values = np.log(np.abs(values) + epsilon)
    min_log = np.min(log_values)
    max_log = np.max(log_values)
    if max_log == min_log:
        return np.ones_like(log_values)
    normalized = (log_values - min_log) / (max_log - min_log)
    return normalized

def find_anchors(kl_scores: Dict[int, float], top_percentile: float) -> List[Dict]:
    """Find anchor chunks (top percentile by log-normalized KL scores)."""
    if not kl_scores:
        return []
    
    chunk_indices = np.array(list(kl_scores.keys()))
    raw_values = np.array([kl_scores[idx] for idx in chunk_indices])
    normalized_values = log_normalize(raw_values)
    
    n_top = max(1, int(np.ceil(len(normalized_values) * top_percentile)))
    top_indices = np.argsort(-normalized_values)[:n_top]
    
    anchors = []
    for idx in top_indices:
        chunk_idx = int(chunk_indices[idx])
        anchors.append({
            'chunk_idx': chunk_idx,
            'raw_kl': float(raw_values[idx]),
            'normalized_kl': float(normalized_values[idx])
        })
    
    return anchors

def save_anchors_json(anchors: List[Dict], kl_scores: Dict[int, float], 
                      percentile: float, output_file: str):
    """Save anchors to JSON file."""
    raw_values = np.array(list(kl_scores.values()))
    results = {
        'total_chunks': len(kl_scores),
        'num_anchors': len(anchors),
        'top_percentile': percentile,
        'anchors': anchors,
        'summary_statistics': {
            'raw_kl_min': float(np.min(raw_values)),
            'raw_kl_max': float(np.max(raw_values)),
            'raw_kl_mean': float(np.mean(raw_values)),
            'raw_kl_median': float(np.median(raw_values)),
            'raw_kl_std': float(np.std(raw_values))
        }
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    """Main function to generate anchor analysis for multiple percentiles."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    percentiles = [0.05, 0.15, 0.20, 0.25]
    
    print("="*80)
    print("GENERATING MULTI-PERCENTILE ANCHOR ANALYSIS")
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
    
    # Process each percentile
    for percentile in percentiles:
        print(f"\n{'='*80}")
        print(f"PROCESSING {percentile*100:.0f}% PERCENTILE")
        print(f"{'='*80}")
        
        # Find anchors
        print(f"\nFinding top {percentile*100:.0f}% anchor chunks...")
        anchors = find_anchors(kl_scores, top_percentile=percentile)
        print(f"✓ Found {len(anchors)} anchor chunks")
        
        # Save anchors JSON
        percentile_str = f"{int(percentile*100):02d}percent"
        anchors_file = str(PROJECT_ROOT / f"visualizations/analysis/anchors/anchors_{percentile_str}.json")
        save_anchors_json(anchors, kl_scores, percentile, anchors_file)
        print(f"✓ Saved anchors to {anchors_file}")
        
        # Generate anchor statistics plot
        print(f"\nGenerating anchor statistics plot...")
        plot_script = str(PROJECT_ROOT / "plot/plot_anchor_statistics.py")
        try:
            result = subprocess.run(
                [sys.executable, plot_script, anchors_file, percentile_str],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            if result.returncode == 0:
                print(f"✓ Generated anchor statistics plot")
            else:
                print(f"✗ Error generating plot: {result.stderr}")
        except Exception as e:
            print(f"✗ Error running plot script: {e}")
        
        # Generate accuracy table
        print(f"\nGenerating accuracy table...")
        accuracy_script = str(PROJECT_ROOT / "plot/generate_anchor_accuracy_table.py")
        try:
            result = subprocess.run(
                [sys.executable, accuracy_script, anchors_file, percentile_str],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            if result.returncode == 0:
                print(f"✓ Generated accuracy table")
            else:
                print(f"✗ Error generating accuracy table: {result.stderr}")
        except Exception as e:
            print(f"✗ Error running accuracy script: {e}")
    
    # Run analysis on all accuracy tables
    print(f"\n{'='*80}")
    print("RUNNING STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    
    analysis_script = str(PROJECT_ROOT / "plot/analyze_accuracy_tables.py")
    try:
        result = subprocess.run(
            [sys.executable, analysis_script],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            print(f"✓ Completed statistical analysis")
        else:
            print(f"✗ Error in analysis: {result.stderr}")
    except Exception as e:
        print(f"✗ Error running analysis script: {e}")
    
    print(f"\n{'='*80}")
    print("MULTI-PERCENTILE ANCHOR ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

