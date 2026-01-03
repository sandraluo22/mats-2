#!/usr/bin/env python3
"""
Plot control-time accuracy vs. post-intervention correctness.
Bins chunks by control accuracy (deciles) and plots mean correctness per bin.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent

def normalize_answer(answer_str):
    """Normalize answer string to extract numeric value."""
    import re
    ans_str = str(answer_str).lower().strip()
    ans_str = re.sub(r'\\boxed\{([^}]+)\}', r'\1', ans_str)
    ans_str = re.sub(r'[^\d.]', '', ans_str)
    try:
        return float(ans_str)
    except:
        return None

def check_correctness(answer, ground_truth):
    """Check if answer matches ground truth (loose matching)."""
    answer_norm = normalize_answer(answer)
    gt_norm = normalize_answer(ground_truth)
    
    if answer_norm is not None and gt_norm is not None:
        return abs(answer_norm - gt_norm) < 0.01
    return False

def load_chunk_rollout_accuracy(dataset, chunk_idx: int, ground_truth: str) -> float:
    """Load accuracy from normal rollouts for a chunk."""
    for ex in dataset:
        path = ex.get('path', '')
        if f'chunk_{chunk_idx}/solutions.json' in path:
            try:
                solutions = json.loads(ex.get('content', '[]'))
                if not solutions:
                    return None
                
                correct_count = 0
                total_count = 0
                
                for solution in solutions:
                    if isinstance(solution, dict):
                        answer = solution.get('answer', '')
                        is_correct = solution.get('is_correct', False)
                        
                        if is_correct is None or (not isinstance(is_correct, bool) and not is_correct):
                            is_correct = check_correctness(answer, ground_truth)
                        
                        total_count += 1
                        if is_correct:
                            correct_count += 1
                
                if total_count > 0:
                    return correct_count / total_count
                return None
            except:
                return None
    return None

def get_all_chunk_indices(dataset) -> List[int]:
    """Get all chunk indices that have rollouts."""
    chunk_indices = set()
    for ex in dataset:
        path = ex.get('path', '')
        import re
        match = re.search(r'chunk_(\d+)/solutions\.json', path)
        if match:
            chunk_indices.add(int(match.group(1)))
    return sorted(list(chunk_indices))

def bin_data(control_accuracies: List[float], intervention_correctness: List[bool], 
              num_bins: int = 10) -> Tuple[List[float], List[float], List[float]]:
    """
    Bin data by control accuracy and calculate mean correctness per bin.
    
    Returns:
        bin_centers: Center of each bin
        bin_means: Mean correctness in each bin
        bin_stds: Standard error of mean in each bin
    """
    # Filter out None values
    valid_data = [(ca, ic) for ca, ic in zip(control_accuracies, intervention_correctness) 
                  if ca is not None and ic is not None]
    
    if not valid_data:
        return [], [], []
    
    control_vals = np.array([d[0] for d in valid_data])
    intervention_vals = np.array([d[1] for d in valid_data], dtype=float)
    
    # Create bins (deciles) - using percentiles for equal-frequency bins
    # This ensures each bin has approximately the same number of data points
    bin_edges = np.percentile(control_vals, np.linspace(0, 100, num_bins + 1))
    bin_edges[0] = 0.0  # Ensure first edge is exactly 0
    bin_edges[-1] = 1.0  # Ensure last edge is exactly 1
    bin_indices = np.digitize(control_vals, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for i in range(num_bins):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            bin_vals = intervention_vals[mask]
            bin_control_vals = control_vals[mask]
            
            bin_centers.append(np.mean(bin_control_vals))
            bin_means.append(np.mean(bin_vals))
            # Standard error of mean
            bin_stds.append(stats.sem(bin_vals) if len(bin_vals) > 1 else 0)
            bin_counts.append(len(bin_vals))
        else:
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_means.append(0)
            bin_stds.append(0)
            bin_counts.append(0)
    
    return bin_centers, bin_means, bin_stds, bin_counts

def main():
    """Main function to create the plot."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    
    print("="*80)
    print("PLOTTING CONTROL vs INTERVENTION ACCURACY")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ Loaded dataset")
    
    # Load flip experiments
    print("\nLoading flip experiments...")
    with open(flip_file, 'r') as f:
        flip_data = json.load(f)
    
    ground_truth = flip_data.get('ground_truth_answer', '')
    flip_experiments = flip_data.get('experiments', [])
    
    # Load ablate experiments
    print("Loading ablate experiments...")
    with open(ablate_file, 'r') as f:
        ablate_data = json.load(f)
    
    ablate_experiments = ablate_data.get('experiments', [])
    
    # Get all chunk indices
    all_chunk_indices = get_all_chunk_indices(dataset)
    print(f"✓ Found {len(all_chunk_indices)} chunks with rollouts")
    
    # Collect data for flip experiments
    print("\nCollecting flip experiment data...")
    flip_control_accuracies = []
    flip_intervention_correctness = []
    
    for exp in flip_experiments:
        chunk_idx = exp.get('flipped_chunk_idx')
        if chunk_idx is not None:
            control_acc = load_chunk_rollout_accuracy(dataset, chunk_idx, ground_truth)
            is_correct = exp.get('is_correct', False)
            
            if control_acc is not None:
                flip_control_accuracies.append(control_acc)
                flip_intervention_correctness.append(1.0 if is_correct else 0.0)
    
    print(f"✓ Collected {len(flip_control_accuracies)} flip data points")
    
    # Collect data for ablate experiments
    print("\nCollecting ablate experiment data...")
    ablate_control_accuracies = []
    ablate_intervention_correctness = []
    
    for exp in ablate_experiments:
        chunk_idx = exp.get('ablated_chunk_idx')
        if chunk_idx is not None:
            control_acc = load_chunk_rollout_accuracy(dataset, chunk_idx, ground_truth)
            is_correct = exp.get('is_correct', False)
            
            if control_acc is not None:
                ablate_control_accuracies.append(control_acc)
                ablate_intervention_correctness.append(1.0 if is_correct else 0.0)
    
    print(f"✓ Collected {len(ablate_control_accuracies)} ablate data points")
    
    # Bin the data
    print("\nBinning data...")
    flip_bin_centers, flip_bin_means, flip_bin_stds, flip_bin_counts = bin_data(
        flip_control_accuracies, flip_intervention_correctness, num_bins=10
    )
    ablate_bin_centers, ablate_bin_means, ablate_bin_stds, ablate_bin_counts = bin_data(
        ablate_control_accuracies, ablate_intervention_correctness, num_bins=10
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot binned means with error bars
    if flip_bin_centers:
        # Plot error bars separately with lower opacity
        for x, y, err in zip(flip_bin_centers, flip_bin_means, flip_bin_stds):
            if err > 0:
                # Vertical line for error bar
                ax.plot([x, x], [y - err, y + err], color='#A23B72', 
                       linewidth=1.5, alpha=0.3, zorder=2)
                # Caps at top and bottom
                ax.plot([x, x], [y - err, y - err], '_', color='#A23B72', 
                       markersize=8, alpha=0.3, zorder=2)
                ax.plot([x, x], [y + err, y + err], '_', color='#A23B72', 
                       markersize=8, alpha=0.3, zorder=2)
        
        # Plot main line and markers with full opacity
        ax.errorbar(flip_bin_centers, flip_bin_means, yerr=None, 
                   fmt='o-', color='#A23B72', linewidth=2.5, markersize=10,
                   label='Flip Chunk', zorder=3, alpha=0.9)
    
    # Plot ablate data
    if ablate_bin_centers:
        # Plot error bars separately with lower opacity
        for x, y, err in zip(ablate_bin_centers, ablate_bin_means, ablate_bin_stds):
            if err > 0:
                # Vertical line for error bar
                ax.plot([x, x], [y - err, y + err], color='#F18F01', 
                       linewidth=1.5, alpha=0.3, zorder=2)
                # Caps at top and bottom
                ax.plot([x, x], [y - err, y - err], '_', color='#F18F01', 
                       markersize=8, alpha=0.3, zorder=2)
                ax.plot([x, x], [y + err, y + err], '_', color='#F18F01', 
                       markersize=8, alpha=0.3, zorder=2)
        
        # Plot main line and markers with full opacity
        ax.errorbar(ablate_bin_centers, ablate_bin_means, yerr=None,
                   fmt='s-', color='#F18F01', linewidth=2.5, markersize=10,
                   label='Ablate Uncertainty', zorder=3, alpha=0.9)
    
    # Add diagonal reference line (y = x)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='y = x (reference)', zorder=1)
    
    # Formatting
    ax.set_xlabel('Control-Time Accuracy (Reliability)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Post-Intervention Correctness', fontsize=14, fontweight='bold')
    ax.set_title('Control-Time Accuracy vs. Post-Intervention Correctness\n(Binned by Deciles)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Format axes as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
    
    # Add bin count annotations (optional, can be removed if too cluttered)
    # for i, (x, y, count) in enumerate(zip(flip_bin_centers, flip_bin_means, flip_bin_counts)):
    #     if count > 0:
    #         ax.annotate(f'n={count}', (x, y), textcoords="offset points",
    #                    xytext=(0,15), ha='center', fontsize=8, color='#A23B72')
    
    plt.tight_layout()
    
    # Save plot
    output_file = PROJECT_ROOT / "visualizations/chunk_statistics_plots/control_vs_intervention_accuracy.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plot to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nFlip Experiments:")
    print(f"  Total data points: {len(flip_control_accuracies)}")
    print(f"  Mean control accuracy: {np.mean(flip_control_accuracies):.3f}")
    print(f"  Mean intervention correctness: {np.mean(flip_intervention_correctness):.3f}")
    
    print(f"\nAblate Experiments:")
    print(f"  Total data points: {len(ablate_control_accuracies)}")
    print(f"  Mean control accuracy: {np.mean(ablate_control_accuracies):.3f}")
    print(f"  Mean intervention correctness: {np.mean(ablate_intervention_correctness):.3f}")
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

