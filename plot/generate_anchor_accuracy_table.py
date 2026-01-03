#!/usr/bin/env python3
"""
Generate accuracy comparison table for anchor chunks.
Compares each anchor chunk's experiment correctness to control and calculates overall accuracy.
"""

import json
import os
from datasets import load_from_disk
from typing import Dict, List, Tuple, Set
from pathlib import Path

# Get project root directory (parent of plot folder)
PROJECT_ROOT = Path(__file__).parent.parent

def normalize_answer(answer_str):
    """Normalize answer string to extract numeric value."""
    import re
    ans_str = str(answer_str).lower().strip()
    # Remove boxed, etc.
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

def load_chunk_rollout_accuracy(dataset, chunk_idx: int, ground_truth: str) -> float:
    """Load accuracy from normal rollouts for a chunk."""
    import re
    
    # Find solutions.json for this chunk
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
                        
                        # If is_correct not provided, check manually
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

def calculate_overall_control_accuracy(dataset, chunk_indices: List[int], ground_truth: str) -> float:
    """Calculate overall control accuracy across all chunks."""
    all_accuracies = []
    total_rollouts = 0
    correct_rollouts = 0
    
    for chunk_idx in chunk_indices:
        acc = load_chunk_rollout_accuracy(dataset, chunk_idx, ground_truth)
        if acc is not None:
            all_accuracies.append(acc)
            # Also count rollouts
            for ex in dataset:
                path = ex.get('path', '')
                if f'chunk_{chunk_idx}/solutions.json' in path:
                    try:
                        solutions = json.loads(ex.get('content', '[]'))
                        if isinstance(solutions, list):
                            for solution in solutions:
                                if isinstance(solution, dict):
                                    total_rollouts += 1
                                    answer = solution.get('answer', '')
                                    is_correct = solution.get('is_correct', False)
                                    if is_correct is None or (not isinstance(is_correct, bool) and not is_correct):
                                        is_correct = check_correctness(answer, ground_truth)
                                    if is_correct:
                                        correct_rollouts += 1
                    except:
                        pass
                    break
    
    if total_rollouts > 0:
        return correct_rollouts / total_rollouts
    elif all_accuracies:
        return sum(all_accuracies) / len(all_accuracies)
    return None

def generate_anchor_table(anchors_file: str, flip_file: str, ablate_file: str, output_file: str):
    """Generate accuracy table for anchor chunk experiments."""
    # Load anchors
    anchor_set = load_anchors(anchors_file)
    
    # Load experiment results
    flip_data = {}
    ablate_data = {}
    
    if Path(flip_file).exists():
        with open(flip_file, 'r') as f:
            flip_data = json.load(f)
    
    if Path(ablate_file).exists():
        with open(ablate_file, 'r') as f:
            ablate_data = json.load(f)
    
    # Get problem info from flip or ablate data
    problem_id = flip_data.get('problem_id') or ablate_data.get('problem_id', 'problem_1591')
    ground_truth = flip_data.get('ground_truth_answer') or ablate_data.get('ground_truth_answer', '')
    
    # Load dataset to get control accuracy and chunk rollout accuracies
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    dataset = load_from_disk(dataset_path)
    
    # Get experiments for anchor chunks
    flip_experiments = {}
    ablate_experiments = {}
    
    for exp in flip_data.get('experiments', []):
        chunk_idx = exp.get('flipped_chunk_idx')
        if chunk_idx is not None and chunk_idx in anchor_set:
            flip_experiments[chunk_idx] = exp
    
    for exp in ablate_data.get('experiments', []):
        chunk_idx = exp.get('ablated_chunk_idx')
        if chunk_idx is not None and chunk_idx in anchor_set:
            ablate_experiments[chunk_idx] = exp
    
    # Get all anchor chunk indices that have experiments
    anchor_chunk_indices = sorted(set(flip_experiments.keys()) | set(ablate_experiments.keys()))
    
    if not anchor_chunk_indices:
        print("No anchor chunks with experiments found.")
        return
    
    # Calculate overall control accuracy across all anchor chunks
    overall_control_accuracy = calculate_overall_control_accuracy(dataset, anchor_chunk_indices, ground_truth)
    
    # Create table data
    table_rows = []
    table_rows.append("="*80)
    table_rows.append("ANCHOR CHUNK EXPERIMENTS - ACCURACY COMPARISON")
    table_rows.append("="*80)
    table_rows.append("")
    table_rows.append(f"Problem ID: {problem_id}")
    table_rows.append(f"Ground Truth: {ground_truth}")
    table_rows.append(f"Total Anchor Chunks: {len(anchor_set)}")
    table_rows.append(f"Anchor Chunks with Experiments: {len(anchor_chunk_indices)}")
    table_rows.append("")
    table_rows.append(f"{'Chunk Index':<15} {'Control Correct %':<20} {'Flip Correct':<15} {'Ablate Correct':<15}")
    table_rows.append("-"*80)
    
    # Process each anchor chunk
    flip_correct_count = 0
    flip_total_count = 0
    ablate_correct_count = 0
    ablate_total_count = 0
    
    for chunk_idx in anchor_chunk_indices:
        # Get control accuracy for this chunk
        control_acc = load_chunk_rollout_accuracy(dataset, chunk_idx, ground_truth)
        control_acc_str = f"{control_acc*100:.2f}%" if control_acc is not None else "N/A"
        
        # Get flip experiment result
        flip_correct = None
        if chunk_idx in flip_experiments:
            exp = flip_experiments[chunk_idx]
            final_answer = exp.get('final_answer', '')
            flip_correct = check_correctness(final_answer, ground_truth)
            flip_total_count += 1
            if flip_correct:
                flip_correct_count += 1
        
        flip_str = "True" if flip_correct is True else ("False" if flip_correct is False else "N/A")
        
        # Get ablate experiment result
        ablate_correct = None
        if chunk_idx in ablate_experiments:
            exp = ablate_experiments[chunk_idx]
            final_answer = exp.get('final_answer', '')
            ablate_correct = check_correctness(final_answer, ground_truth)
            ablate_total_count += 1
            if ablate_correct:
                ablate_correct_count += 1
        
        ablate_str = "True" if ablate_correct is True else ("False" if ablate_correct is False else "N/A")
        
        table_rows.append(f"{chunk_idx:<15} {control_acc_str:<20} {flip_str:<15} {ablate_str:<15}")
    
    # Add summary rows
    table_rows.append("")
    table_rows.append("-"*80)
    overall_control_str = f"{overall_control_accuracy*100:.2f}%" if overall_control_accuracy is not None else "N/A"
    table_rows.append(f"{'Overall Control':<15} {overall_control_str:<20} {'':<15} {'':<15}")
    
    if flip_total_count > 0:
        flip_acc = flip_correct_count / flip_total_count
        flip_str = f"{flip_acc*100:.2f}% ({flip_correct_count}/{flip_total_count})"
        table_rows.append(f"{'Flip Accuracy':<15} {'':<20} {flip_str:<15} {'':<15}")
    
    if ablate_total_count > 0:
        ablate_acc = ablate_correct_count / ablate_total_count
        ablate_str = f"{ablate_acc*100:.2f}% ({ablate_correct_count}/{ablate_total_count})"
        table_rows.append(f"{'Ablate Accuracy':<15} {'':<20} {'':<15} {ablate_str:<15}")
    
    table_rows.append("")
    table_rows.append("="*80)
    
    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(table_rows))
    
    print(f"✓ Generated anchor accuracy table: {output_file}")
    print(f"  - {len(anchor_chunk_indices)} anchor chunks with experiments")
    print(f"  - {flip_total_count} flip experiments")
    print(f"  - {ablate_total_count} ablate experiments")

def main():
    """Main function."""
    import sys
    
    # Allow command-line arguments for anchors file and percentile string
    anchors_file = None
    percentile_str = None
    if len(sys.argv) > 1:
        anchors_file = sys.argv[1]
    if len(sys.argv) > 2:
        percentile_str = sys.argv[2]
    
    if anchors_file is None:
        anchors_file = str(PROJECT_ROOT / "visualizations/analysis/anchors.json")
    
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    
    if percentile_str:
        output_file = str(PROJECT_ROOT / f"visualizations/analysis/anchors/anchor_accuracy_table_{percentile_str}.txt")
    else:
        output_file = str(PROJECT_ROOT / "visualizations/analysis/anchor_accuracy_table.txt")
    
    print("="*80)
    print("GENERATING ANCHOR ACCURACY TABLE")
    if percentile_str:
        print(f"Percentile: {percentile_str}")
    print("="*80)
    
    generate_anchor_table(anchors_file, flip_file, ablate_file, output_file)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

