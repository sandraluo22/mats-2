#!/usr/bin/env python3
"""
Generate accuracy comparison tables for flip and ablate experiments.
Compares each experiment's correctness to control and calculates overall accuracy.
"""

import json
import os
from datasets import load_from_disk
from typing import Dict, List, Tuple
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

def load_control_accuracy(dataset, problem_id: str, ground_truth: str) -> Tuple[bool, str]:
    """
    Load control accuracy by checking base_solution.json.
    Control is the original incorrect_base_solution.
    Returns: (is_correct, final_answer)
    """
    for ex in dataset:
        path = ex.get('path', '')
        if 'base_solution.json' in path and 'incorrect_base_solution' in path and 'forced_answer' not in path:
            if problem_id in path:
                solution = json.loads(ex.get('content', '{}'))
                final_answer = solution.get('answer', '')
                is_correct = check_correctness(final_answer, ground_truth)
                return (is_correct, final_answer)
    return (None, None)

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

def generate_flip_table(flip_file: str, output_file: str):
    """Generate accuracy table for flip experiments."""
    with open(flip_file, 'r') as f:
        flip_data = json.load(f)
    
    problem_id = flip_data.get('problem_id', 'problem_1591')
    ground_truth = flip_data.get('ground_truth_answer', '')
    
    # Load dataset to get control accuracy and chunk rollout accuracies
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    dataset = load_from_disk(dataset_path)
    
    experiments = flip_data.get('experiments', [])
    
    # Get all chunk indices from experiments
    chunk_indices = [exp.get('flipped_chunk_idx') for exp in experiments if exp.get('flipped_chunk_idx') is not None]
    
    # Calculate overall control accuracy across all chunks
    overall_control_accuracy = calculate_overall_control_accuracy(dataset, chunk_indices, ground_truth)
    
    # Create table data
    table_rows = []
    table_rows.append("="*80)
    table_rows.append("FLIP CHUNK EXPERIMENTS - ACCURACY COMPARISON")
    table_rows.append("="*80)
    table_rows.append("")
    table_rows.append(f"Problem ID: {problem_id}")
    table_rows.append(f"Ground Truth: {ground_truth}")
    table_rows.append("")
    table_rows.append(f"{'Chunk Index':<15} {'Control Correct %':<20} {'Experiment Correct':<20}")
    table_rows.append("-"*55)
    
    correct_count = 0
    total_count = 0
    
    for exp in sorted(experiments, key=lambda x: x.get('flipped_chunk_idx', 0)):
        chunk_idx = exp.get('flipped_chunk_idx')
        is_correct = exp.get('is_correct', False)
        
        if chunk_idx is not None:
            total_count += 1
            if is_correct:
                correct_count += 1
            
            # Get control rollout accuracy for this chunk
            control_rollout_acc = load_chunk_rollout_accuracy(dataset, chunk_idx, ground_truth)
            if control_rollout_acc is not None:
                control_acc_str = f"{control_rollout_acc*100:.2f}%"
            else:
                control_acc_str = "N/A"
            
            table_rows.append(f"{chunk_idx:<15} {control_acc_str:<20} {str(is_correct):<20}")
    
    # Calculate overall accuracy
    if total_count > 0:
        overall_accuracy = correct_count / total_count
        table_rows.append("")
        table_rows.append("-"*55)
        table_rows.append(f"{'Overall Accuracy':<15} {overall_accuracy:.4f} ({correct_count}/{total_count})")
    
    # Add control comparison
    table_rows.append("")
    if overall_control_accuracy is not None:
        table_rows.append(f"{'Control Overall':<15} {overall_control_accuracy:.4f} (across all chunks)")
        table_rows.append(f"{'Flip Overall':<15} {overall_accuracy:.4f} ({correct_count}/{total_count})")
        table_rows.append(f"{'Control vs Flip':<15} Control: {overall_control_accuracy:.4f}, Flip: {overall_accuracy:.4f}, Diff: {overall_accuracy - overall_control_accuracy:+.4f}")
    
    table_rows.append("")
    table_rows.append("="*80)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(table_rows))
    
    print(f"✓ Generated flip accuracy table: {output_file}")
    return overall_accuracy if total_count > 0 else None

def generate_ablate_table(ablate_file: str, output_file: str):
    """Generate accuracy table for ablate experiments."""
    with open(ablate_file, 'r') as f:
        ablate_data = json.load(f)
    
    problem_id = ablate_data.get('problem_id', 'problem_1591')
    ground_truth = ablate_data.get('ground_truth_answer', '')
    
    # Load dataset to get control accuracy and chunk rollout accuracies
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    dataset = load_from_disk(dataset_path)
    
    experiments = ablate_data.get('experiments', [])
    
    # Get all chunk indices from experiments
    chunk_indices = [exp.get('ablated_chunk_idx') for exp in experiments if exp.get('ablated_chunk_idx') is not None]
    
    # Calculate overall control accuracy across all chunks
    overall_control_accuracy = calculate_overall_control_accuracy(dataset, chunk_indices, ground_truth)
    
    # Create table data
    table_rows = []
    table_rows.append("="*80)
    table_rows.append("ABLATE UNCERTAINTY EXPERIMENTS - ACCURACY COMPARISON")
    table_rows.append("="*80)
    table_rows.append("")
    table_rows.append(f"Problem ID: {problem_id}")
    table_rows.append(f"Ground Truth: {ground_truth}")
    table_rows.append("")
    table_rows.append(f"{'Chunk Index':<15} {'Control Correct %':<20} {'Experiment Correct':<20}")
    table_rows.append("-"*55)
    
    correct_count = 0
    total_count = 0
    
    for exp in sorted(experiments, key=lambda x: x.get('ablated_chunk_idx', 0)):
        chunk_idx = exp.get('ablated_chunk_idx')
        is_correct = exp.get('is_correct', False)
        
        if chunk_idx is not None:
            total_count += 1
            if is_correct:
                correct_count += 1
            
            # Get control rollout accuracy for this chunk
            control_rollout_acc = load_chunk_rollout_accuracy(dataset, chunk_idx, ground_truth)
            if control_rollout_acc is not None:
                control_acc_str = f"{control_rollout_acc*100:.2f}%"
            else:
                control_acc_str = "N/A"
            
            table_rows.append(f"{chunk_idx:<15} {control_acc_str:<20} {str(is_correct):<20}")
    
    # Calculate overall accuracy
    if total_count > 0:
        overall_accuracy = correct_count / total_count
        table_rows.append("")
        table_rows.append("-"*55)
        table_rows.append(f"{'Overall Accuracy':<15} {overall_accuracy:.4f} ({correct_count}/{total_count})")
    
    # Add control comparison
    table_rows.append("")
    if overall_control_accuracy is not None:
        table_rows.append(f"{'Control Overall':<15} {overall_control_accuracy:.4f} (across all chunks)")
        table_rows.append(f"{'Ablate Overall':<15} {overall_accuracy:.4f} ({correct_count}/{total_count})")
        table_rows.append(f"{'Control vs Ablate':<15} Control: {overall_control_accuracy:.4f}, Ablate: {overall_accuracy:.4f}, Diff: {overall_accuracy - overall_control_accuracy:+.4f}")
    
    table_rows.append("")
    table_rows.append("="*80)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(table_rows))
    
    print(f"✓ Generated ablate accuracy table: {output_file}")
    return overall_accuracy if total_count > 0 else None

def main():
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    output_dir = str(PROJECT_ROOT / "visualizations/analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("GENERATING ACCURACY COMPARISON TABLES")
    print("="*80)
    
    # Generate flip table
    if os.path.exists(flip_file):
        flip_output = os.path.join(output_dir, "flip_accuracy_table.txt")
        flip_accuracy = generate_flip_table(flip_file, flip_output)
    else:
        print(f"✗ {flip_file} not found")
        flip_accuracy = None
    
    # Generate ablate table
    if os.path.exists(ablate_file):
        ablate_output = os.path.join(output_dir, "ablate_accuracy_table.txt")
        ablate_accuracy = generate_ablate_table(ablate_file, ablate_output)
    else:
        print(f"✗ {ablate_file} not found")
        ablate_accuracy = None
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if flip_accuracy is not None:
        print(f"Flip experiments accuracy: {flip_accuracy:.4f}")
    if ablate_accuracy is not None:
        print(f"Ablate experiments accuracy: {ablate_accuracy:.4f}")

if __name__ == '__main__':
    main()

