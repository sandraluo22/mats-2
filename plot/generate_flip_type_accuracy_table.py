#!/usr/bin/env python3
"""
Generate accuracy comparison table for flip experiments grouped by flip type.
"""

import json
import os
from datasets import load_from_disk
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict

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

def generate_flip_type_table(flip_file: str, flip_types_file: str, output_file: str):
    """Generate accuracy table for flip experiments grouped by flip type."""
    with open(flip_file, 'r') as f:
        flip_data = json.load(f)
    
    problem_id = flip_data.get('problem_id', 'problem_1591')
    ground_truth = flip_data.get('ground_truth_answer', '')
    
    # Load flip types
    flip_type_map = load_flip_types(flip_types_file)
    
    # Load dataset to get control accuracy and chunk rollout accuracies
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    dataset = load_from_disk(dataset_path)
    
    experiments = flip_data.get('experiments', [])
    
    # Group experiments by flip type
    experiments_by_type = defaultdict(list)
    for exp in experiments:
        chunk_idx = exp.get('flipped_chunk_idx')
        if chunk_idx is not None:
            flip_type = flip_type_map.get(chunk_idx, 'Unknown')
            experiments_by_type[flip_type].append(exp)
    
    # Get all chunk indices from experiments
    chunk_indices = [exp.get('flipped_chunk_idx') for exp in experiments if exp.get('flipped_chunk_idx') is not None]
    
    # Calculate overall control accuracy across all chunks
    overall_control_accuracy = calculate_overall_control_accuracy(dataset, chunk_indices, ground_truth)
    
    # Create table data
    table_rows = []
    table_rows.append("="*80)
    table_rows.append("FLIP CHUNK EXPERIMENTS - ACCURACY BY FLIP TYPE")
    table_rows.append("="*80)
    table_rows.append("")
    table_rows.append(f"Problem ID: {problem_id}")
    table_rows.append(f"Ground Truth: {ground_truth}")
    table_rows.append("")
    
    # Summary by flip type
    table_rows.append("SUMMARY BY FLIP TYPE:")
    table_rows.append("-"*80)
    table_rows.append(f"{'Flip Type':<50} {'Count':<10} {'Correct':<10} {'Accuracy':<10}")
    table_rows.append("-"*80)
    
    type_summaries = []
    for flip_type in sorted(experiments_by_type.keys()):
        type_experiments = experiments_by_type[flip_type]
        correct_count = sum(1 for exp in type_experiments if exp.get('is_correct', False))
        total_count = len(type_experiments)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        type_summaries.append({
            'type': flip_type,
            'count': total_count,
            'correct': correct_count,
            'accuracy': accuracy
        })
        
        table_rows.append(f"{flip_type:<50} {total_count:<10} {correct_count:<10} {accuracy*100:.2f}%")
    
    table_rows.append("")
    table_rows.append("="*80)
    table_rows.append("DETAILED BREAKDOWN BY FLIP TYPE")
    table_rows.append("="*80)
    table_rows.append("")
    
    # Detailed breakdown by flip type
    for flip_type in sorted(experiments_by_type.keys()):
        type_experiments = experiments_by_type[flip_type]
        
        table_rows.append(f"Flip Type: {flip_type}")
        table_rows.append("-"*80)
        table_rows.append(f"{'Chunk Index':<15} {'Control Correct %':<20} {'Experiment Correct':<20}")
        table_rows.append("-"*80)
        
        correct_count = 0
        total_count = 0
        
        for exp in sorted(type_experiments, key=lambda x: x.get('flipped_chunk_idx', 0)):
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
        
        # Type summary
        if total_count > 0:
            type_accuracy = correct_count / total_count
            table_rows.append("-"*80)
            table_rows.append(f"{'Type Accuracy':<15} {type_accuracy:.4f} ({correct_count}/{total_count})")
        
        table_rows.append("")
    
    # Overall summary
    table_rows.append("="*80)
    table_rows.append("OVERALL SUMMARY")
    table_rows.append("="*80)
    
    total_correct = sum(1 for exp in experiments if exp.get('is_correct', False))
    total_experiments = len([exp for exp in experiments if exp.get('flipped_chunk_idx') is not None])
    
    if total_experiments > 0:
        overall_accuracy = total_correct / total_experiments
        table_rows.append(f"Overall Flip Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_experiments})")
    
    if overall_control_accuracy is not None:
        table_rows.append(f"Overall Control Accuracy: {overall_control_accuracy:.4f} (across all chunks)")
        if total_experiments > 0:
            table_rows.append(f"Control vs Flip: Control: {overall_control_accuracy:.4f}, Flip: {overall_accuracy:.4f}, Diff: {overall_accuracy - overall_control_accuracy:+.4f}")
    
    table_rows.append("")
    table_rows.append("="*80)
    
    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(table_rows))
    
    print(f"✓ Generated flip type accuracy table: {output_file}")
    print(f"  - {len(experiments_by_type)} flip types")
    print(f"  - {total_experiments} total experiments")
    
    return overall_accuracy if total_experiments > 0 else None

def main():
    """Main function."""
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    flip_types_file = str(PROJECT_ROOT / "flip_types.txt")
    output_file = str(PROJECT_ROOT / "visualizations/analysis/flip_type_accuracy_table.txt")
    
    print("="*80)
    print("GENERATING FLIP TYPE ACCURACY TABLE")
    print("="*80)
    
    generate_flip_type_table(flip_file, flip_types_file, output_file)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

