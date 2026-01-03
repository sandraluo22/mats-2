#!/usr/bin/env python3
"""
Calculate accuracy statistics for non-anchor chunks across all percentile thresholds.
"""

import json
from pathlib import Path
from datasets import load_from_disk
from typing import Dict, Set, List
from collections import defaultdict

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

def calculate_overall_control_accuracy(dataset, chunk_indices: List[int], ground_truth: str) -> float:
    """Calculate overall control accuracy across all chunks."""
    total_rollouts = 0
    correct_rollouts = 0
    
    for chunk_idx in chunk_indices:
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
    return None

def get_all_chunk_indices(dataset) -> Set[int]:
    """Get all chunk indices that have rollouts."""
    chunk_indices = set()
    for ex in dataset:
        path = ex.get('path', '')
        import re
        match = re.search(r'chunk_(\d+)/solutions\.json', path)
        if match:
            chunk_indices.add(int(match.group(1)))
    return chunk_indices

def main():
    """Main function to calculate non-anchor accuracies."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    
    print("="*80)
    print("CALCULATING NON-ANCHOR ACCURACIES")
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
    
    # Get all chunk indices
    all_chunk_indices = get_all_chunk_indices(dataset)
    print(f"✓ Found {len(all_chunk_indices)} chunks with rollouts")
    
    # Process each percentile
    percentiles = [
        ('05percent', 0.05),
        ('10percent', 0.10),  # Original
        ('15percent', 0.15),
        ('20percent', 0.20),
        ('25percent', 0.25)
    ]
    
    results = {}
    
    for percentile_str, percentile_val in percentiles:
        print(f"\n{'='*80}")
        print(f"PROCESSING {percentile_val*100:.0f}% PERCENTILE")
        print(f"{'='*80}")
        
        # Load anchors for this percentile
        if percentile_str == '10percent':
            anchors_file = str(PROJECT_ROOT / "visualizations/analysis/anchors.json")
        else:
            anchors_file = str(PROJECT_ROOT / f"visualizations/analysis/anchors/anchors_{percentile_str}.json")
        
        anchor_set = load_anchors(anchors_file)
        print(f"✓ Loaded {len(anchor_set)} anchor chunks")
        
        # Get non-anchor chunks
        non_anchor_chunks = all_chunk_indices - anchor_set
        print(f"✓ Found {len(non_anchor_chunks)} non-anchor chunks")
        
        # Get flip experiments for non-anchors
        non_anchor_flip_experiments = [
            exp for exp in flip_experiments
            if exp.get('flipped_chunk_idx') in non_anchor_chunks
        ]
        
        # Calculate flip accuracy for non-anchors
        flip_correct = 0
        flip_total = 0
        for exp in non_anchor_flip_experiments:
            chunk_idx = exp.get('flipped_chunk_idx')
            if chunk_idx is not None:
                flip_total += 1
                is_correct = exp.get('is_correct', False)
                if is_correct:
                    flip_correct += 1
        
        flip_accuracy = flip_correct / flip_total if flip_total > 0 else None
        
        # Calculate control accuracy for non-anchors
        non_anchor_list = sorted(list(non_anchor_chunks))
        control_accuracy = calculate_overall_control_accuracy(dataset, non_anchor_list, ground_truth)
        
        # Get ablate experiments for non-anchors
        with open(ablate_file, 'r') as f:
            ablate_data = json.load(f)
        ablate_experiments = ablate_data.get('experiments', [])
        
        non_anchor_ablate_experiments = [
            exp for exp in ablate_experiments
            if exp.get('ablated_chunk_idx') in non_anchor_chunks
        ]
        
        ablate_correct = 0
        ablate_total = 0
        for exp in non_anchor_ablate_experiments:
            chunk_idx = exp.get('ablated_chunk_idx')
            if chunk_idx is not None:
                ablate_total += 1
                is_correct = exp.get('is_correct', False)
                if is_correct:
                    ablate_correct += 1
        
        ablate_accuracy = ablate_correct / ablate_total if ablate_total > 0 else None
        
        results[percentile_str] = {
            'non_anchor_count': len(non_anchor_chunks),
            'flip_accuracy': flip_accuracy,
            'flip_correct': flip_correct,
            'flip_total': flip_total,
            'control_accuracy': control_accuracy,
            'ablate_accuracy': ablate_accuracy,
            'ablate_correct': ablate_correct,
            'ablate_total': ablate_total
        }
        
        print(f"\nNon-Anchor Results:")
        print(f"  Count: {len(non_anchor_chunks)}")
        if flip_accuracy is not None:
            print(f"  Flip Accuracy: {flip_accuracy*100:.2f}% ({flip_correct}/{flip_total})")
        if control_accuracy is not None:
            print(f"  Control Accuracy: {control_accuracy*100:.2f}%")
        if ablate_accuracy is not None:
            print(f"  Ablate Accuracy: {ablate_accuracy*100:.2f}% ({ablate_correct}/{ablate_total})")
    
    # Save results
    output_file = str(PROJECT_ROOT / "visualizations/analysis/anchors/non_anchor_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to {output_file}")
    print("\n" + "="*80)
    print("NON-ANCHOR ACCURACY CALCULATION COMPLETE")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()

