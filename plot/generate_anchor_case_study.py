#!/usr/bin/env python3
"""
Generate case study JSON for anchor chunks with their flipped/ablated full_cot.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

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

def load_experiment_results(flip_file: str, ablate_file: str) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """Load experiment results with full_cot."""
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
                            'full_cot': full_cot,
                            'final_answer': exp.get('final_answer', ''),
                            'is_correct': exp.get('is_correct', False)
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
                            'full_cot': full_cot,
                            'final_answer': exp.get('final_answer', ''),
                            'is_correct': exp.get('is_correct', False)
                        }
    
    return flip_results, ablate_results

def main():
    """Main function to generate anchor case study."""
    anchors_file = str(PROJECT_ROOT / "visualizations/analysis/anchors.json")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    output_file = str(PROJECT_ROOT / "visualizations/analysis/anchor_case_study.json")
    
    print("="*80)
    print("GENERATING ANCHOR CASE STUDY")
    print("="*80)
    
    # Load anchors
    print("\nLoading anchors...")
    anchor_set = load_anchors(anchors_file)
    print(f"✓ Loaded {len(anchor_set)} anchor chunks")
    
    # Load experiment results
    print("Loading experiment results...")
    flip_results, ablate_results = load_experiment_results(flip_file, ablate_file)
    print(f"✓ Loaded {len(flip_results)} flip experiments")
    print(f"✓ Loaded {len(ablate_results)} ablate experiments")
    
    # Find anchor chunks with experiments
    anchor_case_study = {
        'total_anchors': len(anchor_set),
        'anchors_with_flip_experiments': [],
        'anchors_with_ablate_experiments': []
    }
    
    # Process flip experiments for anchors
    for chunk_idx in sorted(anchor_set):
        if chunk_idx in flip_results:
            anchor_case_study['anchors_with_flip_experiments'].append({
                'chunk_idx': chunk_idx,
                'experiment_type': 'flip_chunk',
                'full_cot': flip_results[chunk_idx]['full_cot'],
                'final_answer': flip_results[chunk_idx]['final_answer'],
                'is_correct': flip_results[chunk_idx]['is_correct']
            })
    
    # Process ablate experiments for anchors
    for chunk_idx in sorted(anchor_set):
        if chunk_idx in ablate_results:
            anchor_case_study['anchors_with_ablate_experiments'].append({
                'chunk_idx': chunk_idx,
                'experiment_type': 'ablate_uncertainty',
                'full_cot': ablate_results[chunk_idx]['full_cot'],
                'final_answer': ablate_results[chunk_idx]['final_answer'],
                'is_correct': ablate_results[chunk_idx]['is_correct']
            })
    
    print(f"\n✓ Found {len(anchor_case_study['anchors_with_flip_experiments'])} anchors with flip experiments")
    print(f"✓ Found {len(anchor_case_study['anchors_with_ablate_experiments'])} anchors with ablate experiments")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(anchor_case_study, f, indent=2)
    
    print("\n" + "="*80)
    print("ANCHOR CASE STUDY GENERATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()

