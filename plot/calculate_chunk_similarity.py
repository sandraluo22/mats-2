#!/usr/bin/env python3
"""
Calculate percentage similarity (overlap) between chunk index sets:
- Repeated final answers
- Anchors
- Ablate accuracy table
"""

import json
import re
from pathlib import Path
from typing import Set

PROJECT_ROOT = Path(__file__).parent.parent

def load_repeated_final_answers(repeated_file: str) -> Set[int]:
    """Load chunk indices from repeated final answers JSON."""
    chunk_indices = set()
    
    if not Path(repeated_file).exists():
        return chunk_indices
    
    with open(repeated_file, 'r') as f:
        data = json.load(f)
    
    # Extract chunk indices from the data
    for entry in data.get('chunks', []):
        chunk_idx = entry.get('chunk_idx')
        if chunk_idx is not None:
            chunk_indices.add(chunk_idx)
    
    return chunk_indices

def load_anchors(anchors_file: str) -> Set[int]:
    """Load anchor chunk indices."""
    chunk_indices = set()
    
    if not Path(anchors_file).exists():
        return chunk_indices
    
    with open(anchors_file, 'r') as f:
        data = json.load(f)
    
    for anchor in data.get('anchors', []):
        chunk_idx = anchor.get('chunk_idx')
        if chunk_idx is not None:
            chunk_indices.add(chunk_idx)
    
    return chunk_indices

def load_ablate_indices(ablate_table_file: str) -> Set[int]:
    """Load chunk indices from ablate accuracy table."""
    chunk_indices = set()
    
    if not Path(ablate_table_file).exists():
        return chunk_indices
    
    with open(ablate_table_file, 'r') as f:
        content = f.read()
    
    # Parse the table
    lines = content.split('\n')
    in_data_section = False
    
    for line in lines:
        if 'Chunk Index' in line:
            in_data_section = True
            continue
        
        if 'Overall Accuracy' in line or 'Control Overall' in line or 'Ablate Overall' in line:
            break
        
        if in_data_section and line.strip() and not line.strip().startswith('-'):
            parts = line.split()
            if len(parts) >= 1:
                try:
                    chunk_idx = int(parts[0])
                    chunk_indices.add(chunk_idx)
                except (ValueError, IndexError):
                    continue
    
    return chunk_indices

def calculate_similarity(set1: Set[int], set2: Set[int]) -> dict:
    """
    Calculate similarity metrics between two sets.
    
    Returns:
        Dictionary with various similarity metrics
    """
    intersection = set1 & set2
    union = set1 | set2
    
    # Jaccard similarity (intersection / union)
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # Overlap coefficient (intersection / min size)
    min_size = min(len(set1), len(set2))
    overlap_coef = len(intersection) / min_size if min_size > 0 else 0.0
    
    # Percentage of set1 in set2
    pct_set1_in_set2 = len(intersection) / len(set1) if len(set1) > 0 else 0.0
    
    # Percentage of set2 in set1
    pct_set2_in_set1 = len(intersection) / len(set2) if len(set2) > 0 else 0.0
    
    return {
        'intersection_size': len(intersection),
        'set1_size': len(set1),
        'set2_size': len(set2),
        'union_size': len(union),
        'jaccard_similarity': jaccard,
        'jaccard_percentage': jaccard * 100,
        'overlap_coefficient': overlap_coef,
        'overlap_percentage': overlap_coef * 100,
        'pct_set1_in_set2': pct_set1_in_set2 * 100,
        'pct_set2_in_set1': pct_set2_in_set1 * 100,
        'intersection': sorted(intersection)
    }

def main():
    """Main function to calculate similarities."""
    repeated_file = str(PROJECT_ROOT / "visualizations/analysis/repeated_final_answers.json")
    anchors_file = str(PROJECT_ROOT / "visualizations/analysis/anchors.json")
    ablate_table_file = str(PROJECT_ROOT / "visualizations/analysis/ablate_accuracy_table.txt")
    output_file = str(PROJECT_ROOT / "visualizations/analysis/chunk_similarity_analysis.txt")
    
    print("="*80)
    print("CALCULATING CHUNK INDEX SIMILARITY")
    print("="*80)
    
    # Load chunk indices
    print("\nLoading chunk indices...")
    repeated_indices = load_repeated_final_answers(repeated_file)
    print(f"  Repeated final answers: {len(repeated_indices)} chunks")
    print(f"    Indices: {sorted(repeated_indices)}")
    
    anchor_indices = load_anchors(anchors_file)
    print(f"  Anchors: {len(anchor_indices)} chunks")
    print(f"    Indices: {sorted(anchor_indices)}")
    
    ablate_indices = load_ablate_indices(ablate_table_file)
    print(f"  Ablate accuracy: {len(ablate_indices)} chunks")
    print(f"    Indices: {sorted(ablate_indices)}")
    
    # Calculate pairwise similarities
    print("\n" + "="*80)
    print("SIMILARITY ANALYSIS")
    print("="*80)
    
    # Repeated Final Answers vs Anchors
    print("\n1. Repeated Final Answers vs Anchors:")
    sim_ra = calculate_similarity(repeated_indices, anchor_indices)
    print(f"   Intersection: {sim_ra['intersection_size']} chunks")
    print(f"   Jaccard Similarity: {sim_ra['jaccard_percentage']:.2f}%")
    print(f"   Overlap Coefficient: {sim_ra['overlap_percentage']:.2f}%")
    print(f"   % of Repeated in Anchors: {sim_ra['pct_set1_in_set2']:.2f}%")
    print(f"   % of Anchors in Repeated: {sim_ra['pct_set2_in_set1']:.2f}%")
    if sim_ra['intersection']:
        print(f"   Common indices: {sim_ra['intersection']}")
    
    # Repeated Final Answers vs Ablate
    print("\n2. Repeated Final Answers vs Ablate:")
    sim_rb = calculate_similarity(repeated_indices, ablate_indices)
    print(f"   Intersection: {sim_rb['intersection_size']} chunks")
    print(f"   Jaccard Similarity: {sim_rb['jaccard_percentage']:.2f}%")
    print(f"   Overlap Coefficient: {sim_rb['overlap_percentage']:.2f}%")
    print(f"   % of Repeated in Ablate: {sim_rb['pct_set1_in_set2']:.2f}%")
    print(f"   % of Ablate in Repeated: {sim_rb['pct_set2_in_set1']:.2f}%")
    if sim_rb['intersection']:
        print(f"   Common indices: {sim_rb['intersection']}")
    
    # Anchors vs Ablate
    print("\n3. Anchors vs Ablate:")
    sim_ab = calculate_similarity(anchor_indices, ablate_indices)
    print(f"   Intersection: {sim_ab['intersection_size']} chunks")
    print(f"   Jaccard Similarity: {sim_ab['jaccard_percentage']:.2f}%")
    print(f"   Overlap Coefficient: {sim_ab['overlap_percentage']:.2f}%")
    print(f"   % of Anchors in Ablate: {sim_ab['pct_set1_in_set2']:.2f}%")
    print(f"   % of Ablate in Anchors: {sim_ab['pct_set2_in_set1']:.2f}%")
    if sim_ab['intersection']:
        print(f"   Common indices: {sim_ab['intersection']}")
    
    # All three sets intersection
    print("\n4. All Three Sets (Repeated, Anchors, Ablate):")
    all_intersection = repeated_indices & anchor_indices & ablate_indices
    print(f"   Triple intersection: {len(all_intersection)} chunks")
    if all_intersection:
        print(f"   Common indices: {sorted(all_intersection)}")
    
    # Write results to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CHUNK INDEX SIMILARITY ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("SET SIZES:\n")
        f.write(f"  Repeated Final Answers: {len(repeated_indices)} chunks\n")
        f.write(f"  Anchors: {len(anchor_indices)} chunks\n")
        f.write(f"  Ablate Accuracy: {len(ablate_indices)} chunks\n\n")
        
        f.write("SET INDICES:\n")
        f.write(f"  Repeated Final Answers: {sorted(repeated_indices)}\n")
        f.write(f"  Anchors: {sorted(anchor_indices)}\n")
        f.write(f"  Ablate Accuracy: {sorted(ablate_indices)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("PAIRWISE SIMILARITY METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. REPEATED FINAL ANSWERS vs ANCHORS\n")
        f.write("-"*80 + "\n")
        f.write(f"Intersection size: {sim_ra['intersection_size']} chunks\n")
        f.write(f"Jaccard Similarity: {sim_ra['jaccard_percentage']:.2f}% (intersection/union)\n")
        f.write(f"Overlap Coefficient: {sim_ra['overlap_percentage']:.2f}% (intersection/min_size)\n")
        f.write(f"% of Repeated in Anchors: {sim_ra['pct_set1_in_set2']:.2f}%\n")
        f.write(f"% of Anchors in Repeated: {sim_ra['pct_set2_in_set1']:.2f}%\n")
        if sim_ra['intersection']:
            f.write(f"Common indices: {sim_ra['intersection']}\n")
        f.write("\n")
        
        f.write("2. REPEATED FINAL ANSWERS vs ABLATE\n")
        f.write("-"*80 + "\n")
        f.write(f"Intersection size: {sim_rb['intersection_size']} chunks\n")
        f.write(f"Jaccard Similarity: {sim_rb['jaccard_percentage']:.2f}% (intersection/union)\n")
        f.write(f"Overlap Coefficient: {sim_rb['overlap_percentage']:.2f}% (intersection/min_size)\n")
        f.write(f"% of Repeated in Ablate: {sim_rb['pct_set1_in_set2']:.2f}%\n")
        f.write(f"% of Ablate in Repeated: {sim_rb['pct_set2_in_set1']:.2f}%\n")
        if sim_rb['intersection']:
            f.write(f"Common indices: {sim_rb['intersection']}\n")
        f.write("\n")
        
        f.write("3. ANCHORS vs ABLATE\n")
        f.write("-"*80 + "\n")
        f.write(f"Intersection size: {sim_ab['intersection_size']} chunks\n")
        f.write(f"Jaccard Similarity: {sim_ab['jaccard_percentage']:.2f}% (intersection/union)\n")
        f.write(f"Overlap Coefficient: {sim_ab['overlap_percentage']:.2f}% (intersection/min_size)\n")
        f.write(f"% of Anchors in Ablate: {sim_ab['pct_set1_in_set2']:.2f}%\n")
        f.write(f"% of Ablate in Anchors: {sim_ab['pct_set2_in_set1']:.2f}%\n")
        if sim_ab['intersection']:
            f.write(f"Common indices: {sim_ab['intersection']}\n")
        f.write("\n")
        
        f.write("4. ALL THREE SETS INTERSECTION\n")
        f.write("-"*80 + "\n")
        f.write(f"Triple intersection size: {len(all_intersection)} chunks\n")
        if all_intersection:
            f.write(f"Common indices (in all three): {sorted(all_intersection)}\n")
        else:
            f.write("No chunks appear in all three sets.\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n✓ Saved similarity analysis to: {output_file}")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

