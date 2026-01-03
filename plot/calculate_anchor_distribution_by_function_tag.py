#!/usr/bin/env python3
"""
Calculate anchor distribution by function tag across different percentile thresholds.
"""

import json
from pathlib import Path
from datasets import load_from_disk
from typing import Dict, Set, List
from collections import defaultdict

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

def load_chunk_function_tags(dataset) -> Dict[int, str]:
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
                        # Use first function tag if multiple exist, or 'unknown' if empty
                        if isinstance(function_tags, list) and len(function_tags) > 0:
                            chunk_tags[chunk_idx] = function_tags[0]
                        else:
                            chunk_tags[chunk_idx] = 'unknown'
                break
            except json.JSONDecodeError:
                continue
    return chunk_tags

def main():
    """Main function to calculate anchor distribution by function tag."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    
    print("="*80)
    print("CALCULATING ANCHOR DISTRIBUTION BY FUNCTION TAG")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ Loaded dataset")
    
    # Load function tags
    print("\nLoading function tags...")
    chunk_function_tags = load_chunk_function_tags(dataset)
    print(f"✓ Loaded function tags for {len(chunk_function_tags)} chunks")
    
    # Process each percentile
    percentiles = [
        ('05percent', '5%'),
        ('10percent', '10% (Original)'),
        ('15percent', '15%'),
        ('20percent', '20%'),
        ('25percent', '25%')
    ]
    
    # Store distribution data
    distribution_data = defaultdict(lambda: defaultdict(int))
    all_function_tags = set()
    
    for percentile_str, percentile_label in percentiles:
        print(f"\nProcessing {percentile_label}...")
        
        # Load anchors for this percentile
        if percentile_str == '10percent':
            anchors_file = str(PROJECT_ROOT / "visualizations/analysis/anchors.json")
        else:
            anchors_file = str(PROJECT_ROOT / f"visualizations/analysis/anchors/anchors_{percentile_str}.json")
        
        anchor_set = load_anchors(anchors_file)
        print(f"✓ Loaded {len(anchor_set)} anchor chunks")
        
        # Count anchors by function tag
        for chunk_idx in anchor_set:
            function_tag = chunk_function_tags.get(chunk_idx, 'unknown')
            all_function_tags.add(function_tag)
            distribution_data[percentile_label][function_tag] += 1
    
    # Create table
    all_function_tags = sorted(list(all_function_tags))
    
    # Print table
    print("\n" + "="*80)
    print("ANCHOR DISTRIBUTION BY FUNCTION TAG")
    print("="*80)
    
    # Header
    header = f"{'Function Tag':<30}"
    for percentile_str, percentile_label in percentiles:
        header += f" {percentile_label:<20}"
    print(header)
    print("-" * 150)
    
    # Rows
    table_rows = []
    for function_tag in all_function_tags:
        row = f"{function_tag:<30}"
        row_data = [function_tag]
        for percentile_str, percentile_label in percentiles:
            count = distribution_data[percentile_label][function_tag]
            row += f" {count:<20}"
            row_data.append(count)
        print(row)
        table_rows.append(row_data)
    
    # Total row
    print("-" * 150)
    total_row = f"{'TOTAL':<30}"
    total_row_data = ['TOTAL']
    for percentile_str, percentile_label in percentiles:
        total = sum(distribution_data[percentile_label].values())
        total_row += f" {total:<20}"
        total_row_data.append(total)
    print(total_row)
    table_rows.append(total_row_data)
    
    # Save to file
    output_file = str(PROJECT_ROOT / "visualizations/analysis/anchors/anchor_distribution_by_function_tag.txt")
    with open(output_file, 'w') as f:
        f.write("="*120 + "\n")
        f.write("ANCHOR DISTRIBUTION BY FUNCTION TAG\n")
        f.write("="*120 + "\n\n")
        
        # Header
        f.write(f"{'Function Tag':<30}")
        for percentile_str, percentile_label in percentiles:
            f.write(f" {percentile_label:<20}")
        f.write("\n")
        f.write("-" * 150 + "\n")
        
        # Rows
        for function_tag in all_function_tags:
            f.write(f"{function_tag:<30}")
            for percentile_str, percentile_label in percentiles:
                count = distribution_data[percentile_label][function_tag]
                f.write(f" {count:<20}")
            f.write("\n")
        
        # Total row
        f.write("-" * 150 + "\n")
        f.write(f"{'TOTAL':<30}")
        for percentile_str, percentile_label in percentiles:
            total = sum(distribution_data[percentile_label].values())
            f.write(f" {total:<20}")
        f.write("\n")
        f.write("="*150 + "\n")
    
    print(f"\n✓ Saved distribution table to {output_file}")
    
    # Also save as JSON for easier processing
    json_output = {}
    for percentile_str, percentile_label in percentiles:
        json_output[percentile_str] = dict(distribution_data[percentile_label])
    
    json_file = str(PROJECT_ROOT / "visualizations/analysis/anchors/anchor_distribution_by_function_tag.json")
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"✓ Saved distribution data to {json_file}")
    
    print("\n" + "="*80)
    print("ANCHOR DISTRIBUTION CALCULATION COMPLETE")
    print("="*80)
    
    return table_rows, all_function_tags, percentiles

if __name__ == "__main__":
    main()

