#!/usr/bin/env python3
"""
Calculate anchor distribution by flip type across different percentile thresholds.
"""

import json
from pathlib import Path
from typing import Dict, Set
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

def main():
    """Main function to calculate anchor distribution by flip type."""
    flip_types_file = str(PROJECT_ROOT / "flip_types.txt")
    
    print("="*80)
    print("CALCULATING ANCHOR DISTRIBUTION BY FLIP TYPE")
    print("="*80)
    
    # Load flip types
    print("\nLoading flip types...")
    chunk_flip_types = load_flip_types(flip_types_file)
    print(f"✓ Loaded flip types for {len(chunk_flip_types)} chunks")
    
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
    all_flip_types = set()
    
    for percentile_str, percentile_label in percentiles:
        print(f"\nProcessing {percentile_label}...")
        
        # Load anchors for this percentile
        if percentile_str == '10percent':
            anchors_file = str(PROJECT_ROOT / "visualizations/analysis/anchors.json")
        else:
            anchors_file = str(PROJECT_ROOT / f"visualizations/analysis/anchors/anchors_{percentile_str}.json")
        
        anchor_set = load_anchors(anchors_file)
        print(f"✓ Loaded {len(anchor_set)} anchor chunks")
        
        # Count anchors by flip type
        for chunk_idx in anchor_set:
            flip_type = chunk_flip_types.get(chunk_idx, 'Unknown')
            all_flip_types.add(flip_type)
            distribution_data[percentile_label][flip_type] += 1
    
    # Create table
    all_flip_types = sorted(list(all_flip_types))
    
    # Print table
    print("\n" + "="*80)
    print("ANCHOR DISTRIBUTION BY FLIP TYPE")
    print("="*80)
    
    # Header
    header = f"{'Flip Type':<50}"
    for percentile_str, percentile_label in percentiles:
        header += f" {percentile_label:<20}"
    print(header)
    print("-" * 170)
    
    # Rows
    table_rows = []
    for flip_type in all_flip_types:
        row = f"{flip_type:<50}"
        row_data = [flip_type]
        for percentile_str, percentile_label in percentiles:
            count = distribution_data[percentile_label][flip_type]
            row += f" {count:<20}"
            row_data.append(count)
        print(row)
        table_rows.append(row_data)
    
    # Total row
    print("-" * 170)
    total_row = f"{'TOTAL':<50}"
    total_row_data = ['TOTAL']
    for percentile_str, percentile_label in percentiles:
        total = sum(distribution_data[percentile_label].values())
        total_row += f" {total:<20}"
        total_row_data.append(total)
    print(total_row)
    table_rows.append(total_row_data)
    
    # Save to file
    output_file = str(PROJECT_ROOT / "visualizations/analysis/anchors/anchor_distribution_by_flip_type.txt")
    with open(output_file, 'w') as f:
        f.write("="*170 + "\n")
        f.write("ANCHOR DISTRIBUTION BY FLIP TYPE\n")
        f.write("="*170 + "\n\n")
        
        # Header
        f.write(f"{'Flip Type':<50}")
        for percentile_str, percentile_label in percentiles:
            f.write(f" {percentile_label:<20}")
        f.write("\n")
        f.write("-" * 170 + "\n")
        
        # Rows
        for flip_type in all_flip_types:
            f.write(f"{flip_type:<50}")
            for percentile_str, percentile_label in percentiles:
                count = distribution_data[percentile_label][flip_type]
                f.write(f" {count:<20}")
            f.write("\n")
        
        # Total row
        f.write("-" * 170 + "\n")
        f.write(f"{'TOTAL':<50}")
        for percentile_str, percentile_label in percentiles:
            total = sum(distribution_data[percentile_label].values())
            f.write(f" {total:<20}")
        f.write("\n")
        f.write("="*170 + "\n")
    
    print(f"\n✓ Saved distribution table to {output_file}")
    
    # Also save as JSON for easier processing
    json_output = {}
    for percentile_str, percentile_label in percentiles:
        json_output[percentile_str] = dict(distribution_data[percentile_label])
    
    json_file = str(PROJECT_ROOT / "visualizations/analysis/anchors/anchor_distribution_by_flip_type.json")
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"✓ Saved distribution data to {json_file}")
    
    print("\n" + "="*80)
    print("ANCHOR DISTRIBUTION CALCULATION COMPLETE")
    print("="*80)
    
    return table_rows, all_flip_types, percentiles

if __name__ == "__main__":
    main()

