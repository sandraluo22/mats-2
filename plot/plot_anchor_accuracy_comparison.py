#!/usr/bin/env python3
"""
Plot anchor and non-anchor accuracy comparison across percentile thresholds.
Shows control and flip accuracy for both anchors and non-anchors.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def load_anchor_accuracies():
    """Load anchor accuracies from accuracy tables."""
    percentiles = [
        ('05percent', 5),
        ('10percent', 10),
        ('15percent', 15),
        ('20percent', 20),
        ('25percent', 25)
    ]
    
    anchor_control = []
    anchor_flip = []
    non_anchor_control = []
    non_anchor_flip = []
    percentile_values = []
    
    # Load anchor accuracies
    for percentile_str, percentile_val in percentiles:
        if percentile_str == '10percent':
            table_file = PROJECT_ROOT / "visualizations/analysis/anchor_accuracy_table.txt"
        else:
            table_file = PROJECT_ROOT / f"visualizations/analysis/anchors/anchor_accuracy_table_{percentile_str}.txt"
        
        if table_file.exists():
            with open(table_file, 'r') as f:
                content = f.read()
            
            # Extract control accuracy
            import re
            control_match = re.search(r'Overall Control\s+(\d+\.\d+)%', content)
            if control_match:
                anchor_control.append(float(control_match.group(1)) / 100.0)
            else:
                anchor_control.append(None)
            
            # Extract flip accuracy
            flip_match = re.search(r'Flip Accuracy\s+(\d+\.\d+)%\s+\((\d+)/(\d+)\)', content)
            if flip_match:
                anchor_flip.append(float(flip_match.group(1)) / 100.0)
            else:
                anchor_flip.append(None)
    
    # Load non-anchor accuracies
    non_anchor_file = PROJECT_ROOT / "visualizations/analysis/anchors/non_anchor_results.json"
    if non_anchor_file.exists():
        with open(non_anchor_file, 'r') as f:
            non_anchor_data = json.load(f)
        
        for percentile_str, percentile_val in percentiles:
            percentile_values.append(percentile_val)
            if percentile_str in non_anchor_data:
                data = non_anchor_data[percentile_str]
                non_anchor_control.append(data.get('control_accuracy'))
                non_anchor_flip.append(data.get('flip_accuracy'))
            else:
                non_anchor_control.append(None)
                non_anchor_flip.append(None)
    else:
        # Fallback: calculate from percentiles
        for percentile_str, percentile_val in percentiles:
            percentile_values.append(percentile_val)
            non_anchor_control.append(None)
            non_anchor_flip.append(None)
    
    return percentile_values, anchor_control, anchor_flip, non_anchor_control, non_anchor_flip

def main():
    """Main function to create the plot."""
    print("="*80)
    print("PLOTTING ANCHOR ACCURACY COMPARISON")
    print("="*80)
    
    # Load data
    print("\nLoading accuracy data...")
    percentile_values, anchor_control, anchor_flip, non_anchor_control, non_anchor_flip = load_anchor_accuracies()
    
    print(f"Percentile values: {percentile_values}")
    print(f"Anchor control: {anchor_control}")
    print(f"Anchor flip: {anchor_flip}")
    print(f"Non-anchor control: {non_anchor_control}")
    print(f"Non-anchor flip: {non_anchor_flip}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot lines
    # Anchor lines
    ax.plot(percentile_values, anchor_control, 'o-', color='#2E86AB', linewidth=2.5, 
            markersize=8, label='Anchor Control Accuracy', zorder=3)
    ax.plot(percentile_values, anchor_flip, 's-', color='#A23B72', linewidth=2.5, 
            markersize=8, label='Anchor Flip Accuracy', zorder=3)
    
    # Non-anchor lines
    ax.plot(percentile_values, non_anchor_control, 'o--', color='#06A77D', linewidth=2.5, 
            markersize=8, label='Non-Anchor Control Accuracy', zorder=3)
    ax.plot(percentile_values, non_anchor_flip, 's--', color='#F18F01', linewidth=2.5, 
            markersize=8, label='Non-Anchor Flip Accuracy', zorder=3)
    
    # Formatting
    ax.set_xlabel('Anchor Percentile Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Anchor vs Non-Anchor Accuracy Comparison\nAcross Percentile Thresholds', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis ticks (reverse order: 25% to 5%)
    ax.set_xticks(percentile_values)
    ax.set_xticklabels([f'{p}%' for p in percentile_values])
    ax.invert_xaxis()  # Reverse x-axis so 25% is on left, 5% on right
    
    # Set y-axis to percentage format
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
    
    # Add value labels on points (with smart positioning to avoid overlap)
    offsets = {
        0: {'ac': (0, 12), 'af': (0, -18), 'nac': (0, 12), 'naf': (0, -18)},
        1: {'ac': (0, 12), 'af': (0, -18), 'nac': (0, 12), 'naf': (0, -18)},
        2: {'ac': (0, 12), 'af': (0, -18), 'nac': (0, 12), 'naf': (0, -18)},
        3: {'ac': (0, 12), 'af': (0, -18), 'nac': (0, 12), 'naf': (0, -18)},
        4: {'ac': (0, 12), 'af': (0, -18), 'nac': (0, 12), 'naf': (0, -18)}
    }
    
    for i, (p, ac, af, nac, naf) in enumerate(zip(percentile_values, anchor_control, anchor_flip, 
                                                    non_anchor_control, non_anchor_flip)):
        offset = offsets.get(i, {'ac': (0, 12), 'af': (0, -18), 'nac': (0, 12), 'naf': (0, -18)})
        if ac is not None:
            ax.annotate(f'{ac:.1%}', (p, ac), textcoords="offset points", 
                       xytext=offset['ac'], ha='center', fontsize=9, color='#2E86AB', fontweight='bold')
        if af is not None:
            ax.annotate(f'{af:.1%}', (p, af), textcoords="offset points", 
                       xytext=offset['af'], ha='center', fontsize=9, color='#A23B72', fontweight='bold')
        if nac is not None:
            ax.annotate(f'{nac:.1%}', (p, nac), textcoords="offset points", 
                       xytext=offset['nac'], ha='center', fontsize=9, color='#06A77D', fontweight='bold')
        if naf is not None:
            ax.annotate(f'{naf:.1%}', (p, naf), textcoords="offset points", 
                       xytext=offset['naf'], ha='center', fontsize=9, color='#F18F01', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_file = PROJECT_ROOT / "visualizations/chunk_statistics_plots/anchors/anchor_accuracy_comparison.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plot to {output_file}")
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

