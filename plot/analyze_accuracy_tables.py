#!/usr/bin/env python3
"""
Perform statistical analysis on accuracy tables:
- Chi-square goodness of fit test
- Log likelihood comparison (honest, flipped, ignores probabilities)
"""

import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import List, Tuple, Dict

PROJECT_ROOT = Path(__file__).parent.parent

def parse_accuracy_table(table_file: str) -> List[Tuple[bool, float]]:
    """
    Parse accuracy table and extract (y_i, p_i) pairs.
    
    Returns:
        List of (is_correct, control_correct_pct) tuples
    """
    pairs = []
    
    if not Path(table_file).exists():
        print(f"Warning: Table file not found: {table_file}")
        return pairs
    
    with open(table_file, 'r') as f:
        content = f.read()
    
    # Try different parsing strategies based on table format
    lines = content.split('\n')
    
    # Strategy 1: For tables with "Chunk IDX" header (repeated final answers format)
    if 'Chunk IDX' in content:
        in_data_section = False
        for line in lines:
            if 'Chunk IDX' in line or (line.strip().startswith('-') and in_data_section):
                in_data_section = True
                continue
            
            if 'Overall Control Accuracy' in line or 'Repeated Final Answers Accuracy' in line:
                break
            
            if in_data_section and line.strip() and not line.strip().startswith('-'):
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        last_col = parts[-1]
                        control_pct = float(last_col.rstrip('%'))
                        
                        is_correct = None
                        for i in range(len(parts) - 1, max(0, len(parts) - 5), -1):
                            if parts[i].lower() in ['true', 'false']:
                                is_correct = parts[i].lower() == 'true'
                                break
                        
                        if is_correct is not None:
                            pairs.append((is_correct, control_pct / 100.0))
                    except (ValueError, IndexError):
                        continue
    
    # Strategy 2: For flip/ablate tables with "Chunk Index" header
    elif 'Chunk Index' in content:
        # Check if table has "Control Correct %" column
        has_control_col = 'Control Correct %' in content
        
        # Check if this is an anchor table (has both Flip Correct and Ablate Correct columns)
        is_anchor_table = 'Flip Correct' in content and 'Ablate Correct' in content
        
        in_data_section = False
        for line in lines:
            if 'Chunk Index' in line or (line.strip().startswith('-') and in_data_section):
                in_data_section = True
                continue
            
            if 'Overall Accuracy' in line or 'Control Overall' in line or 'Flip Overall' in line or 'Ablate Overall' in line or 'Flip Accuracy' in line or 'Ablate Accuracy' in line or 'Type Accuracy' in line or 'OVERALL SUMMARY' in line:
                # For flip type tables, continue parsing after "Type Accuracy" lines
                if 'Type Accuracy' in line:
                    continue
                break
            
            if in_data_section and line.strip() and not line.strip().startswith('-'):
                parts = line.split()
                
                if is_anchor_table and has_control_col and len(parts) >= 4:
                    # Format: chunk_idx control_pct flip_correct ablate_correct
                    try:
                        control_pct_str = parts[1].rstrip('%')
                        control_pct = float(control_pct_str)
                        
                        # Use flip_correct if available, otherwise ablate_correct
                        is_correct_str = None
                        for i in range(2, min(4, len(parts))):
                            if parts[i].lower() in ['true', 'false']:
                                is_correct_str = parts[i].lower()
                                break
                        
                        if is_correct_str is not None:
                            is_correct = is_correct_str == 'true'
                            pairs.append((is_correct, control_pct / 100.0))
                    except (ValueError, IndexError):
                        continue
                elif has_control_col and len(parts) >= 3:
                    # Format: chunk_idx control_pct experiment_correct
                    try:
                        control_pct_str = parts[1].rstrip('%')
                        control_pct = float(control_pct_str)
                        
                        is_correct_str = parts[2].lower()
                        if is_correct_str in ['true', 'false']:
                            is_correct = is_correct_str == 'true'
                            pairs.append((is_correct, control_pct / 100.0))
                    except (ValueError, IndexError):
                        continue
                elif not has_control_col and len(parts) >= 2:
                    # Format: chunk_idx experiment_correct (no control column)
                    # Need to load control accuracy from dataset
                    # For now, skip - we'll need to load it differently
                    # Actually, let's try to extract from the table if it exists elsewhere
                    pass
    
    return pairs

def chi_square_goodness_of_fit(y: np.ndarray, p: np.ndarray) -> Dict:
    """
    Perform chi-square goodness of fit test.
    
    Args:
        y: Binary outcomes (0/1)
        p: Predicted probabilities
    
    Returns:
        Dictionary with test results
    """
    n = len(y)
    
    # Create bins (e.g., 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_bins = len(bins) - 1
    
    observed = np.zeros(n_bins)
    expected = np.zeros(n_bins)
    
    for i in range(n):
        # Find which bin p[i] falls into
        bin_idx = 0
        for j in range(n_bins):
            if bins[j] <= p[i] < bins[j+1]:
                bin_idx = j
                break
        if p[i] >= bins[-1]:
            bin_idx = n_bins - 1
        
        observed[bin_idx] += y[i]
        expected[bin_idx] += p[i]
    
    # Also count total observations per bin
    total_per_bin = np.zeros(n_bins)
    for i in range(n):
        bin_idx = 0
        for j in range(n_bins):
            if bins[j] <= p[i] < bins[j+1]:
                bin_idx = j
                break
        if p[i] >= bins[-1]:
            bin_idx = n_bins - 1
        total_per_bin[bin_idx] += 1
    
    # Expected failures per bin
    expected_failures = total_per_bin - expected
    
    # Chi-square statistic
    chi2_stat = 0
    for i in range(n_bins):
        if total_per_bin[i] > 0:
            # Observed successes and failures
            obs_success = observed[i]
            obs_fail = total_per_bin[i] - observed[i]
            exp_success = expected[i]
            exp_fail = expected_failures[i]
            
            if exp_success > 0:
                chi2_stat += (obs_success - exp_success)**2 / exp_success
            if exp_fail > 0:
                chi2_stat += (obs_fail - exp_fail)**2 / exp_fail
    
    # Degrees of freedom: number of bins - 1
    df = n_bins - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    
    return {
        'chi2_statistic': chi2_stat,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'bins': bins,
        'observed': observed.tolist(),
        'expected': expected.tolist(),
        'total_per_bin': total_per_bin.tolist()
    }

def log_likelihood(y: np.ndarray, p: np.ndarray) -> float:
    """
    Calculate log likelihood.
    
    Args:
        y: Binary outcomes (0/1)
        p: Predicted probabilities
    
    Returns:
        Log likelihood
    """
    # Avoid log(0) by clipping probabilities
    p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
    ll = np.sum(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
    return ll

def compare_models(y: np.ndarray, p: np.ndarray) -> Dict:
    """
    Compare three models:
    1. Honest use of probabilities (uses p_i)
    2. Flipped probabilities (uses 1 - p_i)
    3. Ignores probabilities (uses constant mean)
    
    Args:
        y: Binary outcomes (0/1)
        p: Control correctness probabilities
    
    Returns:
        Dictionary with model comparisons
    """
    n = len(y)
    
    # Model 1: Honest use of probabilities
    ll_honest = log_likelihood(y, p)
    
    # Model 2: Flipped probabilities
    p_flipped = 1 - p
    ll_flipped = log_likelihood(y, p_flipped)
    
    # Model 3: Ignores probabilities (constant mean)
    p_constant = np.mean(y)
    p_constant_array = np.full(n, p_constant)
    ll_ignores = log_likelihood(y, p_constant_array)
    
    # Calculate AIC (Akaike Information Criterion)
    # AIC = 2k - 2ln(L), where k is number of parameters
    # For all models, k = 1 (just the probability parameter)
    k = 1
    aic_honest = 2 * k - 2 * ll_honest
    aic_flipped = 2 * k - 2 * ll_flipped
    aic_ignores = 2 * k - 2 * ll_ignores
    
    # Calculate BIC (Bayesian Information Criterion)
    # BIC = k*ln(n) - 2*ln(L)
    bic_honest = k * np.log(n) - 2 * ll_honest
    bic_flipped = k * np.log(n) - 2 * ll_flipped
    bic_ignores = k * np.log(n) - 2 * ll_ignores
    
    return {
        'honest': {
            'log_likelihood': ll_honest,
            'aic': aic_honest,
            'bic': bic_honest
        },
        'flipped': {
            'log_likelihood': ll_flipped,
            'aic': aic_flipped,
            'bic': bic_flipped
        },
        'ignores': {
            'log_likelihood': ll_ignores,
            'aic': aic_ignores,
            'bic': bic_ignores,
            'constant_probability': p_constant
        }
    }

def analyze_table(table_file: str, output_file: str):
    """
    Analyze a single accuracy table.
    """
    print(f"\nAnalyzing: {table_file}")
    
    # Parse table
    pairs = parse_accuracy_table(table_file)
    
    if not pairs:
        print(f"  Warning: No data found in {table_file}")
        return
    
    print(f"  Found {len(pairs)} data points")
    
    # Extract y and p
    y = np.array([1 if correct else 0 for correct, _ in pairs])
    p = np.array([prob for _, prob in pairs])
    
    # Perform chi-square test
    print("  Performing chi-square goodness of fit test...")
    chi2_results = chi_square_goodness_of_fit(y, p)
    
    # Compare models
    print("  Comparing models...")
    model_comparison = compare_models(y, p)
    
    # Write results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"STATISTICAL ANALYSIS: {Path(table_file).name}\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Data Summary:\n")
        f.write(f"  Total observations: {len(pairs)}\n")
        f.write(f"  Correct outcomes: {np.sum(y)}\n")
        f.write(f"  Incorrect outcomes: {len(y) - np.sum(y)}\n")
        f.write(f"  Mean control correctness: {np.mean(p):.4f}\n")
        f.write(f"  Mean outcome: {np.mean(y):.4f}\n")
        f.write("\n")
        
        f.write("="*100 + "\n")
        f.write("CHI-SQUARE GOODNESS OF FIT TEST\n")
        f.write("="*100 + "\n\n")
        f.write(f"Chi-square statistic: {chi2_results['chi2_statistic']:.4f}\n")
        f.write(f"Degrees of freedom: {chi2_results['degrees_of_freedom']}\n")
        f.write(f"P-value: {chi2_results['p_value']:.6f}\n")
        f.write("\n")
        f.write("Bin Analysis:\n")
        f.write(f"{'Bin Range':<20} {'Total Obs':<15} {'Observed Success':<20} {'Expected Success':<20}\n")
        f.write("-"*100 + "\n")
        bins = chi2_results['bins']
        observed = chi2_results['observed']
        expected = chi2_results['expected']
        total_per_bin = chi2_results['total_per_bin']
        for i in range(len(bins) - 1):
            bin_range = f"[{bins[i]:.1f}, {bins[i+1]:.1f})"
            f.write(f"{bin_range:<20} {total_per_bin[i]:<15.0f} {observed[i]:<20.2f} {expected[i]:<20.2f}\n")
        f.write("\n")
        
        f.write("="*100 + "\n")
        f.write("LOG LIKELIHOOD MODEL COMPARISON\n")
        f.write("="*100 + "\n\n")
        
        f.write("Model 1: Honest Use of Probabilities (uses p_i)\n")
        f.write(f"  Log Likelihood: {model_comparison['honest']['log_likelihood']:.4f}\n")
        f.write(f"  AIC: {model_comparison['honest']['aic']:.4f}\n")
        f.write(f"  BIC: {model_comparison['honest']['bic']:.4f}\n")
        f.write("\n")
        
        f.write("Model 2: Flipped Probabilities (uses 1 - p_i)\n")
        f.write(f"  Log Likelihood: {model_comparison['flipped']['log_likelihood']:.4f}\n")
        f.write(f"  AIC: {model_comparison['flipped']['aic']:.4f}\n")
        f.write(f"  BIC: {model_comparison['flipped']['bic']:.4f}\n")
        f.write("\n")
        
        f.write("Model 3: Ignores Probabilities (uses constant mean)\n")
        f.write(f"  Constant Probability: {model_comparison['ignores']['constant_probability']:.4f}\n")
        f.write(f"  Log Likelihood: {model_comparison['ignores']['log_likelihood']:.4f}\n")
        f.write(f"  AIC: {model_comparison['ignores']['aic']:.4f}\n")
        f.write(f"  BIC: {model_comparison['ignores']['bic']:.4f}\n")
        f.write("\n")
        
        f.write("Model Comparison:\n")
        f.write(f"{'Model':<30} {'Log Likelihood':<20} {'AIC':<15} {'BIC':<15}\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Honest (uses p_i)':<30} {model_comparison['honest']['log_likelihood']:<20.4f} {model_comparison['honest']['aic']:<15.4f} {model_comparison['honest']['bic']:<15.4f}\n")
        f.write(f"{'Flipped (uses 1-p_i)':<30} {model_comparison['flipped']['log_likelihood']:<20.4f} {model_comparison['flipped']['aic']:<15.4f} {model_comparison['flipped']['bic']:<15.4f}\n")
        f.write(f"{'Ignores (constant)':<30} {model_comparison['ignores']['log_likelihood']:<20.4f} {model_comparison['ignores']['aic']:<15.4f} {model_comparison['ignores']['bic']:<15.4f}\n")
        f.write("\n")
        
        # Best model (lower AIC/BIC is better)
        best_aic = min(model_comparison['honest']['aic'], 
                      model_comparison['flipped']['aic'],
                      model_comparison['ignores']['aic'])
        best_bic = min(model_comparison['honest']['bic'], 
                      model_comparison['flipped']['bic'],
                      model_comparison['ignores']['bic'])
        
        best_aic_model = 'honest' if model_comparison['honest']['aic'] == best_aic else \
                        ('flipped' if model_comparison['flipped']['aic'] == best_aic else 'ignores')
        best_bic_model = 'honest' if model_comparison['honest']['bic'] == best_bic else \
                        ('flipped' if model_comparison['flipped']['bic'] == best_bic else 'ignores')
        
        f.write("Best Model (by AIC): " + best_aic_model + "\n")
        f.write("Best Model (by BIC): " + best_bic_model + "\n")
        f.write("\n")
        
        f.write("="*100 + "\n")
    
    print(f"  ✓ Saved analysis to {output_file}")

def main():
    """Main function to analyze all accuracy tables."""
    print("="*80)
    print("STATISTICAL ANALYSIS OF ACCURACY TABLES")
    print("="*80)
    
    # Define tables to analyze
    tables = [
        {
            'input': str(PROJECT_ROOT / "visualizations/analysis/flip_accuracy_table.txt"),
            'output': str(PROJECT_ROOT / "visualizations/analysis/flip_accuracy_analysis.txt")
        },
        {
            'input': str(PROJECT_ROOT / "visualizations/analysis/ablate_accuracy_table.txt"),
            'output': str(PROJECT_ROOT / "visualizations/analysis/ablate_accuracy_analysis.txt")
        },
        {
            'input': str(PROJECT_ROOT / "visualizations/analysis/repeated_final_answers_accuracy_table.txt"),
            'output': str(PROJECT_ROOT / "visualizations/analysis/repeated_final_answers_analysis.txt")
        },
        {
            'input': str(PROJECT_ROOT / "visualizations/analysis/anchor_accuracy_table.txt"),
            'output': str(PROJECT_ROOT / "visualizations/analysis/anchor_accuracy_analysis.txt")
        },
        {
            'input': str(PROJECT_ROOT / "visualizations/analysis/flip_type_accuracy_table.txt"),
            'output': str(PROJECT_ROOT / "visualizations/analysis/flip_type_accuracy_analysis.txt")
        }
    ]
    
    # Add percentile anchor tables
    for percentile in ['05percent', '15percent', '20percent', '25percent']:
        anchor_table = PROJECT_ROOT / f"visualizations/analysis/anchors/anchor_accuracy_table_{percentile}.txt"
        if anchor_table.exists():
            tables.append({
                'input': str(anchor_table),
                'output': str(PROJECT_ROOT / f"visualizations/analysis/anchors/anchor_accuracy_analysis_{percentile}.txt")
            })
    
    for table_info in tables:
        analyze_table(table_info['input'], table_info['output'])
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

