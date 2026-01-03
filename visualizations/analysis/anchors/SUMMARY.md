# Multi-Percentile Anchor Analysis Summary

This document summarizes the anchor chunk analysis results for different percentile thresholds (5%, 15%, 20%, 25%) based on log-normalized KL importance scores.

## Results Summary Table

| Percentile | # Anchors | Anchor Flip Acc | Anchor Control Acc | Anchor Ablate Acc | # Non-Anchors | Non-Anchor Flip Acc | Non-Anchor Control Acc | Non-Anchor Ablate Acc |
|------------|-----------|-----------------|-------------------|------------------|---------------|---------------------|----------------------|----------------------|
| **5%** | 8 | 75.00% (6/8) | 78.00% | N/A (0) | 144 | 35.42% (51/144) | 67.80% | 47.06% (8/17) |
| **10%** (Original) | 16 | 81.25% (13/16) | 78.12% | N/A (0) | 136 | 32.35% (44/136) | 67.18% | 47.06% (8/17) |
| **15%** | 23 | 69.57% (16/23) | 76.39% | 0.00% (0/1) | 129 | 31.78% (41/129) | 66.90% | 50.00% (8/16) |
| **20%** | 31 | 70.97% (22/31) | 76.03% | 0.00% (0/1) | 121 | 28.93% (35/121) | 66.36% | 50.00% (8/16) |
| **25%** | 38 | 71.05% (27/38) | 73.79% | 50.00% (1/2) | 114 | 26.32% (30/114) | 66.52% | 46.67% (7/15) |

## Key Observations

1. **Anchor Count**: As expected, the number of anchor chunks increases with the percentile threshold (8 → 16 → 23 → 31 → 38).

2. **Control Accuracy**: The overall control accuracy decreases slightly as more chunks are included (78.00% → 78.12% → 76.39% → 76.03% → 73.79%), with the 10% threshold showing the highest control accuracy.

3. **Flip Accuracy**: Flip experiment accuracy varies across percentiles (75.00% → 81.25% → 69.57% → 70.97% → 71.05%), with the 10% threshold showing the highest flip accuracy at 81.25%.

4. **Ablate Experiments**: Very few ablate experiments exist for anchor chunks (0-2 per percentile), making ablate accuracy statistics less reliable.

5. **Accuracy Gap**: The difference between control and flip accuracy ranges from -3.00% (5%) to -2.74% (25%), indicating that anchor chunks maintain relatively high accuracy even when flipped.

6. **Anchor vs Non-Anchor Comparison**: 
   - **Flip Accuracy**: Anchors show dramatically higher flip accuracy (69.57-81.25%) compared to non-anchors (26.32-35.42%), with a gap of ~40-50 percentage points.
   - **Control Accuracy**: Anchors have higher control accuracy (73.79-78.12%) compared to non-anchors (66.36-67.80%), with a gap of ~6-11 percentage points.
   - **Ablate Accuracy**: Both anchors and non-anchors have limited ablate experiments, but non-anchors show moderate ablate accuracy (46.67-50.00%) while anchors have very few experiments.

## Files Generated

For each percentile (05, 15, 20, 25):

- **Anchor JSON**: `anchors_XXpercent.json` - Contains anchor chunk indices and KL scores
- **Accuracy Table**: `anchor_accuracy_table_XXpercent.txt` - Detailed accuracy comparison table
- **Statistical Analysis**: `anchor_accuracy_analysis_XXpercent.txt` - Chi-square and log-likelihood analysis
- **Statistics Plots**: `visualizations/chunk_statistics_plots/anchors/anchor_statistics_XXpercent/` - Box plots comparing anchor vs non-anchor chunks

## Methodology

Anchors are identified by:
1. Loading `resampling_importance_kl` scores from `chunks_labeled.json`
2. Log-normalizing the KL scores
3. Selecting the top N% of chunks by normalized KL score
4. Generating statistics, accuracy tables, and visualizations for each percentile threshold

