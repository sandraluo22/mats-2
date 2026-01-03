"""
Generate case study JSON by selecting representative chunks for each function tag category
and statistic type (above average, average, below average).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).parent.parent

def load_original_chunks(dataset) -> Dict[int, str]:
    """Load original chunks from incorrect_base_solution."""
    chunks_dict = {}
    chunks_labeled = None
    for ex in dataset:
        path = ex.get('path', '')
        if 'chunks_labeled.json' in path and 'incorrect_base_solution' in path and 'forced_answer' not in path:
            try:
                chunks_labeled = json.loads(ex.get('content', '[]'))
                break
            except json.JSONDecodeError:
                continue
    
    if chunks_labeled:
        for chunk_data in chunks_labeled:
            chunk_idx = chunk_data.get('chunk_idx')
            chunk_text = chunk_data.get('chunk', '')
            if chunk_idx is not None:
                chunks_dict[chunk_idx] = chunk_text
    
    return chunks_dict

def load_chunk_function_tags(dataset) -> Dict[int, List[str]]:
    """Load function tags for each chunk from incorrect_base_solution."""
    chunk_tags = {}
    chunks_labeled = None
    for ex in dataset:
        path = ex.get('path', '')
        if 'chunks_labeled.json' in path and 'incorrect_base_solution' in path and 'forced_answer' not in path:
            try:
                chunks_labeled = json.loads(ex.get('content', '[]'))
                break
            except json.JSONDecodeError:
                continue
    
    if chunks_labeled:
        for chunk_data in chunks_labeled:
            chunk_idx = chunk_data.get('chunk_idx')
            function_tags = chunk_data.get('function_tags', [])
            if chunk_idx is not None:
                chunk_tags[chunk_idx] = function_tags if function_tags else ['unknown']
    
    return chunk_tags

def load_chunk_rollouts(dataset, chunk_idx: int) -> List[Dict]:
    """Load all rollouts for a specific chunk."""
    path_pattern = f'chunk_{chunk_idx}/solutions.json'
    examples = [ex for ex in dataset if path_pattern in ex.get('path', '')]
    
    if not examples:
        return []
    
    content = examples[0].get('content', '')
    if not content:
        return []
    
    try:
        solutions = json.loads(content)
        if isinstance(solutions, list):
            return solutions
        elif isinstance(solutions, dict) and 'solutions' in solutions:
            return solutions['solutions']
        else:
            return []
    except json.JSONDecodeError:
        return []

def count_sentences(text: str) -> int:
    """Count sentences in text (simple heuristic: split by periods)."""
    if not text:
        return 0
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return len(sentences)

def count_uncertainty_words(text: str) -> int:
    """Count occurrences of uncertainty words (case-insensitive)."""
    UNCERTAINTY_WORDS = [
        "wait", "alternatively", "perhaps", "reconsider", "double-check", 
        "unlikely", "maybe", "might", "could be", "possibly", "doubt",
        "uncertain", "unsure", "hmm", "actually", "let me think"
    ]
    text_lower = text.lower()
    total_count = 0
    for word in UNCERTAINTY_WORDS:
        total_count += text_lower.count(word.lower())
    return total_count

def extract_statistics_from_rollouts(rollouts: List[Dict], chunk_idx: int, 
                                     original_chunks_dict: Dict[int, str], 
                                     tokenizer) -> Dict[str, List[float]]:
    """Extract statistics from rollouts, including total counts."""
    stats = {
        'total_sentence_count': [],
        'total_token_count': [],
        'total_uncertainty_word_count': []
    }
    
    # Get all previous chunks text (using actual chunk_idx)
    sorted_original_indices = sorted(original_chunks_dict.keys())
    previous_chunks_list = [original_chunks_dict[idx] for idx in sorted_original_indices if idx < chunk_idx]
    previous_chunks_text = " ".join(previous_chunks_list) if previous_chunks_list else ""
    
    for rollout_data in rollouts:
        rollout_text = rollout_data.get('rollout', '')
        if not rollout_text:
            continue
        
        # Calculate totals (previous chunks + current rollout)
        total_text = previous_chunks_text + " " + rollout_text if previous_chunks_text else rollout_text
        total_sentences = count_sentences(total_text)
        total_uncertainty_words = count_uncertainty_words(total_text)
        
        # Tokenize
        tokens = tokenizer.encode(total_text, add_special_tokens=False)
        total_tokens = len(tokens)
        
        stats['total_sentence_count'].append(total_sentences)
        stats['total_token_count'].append(total_tokens)
        stats['total_uncertainty_word_count'].append(total_uncertainty_words)
    
    return stats

def load_experiment_results(flip_file: str, ablate_file: str, tokenizer) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """Load flip and ablate experiment results and calculate total counts."""
    flip_results = {}
    ablate_results = {}
    
    # Load flip results
    if Path(flip_file).exists():
        with open(flip_file, 'r') as f:
            flip_data = json.load(f)
            experiments = flip_data.get('experiments', [])
            for exp in experiments:
                chunk_idx = exp.get('flipped_chunk_idx')
                if chunk_idx is not None:
                    # Get full_cot text (includes previous chunks + generated continuation)
                    full_cot = exp.get('full_cot', '')
                    
                    # Calculate total counts from full_cot
                    total_sentence_count = count_sentences(full_cot) if full_cot else 0
                    try:
                        total_token_count = len(tokenizer.encode(full_cot, add_special_tokens=False)) if full_cot else 0
                    except Exception:
                        # If tokenization fails (e.g., sequence too long), skip
                        total_token_count = 0
                    total_uncertainty_count = count_uncertainty_words(full_cot) if full_cot else 0
                    
                    flip_results[chunk_idx] = {
                        **exp,  # Keep all original fields
                        'total_sentence_count': total_sentence_count,
                        'total_token_count': total_token_count,
                        'total_uncertainty_word_count': total_uncertainty_count
                    }
    
    # Load ablate results
    if Path(ablate_file).exists():
        with open(ablate_file, 'r') as f:
            ablate_data = json.load(f)
            experiments = ablate_data.get('experiments', [])
            for exp in experiments:
                chunk_idx = exp.get('ablated_chunk_idx')
                if chunk_idx is not None:
                    # Get full_cot text (includes previous chunks + generated continuation)
                    full_cot = exp.get('full_cot', '')
                    
                    # Calculate total counts from full_cot
                    total_sentence_count = count_sentences(full_cot) if full_cot else 0
                    try:
                        total_token_count = len(tokenizer.encode(full_cot, add_special_tokens=False)) if full_cot else 0
                    except Exception:
                        # If tokenization fails (e.g., sequence too long), skip
                        total_token_count = 0
                    total_uncertainty_count = count_uncertainty_words(full_cot) if full_cot else 0
                    
                    ablate_results[chunk_idx] = {
                        **exp,  # Keep all original fields
                        'total_sentence_count': total_sentence_count,
                        'total_token_count': total_token_count,
                        'total_uncertainty_word_count': total_uncertainty_count
                    }
    
    return flip_results, ablate_results

def calculate_experiment_statistics(experiment_data: Dict[int, Dict], chunk_idx: int,
                                   original_chunks_dict: Dict[int, str], tokenizer,
                                   stat_key: str) -> Optional[float]:
    """Calculate a specific statistic for an experiment result."""
    entry = experiment_data.get(chunk_idx)
    if not entry:
        return None
    
    # The full_cot already includes previous chunks, so we can use it directly
    # (as per plot_aggregated_statistics.py logic)
    full_cot = entry.get('full_cot', '')
    if not full_cot:
        return None
    
    # Use full_cot directly (it already includes previous chunks)
    total_text = full_cot
    
    if stat_key == 'total_sentence_count':
        return float(count_sentences(total_text))
    elif stat_key == 'total_token_count':
        try:
            tokens = tokenizer.encode(total_text, add_special_tokens=False)
            return float(len(tokens))
        except Exception:
            # If tokenization fails (e.g., sequence too long), return None
            return None
    elif stat_key == 'total_uncertainty_word_count':
        return float(count_uncertainty_words(total_text))
    
    return None

def find_representative_chunks(
    rollout_data: List[float],
    experiment_data: Dict[int, float],  # chunk_idx -> value
    chunk_function_tags: Dict[int, List[str]],
    tag: str,
    stat_key: str,
    experiment_type: str
) -> Dict[str, Optional[Dict]]:
    """
    Find representative chunks: above average, average, below average.
    
    Returns dict with keys: 'above_average', 'average', 'below_average'
    Each value is either None or a dict with 'chunk_idx' and 'value'.
    """
    result = {
        'above_average': None,
        'average': None,
        'below_average': None
    }
    
    if not rollout_data and not experiment_data:
        return result
    
    # Calculate percentiles for rollout data
    rollout_median = np.median(rollout_data) if rollout_data else None
    rollout_mean = np.mean(rollout_data) if rollout_data else None
    rollout_q75 = np.percentile(rollout_data, 75) if rollout_data else None
    rollout_q25 = np.percentile(rollout_data, 25) if rollout_data else None
    
    # Calculate percentiles for experiment data
    exp_values = list(experiment_data.values())
    exp_median = np.median(exp_values) if exp_values else None
    exp_mean = np.mean(exp_values) if exp_values else None
    exp_q75 = np.percentile(exp_values, 75) if exp_values else None
    exp_q25 = np.percentile(exp_values, 25) if exp_values else None
    
    # Combined reference points
    combined_median = np.median(rollout_data + exp_values) if (rollout_data or exp_values) else None
    combined_mean = np.mean(rollout_data + exp_values) if (rollout_data or exp_values) else None
    rollout_mean = rollout_mean if rollout_mean is not None else combined_mean
    
    # Find chunks that match criteria
    candidates_above = []
    candidates_average = []
    candidates_below = []
    
    for chunk_idx, exp_value in experiment_data.items():
        # Verify this chunk has the correct tag
        tags = chunk_function_tags.get(chunk_idx, ['unknown'])
        chunk_tag = tags[0] if tags else 'unknown'
        if chunk_tag != tag:
            continue
        
        # Above average: value above rollout q75 or above combined median
        # Use more lenient criteria - just need to be in upper third
        if rollout_q75 is not None and exp_value >= rollout_q75:
            candidates_above.append((chunk_idx, exp_value))
        elif combined_median is not None and exp_value >= combined_median * 1.02:
            candidates_above.append((chunk_idx, exp_value))
        elif exp_median is not None and exp_value >= exp_median * 1.1:
            # If no rollout data, use experiment median
            candidates_above.append((chunk_idx, exp_value))
        
        # Average: value near combined median/mean (relaxed tolerance)
        # Use middle third
        if combined_median is not None:
            tolerance = abs(combined_median * 0.3) if combined_median > 0 else 30
            if abs(exp_value - combined_median) <= tolerance:
                candidates_average.append((chunk_idx, exp_value))
        elif rollout_median is not None:
            tolerance = abs(rollout_median * 0.3) if rollout_median > 0 else 30
            if abs(exp_value - rollout_median) <= tolerance:
                candidates_average.append((chunk_idx, exp_value))
        
        # Below average: value below rollout q25 or below combined median
        # Use lower third
        if rollout_q25 is not None and exp_value <= rollout_q25:
            candidates_below.append((chunk_idx, exp_value))
        elif combined_median is not None and exp_value <= combined_median * 0.98:
            candidates_below.append((chunk_idx, exp_value))
        elif exp_median is not None and exp_value <= exp_median * 0.9:
            # If no rollout data, use experiment median
            candidates_below.append((chunk_idx, exp_value))
    
    # Select best candidates (closest to target)
    if candidates_above:
        # Pick one closest to q75 or above median
        target = rollout_q75 if rollout_q75 else (combined_median * 1.15 if combined_median else None)
        if target:
            best = min(candidates_above, key=lambda x: abs(x[1] - target))
            result['above_average'] = {'chunk_idx': best[0], 'value': float(best[1])}
        else:
            # If no target, just pick the highest
            best = max(candidates_above, key=lambda x: x[1])
            result['above_average'] = {'chunk_idx': best[0], 'value': float(best[1])}
    
    if candidates_average:
        # Pick one closest to median
        if combined_median:
            best = min(candidates_average, key=lambda x: abs(x[1] - combined_median))
            result['average'] = {'chunk_idx': best[0], 'value': float(best[1])}
        else:
            # If no median, pick closest to mean
            if rollout_mean:
                best = min(candidates_average, key=lambda x: abs(x[1] - rollout_mean))
                result['average'] = {'chunk_idx': best[0], 'value': float(best[1])}
    
    if candidates_below:
        # Pick one closest to q25 or below median
        target = rollout_q25 if rollout_q25 else (combined_median * 0.85 if combined_median else None)
        if target:
            best = min(candidates_below, key=lambda x: abs(x[1] - target))
            result['below_average'] = {'chunk_idx': best[0], 'value': float(best[1])}
        else:
            # If no target, just pick the lowest
            best = min(candidates_below, key=lambda x: x[1])
            result['below_average'] = {'chunk_idx': best[0], 'value': float(best[1])}
    
    return result

def generate_case_study_json(
    all_rollout_stats_by_chunk: Dict[int, Dict[str, List[float]]],
    all_flip_data: Dict[int, Dict],
    all_ablate_data: Dict[int, Dict],
    chunk_function_tags: Dict[int, List[str]],
    original_chunks_dict: Dict[int, str],
    tokenizer,
    output_file: str
):
    """Generate case study JSON with representative chunks."""
    
    # Group rollout data by function tag
    rollout_data_by_tag = {}
    for chunk_idx, stats_dict in all_rollout_stats_by_chunk.items():
        tags = chunk_function_tags.get(chunk_idx, ['unknown'])
        tag = tags[0] if tags else 'unknown'
        
        if tag not in rollout_data_by_tag:
            rollout_data_by_tag[tag] = {
                'total_sentence_count': [],
                'total_token_count': [],
                'total_uncertainty_word_count': []
            }
        
        for stat_key in rollout_data_by_tag[tag].keys():
            if stat_key in stats_dict:
                rollout_data_by_tag[tag][stat_key].extend(stats_dict[stat_key])
    
    # Get all tags that have data
    all_tags = sorted([tag for tag in rollout_data_by_tag.keys() if any(rollout_data_by_tag[tag].values())])
    print(f"  Found {len(all_tags)} tags with rollout data: {all_tags[:5]}...")
    
    stats = [
        ('total_sentence_count', 'Total Sentence Count'),
        ('total_token_count', 'Total Token Count'),
        ('total_uncertainty_word_count', 'Total Uncertainty Word Count')
    ]
    
    case_study = {
        'flip_chunks': {},
        'ablate_chunks': {},
        'metadata': {
            'description': 'Representative chunks selected for case study analysis',
            'selection_criteria': {
                'above_average': 'Value above the 75th percentile of rollout data or above 110% of combined median',
                'average': 'Value within 15% of the combined median of rollout and experiment data',
                'below_average': 'Value below the 25th percentile of rollout data or below 90% of combined median'
            }
        }
    }
    
    # Process flip chunks
    print("\nProcessing flip chunks...")
    for stat_key, stat_name in stats:
        case_study['flip_chunks'][stat_key] = {}
        
        # Get pre-calculated values from experiment data
        flip_stat_values = {}
        for chunk_idx, flip_entry in all_flip_data.items():
            if stat_key in flip_entry:
                value = flip_entry[stat_key]
                if value is not None and value > 0:  # Only include valid values
                    flip_stat_values[chunk_idx] = float(value)
        
        print(f"  Found {len(flip_stat_values)} flip values for {stat_key}")
        
        for tag in all_tags:
            rollout_data = rollout_data_by_tag[tag].get(stat_key, [])
            if not rollout_data:
                continue
            
            # Get flip values for this tag
            tag_flip_values = {}
            for chunk_idx, value in flip_stat_values.items():
                tags = chunk_function_tags.get(chunk_idx, ['unknown'])
                chunk_tag = tags[0] if tags else 'unknown'
                if chunk_tag == tag:
                    tag_flip_values[chunk_idx] = value
            
            if not tag_flip_values:
                continue
            
            # Debug: print some info
            if len(tag_flip_values) > 0:
                print(f"  {stat_key} - {tag}: {len(rollout_data)} rollout points, {len(tag_flip_values)} experiment points")
            
            # Find representative chunks
            representatives = find_representative_chunks(
                rollout_data, tag_flip_values, chunk_function_tags, tag, stat_key, 'flip'
            )
            
            # Only add if we found at least one representative
            if any(v is not None for v in representatives.values()):
                # Add full_cot for each representative
                representatives_with_cot = {}
                for rep_type, rep_data in representatives.items():
                    if rep_data is not None:
                        chunk_idx = rep_data['chunk_idx']
                        # Get full_cot from experiment data
                        flip_entry = all_flip_data.get(chunk_idx, {})
                        full_cot = flip_entry.get('full_cot', '')
                        representatives_with_cot[rep_type] = {
                            **rep_data,
                            'full_cot': full_cot
                        }
                
                # Create debrief
                debrief = create_debrief(rollout_data, tag_flip_values, representatives, stat_key, tag, 'flip')
                
                case_study['flip_chunks'][stat_key][tag] = {
                    'representatives': representatives_with_cot,
                    'debrief': debrief,
                    'statistic_name': stat_name,
                    'function_tag': tag
                }
    
    # Process ablate chunks
    print("Processing ablate chunks...")
    for stat_key, stat_name in stats:
        case_study['ablate_chunks'][stat_key] = {}
        
        # Get pre-calculated values from experiment data
        ablate_stat_values = {}
        for chunk_idx, ablate_entry in all_ablate_data.items():
            if stat_key in ablate_entry:
                value = ablate_entry[stat_key]
                if value is not None and value > 0:  # Only include valid values
                    ablate_stat_values[chunk_idx] = float(value)
        
        for tag in all_tags:
            rollout_data = rollout_data_by_tag[tag].get(stat_key, [])
            if not rollout_data:
                continue
            
            # Get ablate values for this tag
            tag_ablate_values = {}
            for chunk_idx, value in ablate_stat_values.items():
                tags = chunk_function_tags.get(chunk_idx, ['unknown'])
                chunk_tag = tags[0] if tags else 'unknown'
                if chunk_tag == tag:
                    tag_ablate_values[chunk_idx] = value
            
            if not tag_ablate_values:
                continue
            
            # Find representative chunks
            representatives = find_representative_chunks(
                rollout_data, tag_ablate_values, chunk_function_tags, tag, stat_key, 'ablate'
            )
            
            # Only add if we found at least one representative
            if any(v is not None for v in representatives.values()):
                # Add full_cot for each representative
                representatives_with_cot = {}
                for rep_type, rep_data in representatives.items():
                    if rep_data is not None:
                        chunk_idx = rep_data['chunk_idx']
                        # Get full_cot from experiment data
                        ablate_entry = all_ablate_data.get(chunk_idx, {})
                        full_cot = ablate_entry.get('full_cot', '')
                        representatives_with_cot[rep_type] = {
                            **rep_data,
                            'full_cot': full_cot
                        }
                
                # Create debrief
                debrief = create_debrief(rollout_data, tag_ablate_values, representatives, stat_key, tag, 'ablate')
                
                case_study['ablate_chunks'][stat_key][tag] = {
                    'representatives': representatives_with_cot,
                    'debrief': debrief,
                    'statistic_name': stat_name,
                    'function_tag': tag
                }
    
    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(case_study, f, indent=2)
    
    print(f"\n✓ Saved case study to {output_path}")
    return case_study

def create_debrief(rollout_data: List[float], experiment_data: Dict[int, float],
                 representatives: Dict[str, Optional[Dict]], stat_key: str, tag: str,
                 experiment_type: str) -> str:
    """Create a debrief explaining how chunks were selected."""
    rollout_median = np.median(rollout_data) if rollout_data else None
    rollout_mean = np.mean(rollout_data) if rollout_data else None
    rollout_q75 = np.percentile(rollout_data, 75) if rollout_data else None
    rollout_q25 = np.percentile(rollout_data, 25) if rollout_data else None
    
    exp_values = list(experiment_data.values())
    exp_median = np.median(exp_values) if exp_values else None
    exp_mean = np.mean(exp_values) if exp_values else None
    
    debrief_parts = []
    debrief_parts.append(f"Selection for {tag} category, {stat_key} statistic ({experiment_type} experiments):")
    debrief_parts.append(f"  - Rollout data: median={rollout_median:.1f}, mean={rollout_mean:.1f}, Q75={rollout_q75:.1f}, Q25={rollout_q25:.1f}")
    debrief_parts.append(f"  - Experiment data: median={exp_median:.1f}, mean={exp_mean:.1f}, n={len(exp_values)}")
    
    if representatives['above_average']:
        chunk_idx = representatives['above_average']['chunk_idx']
        value = representatives['above_average']['value']
        debrief_parts.append(f"  - Above average: Chunk {chunk_idx} (value={value:.1f}, above Q75={rollout_q75:.1f})")
    
    if representatives['average']:
        chunk_idx = representatives['average']['chunk_idx']
        value = representatives['average']['value']
        debrief_parts.append(f"  - Average: Chunk {chunk_idx} (value={value:.1f}, near median={rollout_median:.1f})")
    
    if representatives['below_average']:
        chunk_idx = representatives['below_average']['chunk_idx']
        value = representatives['below_average']['value']
        debrief_parts.append(f"  - Below average: Chunk {chunk_idx} (value={value:.1f}, below Q25={rollout_q25:.1f})")
    
    return "\n".join(debrief_parts)

def main():
    """Main function to generate case study JSON."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    output_file = str(PROJECT_ROOT / "visualizations/analysis/case_study.json")
    
    print("="*80)
    print("GENERATING CASE STUDY JSON")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ Loaded dataset with {len(dataset)} entries")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    print("✓ Loaded tokenizer")
    
    # Load function tags
    print("Loading function tags...")
    chunk_function_tags = load_chunk_function_tags(dataset)
    print(f"✓ Loaded function tags for {len(chunk_function_tags)} chunks")
    
    # Load original chunks
    print("Loading original chunks...")
    original_chunks_dict = load_original_chunks(dataset)
    print(f"✓ Loaded {len(original_chunks_dict)} original chunks")
    
    # Load experiment results
    print("Loading experiment results...")
    flip_results, ablate_results = load_experiment_results(flip_file, ablate_file, tokenizer)
    print(f"✓ Loaded {len(flip_results)} flip results and {len(ablate_results)} ablate results")
    
    # Find all chunk indices
    import re
    chunk_indices = set()
    for ex in dataset:
        path = ex.get('path', '')
        match = re.search(r'chunk_(\d+)/solutions\.json', path)
        if match:
            chunk_indices.add(int(match.group(1)))
    
    chunk_indices = sorted(chunk_indices)
    print(f"✓ Found {len(chunk_indices)} chunks (indices: {min(chunk_indices)} to {max(chunk_indices)})")
    
    # Aggregate statistics across all chunks
    print("\nAggregating statistics...")
    all_rollout_stats_by_chunk = {}
    
    chunks_processed = 0
    for chunk_idx in chunk_indices:
        rollouts = load_chunk_rollouts(dataset, chunk_idx)
        if not rollouts:
            continue
        
        chunks_processed += 1
        if chunks_processed % 10 == 0:
            print(f"  Processed {chunks_processed}/{len(chunk_indices)} chunks...", end='\r')
        
        rollout_stats = extract_statistics_from_rollouts(rollouts, chunk_idx, original_chunks_dict, tokenizer)
        all_rollout_stats_by_chunk[chunk_idx] = rollout_stats
    
    print(f"\n✓ Processed {chunks_processed} chunks")
    
    # Store experiment data (already has statistics calculated in load_experiment_results)
    print("\nPreparing experiment data...")
    all_flip_data = flip_results  # Already contains full_cot and statistics
    all_ablate_data = ablate_results  # Already contains full_cot and statistics
    
    print(f"✓ Calculated statistics for {len(all_flip_data)} flip chunks and {len(all_ablate_data)} ablate chunks")
    
    # Generate case study JSON
    print("\nGenerating case study JSON...")
    case_study = generate_case_study_json(
        all_rollout_stats_by_chunk,
        all_flip_data,
        all_ablate_data,
        chunk_function_tags,
        original_chunks_dict,
        tokenizer,
        output_file
    )
    
    print("\n" + "="*80)
    print("CASE STUDY GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    main()

