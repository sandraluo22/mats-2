#!/usr/bin/env python3
"""
Calculate statistics for each chunk in problem 1591 based on 100 normal rollouts.

For each chunk, calculates:
- Average sentence count
- Average token count
- Average uncertainty word count
- Average uncertainty occurrences
- Average accuracy in final answer
- Standard deviations for all of the above

Statistics are based on the 100 generated rollouts from each chunk (normal, not ablated).
"""

import json
import re
import statistics
from datasets import load_from_disk
from transformers import AutoTokenizer
from typing import Dict, List, Tuple

# Uncertainty indicators (same as in ablate_uncertainty_and_evaluate.py)
UNCERTAINTY_WORDS = [
    "wait", "alternatively", "perhaps", "reconsider", "double-check", 
    "unlikely", "maybe", "might", "could be", "possibly", "doubt",
    "uncertain", "unsure", "hmm", "actually", "let me think"
]

def count_sentences(text: str) -> int:
    """Count sentences in text (simple heuristic: split by periods)."""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return len(sentences)

def count_uncertainty_words(text: str) -> Tuple[int, List[str]]:
    """
    Count uncertainty words in text (case insensitive).
    
    Returns:
        Tuple of (total_count, list of occurrences as "word:count")
    """
    text_lower = text.lower()
    total_count = 0
    occurrences = []
    
    for word in UNCERTAINTY_WORDS:
        count = text_lower.count(word.lower())
        if count > 0:
            total_count += count
            occurrences.append(f"{word}:{count}")
    
    return total_count, occurrences

def extract_chunk_number(path: str) -> int:
    """Extract chunk number from path like 'chunk_0/solutions.json'."""
    match = re.search(r'chunk_(\d+)/solutions\.json', path)
    if match:
        return int(match.group(1))
    return -1

def load_chunk_solutions(dataset, chunk_idx: int) -> List[Dict]:
    """
    Load solutions for a specific chunk from the dataset.
    
    Args:
        dataset: The loaded dataset
        chunk_idx: The chunk index (0-based)
    
    Returns:
        List of solution dictionaries
    """
    path_pattern = f'chunk_{chunk_idx}/solutions.json'
    examples = [ex for ex in dataset if path_pattern in ex.get('path', '')]
    
    if not examples:
        return []
    
    # Load the JSON content
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
        print(f"Warning: Could not parse JSON for chunk {chunk_idx}")
        return []

def load_original_chunks(dataset) -> Dict[int, str]:
    """
    Load original chunks from chunks_labeled.json, using actual chunk_idx from data.
    
    Returns:
        Dictionary mapping chunk_idx to chunk text
    """
    chunks_dict = {}
    
    # Find chunks_labeled.json
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
        # Use actual chunk_idx from data (don't assume sequential)
        for chunk_data in chunks_labeled:
            chunk_idx = chunk_data.get('chunk_idx')
            if chunk_idx is not None:
                chunk_text = chunk_data.get('chunk', '')
                if chunk_text:
                    chunks_dict[chunk_idx] = chunk_text
    
    return chunks_dict

def calculate_statistics_for_chunk(
    solutions: List[Dict],
    tokenizer: AutoTokenizer,
    ground_truth_answer: str,
    chunk_idx: int,
    original_chunks_dict: Dict[int, str]
) -> Dict:
    """
    Calculate statistics for a chunk based on its solutions, including total counts.
    
    Args:
        solutions: List of solution dictionaries from the dataset
        tokenizer: Tokenizer for counting tokens
        ground_truth_answer: Ground truth answer for accuracy calculation
        chunk_idx: Current chunk index
        original_chunks: List of all original chunk texts
    
    Returns:
        Dictionary with statistics (both rollout-only and total)
    """
    if not solutions:
        return None
    
    sentence_counts = []
    token_counts = []
    uncertainty_word_counts = []
    uncertainty_occurrence_counts = []
    accuracies = []
    
    # Total counts (include all previous chunks + current rollout)
    total_sentence_counts = []
    total_token_counts = []
    total_uncertainty_word_counts = []
    total_uncertainty_occurrence_counts = []
    
    # Get text from all chunks before current chunk (using actual chunk_idx)
    sorted_original_indices = sorted(original_chunks_dict.keys())
    previous_chunks_list = [original_chunks_dict[idx] for idx in sorted_original_indices if idx < chunk_idx]
    previous_chunks_text = " ".join(previous_chunks_list) if previous_chunks_list else ""
    
    for solution in solutions:
        # Get the rollout text (the generated continuation)
        rollout_text = solution.get('rollout', '')
        if not rollout_text:
            continue
        
        # Count sentences in rollout only
        sentence_count = count_sentences(rollout_text)
        sentence_counts.append(sentence_count)
        
        # Count tokens in rollout only
        token_count = len(tokenizer.encode(rollout_text))
        token_counts.append(token_count)
        
        # Count uncertainty words in rollout only
        uncertainty_count, occurrences = count_uncertainty_words(rollout_text)
        uncertainty_word_counts.append(uncertainty_count)
        uncertainty_occurrence_counts.append(len(occurrences))  # Number of unique uncertainty word types
        
        # Calculate total counts (previous chunks + current rollout)
        full_cot_text = previous_chunks_text + " " + rollout_text if previous_chunks_text else rollout_text
        total_sentence_count = count_sentences(full_cot_text)
        total_token_count = len(tokenizer.encode(full_cot_text))
        total_uncertainty_count, total_occurrences_list = count_uncertainty_words(full_cot_text)
        
        total_sentence_counts.append(total_sentence_count)
        total_token_counts.append(total_token_count)
        total_uncertainty_word_counts.append(total_uncertainty_count)
        total_uncertainty_occurrence_counts.append(len(total_occurrences_list))
        
        # Check accuracy
        answer = solution.get('answer', '')
        is_correct = solution.get('is_correct', False)
        
        # If is_correct is not provided, try to check manually
        if is_correct is None or (not isinstance(is_correct, bool) and not is_correct):
            # Try to check correctness manually
            is_correct = check_correctness(answer, ground_truth_answer)
        
        accuracies.append(1.0 if is_correct else 0.0)
    
    if not sentence_counts:
        return None
    
    # Calculate averages for rollout-only stats
    avg_sentence_count = statistics.mean(sentence_counts)
    avg_token_count = statistics.mean(token_counts)
    avg_uncertainty_word_count = statistics.mean(uncertainty_word_counts)
    avg_uncertainty_occurrences = statistics.mean(uncertainty_occurrence_counts)
    avg_accuracy = statistics.mean(accuracies)
    
    # Calculate standard deviations for rollout-only stats
    std_sentence_count = statistics.stdev(sentence_counts) if len(sentence_counts) > 1 else 0.0
    std_token_count = statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0
    std_uncertainty_word_count = statistics.stdev(uncertainty_word_counts) if len(uncertainty_word_counts) > 1 else 0.0
    std_uncertainty_occurrences = statistics.stdev(uncertainty_occurrence_counts) if len(uncertainty_occurrence_counts) > 1 else 0.0
    std_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    
    # Calculate averages for total stats
    avg_total_sentence_count = statistics.mean(total_sentence_counts)
    avg_total_token_count = statistics.mean(total_token_counts)
    avg_total_uncertainty_word_count = statistics.mean(total_uncertainty_word_counts)
    avg_total_uncertainty_occurrences = statistics.mean(total_uncertainty_occurrence_counts)
    
    # Calculate standard deviations for total stats
    std_total_sentence_count = statistics.stdev(total_sentence_counts) if len(total_sentence_counts) > 1 else 0.0
    std_total_token_count = statistics.stdev(total_token_counts) if len(total_token_counts) > 1 else 0.0
    std_total_uncertainty_word_count = statistics.stdev(total_uncertainty_word_counts) if len(total_uncertainty_word_counts) > 1 else 0.0
    std_total_uncertainty_occurrences = statistics.stdev(total_uncertainty_occurrence_counts) if len(total_uncertainty_occurrence_counts) > 1 else 0.0
    
    return {
        'chunk_idx': None,  # Will be set by caller
        'num_rollouts': len(sentence_counts),
        'avg_sentence_count': avg_sentence_count,
        'std_sentence_count': std_sentence_count,
        'avg_token_count': avg_token_count,
        'std_token_count': std_token_count,
        'avg_uncertainty_word_count': avg_uncertainty_word_count,
        'std_uncertainty_word_count': std_uncertainty_word_count,
        'avg_uncertainty_occurrences': avg_uncertainty_occurrences,
        'std_uncertainty_occurrences': std_uncertainty_occurrences,
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'avg_total_sentence_count': avg_total_sentence_count,
        'std_total_sentence_count': std_total_sentence_count,
        'avg_total_token_count': avg_total_token_count,
        'std_total_token_count': std_total_token_count,
        'avg_total_uncertainty_word_count': avg_total_uncertainty_word_count,
        'std_total_uncertainty_word_count': std_total_uncertainty_word_count,
        'avg_total_uncertainty_occurrences': avg_total_uncertainty_occurrences,
        'std_total_uncertainty_occurrences': std_total_uncertainty_occurrences
    }

def check_correctness(final_answer: str, ground_truth_answer: str) -> bool:
    """Check if the final answer matches ground truth (loose matching)."""
    # Normalize: remove whitespace, convert to lowercase
    final_norm = re.sub(r'\s+', '', str(final_answer).lower())
    gt_norm = re.sub(r'\s+', '', str(ground_truth_answer).lower())
    
    # Try exact match
    if final_norm == gt_norm:
        return True
    
    # Try to extract numbers and compare
    final_numbers = re.findall(r'\d+\.?\d*', final_norm)
    gt_numbers = re.findall(r'\d+\.?\d*', gt_norm)
    
    if final_numbers and gt_numbers:
        return final_numbers[-1] == gt_numbers[-1]
    
    return False

def get_ground_truth_answer(dataset) -> str:
    """Get ground truth answer from problem.json."""
    problem_files = [ex for ex in dataset if 'problem.json' in ex.get('path', '')]
    
    if not problem_files:
        return None
    
    # Prefer non-forced version
    problem_file = None
    for ex in problem_files:
        if 'forced_answer' not in ex.get('path', '').lower():
            problem_file = ex
            break
    
    if not problem_file:
        problem_file = problem_files[0]
    
    try:
        problem_data = json.loads(problem_file.get('content', '{}'))
        return problem_data.get('gt_answer', '')
    except json.JSONDecodeError:
        return None

def get_all_chunk_indices(dataset) -> List[int]:
    """Get all chunk indices from the dataset."""
    chunk_indices = set()
    
    for ex in dataset:
        path = ex.get('path', '')
        chunk_idx = extract_chunk_number(path)
        if chunk_idx >= 0:
            chunk_indices.add(chunk_idx)
    
    return sorted(chunk_indices)

def write_results_to_json(results: Dict, output_file: str) -> bool:
    """
    Write results dictionary to JSON file with proper formatting and error handling.
    
    Args:
        results: Dictionary containing all statistics and metadata
        output_file: Path to output JSON file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Verify file was written successfully
        import os
        if not os.path.exists(output_file):
            print(f"✗ Error: File {output_file} was not created")
            return False
        
        file_size = os.path.getsize(output_file)
        if file_size == 0:
            print(f"✗ Error: File {output_file} is empty")
            return False
        
        return True
        
    except json.JSONEncodeError as e:
        print(f"✗ JSON encoding error: {e}")
        return False
    except IOError as e:
        print(f"✗ File I/O error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error writing JSON: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to calculate statistics for all chunks."""
    import sys
    
    problem_id = "problem_1591"
    dataset_path = f"{problem_id}_dataset"
    output_file = "chunk_statistics.json"
    
    print("="*80)
    print("CHUNK STATISTICS CALCULATOR")
    print("="*80)
    print(f"\nLoading dataset from: {dataset_path}")
    
    try:
        dataset = load_from_disk(dataset_path)
        print(f"✓ Loaded dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Get ground truth answer
    print("\nExtracting ground truth answer...")
    ground_truth = get_ground_truth_answer(dataset)
    if not ground_truth:
        print("Warning: Could not find ground truth answer, accuracy will be based on is_correct field only")
    else:
        print(f"✓ Ground truth answer: {ground_truth}")
    
    # Get all chunk indices
    print("\nFinding all chunks...")
    chunk_indices = get_all_chunk_indices(dataset)
    print(f"✓ Found {len(chunk_indices)} chunks (indices: {min(chunk_indices)} to {max(chunk_indices)})")
    
    # Load original chunks for calculating totals
    print("\nLoading original chunks...")
    original_chunks_dict = load_original_chunks(dataset)
    print(f"✓ Loaded {len(original_chunks_dict)} original chunks (indices: {min(original_chunks_dict.keys())} to {max(original_chunks_dict.keys())})")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Calculate statistics for each chunk
    print("\n" + "="*80)
    print("CALCULATING STATISTICS")
    print("="*80)
    
    all_statistics = []
    
    for chunk_idx in chunk_indices:
        print(f"\nProcessing chunk {chunk_idx}...")
        
        # Load solutions for this chunk
        solutions = load_chunk_solutions(dataset, chunk_idx)
        
        if not solutions:
            print(f"  Warning: No solutions found for chunk {chunk_idx}")
            continue
        
        print(f"  Found {len(solutions)} rollouts")
        
        # Calculate statistics (including totals)
        stats = calculate_statistics_for_chunk(solutions, tokenizer, ground_truth or "", chunk_idx, original_chunks_dict)
        
        if stats:
            stats['chunk_idx'] = chunk_idx
            all_statistics.append(stats)
            
            print(f"  ✓ Calculated statistics:")
            print(f"    - Avg sentences: {stats['avg_sentence_count']:.2f} ± {stats['std_sentence_count']:.2f}")
            print(f"    - Avg total sentences: {stats['avg_total_sentence_count']:.2f} ± {stats['std_total_sentence_count']:.2f}")
            print(f"    - Avg tokens: {stats['avg_token_count']:.2f} ± {stats['std_token_count']:.2f}")
            print(f"    - Avg total tokens: {stats['avg_total_token_count']:.2f} ± {stats['std_total_token_count']:.2f}")
            print(f"    - Avg uncertainty words: {stats['avg_uncertainty_word_count']:.2f} ± {stats['std_uncertainty_word_count']:.2f}")
            print(f"    - Avg total uncertainty words: {stats['avg_total_uncertainty_word_count']:.2f} ± {stats['std_total_uncertainty_word_count']:.2f}")
            print(f"    - Avg uncertainty occurrences: {stats['avg_uncertainty_occurrences']:.2f} ± {stats['std_uncertainty_occurrences']:.2f}")
            print(f"    - Avg total uncertainty occurrences: {stats['avg_total_uncertainty_occurrences']:.2f} ± {stats['std_total_uncertainty_occurrences']:.2f}")
            print(f"    - Avg accuracy: {stats['avg_accuracy']:.3f} ± {stats['std_accuracy']:.3f}")
        else:
            print(f"  Warning: Could not calculate statistics for chunk {chunk_idx}")
    
    # Prepare JSON results structure
    print("\n" + "="*80)
    print("PREPARING JSON RESULTS")
    print("="*80)
    
    results = {
        'problem_id': problem_id,
        'ground_truth_answer': ground_truth,
        'num_chunks': len(all_statistics),
        'total_rollouts_analyzed': sum(s['num_rollouts'] for s in all_statistics),
        'statistics': sorted(all_statistics, key=lambda x: x['chunk_idx'])
    }
    
    # Add metadata
    results['metadata'] = {
        'uncertainty_words': UNCERTAINTY_WORDS,
        'tokenizer_model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'sentence_counting_method': 'split by periods',
        'accuracy_calculation': 'based on is_correct field and ground truth comparison'
    }
    
    # Save results to JSON file
    print("\n" + "="*80)
    print("SAVING RESULTS TO JSON")
    print("="*80)
    
    if write_results_to_json(results, output_file):
        # Verify file was written
        import os
        file_size = os.path.getsize(output_file)
        print(f"\n✓ Results saved to: {output_file}")
        print(f"  File size: {file_size / 1024:.2f} KB")
        print(f"  JSON structure:")
        print(f"    - problem_id: {results['problem_id']}")
        print(f"    - ground_truth_answer: {results['ground_truth_answer']}")
        print(f"    - num_chunks: {results['num_chunks']}")
        print(f"    - total_rollouts_analyzed: {results['total_rollouts_analyzed']}")
        print(f"    - statistics: array of {len(results['statistics'])} chunk statistics")
        print(f"    - metadata: additional information about the calculation")
    else:
        print(f"\n✗ Failed to save results to {output_file}")
        return
    
    print(f"\nSummary:")
    print(f"  - Processed {len(all_statistics)} chunks")
    print(f"  - Total rollouts analyzed: {sum(s['num_rollouts'] for s in all_statistics)}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Chunk':<8} {'Rollouts':<10} {'Avg Sent':<12} {'Avg Tokens':<15} {'Avg Unc Words':<18} {'Avg Unc Occ':<15} {'Avg Acc':<10}")
    print("-" * 80)
    for stats in sorted(all_statistics, key=lambda x: x['chunk_idx']):
        print(f"{stats['chunk_idx']:<8} {stats['num_rollouts']:<10} "
              f"{stats['avg_sentence_count']:.2f}±{stats['std_sentence_count']:.2f}  "
              f"{stats['avg_token_count']:.2f}±{stats['std_token_count']:.2f}  "
              f"{stats['avg_uncertainty_word_count']:.2f}±{stats['std_uncertainty_word_count']:.2f}  "
              f"{stats['avg_uncertainty_occurrences']:.2f}±{stats['std_uncertainty_occurrences']:.2f}  "
              f"{stats['avg_accuracy']:.3f}±{stats['std_accuracy']:.3f}")

if __name__ == "__main__":
    main()

