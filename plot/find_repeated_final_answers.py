#!/usr/bin/env python3
"""
Find chunks with repeated "final answer" patterns and extract their information.
"""

import json
import re
from pathlib import Path
from typing import Dict, List
from datasets import load_from_disk
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent

def count_sentences(text: str) -> int:
    """Count sentences in text."""
    if not text:
        return 0
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return len(sentences)

def count_uncertainty_words(text: str) -> int:
    """Count uncertainty words in text."""
    UNCERTAINTY_WORDS = [
        "wait", "alternatively", "perhaps", "reconsider", "double-check", 
        "unlikely", "maybe", "might", "could be", "possibly", "doubt",
        "uncertain", "unsure", "hmm", "actually", "let me think"
    ]
    if not text:
        return 0
    text_lower = text.lower()
    total_count = 0
    for word in UNCERTAINTY_WORDS:
        total_count += text_lower.count(word.lower())
    return total_count

def has_repeated_final_answer(text: str, min_repeats: int = 3) -> bool:
    """
    Check if text has repeated "final answer" patterns.
    Looks for patterns like "final answer" appearing multiple times in sequence.
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Pattern 1: "final answer" repeated multiple times
    pattern1 = r'(final\s+answer\s*){' + str(min_repeats) + r',}'
    if re.search(pattern1, text_lower, re.IGNORECASE):
        return True
    
    # Pattern 2: "**Final Answer**" repeated multiple times
    pattern2 = r'(\*\*Final\s+Answer\*\*\s*){' + str(min_repeats) + r',}'
    if re.search(pattern2, text_lower, re.IGNORECASE):
        return True
    
    # Pattern 3: "Final Answer:" repeated multiple times
    pattern3 = r'(Final\s+Answer\s*:?\s*){' + str(min_repeats) + r',}'
    if re.search(pattern3, text_lower, re.IGNORECASE):
        return True
    
    # Pattern 4: Look for "final answer" appearing at least min_repeats times in close proximity
    # (within 200 characters of each other)
    matches = list(re.finditer(r'final\s+answer', text_lower, re.IGNORECASE))
    if len(matches) >= min_repeats:
        # Check if they're in close proximity
        for i in range(len(matches) - min_repeats + 1):
            start_pos = matches[i].start()
            end_pos = matches[i + min_repeats - 1].end()
            if end_pos - start_pos < 500:  # Within 500 characters
                return True
    
    return False

def load_chunk_function_tags(dataset) -> Dict[int, List[str]]:
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
                        chunk_tags[chunk_idx] = function_tags if function_tags else ['unknown']
                break
            except json.JSONDecodeError:
                continue
    return chunk_tags

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
    """Main function to find chunks with repeated final answers."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_file = str(PROJECT_ROOT / "flip_chunk_results.json")
    ablate_file = str(PROJECT_ROOT / "ablate_uncertainty_results.json")
    flip_types_file = str(PROJECT_ROOT / "flip_types.txt")
    output_file = str(PROJECT_ROOT / "visualizations/analysis/repeated_final_answers.json")
    
    print("="*80)
    print("FINDING CHUNKS WITH REPEATED FINAL ANSWERS")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ Loaded dataset")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    print("✓ Loaded tokenizer")
    
    # Load function tags and flip types
    print("Loading function tags and flip types...")
    chunk_function_tags = load_chunk_function_tags(dataset)
    chunk_flip_types = load_flip_types(flip_types_file)
    print(f"✓ Loaded function tags for {len(chunk_function_tags)} chunks")
    print(f"✓ Loaded flip types for {len(chunk_flip_types)} chunks")
    
    # Load experiment results
    print("\nLoading experiment results...")
    results = []
    
    # Process flip results
    if Path(flip_file).exists():
        with open(flip_file, 'r') as f:
            flip_data = json.load(f)
            for exp in flip_data.get('experiments', []):
                chunk_idx = exp.get('flipped_chunk_idx')
                full_cot = exp.get('full_cot', '')
                
                if chunk_idx is not None and full_cot and has_repeated_final_answer(full_cot):
                    # Calculate statistics
                    total_sentence_count = count_sentences(full_cot)
                    total_token_count = len(tokenizer.encode(full_cot, add_special_tokens=False))
                    total_uncertainty_word_count = count_uncertainty_words(full_cot)
                    
                    # Get function tag
                    tags = chunk_function_tags.get(chunk_idx, ['unknown'])
                    function_tag = tags[0] if tags else 'unknown'
                    
                    # Get flip type
                    flip_type = chunk_flip_types.get(chunk_idx, 'unknown')
                    
                    results.append({
                        'chunk_idx': chunk_idx,
                        'experiment_type': 'flip_chunk',
                        'flip_type': flip_type,
                        'function_tag': function_tag,
                        'total_sentence_count': total_sentence_count,
                        'total_token_count': total_token_count,
                        'total_uncertainty_word_count': total_uncertainty_word_count,
                        'final_answer': exp.get('final_answer', ''),
                        'is_correct': exp.get('is_correct', False),
                        'full_cot': full_cot
                    })
    
    # Process ablate results
    if Path(ablate_file).exists():
        with open(ablate_file, 'r') as f:
            ablate_data = json.load(f)
            for exp in ablate_data.get('experiments', []):
                chunk_idx = exp.get('ablated_chunk_idx')
                full_cot = exp.get('full_cot', '')
                
                if chunk_idx is not None and full_cot and has_repeated_final_answer(full_cot):
                    # Calculate statistics
                    total_sentence_count = count_sentences(full_cot)
                    total_token_count = len(tokenizer.encode(full_cot, add_special_tokens=False))
                    total_uncertainty_word_count = count_uncertainty_words(full_cot)
                    
                    # Get function tag
                    tags = chunk_function_tags.get(chunk_idx, ['unknown'])
                    function_tag = tags[0] if tags else 'unknown'
                    
                    # Get flip type
                    flip_type = chunk_flip_types.get(chunk_idx, 'unknown')
                    
                    results.append({
                        'chunk_idx': chunk_idx,
                        'experiment_type': 'ablate_uncertainty',
                        'flip_type': flip_type,
                        'function_tag': function_tag,
                        'total_sentence_count': total_sentence_count,
                        'total_token_count': total_token_count,
                        'total_uncertainty_word_count': total_uncertainty_word_count,
                        'final_answer': exp.get('final_answer', ''),
                        'is_correct': exp.get('is_correct', False),
                        'full_cot': full_cot
                    })
    
    print(f"\n✓ Found {len(results)} chunks with repeated final answers")
    
    # Sort by chunk_idx
    results.sort(key=lambda x: x['chunk_idx'])
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'total_found': len(results),
        'chunks': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    if results:
        print("\nSummary:")
        print("-" * 80)
        print(f"{'Chunk IDX':<12} {'Type':<20} {'Flip Type':<30} {'Function Tag':<20} {'Correct':<10}")
        print("-" * 80)
        for r in results:
            print(f"{r['chunk_idx']:<12} {r['experiment_type']:<20} {r['flip_type'][:28]:<30} {r['function_tag']:<20} {str(r['is_correct']):<10}")

if __name__ == "__main__":
    main()

