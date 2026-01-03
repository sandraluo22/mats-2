#!/usr/bin/env python3
"""
Update case study JSONs with additional statistics:
- Sentence length
- Token length
- Uncertainty count
- Function tag
- Flip type (for flip chunks only)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_from_disk
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent

UNCERTAINTY_WORDS = [
    "wait", "alternatively", "perhaps", "reconsider", "double-check", 
    "unlikely", "maybe", "might", "could be", "possibly", "doubt",
    "uncertain", "unsure", "hmm", "actually", "let me think"
]

def count_sentences(text: str) -> int:
    """Count sentences in text (simple heuristic: split by periods)."""
    if not text:
        return 0
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return len(sentences)

def count_uncertainty_words(text: str) -> int:
    """Count occurrences of uncertainty words (case-insensitive)."""
    if not text:
        return 0
    text_lower = text.lower()
    total_count = 0
    for word in UNCERTAINTY_WORDS:
        total_count += text_lower.count(word.lower())
    return total_count

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

def update_entry_statistics(entry: Dict, chunk_idx: int, tokenizer, 
                           chunk_function_tags: Dict[int, List[str]],
                           flip_types: Dict[int, str], experiment_type: str) -> Dict:
    """Update a single entry with statistics."""
    full_cot = entry.get('full_cot', '')
    
    # Calculate statistics
    sentence_count = count_sentences(full_cot)
    uncertainty_count = count_uncertainty_words(full_cot)
    
    # Token count
    try:
        tokens = tokenizer.encode(full_cot, add_special_tokens=False)
        token_count = len(tokens)
    except Exception:
        token_count = 0
    
    # Get function tag
    tags = chunk_function_tags.get(chunk_idx, ['unknown'])
    function_tag = tags[0] if tags else 'unknown'
    
    # Get flip type (only for flip chunks)
    flip_type = flip_types.get(chunk_idx) if experiment_type == 'flip' else None
    
    # Update entry
    updated_entry = entry.copy()
    updated_entry['sentence_count'] = sentence_count
    updated_entry['token_count'] = token_count
    updated_entry['uncertainty_count'] = uncertainty_count
    updated_entry['function_tag'] = function_tag
    if flip_type:
        updated_entry['flip_type'] = flip_type
    
    return updated_entry

def update_case_study_json(case_study_file: str, tokenizer, 
                          chunk_function_tags: Dict[int, List[str]],
                          flip_types: Dict[int, str]):
    """Update case_study.json with statistics."""
    print(f"\nUpdating {case_study_file}...")
    
    with open(case_study_file, 'r') as f:
        data = json.load(f)
    
    updated_count = 0
    
    # Update flip chunks
    if 'flip_chunks' in data:
        for stat_name, stat_data in data['flip_chunks'].items():
            # stat_data is a dict with function tags as keys
            for function_tag, tag_data in stat_data.items():
                if isinstance(tag_data, dict) and 'representatives' in tag_data:
                    for rep_type in ['above_average', 'average', 'below_average']:
                        if rep_type in tag_data['representatives']:
                            entry = tag_data['representatives'][rep_type]
                            if entry and isinstance(entry, dict) and 'chunk_idx' in entry:
                                chunk_idx = entry['chunk_idx']
                                updated_entry = update_entry_statistics(
                                    entry, chunk_idx, tokenizer, 
                                    chunk_function_tags, flip_types, 'flip'
                                )
                                tag_data['representatives'][rep_type] = updated_entry
                                updated_count += 1
                                if updated_count % 10 == 0:
                                    print(f"  Updated {updated_count} entries...")
    
    # Update ablate chunks
    if 'ablate_chunks' in data:
        for stat_name, stat_data in data['ablate_chunks'].items():
            # stat_data is a dict with function tags as keys
            for function_tag, tag_data in stat_data.items():
                if isinstance(tag_data, dict) and 'representatives' in tag_data:
                    for rep_type in ['above_average', 'average', 'below_average']:
                        if rep_type in tag_data['representatives']:
                            entry = tag_data['representatives'][rep_type]
                            if entry and isinstance(entry, dict) and 'chunk_idx' in entry:
                                chunk_idx = entry['chunk_idx']
                                updated_entry = update_entry_statistics(
                                    entry, chunk_idx, tokenizer, 
                                    chunk_function_tags, flip_types, 'ablate'
                                )
                                tag_data['representatives'][rep_type] = updated_entry
                                updated_count += 1
                                if updated_count % 10 == 0:
                                    print(f"  Updated {updated_count} entries...")
    
    # Save updated file
    with open(case_study_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Updated {case_study_file} ({updated_count} entries)")

def update_anchor_case_study_json(anchor_case_study_file: str, tokenizer,
                                  chunk_function_tags: Dict[int, List[str]],
                                  flip_types: Dict[int, str]):
    """Update anchor_case_study.json with statistics."""
    print(f"\nUpdating {anchor_case_study_file}...")
    
    with open(anchor_case_study_file, 'r') as f:
        data = json.load(f)
    
    updated_count = 0
    
    # Update flip chunks
    if 'anchors_with_flip_experiments' in data:
        for entry in data['anchors_with_flip_experiments']:
            if 'chunk_idx' in entry:
                chunk_idx = entry['chunk_idx']
                updated_entry = update_entry_statistics(
                    entry, chunk_idx, tokenizer,
                    chunk_function_tags, flip_types, 'flip'
                )
                # Replace the entry
                for i, e in enumerate(data['anchors_with_flip_experiments']):
                    if e.get('chunk_idx') == chunk_idx:
                        data['anchors_with_flip_experiments'][i] = updated_entry
                        updated_count += 1
                        break
    
    # Update ablate chunks
    if 'anchors_with_ablate_experiments' in data:
        for entry in data['anchors_with_ablate_experiments']:
            if 'chunk_idx' in entry:
                chunk_idx = entry['chunk_idx']
                updated_entry = update_entry_statistics(
                    entry, chunk_idx, tokenizer,
                    chunk_function_tags, flip_types, 'ablate'
                )
                # Replace the entry
                for i, e in enumerate(data['anchors_with_ablate_experiments']):
                    if e.get('chunk_idx') == chunk_idx:
                        data['anchors_with_ablate_experiments'][i] = updated_entry
                        updated_count += 1
                        break
    
    # Save updated file
    with open(anchor_case_study_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Updated {anchor_case_study_file} ({updated_count} entries)")

def main():
    """Main function to update case study JSONs."""
    problem_id = "problem_1591"
    dataset_path = str(PROJECT_ROOT / f"{problem_id}_dataset")
    flip_types_file = str(PROJECT_ROOT / "flip_types.txt")
    case_study_file = str(PROJECT_ROOT / "visualizations/analysis/case_study.json")
    anchor_case_study_file = str(PROJECT_ROOT / "visualizations/analysis/anchor_case_study.json")
    
    print("="*80)
    print("UPDATING CASE STUDY STATISTICS")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ Loaded dataset")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    print("✓ Tokenizer loaded")
    
    # Load function tags
    print("\nLoading function tags...")
    chunk_function_tags = load_chunk_function_tags(dataset)
    print(f"✓ Loaded function tags for {len(chunk_function_tags)} chunks")
    
    # Load flip types
    print("\nLoading flip types...")
    flip_types = load_flip_types(flip_types_file)
    print(f"✓ Loaded flip types for {len(flip_types)} chunks")
    
    # Update case_study.json
    if Path(case_study_file).exists():
        update_case_study_json(case_study_file, tokenizer, chunk_function_tags, flip_types)
    else:
        print(f"Warning: {case_study_file} not found")
    
    # Update anchor_case_study.json
    if Path(anchor_case_study_file).exists():
        update_anchor_case_study_json(anchor_case_study_file, tokenizer, chunk_function_tags, flip_types)
    else:
        print(f"Warning: {anchor_case_study_file} not found")
    
    print("\n" + "="*80)
    print("UPDATE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

