#!/usr/bin/env python3
"""
Clean padding patterns from flip_chunk_results.json and ablate_uncertainty_results.json.

Removes patterns like:
- "\n**Final Answer**\n\\boxed{...}\n"
- Repeated final answer patterns at the end

Then recalculates sentence count, token count, and uncertainty word count.
"""

import json
import re
from pathlib import Path
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent

# Uncertainty words for counting
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

def clean_padding_patterns(text: str) -> str:
    """
    Remove duplicate padding patterns from the end of text, keeping only one final answer.
    
    Removes repeated "\n**Final Answer**\n\\boxed{...}\n\\ )\n" patterns, keeping only the first one.
    """
    if not text:
        return text
    
    text_rstrip = text.rstrip()
    
    # Find all **Final Answer** positions
    fa_positions = [m.start() for m in re.finditer(r'\*\*Final Answer\*\*', text_rstrip)]
    
    if len(fa_positions) <= 1:
        # Only one or zero occurrences, no duplicates
        return text_rstrip
    
    # Find where duplicates start (consecutive occurrences with small gaps)
    duplicate_start_idx = len(fa_positions) - 1
    for i in range(len(fa_positions) - 2, -1, -1):
        gap = fa_positions[i+1] - fa_positions[i]
        if gap < 100:  # Small gap indicates duplicates
            duplicate_start_idx = i
        else:
            break
    
    # If we found duplicates, remove them but keep one
    if duplicate_start_idx < len(fa_positions) - 1:
        # Find where to keep up to (before duplicates start)
        if duplicate_start_idx > 0:
            # Find the end of the last non-duplicate final answer
            prev_fa_pos = fa_positions[duplicate_start_idx - 1]
            after_prev = text_rstrip[prev_fa_pos:]
            # Find \boxed and the end of that block
            block_end_match = re.search(r'\\boxed\{[^}]+\}[^\n]*(?:\\ \\))?[^\n]*\n', after_prev)
            if block_end_match:
                keep_end = prev_fa_pos + block_end_match.end()
            else:
                # Fallback: find next newline after boxed
                boxed_match = re.search(r'\\boxed\{[^}]+\}', after_prev)
                if boxed_match:
                    after_boxed = after_prev[boxed_match.end():]
                    newline_match = re.search(r'\n', after_boxed)
                    if newline_match:
                        keep_end = prev_fa_pos + boxed_match.end() + newline_match.end()
                    else:
                        keep_end = prev_fa_pos + boxed_match.end()
                else:
                    keep_end = prev_fa_pos
        else:
            # All are duplicates, keep only the first one
            first_fa_pos = fa_positions[0]
            after_first = text_rstrip[first_fa_pos:]
            block_end_match = re.search(r'\\boxed\{[^}]+\}[^\n]*(?:\\ \\))?[^\n]*\n', after_first)
            if block_end_match:
                keep_end = first_fa_pos + block_end_match.end()
            else:
                keep_end = first_fa_pos
        
        # Get the first duplicate to keep (just one occurrence)
        first_dup_pos = fa_positions[duplicate_start_idx]
        dup_block = text_rstrip[first_dup_pos:]
        # Match the pattern: **Final Answer** ... \boxed{...} ... \n
        first_dup_match = re.search(r'\*\*Final Answer\*\*[^\n]*\\boxed\{[^}]+\}[^\n]*(?:\\ \\))?[^\n]*\n', dup_block)
        if first_dup_match:
            keep_one = first_dup_match.group()
            cleaned = text_rstrip[:keep_end] + keep_one
            return cleaned.rstrip()
        else:
            # Fallback: just keep up to keep_end
            return text_rstrip[:keep_end].rstrip()
    
    # No duplicates found
    return text_rstrip

def clean_and_recalculate_json(json_file: Path, tokenizer: AutoTokenizer) -> dict:
    """Clean padding patterns and recalculate statistics in a JSON file."""
    print(f"\nProcessing {json_file.name}...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    experiments = data.get('experiments', [])
    cleaned_count = 0
    total_removed_chars = 0
    
    for exp in experiments:
        full_cot = exp.get('full_cot', '')
        if not full_cot:
            continue
        
        original_len = len(full_cot)
        cleaned_cot = clean_padding_patterns(full_cot)
        removed_chars = original_len - len(cleaned_cot)
        
        if removed_chars > 0:
            cleaned_count += 1
            total_removed_chars += removed_chars
            
            # Update full_cot
            exp['full_cot'] = cleaned_cot
            
            # Recalculate statistics
            exp['sentence_count'] = count_sentences(cleaned_cot)
            try:
                exp['token_count'] = len(tokenizer.encode(cleaned_cot, add_special_tokens=False))
            except Exception:
                # If tokenization fails, keep original or set to 0
                exp['token_count'] = 0
            exp['uncertainty_word_count'] = count_uncertainty_words(cleaned_cot)
    
    print(f"  Cleaned {cleaned_count} out of {len(experiments)} experiments")
    print(f"  Total characters removed: {total_removed_chars:,}")
    
    return data

def main():
    """Main function to clean padding patterns from JSON files."""
    print("="*80)
    print("CLEANING PADDING PATTERNS FROM EXPERIMENT RESULTS")
    print("="*80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    print("✓ Tokenizer loaded")
    
    # Files to clean
    flip_file = PROJECT_ROOT / "flip_chunk_results.json"
    ablate_file = PROJECT_ROOT / "ablate_uncertainty_results.json"
    
    # Clean flip results
    if flip_file.exists():
        flip_data = clean_and_recalculate_json(flip_file, tokenizer)
        # Backup original
        backup_file = PROJECT_ROOT / "flip_chunk_results.json.backup"
        if not backup_file.exists():
            import shutil
            shutil.copy(flip_file, backup_file)
            print(f"  Created backup: {backup_file.name}")
        
        # Save cleaned version
        with open(flip_file, 'w') as f:
            json.dump(flip_data, f, indent=2)
        print(f"  ✓ Saved cleaned {flip_file.name}")
    else:
        print(f"  ⚠ {flip_file.name} not found")
    
    # Clean ablate results
    if ablate_file.exists():
        ablate_data = clean_and_recalculate_json(ablate_file, tokenizer)
        # Backup original
        backup_file = PROJECT_ROOT / "ablate_uncertainty_results.json.backup"
        if not backup_file.exists():
            import shutil
            shutil.copy(ablate_file, backup_file)
            print(f"  Created backup: {backup_file.name}")
        
        # Save cleaned version
        with open(ablate_file, 'w') as f:
            json.dump(ablate_data, f, indent=2)
        print(f"  ✓ Saved cleaned {ablate_file.name}")
    else:
        print(f"  ⚠ {ablate_file.name} not found")
    
    print("\n" + "="*80)
    print("CLEANING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Regenerate case_study.json: python3 plot/generate_case_study.py")
    print("  2. Regenerate accuracy tables: python3 plot/generate_accuracy_tables.py")
    print("  3. Regenerate visualizations: python3 plot/plot_aggregated_statistics.py")
    print("  4. Regenerate chunk statistics plots: python3 plot/plot_chunk_statistics.py")

if __name__ == "__main__":
    main()

