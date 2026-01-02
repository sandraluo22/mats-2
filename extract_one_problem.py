#!/usr/bin/env python3
"""
Extract ONE problem from the dataset and save it locally.
This uses streaming mode to avoid loading the full dataset.

Usage:
    python3 extract_one_problem.py problem_1591
    python3 extract_one_problem.py problem_1591 --output problem_1591.json
"""

import sys
import json
import os
import shutil
from datasets import Dataset
from load_dataset import load_llama8b_dataset

def extract_one_problem(problem_id, output_file=None):
    """
    Extract a single problem from the dataset using streaming mode.
    Saves to a local JSON file and returns a Dataset object.
    """
    if output_file is None:
        output_file = f"{problem_id}.json"
    
    print(f"Extracting problem: {problem_id}")
    print(f"Using streaming mode (won't load full dataset)...")
    print()
    
    # Use the existing function which has retry logic
    print("Loading dataset (this may take a while)...")
    dataset = load_llama8b_dataset(problem_ids=[problem_id], use_streaming=True)
    
    # DATA IS NOW IN MEMORY as a Dataset object
    print(f"\n✓ Found dataset with {len(dataset)} examples")
    
    if len(dataset) == 0:
        print(f"\nERROR: No examples found for {problem_id}")
        print("Make sure the problem ID exists in the dataset.")
        return None
    
    # Save Dataset format first (faster, no conversion needed)
    dataset_dir = output_file.replace('.json', '_dataset')
    print(f"\nSaving Dataset format to {dataset_dir}...")
    dataset.save_to_disk(dataset_dir)
    print(f"✓ Saved Dataset format ({len(dataset)} examples)")
    
    # Convert to list and save as JSON (slower, but human-readable)
    print(f"Converting to JSON format (this may take a moment)...")
    examples = [dict(example) for example in dataset]
    
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    file_size = os.path.getsize(output_file) / (1024**2)  # MB
    print(f"✓ Saved {len(examples)} examples to {output_file} ({file_size:.1f} MB)")
    
    # Now that everything is saved, delete the cache to free up space
    print(f"\nDeleting 19GB raw dataset cache to free up space...")
    cache_paths_to_check = [
        os.path.join(os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")), "datasets--uzaymacar--math-rollouts"),
        os.path.expanduser("~/.cache/huggingface/hub/datasets--uzaymacar--math-rollouts"),
        "/workspace/.cache/huggingface/hub/datasets--uzaymacar--math-rollouts",
        "/root/.cache/huggingface/hub/datasets--uzaymacar--math-rollouts",
    ]
    
    deleted_any = False
    for raw_cache_path in cache_paths_to_check:
        if os.path.exists(raw_cache_path):
            try:
                shutil.rmtree(raw_cache_path)
                print(f"✓ Deleted {raw_cache_path}")
                print("  (Freed approximately 19GB of space)")
                deleted_any = True
            except Exception as e:
                print(f"⚠ Warning: Could not delete {raw_cache_path}: {e}")
    
    if not deleted_any:
        print("Cache already deleted or doesn't exist.")
    
    return dataset

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_one_problem.py <problem_id> [--output filename.json]")
        print("Example: python3 extract_one_problem.py problem_1591")
        sys.exit(1)
    
    problem_id = sys.argv[1]
    output_file = None
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
    
    dataset = extract_one_problem(problem_id, output_file)
    
    if dataset:
        # Get output_file from the function (it was set inside extract_one_problem)
        actual_output = output_file if output_file else f"{problem_id}.json"
        dataset_dir = actual_output.replace('.json', '_dataset')
        
        print()
        print("=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"You can now use the extracted problem:")
        print(f"  from datasets import load_from_disk")
        print(f"  dataset = load_from_disk('{dataset_dir}')")
        print()
        print("Or load the JSON file directly:")
        print(f"  import json")
        print(f"  with open('{actual_output}') as f:")
        print(f"    data = json.load(f)")
        print()
        print("(Cache was deleted after saving to free up space)")

