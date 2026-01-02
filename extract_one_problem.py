#!/usr/bin/env python3
"""
Extract ONE problem from the dataset and save it locally.
By default uses streaming mode (faster for one-time extraction of a single problem).

Usage:
    python3 extract_one_problem.py problem_1591
    python3 extract_one_problem.py problem_1591 --output problem_1591.json
    python3 extract_one_problem.py problem_1591 --no-streaming  # Use non-streaming (downloads full dataset first)
"""

import sys
import json
import os
import shutil
from datasets import Dataset
from load_dataset import load_llama8b_dataset

def extract_one_problem(problem_id, output_file=None, use_streaming=True):
    """
    Extract a single problem from the dataset.
    
    Args:
        problem_id: The problem ID to extract
        output_file: Optional output file path
        use_streaming: If True, use streaming mode (faster for one-time extraction, doesn't cache full dataset).
                      If False, use non-streaming mode (downloads full ~19GB dataset first, then fast filtering).
                      Note: Since cache is deleted after extraction, streaming is usually faster for single problems.
    
    Saves to a local JSON file and returns a Dataset object.
    """
    if output_file is None:
        output_file = f"{problem_id}.json"
    
    print(f"Extracting problem: {problem_id}")
    if use_streaming:
        print(f"Using streaming mode (faster for one-time extraction, no full dataset cache)...")
    else:
        print(f"Using non-streaming mode (will download ~19GB dataset first, then fast filtering)...")
        print(f"  Note: Cache will be deleted after extraction, so this is slower for single problems.")
    print()
    
    # Use the existing function which has retry logic
    print("Loading dataset (this may take a while)...")
    try:
        dataset = load_llama8b_dataset(problem_ids=[problem_id], use_streaming=use_streaming)
    except ValueError as e:
        if "No llama-8b data found" in str(e):
            print(f"\n✗ ERROR: {e}")
            print(f"\nTroubleshooting:")
            print(f"  1. Verify the problem ID '{problem_id}' exists in the dataset")
            print(f"  2. Check that the problem has llama-8b data")
            print(f"  3. Try a different problem ID")
            return None
        else:
            raise
    
    # DATA IS NOW IN MEMORY as a Dataset object
    print(f"\n✓ Found dataset with {len(dataset)} examples")
    
    if len(dataset) == 0:
        print(f"\n✗ ERROR: No examples found for {problem_id}")
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
        print("Usage: python3 extract_one_problem.py <problem_id> [--output filename.json] [--no-streaming]")
        print("Example: python3 extract_one_problem.py problem_1591")
        print("         python3 extract_one_problem.py problem_1591 --no-streaming  # Download full dataset first")
        sys.exit(1)
    
    problem_id = sys.argv[1]
    output_file = None
    use_streaming = "--no-streaming" not in sys.argv  # Default to streaming
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
    
    dataset = extract_one_problem(problem_id, output_file, use_streaming=use_streaming)
    
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

