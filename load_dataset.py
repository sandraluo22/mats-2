"""
Load the math-rollouts dataset from Hugging Face (llama-8b only).
Dataset: https://huggingface.co/datasets/uzaymacar/math-rollouts
"""

import os

# Set cache directories BEFORE importing datasets library
# NOTE: Temporarily disabled /workspace cache due to quota issues
# Using default /root/.cache/huggingface which now has 19GB free
# if os.path.exists("/workspace") and os.access("/workspace", os.W_OK):
#     # Set XDG_CACHE_HOME first (most fundamental, affects many tools)
#     os.environ["XDG_CACHE_HOME"] = "/workspace/.cache"
#     # Set all HuggingFace cache locations to /workspace
#     os.environ["HF_DATASETS_CACHE"] = "/workspace/.cache/huggingface/datasets"
#     os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
#     os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"
#     os.environ["TMPDIR"] = "/workspace/tmp"
#     # Create directories
#     os.makedirs("/workspace/.cache/huggingface/datasets", exist_ok=True)
#     os.makedirs("/workspace/.cache/huggingface/hub", exist_ok=True)
#     os.makedirs("/workspace/tmp", exist_ok=True)

from datasets import load_dataset
import time
from requests.exceptions import ReadTimeout, ConnectionError

def load_llama8b_dataset(problem_ids=None, max_problems=None, max_retries=3, use_streaming=True):
    """
    Load only the llama-8b portion of the math-rollouts dataset from Hugging Face.
    Uses streaming mode by default to avoid generating large Arrow files.
    
    Args:
        problem_ids: Optional list of specific problem IDs to load (e.g., ["problem_330", "problem_331"]).
                    If None, loads all problems.
        max_problems: Optional maximum number of problems to load. If specified, takes first N unique problems.
        max_retries: Maximum number of retry attempts for network errors (default: 3)
        use_streaming: Use streaming mode to avoid processing full dataset (default: True)
    
    Returns:
        Dataset: The filtered dataset containing only llama-8b data (and optionally only specified problems)
    """
    print("Loading math-rollouts dataset (llama-8b only)...")
    if use_streaming:
        print("Using streaming mode (avoids generating large Arrow cache files)")
    
    # Set longer timeout for HuggingFace Hub (in seconds)
    # This helps with slow network connections
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes
    
    # Note: Cache directories are set at module import time (above)
    # Using default /root cache location (now has 19GB free after cleanup)
    
    # Load the dataset with streaming mode
    for attempt in range(max_retries):
        try:
            dataset = load_dataset("uzaymacar/math-rollouts", streaming=use_streaming)
            break  # Success, exit retry loop
        except (ReadTimeout, ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Network error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # Last attempt failed
                print(f"\nFailed after {max_retries} attempts. Last error: {e}")
                print("\nTroubleshooting tips:")
                print("  1. Check your internet connection")
                print("  2. Try again later (HuggingFace servers may be busy)")
                print("  3. Use a VPN if you're behind a firewall")
                print("  4. Check if the dataset exists: https://huggingface.co/datasets/uzaymacar/math-rollouts")
                raise
    
    if use_streaming:
        # With streaming, we need to iterate and collect examples
        from datasets import Dataset
        
        # Handle DatasetDict or single Dataset
        if isinstance(dataset, dict):
            splits_to_process = dataset.items()
        else:
            splits_to_process = [("default", dataset)]
        
        llama8b_examples = {}
        
        for split_name, split_data in splits_to_process:
            print(f"Processing split: {split_name} (streaming mode)...")
            examples = []
            problem_ids_seen = set()
            problem_ids_to_keep_set = set(problem_ids) if problem_ids else None
            
            # For max_problems, first pass: collect problem IDs
            if max_problems is not None and problem_ids_to_keep_set is None:
                print(f"  First pass: Collecting {max_problems} unique problem IDs...")
                for example in split_data:
                    path = example.get("path", "")
                    if "llama-8b" not in path.lower():
                        continue
                    if "problem_" in path:
                        parts = path.split("/")
                        for part in parts:
                            if part.startswith("problem_"):
                                problem_ids_seen.add(part)
                                if len(problem_ids_seen) % 10 == 0:
                                    print(f"    Found {len(problem_ids_seen)} problem IDs...", end='\r')
                                break
                    if len(problem_ids_seen) >= max_problems:
                        break
                print(f"\n  Found {len(problem_ids_seen)} unique problem IDs")
                problem_ids_to_keep_set = problem_ids_seen
            
            # Second pass: collect all examples for the selected problem IDs
            if problem_ids_to_keep_set is not None:
                print(f"  Collecting examples for {len(problem_ids_to_keep_set)} problem(s)...")
                # Need to reload for second pass since iterator is exhausted
                dataset_reload = load_dataset("uzaymacar/math-rollouts", streaming=True)
                if isinstance(dataset_reload, dict):
                    split_data = dataset_reload[split_name] if split_name in dataset_reload else list(dataset_reload.values())[0]
                else:
                    split_data = dataset_reload
            
            for example in split_data:
                path = example.get("path", "")
                
                # Filter for llama-8b only
                if "llama-8b" not in path.lower():
                    continue
                
                # Filter by problem IDs if specified
                if problem_ids_to_keep_set is not None:
                    example_problem_id = None
                    if "problem_" in path:
                        parts = path.split("/")
                        for part in parts:
                            if part.startswith("problem_"):
                                example_problem_id = part
                                break
                    
                    if example_problem_id not in problem_ids_to_keep_set:
                        continue
                
                # Add this example (convert to dict for Dataset.from_list)
                examples.append(dict(example))
                
                # Progress indicator
                if len(examples) % 100 == 0:
                    print(f"    Collected {len(examples)} examples...", end='\r')
            
            if examples:
                print(f"\n  Collected {len(examples)} llama-8b examples")
                llama8b_examples[split_name] = examples
        
        # Convert collected examples to Dataset
        if not llama8b_examples:
            raise ValueError("No llama-8b data found in the dataset")
        
        if len(llama8b_examples) == 1:
            # Single split - return Dataset
            split_name, examples = list(llama8b_examples.items())[0]
            print(f"Creating Dataset from {len(examples)} examples...")
            return Dataset.from_list(examples)
        else:
            # Multiple splits - return DatasetDict
            from datasets import DatasetDict
            dataset_dict = {}
            for split_name, examples in llama8b_examples.items():
                print(f"Creating Dataset for split '{split_name}' from {len(examples)} examples...")
                dataset_dict[split_name] = Dataset.from_list(examples)
            return DatasetDict(dataset_dict)
    
    else:
        # Non-streaming mode (original implementation)
        # Filter to only include llama-8b files
        # The dataset structure has paths like "deepseek-r1-distill-llama-8b/..."
        llama8b_dataset = {}
        
        for split_name, split_data in dataset.items():
            print(f"Filtering split: {split_name}")
            
            # Filter rows where path contains "llama-8b" (case-insensitive)
            filtered = split_data.filter(
                lambda example: "llama-8b" in example.get("path", "").lower()
            )
            
            # Further filter by problem IDs if specified
            if problem_ids is not None:
                problem_ids_set = set(problem_ids)
                filtered = filtered.filter(
                    lambda example: any(
                        f"/{pid}/" in example.get("path", "") or example.get("path", "").endswith(f"/{pid}")
                        for pid in problem_ids_set
                    )
                )
                print(f"  Filtered to {len(filtered)} files matching {len(problem_ids)} problem IDs")
            elif max_problems is not None:
                # Extract unique problem IDs and limit
                # We need to iterate through the dataset to collect problem IDs
                print(f"  Collecting problem IDs (this may take a moment)...")
                problem_ids_to_keep = set()
                
                # Iterate through the filtered dataset to collect problem IDs
                for example in filtered:
                    path = example.get("path", "")
                    if "problem_" in path:
                        parts = path.split("/")
                        for part in parts:
                            if part.startswith("problem_"):
                                problem_ids_to_keep.add(part)
                                if len(problem_ids_to_keep) >= max_problems:
                                    break
                        if len(problem_ids_to_keep) >= max_problems:
                            break
                
                print(f"  Found {len(problem_ids_to_keep)} unique problems, filtering to those...")
                
                # Filter to only those problems
                problem_ids_list = list(problem_ids_to_keep)
                filtered = filtered.filter(
                    lambda example: any(
                        f"/{pid}/" in example.get("path", "") or example.get("path", "").endswith(f"/{pid}")
                        for pid in problem_ids_list
                    )
                )
                print(f"  Limited to {len(problem_ids_to_keep)} problems ({len(filtered)} files)")
            
            if len(filtered) > 0:
                llama8b_dataset[split_name] = filtered
                print(f"  Found {len(filtered)} llama-8b examples in {split_name}")
        
        # If we have splits, return as DatasetDict, otherwise return the single split
        if len(llama8b_dataset) > 1:
            from datasets import DatasetDict
            return DatasetDict(llama8b_dataset)
        elif len(llama8b_dataset) == 1:
            return list(llama8b_dataset.values())[0]
        else:
            raise ValueError("No llama-8b data found in the dataset")

if __name__ == "__main__":
    import sys
    
    # Check if user wants to load only a few problems
    if len(sys.argv) > 1:
        if sys.argv[1] == "--max-problems":
            max_problems = int(sys.argv[2])
            print(f"Loading only first {max_problems} problems...")
            dataset = load_llama8b_dataset(max_problems=max_problems)
        elif sys.argv[1] == "--problems":
            problem_ids = sys.argv[2].split(",")
            print(f"Loading specified problems: {problem_ids}")
            dataset = load_llama8b_dataset(problem_ids=problem_ids)
        else:
            print("Usage:")
            print("  python load_dataset.py                    # Load all problems")
            print("  python load_dataset.py --max-problems 5   # Load first 5 problems")
            print("  python load_dataset.py --problems problem_330,problem_331  # Load specific problems")
            sys.exit(1)
    else:
        # Load all llama-8b data
        dataset = load_llama8b_dataset()
    
    # Display basic information
    print("\nDataset loaded successfully!")
    
    if isinstance(dataset, dict):
        print(f"Dataset splits: {list(dataset.keys())}")
        
        # Show info about each split
        for split_name, split_data in dataset.items():
            print(f"\n{split_name} split info:")
            print(f"  Number of rows: {len(split_data)}")
            print(f"  Features: {list(split_data.features.keys())}")
            
            # Show first example
            if len(split_data) > 0:
                print(f"\n  First example keys: {list(split_data[0].keys())}")
                print(f"\n  First example preview:")
                for key, value in list(split_data[0].items())[:5]:  # Show first 5 keys
                    if isinstance(value, str) and len(value) > 200:
                        print(f"    {key}: {value[:200]}...")
                    else:
                        print(f"    {key}: {value}")
    else:
        print(f"Number of rows: {len(dataset)}")
        print(f"Features: {list(dataset.features.keys())}")
        
        # Show first example
        if len(dataset) > 0:
            print(f"\nFirst example keys: {list(dataset[0].keys())}")
            print(f"\nFirst example preview:")
            for key, value in list(dataset[0].items())[:5]:  # Show first 5 keys
                if isinstance(value, str) and len(value) > 200:
                    print(f"  {key}: {value[:200]}...")
                else:
                    print(f"  {key}: {value}")
