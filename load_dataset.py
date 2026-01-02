"""
Load the math-rollouts dataset from Hugging Face (llama-8b only).
Dataset: https://huggingface.co/datasets/uzaymacar/math-rollouts
"""

from datasets import load_dataset

def load_llama8b_dataset(problem_ids=None, max_problems=None):
    """
    Load only the llama-8b portion of the math-rollouts dataset from Hugging Face.
    
    Args:
        problem_ids: Optional list of specific problem IDs to load (e.g., ["problem_330", "problem_331"]).
                    If None, loads all problems.
        max_problems: Optional maximum number of problems to load. If specified, takes first N unique problems.
    
    Returns:
        Dataset: The filtered dataset containing only llama-8b data (and optionally only specified problems)
    """
    print("Loading math-rollouts dataset (llama-8b only)...")
    
    # Load the full dataset
    dataset = load_dataset("uzaymacar/math-rollouts")
    
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
