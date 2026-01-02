"""
Download the first 10 problems from the math-rollouts dataset (llama-8b only).
This will download all files associated with these problems including all solutions.
"""

from load_dataset import load_llama8b_dataset
import json

def download_10_problems():
    """
    Download the first 10 problems with all their solutions.
    Also saves the problem IDs to a file for quick access later.
    """
    print("="*80)
    print("Downloading first 10 problems from math-rollouts dataset (llama-8b)")
    print("="*80)
    print("\nThis will download:")
    print("  - problem.json files")
    print("  - chunks_labeled.json files")
    print("  - All solutions.json files in each chunk directory")
    print("\nStarting download...\n")
    
    # Load first 10 problems
    dataset = load_llama8b_dataset(max_problems=10)
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    
    # Extract problem IDs
    print("\nExtracting problem IDs...")
    problem_ids = set()
    problems_by_type = {}
    
    for example in dataset:
        path = example.get("path", "")
        if "problem_" in path:
            parts = path.split("/")
            for part in parts:
                if part.startswith("problem_"):
                    problem_ids.add(part)
                    break
    
    sorted_problem_ids = sorted(list(problem_ids))
    
    # Save problem IDs to file
    problems_info = {
        "problem_ids": sorted_problem_ids,
        "total_problems": len(sorted_problem_ids),
        "max_problems": 10
    }
    
    with open("downloaded_problems.json", "w") as f:
        json.dump(problems_info, f, indent=2)
    
    # Display summary
    if hasattr(dataset, '__len__'):
        print(f"\nTotal files downloaded: {len(dataset)}")
        print(f"\nProblem IDs downloaded ({len(sorted_problem_ids)} problems):")
        for pid in sorted_problem_ids:
            print(f"  {pid}")
        print(f"\nProblem list saved to: downloaded_problems.json")
        print(f"\nDataset is now cached locally and ready to use.")
        print(f"\nYou can now use this dataset in your code like this:")
        print(f"  from load_dataset import load_llama8b_dataset")
        print(f"  dataset = load_llama8b_dataset(max_problems=10)")
    
    return dataset, sorted_problem_ids

if __name__ == "__main__":
    dataset, problem_ids = download_10_problems()

