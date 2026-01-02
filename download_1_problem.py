"""
Download just ONE problem to test - pleaaaaeaeaeaeaese
"""

from load_dataset import load_llama8b_dataset
import json
import sys

def download_1_problem():
    """Download the first problem only."""
    print("="*80)
    print("Downloading FIRST problem from math-rollouts dataset (llama-8b only)")
    print("="*80)
    print("\nThis will download:")
    print("  - problem.json file")
    print("  - chunks_labeled.json file")
    print("  - All solutions.json files for this problem")
    print("\nStarting download...\n")
    
    # Load first 1 problem
    dataset = load_llama8b_dataset(max_problems=1)
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    
    # Extract problem ID
    print("\nExtracting problem ID...")
    problem_ids = set()
    
    for example in dataset:
        path = example.get("path", "")
        if "problem_" in path:
            parts = path.split("/")
            for part in parts:
                if part.startswith("problem_"):
                    problem_ids.add(part)
                    break
    
    sorted_problem_ids = sorted(list(problem_ids))
    
    # Save problem ID to file
    problems_info = {
        "problem_ids": sorted_problem_ids,
        "total_problems": len(sorted_problem_ids),
        "max_problems": 1
    }
    
    with open("downloaded_problems.json", "w") as f:
        json.dump(problems_info, f, indent=2)
    
    # Display summary
    if hasattr(dataset, '__len__'):
        print(f"\nTotal files downloaded: {len(dataset)}")
        print(f"\nProblem ID: {sorted_problem_ids[0] if sorted_problem_ids else 'N/A'}")
        print(f"\nProblem list saved to: downloaded_problems.json")
        print(f"\nYou can now use this problem ID in your code!")
    
    return dataset, sorted_problem_ids

if __name__ == "__main__":
    dataset, problem_ids = download_1_problem()

