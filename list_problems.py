"""
List all problems available in the downloaded dataset.
Reads from downloaded_problems.json file (created by download_10_problems.py).
"""

import json
import os

def list_problems():
    """
    List all problem IDs from the saved downloaded_problems.json file.
    """
    problems_file = "downloaded_problems.json"
    
    if not os.path.exists(problems_file):
        print("="*80)
        print("ERROR: No downloaded problems found!")
        print("="*80)
        print(f"\n{problems_file} file not found.")
        print("Please run download_10_problems.py first to download and save the problem list.")
        print("\nUsage:")
        print("  python download_10_problems.py")
        return []
    
    print("Reading problem list from downloaded_problems.json...")
    
    try:
        with open(problems_file, "r") as f:
            problems_info = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse {problems_file}: {e}")
        return []
    
    problem_ids = problems_info.get("problem_ids", [])
    total_problems = problems_info.get("total_problems", len(problem_ids))
    max_problems = problems_info.get("max_problems", "unknown")
    
    # Print results
    print("\n" + "="*80)
    print("DOWNLOADED PROBLEMS")
    print("="*80)
    
    print(f"\nTotal problems downloaded: {total_problems} (max_problems={max_problems})")
    print(f"\nProblem IDs:")
    for pid in problem_ids:
        print(f"  {pid}")
    
    print("\n" + "="*80)
    print("NOTE")
    print("="*80)
    print("\nAll these problems should have forced_answer data available for plotting.")
    print("\nTo use with plot_kl_vs_accuracy.py:")
    if problem_ids:
        print(f"  python plot_kl_vs_accuracy.py {problem_ids[0]}")
    
    return problem_ids

if __name__ == "__main__":
    problem_ids = list_problems()

