"""
Quick script to check how many problems are in the dataset and explore the structure.
"""

from datasets import load_dataset
from collections import defaultdict

def check_dataset_info():
    """
    Check how many problems are in the dataset.
    """
    print("Loading dataset to check structure...")
    print("(This may take a minute on first run)")
    
    dataset = load_dataset("uzaymacar/math-rollouts")
    
    # Get the default split
    split_data = dataset.get("default", None)
    if split_data is None:
        split_data = list(dataset.values())[0]
    
    print(f"\nTotal files/rows in dataset: {len(split_data)}")
    
    # Filter for llama-8b
    llama8b_data = split_data.filter(
        lambda example: "llama-8b" in example.get("path", "").lower()
    )
    
    print(f"Llama-8b files: {len(llama8b_data)}")
    
    # Extract problem IDs from paths
    # Paths look like: "deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95/correct_base_solution/problem_330/problem.json"
    problem_ids = set()
    problem_paths = defaultdict(set)
    
    for example in llama8b_data:
        path = example.get("path", "")
        
        # Extract problem ID (e.g., "problem_330")
        if "problem_" in path:
            parts = path.split("/")
            for part in parts:
                if part.startswith("problem_"):
                    problem_id = part
                    problem_ids.add(problem_id)
                    # Track what files are associated with this problem
                    if "problem.json" in path:
                        problem_paths[problem_id].add("problem.json")
                    elif "chunks_labeled.json" in path:
                        problem_paths[problem_id].add("chunks_labeled.json")
                    elif "solutions.json" in path:
                        problem_paths[problem_id].add("solutions.json")
                    break
    
    print(f"\nUnique problems found: {len(problem_ids)}")
    
    # Group by solution type
    solution_types = defaultdict(int)
    for example in llama8b_data:
        path = example.get("path", "")
        if "problem_" in path:
            if "correct_base_solution" in path and "forced_answer" in path:
                solution_types["correct_base_solution_forced_answer"] += 1
            elif "incorrect_base_solution" in path and "forced_answer" in path:
                solution_types["incorrect_base_solution_forced_answer"] += 1
            elif "correct_base_solution" in path and "sanity" in path:
                solution_types["correct_base_solution_sanity"] += 1
            elif "correct_base_solution" in path:
                solution_types["correct_base_solution"] += 1
            elif "incorrect_base_solution" in path:
                solution_types["incorrect_base_solution"] += 1
    
    print(f"\nFiles by solution type:")
    for sol_type, count in sorted(solution_types.items()):
        print(f"  {sol_type}: {count} files")
    
    # Count problems per solution type
    problems_by_type = defaultdict(set)
    for example in llama8b_data:
        path = example.get("path", "")
        if "problem_" in path:
            parts = path.split("/")
            problem_id = None
            for part in parts:
                if part.startswith("problem_"):
                    problem_id = part
                    break
            
            if problem_id:
                if "correct_base_solution" in path and "forced_answer" in path:
                    problems_by_type["correct_base_solution_forced_answer"].add(problem_id)
                elif "incorrect_base_solution" in path and "forced_answer" in path:
                    problems_by_type["incorrect_base_solution_forced_answer"].add(problem_id)
                elif "correct_base_solution" in path and "sanity" in path:
                    problems_by_type["correct_base_solution_sanity"].add(problem_id)
                elif "correct_base_solution" in path:
                    problems_by_type["correct_base_solution"].add(problem_id)
                elif "incorrect_base_solution" in path:
                    problems_by_type["incorrect_base_solution"].add(problem_id)
    
    print(f"\nProblems by solution type:")
    for sol_type, problem_set in sorted(problems_by_type.items()):
        print(f"  {sol_type}: {len(problem_set)} problems")
    
    # Show some example problem IDs
    sorted_problem_ids = sorted(list(problem_ids))[:10]
    print(f"\nFirst 10 problem IDs: {sorted_problem_ids}")
    
    return problem_ids, llama8b_data

if __name__ == "__main__":
    problem_ids, llama8b_data = check_dataset_info()

