"""
Plot KL importance vs average accuracy for each sentence in the chain of thought.
For each chunk (sentence) in the COT:
  - X-axis: KL importance (from chunks_labeled.json)
  - Y-axis: Average accuracy over forced answer solutions (from solutions.json)
"""

from load_dataset import load_llama8b_dataset
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

def extract_problem_id_from_path(path):
    """Extract problem ID from a file path."""
    parts = path.split("/")
    for part in parts:
        if part.startswith("problem_"):
            return part
    return None

def load_problem_data(dataset, problem_id, use_forced_answer=True):
    """
    Load all data for a specific problem from the dataset.
    
    Args:
        dataset: The dataset to search
        problem_id: Problem ID to load
        use_forced_answer: If True, only load from forced_answer directories
    
    Returns:
        dict with keys: 'problem_json', 'chunks_labeled', 'chunk_solutions'
    """
    problem_data = {
        'problem_json': None,
        'chunks_labeled': None,
        'chunk_solutions': defaultdict(list)  # chunk_idx -> list of solutions
    }
    
    # Filter dataset for this problem
    problem_files = dataset.filter(
        lambda example: problem_id in example.get("path", "")
    )
    
    # Further filter for forced_answer if requested
    if use_forced_answer:
        problem_files = problem_files.filter(
            lambda example: "forced_answer" in example.get("path", "").lower()
        )
    
    for example in problem_files:
        path = example.get("path", "")
        content = example.get("content", "")
        
        if "problem.json" in path:
            try:
                problem_data['problem_json'] = json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse problem.json for {problem_id}")
        
        elif "chunks_labeled.json" in path:
            try:
                problem_data['chunks_labeled'] = json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse chunks_labeled.json for {problem_id}")
        
        elif "solutions.json" in path:
            # Extract chunk index from path like "chunk_0/solutions.json"
            try:
                parts = path.split("/")
                for part in parts:
                    if part.startswith("chunk_") and part != "chunks_labeled.json":
                        chunk_idx = int(part.replace("chunk_", ""))
                        solutions = json.loads(content)
                        if isinstance(solutions, list):
                            problem_data['chunk_solutions'][chunk_idx] = solutions
                        break
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Could not parse solutions.json at {path}: {e}")
    
    return problem_data

def plot_kl_vs_accuracy(problem_data, problem_id, kl_metric="resampling_importance_kl"):
    """
    Plot KL importance vs average accuracy for each chunk.
    
    Args:
        problem_data: Dictionary with problem data
        problem_id: Problem ID for title
        kl_metric: Which KL metric to use (default: "resampling_importance_kl")
                   Options: "resampling_importance_kl", "counterfactual_importance_kl", 
                           "forced_importance_kl"
    """
    chunks_labeled = problem_data.get('chunks_labeled')
    chunk_solutions = problem_data.get('chunk_solutions', {})
    
    if chunks_labeled is None:
        print(f"Error: No chunks_labeled.json found for {problem_id}")
        return
    
    if not isinstance(chunks_labeled, list):
        print(f"Error: chunks_labeled.json is not a list for {problem_id}")
        return
    
    # Collect data points
    kl_values = []
    avg_accuracy_values = []
    chunk_indices = []
    
    for chunk_data in chunks_labeled:
        if not isinstance(chunk_data, dict):
            continue
        
        chunk_idx = chunk_data.get('chunk_idx')
        if chunk_idx is None:
            continue
        
        # Get KL importance
        kl_value = chunk_data.get(kl_metric)
        if kl_value is None:
            print(f"Warning: No {kl_metric} for chunk {chunk_idx}, skipping")
            continue
        
        # Get solutions for this chunk
        solutions = chunk_solutions.get(chunk_idx, [])
        if not solutions:
            print(f"Warning: No solutions found for chunk {chunk_idx}, skipping")
            continue
        
        # Compute average accuracy (is_correct = True means accuracy = 1.0)
        accuracies = []
        for solution in solutions:
            if isinstance(solution, dict):
                is_correct = solution.get('is_correct', False)
                accuracies.append(1.0 if is_correct else 0.0)
        
        if not accuracies:
            print(f"Warning: No valid accuracy values for chunk {chunk_idx}, skipping")
            continue
        
        avg_accuracy = np.mean(accuracies)
        
        # Print info for debugging (only first few chunks)
        if len(kl_values) < 3:
            print(f"  Chunk {chunk_idx}: KL={kl_value:.4f}, {len(accuracies)} solutions, avg_acc={avg_accuracy:.4f}")
        
        kl_values.append(kl_value)
        avg_accuracy_values.append(avg_accuracy)
        chunk_indices.append(chunk_idx)
    
    if not kl_values:
        print(f"Error: No valid data points to plot for {problem_id}")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(kl_values, avg_accuracy_values, alpha=0.6, s=100)
    
    # Add labels
    plt.xlabel(f'KL Importance ({kl_metric})', fontsize=12)
    plt.ylabel('Average Accuracy (over forced solutions)', fontsize=12)
    plt.title(f'KL Importance vs Average Accuracy\nProblem: {problem_id}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add chunk index annotations (optional - can be commented out if too cluttered)
    # for i, (kl, acc, idx) in enumerate(zip(kl_values, avg_accuracy_values, chunk_indices)):
    #     plt.annotate(f'C{idx}', (kl, acc), fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"{problem_id}_kl_vs_accuracy.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_filename}")
    
    # Also show the plot
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary for {problem_id}:")
    print(f"  Total chunks plotted: {len(kl_values)}")
    print(f"  KL importance range: [{min(kl_values):.4f}, {max(kl_values):.4f}]")
    print(f"  Average accuracy range: [{min(avg_accuracy_values):.4f}, {max(avg_accuracy_values):.4f}]")
    print(f"  Correlation: {np.corrcoef(kl_values, avg_accuracy_values)[0,1]:.4f}")

def main(problem_id=None, max_problems=10, kl_metric="resampling_importance_kl"):
    """
    Main function to plot KL vs accuracy for a problem.
    
    Args:
        problem_id: Specific problem ID to plot (e.g., "problem_330")
                   If None, will use the first problem found
        max_problems: Number of problems to load (only used if problem_id is None)
        kl_metric: Which KL metric to use
    """
    print("Loading dataset...")
    dataset = load_llama8b_dataset(max_problems=max_problems)
    
    if problem_id is None:
        # Find the first problem ID
        print("Finding first problem...")
        for example in dataset:
            path = example.get("path", "")
            if "problem.json" in path:
                problem_id = extract_problem_id_from_path(path)
                if problem_id:
                    print(f"Using problem: {problem_id}")
                    break
        
        if problem_id is None:
            print("Error: Could not find any problem in the dataset")
            return
    
    print(f"\nLoading data for {problem_id} (forced_answer version)...")
    problem_data = load_problem_data(dataset, problem_id, use_forced_answer=True)
    
    # Check if we have the required data
    if problem_data['chunks_labeled'] is None:
        print(f"Error: Could not load chunks_labeled.json for {problem_id}")
        return
    
    if not problem_data['chunk_solutions']:
        print(f"Error: Could not find any solutions.json files for {problem_id}")
        print(f"Note: Make sure you're using the 'forced_answer' version of the dataset")
        return
    
    print(f"\nFound {len(problem_data['chunks_labeled'])} chunks")
    print(f"Found solutions for {len(problem_data['chunk_solutions'])} chunks")
    
    # Create the plot
    plot_kl_vs_accuracy(problem_data, problem_id, kl_metric=kl_metric)

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    problem_id = None
    kl_metric = "resampling_importance_kl"
    
    if len(sys.argv) > 1:
        problem_id = sys.argv[1]
    
    if len(sys.argv) > 2:
        kl_metric = sys.argv[2]
        valid_metrics = ["resampling_importance_kl", "counterfactual_importance_kl", "forced_importance_kl"]
        if kl_metric not in valid_metrics:
            print(f"Warning: {kl_metric} not recognized. Using default: resampling_importance_kl")
            print(f"Valid options: {', '.join(valid_metrics)}")
            kl_metric = "resampling_importance_kl"
    
    main(problem_id=problem_id, kl_metric=kl_metric)

