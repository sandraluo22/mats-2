"""
Quick script to get the first problem ID from the dataset using streaming.
This is much faster than downloading everything - just gets the ID.
"""

from datasets import load_dataset

print("Getting first problem ID from dataset (streaming mode)...")
print("This will be fast - we only need to find the first llama-8b problem.\n")

dataset = load_dataset("uzaymacar/math-rollouts", streaming=True)

# Handle DatasetDict or single Dataset
if isinstance(dataset, dict):
    split_data = list(dataset.values())[0]
else:
    split_data = dataset

print("Scanning for first llama-8b problem...")
problem_id = None
count = 0

for example in split_data:
    count += 1
    path = example.get("path", "")
    
    # Filter for llama-8b only
    if "llama-8b" not in path.lower():
        continue
    
    # Look for problem ID
    if "problem_" in path:
        parts = path.split("/")
        for part in parts:
            if part.startswith("problem_"):
                problem_id = part
                break
        
        if problem_id:
            print(f"\n✓ Found first problem ID: {problem_id}")
            print(f"  (Scanned {count} examples to find it)")
            break
    
    if count % 1000 == 0:
        print(f"  Scanned {count} examples...", end='\r')

if problem_id:
    print(f"\n\nFirst problem ID: {problem_id}")
    print(f"\nYou can now use this in your scripts:")
    print(f"  from load_dataset import load_llama8b_dataset")
    print(f"  dataset = load_llama8b_dataset(problem_ids=['{problem_id}'])")
else:
    print("\n✗ Could not find any problem ID")

