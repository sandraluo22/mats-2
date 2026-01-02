#!/usr/bin/env python3
"""
Utility script to load and work with extracted problem JSON files.

Usage:
    python3 load_json.py problem_1591.json                    # Load and show summary
    python3 load_json.py problem_1591.json --info             # Show detailed info
    python3 load_json.py problem_1591.json --count            # Count examples
    python3 load_json.py problem_1591.json --paths             # List all paths
    python3 load_json.py problem_1591.json --to-dataset        # Convert to Dataset format
    python3 load_json.py problem_1591.json --filter problem.json  # Filter by filename
"""

import sys
import json
import os
from datasets import Dataset

def load_json_file(json_path):
    """Load a JSON file and return the data."""
    print(f"Loading {json_path}...")
    if not os.path.exists(json_path):
        print(f"ERROR: File not found: {json_path}")
        return None
    
    file_size = os.path.getsize(json_path) / (1024**2)  # MB
    print(f"File size: {file_size:.1f} MB")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} examples")
    return data

def show_summary(data):
    """Show a summary of the loaded data."""
    if not data:
        print("No data to summarize")
        return
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total examples: {len(data)}")
    
    if len(data) > 0:
        print(f"\nFirst example keys: {list(data[0].keys())}")
        
        # Show first example path if available
        if 'path' in data[0]:
            print(f"\nFirst example path: {data[0]['path']}")
        
        # Count unique problem IDs
        problem_ids = set()
        for example in data:
            path = example.get('path', '')
            if 'problem_' in path:
                parts = path.split('/')
                for part in parts:
                    if part.startswith('problem_'):
                        problem_ids.add(part)
                        break
        
        if problem_ids:
            print(f"\nProblem IDs found: {sorted(problem_ids)}")

def show_info(data):
    """Show detailed information about the data."""
    if not data:
        return
    
    show_summary(data)
    
    print(f"\n{'='*70}")
    print("DETAILED INFO")
    print(f"{'='*70}")
    
    # Analyze structure
    all_keys = set()
    for example in data:
        all_keys.update(example.keys())
    
    print(f"\nAll keys in examples: {sorted(all_keys)}")
    
    # Show sample values for each key
    if len(data) > 0:
        print(f"\nSample values from first example:")
        for key in sorted(all_keys):
            value = data[0].get(key, 'N/A')
            if isinstance(value, str):
                if len(value) > 100:
                    print(f"  {key}: {value[:100]}... (truncated, {len(value)} chars)")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {type(value).__name__} ({value})")

def list_paths(data):
    """List all paths in the data."""
    if not data:
        return
    
    print(f"\n{'='*70}")
    print("PATHS")
    print(f"{'='*70}")
    
    paths = [example.get('path', 'N/A') for example in data]
    for i, path in enumerate(paths[:50], 1):  # Show first 50
        print(f"{i:4d}. {path}")
    
    if len(paths) > 50:
        print(f"\n... and {len(paths) - 50} more paths")

def filter_by_filename(data, filename):
    """Filter examples by filename in path."""
    if not data:
        return []
    
    filtered = [ex for ex in data if filename in ex.get('path', '')]
    print(f"\nFiltered to {len(filtered)} examples containing '{filename}' in path")
    return filtered

def convert_to_dataset(data, output_dir=None):
    """Convert JSON data to HuggingFace Dataset format."""
    if not data:
        print("No data to convert")
        return None
    
    if output_dir is None:
        # Infer from JSON filename
        json_path = sys.argv[1] if len(sys.argv) > 1 else "output"
        output_dir = json_path.replace('.json', '_dataset')
    
    print(f"\nConverting to Dataset format...")
    dataset = Dataset.from_list(data)
    
    print(f"Saving to {output_dir}...")
    dataset.save_to_disk(output_dir)
    print(f"✓ Saved Dataset to {output_dir}")
    print(f"  You can load it with: from datasets import load_from_disk; dataset = load_from_disk('{output_dir}')")
    
    return dataset

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    json_path = sys.argv[1]
    data = load_json_file(json_path)
    
    if data is None:
        sys.exit(1)
    
    # Handle different commands
    if '--info' in sys.argv:
        show_info(data)
    elif '--count' in sys.argv:
        print(f"\nTotal examples: {len(data)}")
    elif '--paths' in sys.argv:
        list_paths(data)
    elif '--to-dataset' in sys.argv:
        output_dir = None
        if '--output' in sys.argv:
            idx = sys.argv.index('--output')
            if idx + 1 < len(sys.argv):
                output_dir = sys.argv[idx + 1]
        convert_to_dataset(data, output_dir)
    elif '--filter' in sys.argv:
        idx = sys.argv.index('--filter')
        if idx + 1 < len(sys.argv):
            filename = sys.argv[idx + 1]
            filtered = filter_by_filename(data, filename)
            if filtered:
                # Save filtered results
                output_file = json_path.replace('.json', f'_filtered_{filename.replace(".", "_")}.json')
                with open(output_file, 'w') as f:
                    json.dump(filtered, f, indent=2)
                print(f"✓ Saved filtered data to {output_file}")
    else:
        # Default: show summary
        show_summary(data)
        print(f"\nUse --info for more details, --paths to list paths, --to-dataset to convert")

if __name__ == "__main__":
    main()

