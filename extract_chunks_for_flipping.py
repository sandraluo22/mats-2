"""
Extract all chunks from flip_chunk_results.json and output them in a format
that's easy to copy-paste for manual flipping.
"""
import json

def extract_chunks_for_flipping(results_file="flip_chunk_results.json"):
    """Extract chunks and output them numbered for easy copy-paste."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    chunks = data.get('original_chunks', [])
    
    print("="*80)
    print(f"CHUNKS FOR PROBLEM: {data.get('problem_id', 'unknown')}")
    print(f"Total chunks: {len(chunks)}")
    print("="*80)
    print("\nCopy the chunks below, flip them, and paste them back in the same format.\n")
    print("="*80)
    
    for i, chunk in enumerate(chunks):
        print(f"CHUNK_{i:03d}:")
        print(chunk)
        print()
    
    print("="*80)
    print("END OF CHUNKS")
    print("="*80)

if __name__ == "__main__":
    import sys
    results_file = sys.argv[1] if len(sys.argv) > 1 else "flip_chunk_results.json"
    extract_chunks_for_flipping(results_file)

