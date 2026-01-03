#!/usr/bin/env python3
"""
Fix chunks by comparing chunks_to_flip.txt with all_chunks_plain.txt.
Preserves all chunks from all_chunks_plain.txt in order.
Adds unchunks from chunks_to_flip.txt in the order they appear there.
"""

import re

def parse_chunks_file(filename):
    """Parse chunks from all_chunks_plain.txt format: CHUNK XXX:"""
    chunks = {}
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'CHUNK\s+(\d+):\s*\n(.*?)(?=\nCHUNK\s+\d+:|$)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        chunk_idx = int(match.group(1))
        chunk_text = match.group(2).strip()
        chunks[chunk_idx] = chunk_text
    
    return chunks

def parse_flip_file(filename):
    """Parse chunks from chunks_to_flip.txt format: CHUNK_XXX:"""
    chunks = {}
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'CHUNK_(\d+):\s*\n(.*?)(?=\nCHUNK_\d+:|$)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        chunk_idx = int(match.group(1))
        chunk_text = match.group(2).strip()
        chunks[chunk_idx] = chunk_text
    
    return chunks

def normalize_text(text):
    """Normalize text for comparison"""
    text = ' '.join(text.split())
    return text.lower().strip()

def main():
    # Load chunks
    print("Loading chunks...")
    actual_chunks = parse_chunks_file('all_chunks_plain_incorrect.txt')
    flip_chunks = parse_flip_file('chunks_to_flip.txt')
    
    print(f"Actual chunks: {len(actual_chunks)} (indices: {min(actual_chunks.keys())} to {max(actual_chunks.keys())})")
    print(f"Flip chunks: {len(flip_chunks)} (indices: {min(flip_chunks.keys())} to {max(flip_chunks.keys())})")
    
    # Create normalized text mapping for actual chunks
    actual_normalized = {}
    for idx, text in actual_chunks.items():
        actual_normalized[idx] = normalize_text(text)
    
    # Step 1: Go through chunks_to_flip.txt in order and identify matches/unchunks
    # Track which flip chunks match which actual chunks
    flip_to_actual = {}  # flip_idx -> actual_idx (or None if no match)
    
    for flip_idx in sorted(flip_chunks.keys()):
        flip_text = flip_chunks[flip_idx]
        flip_normalized = normalize_text(flip_text)
        
        # Check if this matches any actual chunk
        matched_actual_idx = None
        for actual_idx, actual_norm in actual_normalized.items():
            # Exact match or substring match
            if flip_normalized == actual_norm or flip_normalized in actual_norm or actual_norm in flip_normalized:
                matched_actual_idx = actual_idx
                break
        
        flip_to_actual[flip_idx] = matched_actual_idx
    
    # Step 2: Identify unchunks and determine where they should be placed
    # unchunks_by_actual: actual_idx -> list of (flip_idx, text) in order
    unchunks_by_actual = {}
    
    last_matched_actual = None  # Track the last actual chunk we've seen
    
    for flip_idx in sorted(flip_chunks.keys()):
        matched_actual = flip_to_actual[flip_idx]
        
        if matched_actual is not None:
            # This flip chunk matches an actual chunk
            last_matched_actual = matched_actual
        else:
            # This is an unchunk - it should follow the last matched actual chunk
            if last_matched_actual is not None:
                if last_matched_actual not in unchunks_by_actual:
                    unchunks_by_actual[last_matched_actual] = []
                unchunks_by_actual[last_matched_actual].append((flip_idx, flip_chunks[flip_idx]))
            else:
                # Unchunk before any matched chunk - use -1
                if -1 not in unchunks_by_actual:
                    unchunks_by_actual[-1] = []
                unchunks_by_actual[-1].append((flip_idx, flip_chunks[flip_idx]))
    
    print(f"Unchunks found: {sum(len(v) for v in unchunks_by_actual.values())}")
    print(f"Unchunk groups: {len(unchunks_by_actual)}")
    
    # Step 3: Concatenate unchunks for each actual chunk
    concatenated_unchunks = {}
    for actual_idx, un_list in unchunks_by_actual.items():
        # un_list is already in order (from sorted flip_chunks.keys())
        concatenated_text = "\n\n".join([text for _, text in un_list])
        concatenated_unchunks[actual_idx] = concatenated_text
    
    # Step 4: Create output
    # Output all actual chunks in order, with unchunks inserted after them
    output_lines = []
    output_lines.append("="*80)
    output_lines.append("CHUNKS_FIXED: All chunks (actual + unchunks)")
    output_lines.append("="*80)
    output_lines.append("")
    
    # Handle unchunks before all chunks
    if -1 in concatenated_unchunks:
        output_lines.append("UNCHUNK -1:")
        output_lines.append(concatenated_unchunks[-1])
        output_lines.append("")
    
    # Output all actual chunks in order
    for actual_idx in sorted(actual_chunks.keys()):
        # Add actual chunk
        output_lines.append(f"CHUNK {actual_idx:03d}:")
        output_lines.append(actual_chunks[actual_idx])
        output_lines.append("")
        
        # Add unchunk if it follows this actual chunk
        if actual_idx in concatenated_unchunks:
            output_lines.append(f"UNCHUNK {actual_idx:03d}:")
            output_lines.append(concatenated_unchunks[actual_idx])
            output_lines.append("")
    
    # Write output
    with open('chunks_fixed.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n✓ Created chunks_fixed.txt")
    print(f"  Total lines: {len(output_lines)}")
    
    # Show summary
    actual_count = len([l for l in output_lines if l.startswith('CHUNK ')])
    unchunk_count = len([l for l in output_lines if l.startswith('UNCHUNK')])
    print(f"\nSummary:")
    print(f"  Actual chunks in output: {actual_count}")
    print(f"  Unchunk groups in output: {unchunk_count}")
    print(f"  Total entries: {actual_count + unchunk_count}")

if __name__ == '__main__':
    main()
