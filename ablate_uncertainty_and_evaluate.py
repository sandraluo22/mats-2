"""
Ablate uncertainty words from chunks in chain of thought and evaluate model performance.

For each chunk in the original COT:
1. Check if chunk contains uncertainty words
2. If yes, create an ablated version with all uncertainty words removed
3. Run the model with original chunks up to the ablated chunk, then the ablated chunk
4. Let the model generate the rest
5. Track metrics: final answer, correctness, sentence/token length, uncertainty words, full COT
6. Skip chunks that have no uncertainty words
"""

from load_dataset import load_llama8b_dataset
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, List, Optional, Tuple
import time

# Uncertainty indicators to track and ablate
UNCERTAINTY_WORDS = [
    "wait", "alternatively", "perhaps", "reconsider", "double-check", 
    "unlikely", "maybe", "might", "could be", "possibly", "doubt",
    "uncertain", "unsure", "hmm", "actually", "let me think"
]

class UncertaintyAblator:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device=None):
        """
        Initialize the model for uncertainty ablation and generating solutions.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu), auto-detected if None
        """
        print(f"Loading model: {model_name}")
        print("This may take a few minutes...")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        print(f"Model loaded on {device}")
    
    def ablate_uncertainty_words(self, chunk_text: str) -> Tuple[str, bool]:
        """
        Remove all uncertainty words from a chunk.
        
        Args:
            chunk_text: The original chunk text
        
        Returns:
            Tuple of (ablated_chunk_text, has_uncertainty_words)
        """
        chunk_lower = chunk_text.lower()
        has_uncertainty = False
        
        # Check if chunk has any uncertainty words
        for word in UNCERTAINTY_WORDS:
            if word.lower() in chunk_lower:
                has_uncertainty = True
                break
        
        if not has_uncertainty:
            return chunk_text, False
        
        # Remove uncertainty words (case-insensitive)
        ablated_text = chunk_text
        for word in UNCERTAINTY_WORDS:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(word) + r'\b'
            ablated_text = re.sub(pattern, '', ablated_text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        ablated_text = re.sub(r'\s+', ' ', ablated_text).strip()
        
        return ablated_text, True
    
    def solve_with_ablated_chunk(
        self, 
        problem: str, 
        original_chunks_dict: Dict[int, str], 
        chunk_idx: int, 
        ablated_chunk: str,
        max_tokens: int = 7000
    ) -> Dict:
        """
        Solve the problem using original COT up to the ablated chunk, then the ablated chunk.
        
        Args:
            problem: The problem statement
            original_chunks: List of original chunk texts
            chunk_idx: Index of the chunk that was ablated
            ablated_chunk: The uncertainty-ablated chunk
            max_tokens: Maximum tokens to generate
        
        Returns:
            Dictionary with solution details
        """
        # Build the prompt with original chunks up to the ablated chunk, then the ablated chunk
        # Don't provide chunks after the ablated chunk - let the model generate from there
        # Get all chunk indices sorted, and include all chunks with idx <= chunk_idx
        sorted_chunk_indices = sorted(original_chunks_dict.keys())
        cot_parts = []
        for orig_chunk_idx in sorted_chunk_indices:
            if orig_chunk_idx < chunk_idx:
                cot_parts.append(original_chunks_dict[orig_chunk_idx])  # Use original chunk
            elif orig_chunk_idx == chunk_idx:
                cot_parts.append(ablated_chunk)  # Use ablated chunk here
            else:
                break  # Stop after the ablated chunk
        
        # Combine into full chain of thought (only up to the ablated chunk)
        full_cot = " ".join(cot_parts)
        
        prompt = f"""Problem: {problem}

Reasoning step by step:
{full_cot}

Continue solving from here. Therefore, the final answer is \\boxed{{:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]
        max_new_tokens = min(max_tokens - input_length, 6000)
        
        if max_new_tokens <= 0:
            max_new_tokens = 100
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        solution_text = full_response[len(prompt):].strip()
        
        # Build full COT starting from the ablated chunk (ablated chunk + all generated text)
        full_cot_from_ablated = f"{ablated_chunk} {solution_text}".strip()
        
        # Count sentences (simple heuristic: split by periods)
        sentences = [s.strip() for s in solution_text.split('.') if s.strip()]
        sentence_count = len(sentences)
        
        # Count tokens
        token_count = len(self.tokenizer.encode(solution_text))
        
        # Count uncertainty words (case insensitive)
        uncertainty_count = 0
        uncertainty_occurrences = []
        solution_lower = solution_text.lower()
        for word in UNCERTAINTY_WORDS:
            count = solution_lower.count(word.lower())
            if count > 0:
                uncertainty_count += count
                uncertainty_occurrences.append(f"{word}:{count}")
        
        # Extract final answer (try to find boxed answer or last number)
        final_answer = self.extract_answer(solution_text)
        
        return {
            "full_cot": full_cot_from_ablated,  # Full COT starting from ablated chunk
            "sentence_count": sentence_count,
            "token_count": token_count,
            "uncertainty_word_count": uncertainty_count,
            "uncertainty_occurrences": uncertainty_occurrences,
            "final_answer": final_answer,
            "ablated_chunk_idx": chunk_idx,
            "input_token_count": input_length
        }
    
    def extract_answer(self, text: str) -> str:
        """Extract the final answer from the solution text."""
        # Try to find boxed answer
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, text)
        if match:
            return match.group(1)
        
        # Try to find answer at the end
        # Look for "answer is" or "final answer" patterns
        answer_patterns = [
            r'[Ff]inal answer[:\s]+([^\n.]+)',
            r'[Aa]nswer[:\s]+([^\n.]+)',
            r'[Tt]he answer is[:\s]+([^\n.]+)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Return last line if no pattern matches
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            return lines[-1]
        
        return "N/A"

def load_problem_data(dataset, problem_id: str):
    """
    Load problem data including original COT chunks (non-forced version).
    Gets the original chain of thought from incorrect_base_solution (not forced_answer).
    """
    problem_data = {
        'problem_json': None,
        'chunks_labeled': None,
    }
    
    problem_files = dataset.filter(
        lambda example: problem_id in example.get("path", "")
    )
    
    # Prefer non-forced version for original COT
    for example in problem_files:
        path = example.get("path", "")
        content = example.get("content", "")
        
        # Get problem.json (any version, but prefer non-forced)
        if "problem.json" in path:
            if problem_data['problem_json'] is None or "forced_answer" not in path.lower():
                try:
                    problem_data['problem_json'] = json.loads(content)
                except json.JSONDecodeError:
                    pass
        
        # Get chunks_labeled.json from incorrect_base_solution (original COT)
        elif "chunks_labeled.json" in path:
            # Prefer incorrect_base_solution over forced_answer for original COT
            if "incorrect_base_solution" in path and "forced_answer" not in path.lower():
                try:
                    problem_data['chunks_labeled'] = json.loads(content)
                except json.JSONDecodeError:
                    pass
            # Fallback: use any non-forced version if we haven't found one yet
            elif problem_data['chunks_labeled'] is None and "forced_answer" not in path.lower():
                try:
                    problem_data['chunks_labeled'] = json.loads(content)
                except json.JSONDecodeError:
                    pass
    
    return problem_data

def extract_chunks(chunks_labeled: List[Dict]) -> Tuple[Dict[int, str], List[int]]:
    """
    Extract chunk texts from chunks_labeled.json, using actual chunk_idx from data.
    
    Returns:
        Tuple of (dict mapping chunk_idx to chunk_text, sorted list of chunk_indices)
    """
    chunks_dict = {}
    chunk_indices = []
    
    for chunk_data in chunks_labeled:
        if isinstance(chunk_data, dict):
            chunk_idx = chunk_data.get('chunk_idx')
            chunk_text = chunk_data.get('chunk', '')
            if chunk_idx is not None and chunk_text:
                chunks_dict[chunk_idx] = chunk_text
                chunk_indices.append(chunk_idx)
    
    # Sort chunk indices
    chunk_indices = sorted(chunk_indices)
    return chunks_dict, chunk_indices

def check_correctness(final_answer: str, ground_truth_answer: str) -> bool:
    """Check if the final answer matches ground truth (loose matching)."""
    # Normalize: remove whitespace, convert to lowercase
    final_norm = re.sub(r'\s+', '', str(final_answer).lower())
    gt_norm = re.sub(r'\s+', '', str(ground_truth_answer).lower())
    
    # Try exact match
    if final_norm == gt_norm:
        return True
    
    # Try to extract numbers and compare
    final_numbers = re.findall(r'\d+\.?\d*', final_norm)
    gt_numbers = re.findall(r'\d+\.?\d*', gt_norm)
    
    if final_numbers and gt_numbers:
        return final_numbers[-1] == gt_numbers[-1]
    
    return False

def run_experiment(problem_id: str, output_file: str = "ablate_uncertainty_results.json"):
    """
    Run the full experiment: ablate uncertainty words and evaluate.
    
    Args:
        problem_id: Problem ID to use
        output_file: File to save results
    """
    print("="*80)
    print("UNCERTAINTY ABLATION EXPERIMENT")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset and problem {problem_id}...")
    # Try to load from local saved dataset first (faster, no network needed)
    import os
    local_dataset_path = f"{problem_id}_dataset"
    if os.path.exists(local_dataset_path):
        print(f"Loading from local dataset: {local_dataset_path}")
        from datasets import load_from_disk
        dataset = load_from_disk(local_dataset_path)
    else:
        print(f"Local dataset not found, downloading via streaming...")
        # Load only the specific problem ID to save time and space
        dataset = load_llama8b_dataset(problem_ids=[problem_id])
    problem_data = load_problem_data(dataset, problem_id)
    
    if not problem_data['problem_json']:
        print(f"Error: Could not load problem.json for {problem_id}")
        return
    
    if not problem_data['chunks_labeled']:
        print(f"Error: Could not load chunks_labeled.json for {problem_id}")
        return
    
    problem_text = problem_data['problem_json'].get('problem', '')
    ground_truth = problem_data['problem_json'].get('gt_answer', '')
    
    print(f"\nProblem: {problem_text[:200]}...")
    print(f"Ground truth answer: {ground_truth}")
    
    # Extract chunks (using actual chunk_idx from data)
    chunks_dict, chunk_indices = extract_chunks(problem_data['chunks_labeled'])
    print(f"\nFound {len(chunks_dict)} chunks in original COT (indices: {min(chunk_indices)} to {max(chunk_indices)})")
    
    # Initialize model
    ablator = UncertaintyAblator()
    
    # Store results
    results = {
        "problem_id": problem_id,
        "problem": problem_text,
        "ground_truth_answer": ground_truth,
        "num_chunks": len(chunks_dict),
        "original_chunks": [chunks_dict[idx] for idx in chunk_indices],  # Keep as list for backward compatibility
        "experiments": []
    }
    
    # Process each chunk using actual chunk_idx
    chunks_with_uncertainty = 0
    for chunk_idx in chunk_indices:
        print(f"\n{'='*80}")
        print(f"Processing chunk {chunk_indices.index(chunk_idx) + 1}/{len(chunk_indices)} (chunk_idx: {chunk_idx})")
        print(f"{'='*80}")
        print(f"Original chunk: {chunks_dict[chunk_idx][:150]}...")
        
        # Step 1: Ablate uncertainty words
        print("\nStep 1: Ablating uncertainty words...")
        ablated_chunk, has_uncertainty = ablator.ablate_uncertainty_words(chunks_dict[chunk_idx])
        
        if not has_uncertainty:
            print("  No uncertainty words found in this chunk. Skipping.")
            continue
        
        chunks_with_uncertainty += 1
        print(f"Ablated chunk: {ablated_chunk[:150]}...")
        
        # Step 2: Solve with ablated chunk
        print("\nStep 2: Solving with ablated chunk...")
        try:
            solution_result = ablator.solve_with_ablated_chunk(
                problem_text,
                chunks_dict,
                chunk_idx,
                ablated_chunk,
                max_tokens=7000
            )
            
            # Check correctness
            is_correct = check_correctness(
                solution_result['final_answer'],
                ground_truth
            )
            solution_result['is_correct'] = is_correct
            
            solution_result['ablated_chunk'] = ablated_chunk
            solution_result['original_chunk'] = chunks_dict[chunk_idx]
            
            results['experiments'].append(solution_result)
            
            print(f"\nResults for chunk {chunk_idx}:")
            print(f"  Final answer: {solution_result['final_answer']}")
            print(f"  Correct: {is_correct}")
            print(f"  Sentence count: {solution_result['sentence_count']}")
            print(f"  Token count: {solution_result['token_count']}")
            print(f"  Uncertainty words: {solution_result['uncertainty_word_count']}")
            
            # Save intermediate results
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_file}")
            
        except Exception as e:
            print(f"Error solving with ablated chunk: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"\nProcessed {chunks_with_uncertainty} chunks with uncertainty words (out of {len(chunks)} total chunks)")
    print(f"Results saved to: {output_file}")
    
    # Summary of final answers
    print(f"\n{'='*80}")
    print("FINAL ANSWERS SUMMARY")
    print(f"{'='*80}")
    print(f"Ground truth answer: {ground_truth}")
    print(f"\nChunk | Final Answer | Correct")
    print("-" * 80)
    for exp in results['experiments']:
        chunk_idx = exp.get('ablated_chunk_idx', 'N/A')
        final_answer = exp.get('final_answer', 'N/A')
        is_correct = exp.get('is_correct', False)
        correct_marker = "✓" if is_correct else "✗"
        print(f"  {chunk_idx:4d} | {final_answer:20s} | {correct_marker}")
    
    # Count correct vs incorrect
    correct_count = sum(1 for exp in results['experiments'] if exp.get('is_correct', False))
    incorrect_count = len(results['experiments']) - correct_count
    print(f"\nSummary: {correct_count} correct, {incorrect_count} incorrect out of {len(results['experiments'])} experiments")
    print(f"{'='*80}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ablate_uncertainty_and_evaluate.py <problem_id> [output_file]")
        print("Example: python ablate_uncertainty_and_evaluate.py problem_1591")
        print("Example: python ablate_uncertainty_and_evaluate.py problem_1591 results.json")
        print("\nNote: This script requires:")
        print("  - GPU with at least 16GB VRAM (for 8B model)")
        print("  - Will take significant time (minutes to hours depending on number of chunks)")
        print("  - Results are saved incrementally as each chunk is processed")
        print("  - Only processes chunks that contain uncertainty words")
        sys.exit(1)
    
    problem_id = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "ablate_uncertainty_results.json"
    
    print("\nWARNING: This is computationally intensive!")
    print("  - Model: DeepSeek-R1-Distill-Llama-8B (~16GB GPU memory)")
    print("  - Time: ~2-5 minutes per chunk with uncertainty words")
    print("  - Results saved incrementally to:", output_file)
    print()
    
    run_experiment(problem_id, output_file)

