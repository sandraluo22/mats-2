"""
Flip chunks in chain of thought and evaluate model performance.

For each chunk in the original COT:
1. Generate a "flipped" incorrect version (with arithmetic/logical errors)
2. Run the model with all original chunks except the current one replaced with the flipped version
3. Track metrics: final answer, correctness, sentence/token length, uncertainty words, full COT
"""

from load_dataset import load_llama8b_dataset
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, List, Optional
import time

# Uncertainty indicators to track
UNCERTAINTY_WORDS = [
    "wait", "alternatively", "perhaps", "reconsider", "double-check", 
    "unlikely", "maybe", "might", "could be", "possibly", "doubt",
    "uncertain", "unsure", "hmm", "actually", "let me think"
]

class ChunkFlipper:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device=None):
        """
        Initialize the model for flipping chunks and generating solutions.
        
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
    
    def flip_chunk(self, chunk_text: str, problem_context: str) -> str:
        """
        Generate a "flipped" incorrect version of a chunk.
        Introduces small arithmetic/logical errors.
        
        Args:
            chunk_text: The original chunk text to flip
            problem_context: The problem statement for context
        
        Returns:
            Flipped chunk text with errors introduced
        """
        prompt = f"""You are given a mathematical problem and a reasoning step. 
Your task is to create a slightly incorrect version of this reasoning step by introducing a small error:
- If it's arithmetic, introduce a small calculation mistake (like misplacing a number or sign)
- If it's logical, introduce an incorrect logical step or conclusion
- Keep the overall structure and style similar

Problem: {problem_context}

Original reasoning step: {chunk_text}

Generate a slightly incorrect version of this reasoning step (introduce a subtle error):"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the flipped chunk (remove the prompt)
        flipped_chunk = response[len(prompt):].strip()
        
        return flipped_chunk
    
    def solve_with_flipped_chunk(
        self, 
        problem: str, 
        original_chunks: List[str], 
        chunk_idx: int, 
        flipped_chunk: str,
        max_tokens: int = 7000
    ) -> Dict:
        """
        Solve the problem using original COT but with one chunk flipped.
        
        Args:
            problem: The problem statement
            original_chunks: List of original chunk texts
            chunk_idx: Index of the chunk to replace with flipped version
            flipped_chunk: The flipped incorrect chunk
            max_tokens: Maximum tokens to generate
        
        Returns:
            Dictionary with solution details
        """
        # Build the prompt with original chunks except the flipped one
        cot_parts = []
        for i, chunk in enumerate(original_chunks):
            if i == chunk_idx:
                cot_parts.append(flipped_chunk)  # Use flipped chunk here
            else:
                cot_parts.append(chunk)  # Use original chunks elsewhere
        
        # Combine into full chain of thought (with flipped chunk inserted)
        full_cot = " ".join(cot_parts)
        
        prompt = f"""Problem: {problem}

Reasoning step by step:
{full_cot}

Continue solving from here and provide the final answer:"""
        
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
            "full_cot": solution_text,
            "sentence_count": sentence_count,
            "token_count": token_count,
            "uncertainty_word_count": uncertainty_count,
            "uncertainty_occurrences": uncertainty_occurrences,
            "final_answer": final_answer,
            "flipped_chunk_idx": chunk_idx,
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
    Gets the original chain of thought from correct_base_solution (not forced_answer).
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
        
        # Get chunks_labeled.json from correct_base_solution (original COT)
        elif "chunks_labeled.json" in path:
            # Prefer correct_base_solution over forced_answer for original COT
            if "correct_base_solution" in path and "forced_answer" not in path.lower():
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

def extract_chunks(chunks_labeled: List[Dict]) -> List[str]:
    """Extract chunk texts from chunks_labeled.json."""
    chunks = []
    for chunk_data in sorted(chunks_labeled, key=lambda x: x.get('chunk_idx', 0)):
        if isinstance(chunk_data, dict):
            chunk_text = chunk_data.get('chunk', '')
            if chunk_text:
                chunks.append(chunk_text)
    return chunks

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

def run_experiment(problem_id: str, output_file: str = "flip_chunk_results.json"):
    """
    Run the full experiment: flip chunks and evaluate.
    
    Args:
        problem_id: Problem ID to use
        output_file: File to save results
    """
    print("="*80)
    print("FLIP CHUNK EXPERIMENT")
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
    
    # Extract chunks
    chunks = extract_chunks(problem_data['chunks_labeled'])
    print(f"\nFound {len(chunks)} chunks in original COT")
    
    # Initialize model
    flipper = ChunkFlipper()
    
    # Store results
    results = {
        "problem_id": problem_id,
        "problem": problem_text,
        "ground_truth_answer": ground_truth,
        "num_chunks": len(chunks),
        "original_chunks": chunks,
        "experiments": []
    }
    
    # Process each chunk
    for chunk_idx in range(len(chunks)):
        print(f"\n{'='*80}")
        print(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
        print(f"{'='*80}")
        print(f"Original chunk: {chunks[chunk_idx][:150]}...")
        
        # Step 1: Flip the chunk
        print("\nStep 1: Generating flipped chunk...")
        try:
            flipped_chunk = flipper.flip_chunk(chunks[chunk_idx], problem_text)
            print(f"Flipped chunk: {flipped_chunk[:150]}...")
        except Exception as e:
            print(f"Error flipping chunk: {e}")
            continue
        
        # Step 2: Solve with flipped chunk
        print("\nStep 2: Solving with flipped chunk...")
        try:
            solution_result = flipper.solve_with_flipped_chunk(
                problem_text,
                chunks,
                chunk_idx,
                flipped_chunk,
                max_tokens=7000
            )
            
            # Check correctness
            is_correct = check_correctness(
                solution_result['final_answer'],
                ground_truth
            )
            solution_result['is_correct'] = is_correct
            
            solution_result['flipped_chunk'] = flipped_chunk
            solution_result['original_chunk'] = chunks[chunk_idx]
            
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
            print(f"Error solving with flipped chunk: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"\nProcessed {len(results['experiments'])} chunks")
    print(f"Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python flip_chunks_and_evaluate.py <problem_id> [output_file]")
        print("Example: python flip_chunks_and_evaluate.py problem_330")
        print("\nNote: This script requires:")
        print("  - GPU with at least 16GB VRAM (for 8B model)")
        print("  - Will take significant time (minutes to hours depending on number of chunks)")
        print("  - Results are saved incrementally as each chunk is processed")
        sys.exit(1)
    
    problem_id = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "flip_chunk_results.json"
    
    print("\nWARNING: This is computationally intensive!")
    print("  - Model: DeepSeek-R1-Distill-Llama-8B (~16GB GPU memory)")
    print("  - Time: ~2-5 minutes per chunk")
    print("  - Results saved incrementally to:", output_file)
    print()
    
    run_experiment(problem_id, output_file)

