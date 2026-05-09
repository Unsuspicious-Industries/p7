#!/usr/bin/env python3
"""
Phi-3.5-mini demo: ReasoningEnvironment with CoT + Grammar-constrained output.

Demonstrates the new ReasoningEnvironment that allows the model to:
1. Think freely in <think>...</think> blocks (unconstrained CoT)
2. Produce typed output in <formal>...</formal> blocks (grammar-constrained)

Uses Microsoft's Phi-3.5-mini-instruct.
https://huggingface.co/microsoft/Phi-3.5-mini-instruct
"""

import os
import sys

# Ensure the local `src/` package is importable when running examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import proposition7


# Tasks for the environment to solve
TASKS = [
    {
        "name": "identity",
        "task": "Create the identity function for Int",
        "description": "Should produce λx:Int.x",
    },
    {
        "name": "const_K",
        "task": "Create the K combinator (constant function) that takes an Int, then a Bool, and returns the Int",
        "description": "Should produce λx:Int.λy:Bool.x",
    },
    {
        "name": "apply",
        "task": "Create a function that applies a function f:(Int->Bool) to an argument x:Int",
        "description": "Should produce λf:(Int->Bool).λx:Int.(f x)",
    },
    {
        "name": "compose",
        "task": "Create the B combinator (function composition) for f:(Int->Bool) and g:(Bool->Int)",
        "description": "Should produce something like λf:(Int->Bool).λg:(Bool->Int).λx:Bool.(f (g x))",
    },
    {
        "name": "twice",
        "task": "Create a function that applies f:(Int->Int) twice to x:Int",
        "description": "Should produce λf:(Int->Int).λx:Int.(f (f x))",
    },
]


def print_header(text: str, char: str = "=", width: int = 80):
    print(char * width)
    print(text)
    print(char * width)


def run_environment_demo(env: proposition7.ReasoningEnvironment):
    """Run the ReasoningEnvironment demo with all tasks."""
    
    results = []
    
    for task in TASKS:
        print_header(f"Task: {task['name']}", "=")
        print(f"  Description: {task['description']}")
        print(f"  Task:        {task['task']}")
        print()
        
        try:
            result = env.generate(prompt=task["task"])
            
            print("/".join([result.think_blocks[i].content for i in range(len(result.think_blocks))]))
            
            if result.final_output:
                print(f"\n  Final output:  {result.final_output.content}")
            
            results.append({
                "name": task["name"],
                "complete": result.is_complete,
                "output": result.final_output.content if result.final_output else None,
                "tokens": result.total_tokens,
            })
            
        except Exception as e:
            print(f"\n  ERROR: {e}")
            results.append({
                "name": task["name"],
                "complete": False,
                "output": None,
                "tokens": 0,
                "error": str(e),
            })
        
        print("\n")
    
    return results


def main():
    print_header("proposition7 ReasoningEnvironment Demo with Phi-3.5-mini", "#", 80)
    
    model_name = "Qwen/Qwen3.5-9B"
    grammar_name = "stlc"
    
    print(f"\nLoading {model_name}...")
    print(f"Grammar: {grammar_name}")
    print()
    
    # Show the auto-generated system prompt
    system_prompt = proposition7.build_system_prompt(grammar_name)
    print_header("Auto-generated System Prompt", "-", 60)
    print(system_prompt)
    print()
    
    # Load model
    model = proposition7.ConstrainedModel.from_pretrained(
        model_name,
        grammar=proposition7.get_grammar(grammar_name),
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    print()
    
    # Create reasoning environment
    env = proposition7.ReasoningEnvironment(
        model=model,
        grammar_name=grammar_name,
        think_budget=1024,
        formal_budget=50,
    )
    
    # Run main demo
    print_header("ReasoningEnvironment Demo", "#")
    print("CoT reasoning with <think> blocks, typed output with <formal> blocks\n")
    
    results = run_environment_demo(env)
    
    # Summary
    print_header("SUMMARY", "=")
    print(f"\n{'Task':<15} {'Complete':<10} {'Output':<40}")
    print("-" * 65)
    for r in results:
        output = r['output'][:35] + "..." if r['output'] and len(r['output']) > 35 else (r['output'] or "N/A")
        print(f"{r['name']:<15} {str(r['complete']):<10} {output:<40}")
    
    complete = sum(1 for r in results if r['complete'])
    print("-" * 65)
    print(f"Complete: {complete}/{len(results)}")
    




if __name__ == "__main__":
    main()
