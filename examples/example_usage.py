"""Example usage of the Benchmark Factory Multi-Agent Framework."""
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.task import Task
from src.models.benchmark import Benchmark, BenchmarkSample
from src.orchestrator import BenchmarkOrchestrator


def example_math_problem_transformation():
    """Example: Transforming math problem benchmarks."""
    
    # Define the task
    task = Task(
        description="Solve arithmetic word problems",
        constraints=[
            "Input must be a valid English word problem",
            "Problem must be solvable using basic arithmetic",
            "Numbers must be positive integers"
        ],
        input_space_definition="A string containing an English word problem that can be solved with addition, subtraction, multiplication, or division of positive integers.",
        output_space_definition="A single integer representing the solution to the problem."
    )
    
    # Create a benchmark
    benchmark = Benchmark(
        name="Math Word Problems",
        description="Simple arithmetic word problems",
        samples=[
            BenchmarkSample(
                input="John has 5 apples. He buys 3 more. How many apples does he have?",
                output=8,
                metadata={"operation": "addition", "complexity": "easy"}
            ),
            BenchmarkSample(
                input="Sarah has 10 cookies. She gives away 4. How many cookies are left?",
                output=6,
                metadata={"operation": "subtraction", "complexity": "easy"}
            ),
            BenchmarkSample(
                input="There are 4 boxes with 6 books each. How many books are there in total?",
                output=24,
                metadata={"operation": "multiplication", "complexity": "medium"}
            ),
        ]
    )
    
    # Initialize orchestrator
    orchestrator = BenchmarkOrchestrator(
        model="llama3.1",
        max_iterations=3,
        max_retries_per_sample=5
    )
    
    # Transform benchmark
    print("Starting benchmark transformation...")
    modified_inputs = orchestrator.transform_benchmark(
        benchmark=benchmark,
        task=task,
        num_outputs=5
    )
    
    # Display results
    print(f"\nGenerated {len(modified_inputs)} valid modified inputs:")
    for i, mod_input in enumerate(modified_inputs, 1):
        print(f"\n{i}. Transformation: {mod_input.transformation_applied}")
        print(f"   Original: {mod_input.original_input}")
        print(f"   Modified: {mod_input.modified_input}")
        print(f"   Complexity Score: {mod_input.complexity_score}")
    
    # Display statistics
    stats = orchestrator.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Valid transformations: {stats['valid_transformations']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")


def example_code_problem_transformation():
    """Example: Transforming coding problem benchmarks."""
    
    # Define the task
    task = Task(
        description="Generate valid Python code snippets",
        constraints=[
            "Code must be syntactically valid Python",
            "Code must be executable",
            "No import statements",
            "Use only built-in functions"
        ],
        input_space_definition="A string containing a Python code snippet that uses only built-in functions and is syntactically valid.",
        output_space_definition="The output of executing the code snippet."
    )
    
    # Create a benchmark
    benchmark = Benchmark(
        name="Python Code Problems",
        description="Simple Python coding problems",
        samples=[
            BenchmarkSample(
                input="def add(a, b):\n    return a + b",
                output="Function definition",
                metadata={"type": "function", "complexity": "easy"}
            ),
            BenchmarkSample(
                input="numbers = [1, 2, 3, 4, 5]\nresult = sum(numbers)",
                output=15,
                metadata={"type": "list_operation", "complexity": "easy"}
            ),
        ]
    )
    
    # Initialize orchestrator
    orchestrator = BenchmarkOrchestrator(
        model="llama3.1",
        max_iterations=2,
        max_retries_per_sample=3
    )
    
    # Transform benchmark
    print("Starting benchmark transformation...")
    modified_inputs = orchestrator.transform_benchmark(
        benchmark=benchmark,
        task=task,
        num_outputs=3
    )
    
    # Display results
    print(f"\nGenerated {len(modified_inputs)} valid modified inputs:")
    for i, mod_input in enumerate(modified_inputs, 1):
        print(f"\n{i}. Transformation: {mod_input.transformation_applied}")
        print(f"   Original: {mod_input.original_input}")
        print(f"   Modified: {mod_input.modified_input}")


if __name__ == "__main__":
    print("Example 1: Math Problem Transformation")
    print("=" * 50)
    example_math_problem_transformation()
    
    print("\n\nExample 2: Code Problem Transformation")
    print("=" * 50)
    example_code_problem_transformation()
