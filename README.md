# AgenticBenchmarks Multi-Agent Framework

A multi-agent framework designed to extend the lifespan and reliability of static benchmarks by dynamically modifying them. The framework generates inputs that are similar in complexity but different in syntax and semantics, ensuring that model performance assessment reflects actual capabilities rather than memorization or data leakage.

## Features

- **Multi-Agent Architecture**: Specialized agents for each step of the transformation process
- **Local Model Support**: Uses Ollama for local LLM inference to prevent data leakage
- **Iterative Refinement**: Feedback loops for continuous improvement of transformations
- **Complexity Preservation**: Ensures generated inputs maintain similar complexity levels
- **Constraint Enforcement**: Validates all transformations respect task input space

## Architecture

The framework implements the following methodology:

1. **Sampling**: Selects representative samples from the benchmark
2. **Analysis**: Analyzes benchmark patterns and enhances task constraints
3. **Transformation Rule Design**: Designs rules for generating modified inputs
4. **Transformation Application**: Applies rules to generate syntactically different inputs
5. **Validation**: Checks constraints, input space validity, and complexity matching

With feedback loops connecting:
- Validation → Analysis (refine constraints)
- Validation → Transformation Rule Design (improve rules)
- Analysis → Transformation Rule Design (inform rule creation)

## Installation

### Prerequisites

1. Install [Ollama](https://ollama.ai/) (Optional)
2. Pull a model (e.g., `ollama pull llama3.1`)
3. Python 3.8 or higher

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify Ollama is running
ollama list

# Verify setup (optional)
python scripts/verify_setup.py
```

## Quick Start

```python
from src.models.task import Task
from src.models.benchmark import Benchmark, BenchmarkSample
from src.orchestrator import BenchmarkOrchestrator

# Define your task
task = Task(
    description="Your task description",
    constraints=["constraint1", "constraint2"],
    input_space_definition="Definition of valid input space",
    output_space_definition="Definition of valid output space"
)

# Create your benchmark
benchmark = Benchmark(
    name="My Benchmark",
    samples=[
        BenchmarkSample(input="sample1", output="output1"),
        BenchmarkSample(input="sample2", output="output2"),
    ]
)

# Initialize orchestrator
orchestrator = BenchmarkOrchestrator(
    model="llama3.1",  # Change to your preferred Ollama model
    max_iterations=3,
    max_retries_per_sample=5
)

# Transform benchmark
modified_inputs = orchestrator.transform_benchmark(
    benchmark=benchmark,
    task=task,
    num_outputs=10
)

# Use the modified inputs
for mod_input in modified_inputs:
    print(f"Original: {mod_input.original_input}")
    print(f"Modified: {mod_input.modified_input}")
    print(f"Transformation: {mod_input.transformation_applied}")
```

## Components

### Agents

- **SamplingAgent**: Selects samples from benchmark
- **AnalysisAgent**: Analyzes patterns and enhances constraints
- **TransformationRuleAgent**: Designs transformation rules
- **TransformationAgent**: Applies transformations
- **ValidationAgent**: Validates generated inputs

### Models

- **Task**: Defines task constraints and input/output spaces
- **Benchmark**: Container for benchmark samples
- **BenchmarkSample**: Individual benchmark sample
- **ModifiedInput**: Generated modified input with metadata

## Configuration

You can customize the framework behavior:

```python
from src.config import FrameworkConfig, OllamaConfig

config = FrameworkConfig(
    ollama_config=OllamaConfig(
        model="llama3.1",
        base_url="http://localhost:11434"
    ),
    sampling_strategy="diverse",
    max_iterations=5,
    max_retries_per_sample=10
)
```

## Examples

See `examples/example_usage.py` for complete examples including:
- Math problem transformation
- Code problem transformation

## Methodology

The framework implements a cyclical process:

1. **Benchmark** → **Sampling**: Extract representative samples
2. **Task** + **Sampling** → **Analysis**: Enhance constraints and input space definition
3. **Analysis** → **Transformation Rule Design**: Create transformation rules
4. **Transformation Rule** → **Application**: Generate modified inputs
5. **Application** → **Validation**: Check validity and complexity
6. **Validation** → **Analysis** (feedback loop): Refine constraints
7. **Validation** → **Transformation Rule** (feedback loop): Improve rules

## Requirements

- Python 3.11+ (required for BESSER framework)
- `besser-agentic-framework[llms,extras]>=4.0.0` (installed via requirements.txt)
- Ollama installed and running (for Ollama backend)
- A compatible Ollama model (e.g., llama3.1, mistral, etc.) OR HuggingFace model (for HuggingFace backend)

## BESSER Integration

The framework is now fully integrated with the [BESSER Agentic Framework](https://besser-agentic-framework.readthedocs.io/latest/). All agents can use either:
- **Ollama backend** (default): Uses Ollama directly for LLM calls
- **HuggingFace backend**: Uses BESSER's HuggingFace LLM integration

### Basic Usage with BESSER

The framework automatically integrates with BESSER when available. You can use it in two ways:

#### Option 1: Direct Usage (Automatic BESSER Integration)

```python
from src.orchestrator import BenchmarkOrchestrator

# Uses Ollama backend by default, but integrates with BESSER if available
orchestrator = BenchmarkOrchestrator(
    model="llama3.1",
    llm_backend="ollama"  # or "huggingface" for HuggingFace models
)
```

#### Option 2: Full BESSER Integration with Agent Wrapping

```python
from src.utils.besser_integration import create_besser_orchestrator, get_besser_availability

# Check if BESSER is available
if get_besser_availability():
    # Create orchestrator with BESSER agent wrappers
    orchestrator = create_besser_orchestrator(
        model="llama3.1",
        llm_backend="ollama",  # or "huggingface"
        use_besser_agents=True
    )
else:
    # Fallback to standard orchestrator
    from src.orchestrator import BenchmarkOrchestrator
    orchestrator = BenchmarkOrchestrator(model="llama3.1")
```

### Using HuggingFace Models via BESSER

```python
from src.orchestrator import BenchmarkOrchestrator

# Use HuggingFace models through BESSER
orchestrator = BenchmarkOrchestrator(
    model="microsoft/DialoGPT-medium",  # HuggingFace model ID
    llm_backend="huggingface"
)
```

### Features

- **Automatic fallback**: If BESSER is not available, the framework falls back to direct Ollama calls
- **LLM abstraction**: All agents use a unified LLM interface that works with both backends
- **BESSER agent wrapping**: Optional wrapping of agents with BESSER's Agent class for state management
- **Multi-agent support**: When using BESSER wrappers, agents can leverage BESSER's multi-agent coordination features

## Contributing

This framework is designed to be extensible. You can:
- Add new transformation strategies
- Implement custom validation logic
- Extend agents with domain-specific knowledge
- Add support for different LLM providers
- Enhance BESSER integration

## License

[Add your license here]

## Citation

If you use this framework in your research, please cite:

[Add citation information]
