# AgenticBenchmarks Multi-Agent Framework

A multi-agent framework designed to extend the lifespan and reliability of static benchmarks by dynamically modifying them. The framework generates inputs that are similar in complexity but different in syntax and semantics, ensuring that model performance assessment reflects actual capabilities rather than memorization or data leakage.

## Features

- **Multi-Agent Architecture**: Specialized agents for each step of the transformation process
- **Multiple LLM Backends**: Support for Ollama (local), HuggingFace (via BESSER), and OpenRouter (cloud)
- **Parallel Processing**: Built-in parallel processing for faster benchmark transformation
- **Configuration-Based**: YAML configuration file for easy setup and customization
- **Iterative Refinement**: Feedback loops for continuous improvement of transformations
- **Complexity Preservation**: Ensures generated inputs maintain similar complexity levels
- **Constraint Enforcement**: Validates all transformations respect task input space
- **Predefined Tasks**: Ready-to-use task definitions for common benchmark types
- **Dataset Support**: Load datasets from HuggingFace or local JSON files

## Architecture

The framework implements the following methodology:

1. **Sampling**: Selects representative samples from the benchmark
2. **Analysis**: Analyzes benchmark patterns and enhances task constraints
3. **Transformation Rule Design**: Designs rules for generating modified inputs
4. **Transformation Application**: Applies rules to generate syntactically different inputs
5. **Output Update**: Determines if outputs should change based on transformations
6. **Validation**: Checks constraints, input space validity, and complexity matching

With feedback loops connecting:
- Validation → Analysis (refine constraints)
- Validation → Transformation Rule Design (improve rules)
- Analysis → Transformation Rule Design (inform rule creation)

## Installation

### Prerequisites

- Python 3.8 or higher (Python 3.11+ recommended for BESSER framework)
- For Ollama backend: Install [Ollama](https://ollama.ai/) and pull a model (e.g., `ollama pull llama3.1`)
- For OpenRouter backend: Get an API key from [OpenRouter](https://openrouter.ai/)

### Setup

```bash
# Clone the repository (if applicable)
# cd BenchFactory_MA

# Install dependencies
pip install -r requirements.txt

# Verify Ollama is running (if using Ollama backend)
ollama list

# Verify setup (optional)
python scripts/verify_setup.py
```

### Dependencies

- `besser-agentic-framework[llms,extras]>=4.0.0` - BESSER framework integration
- `ollama>=0.1.0` - Ollama client library
- `pyyaml>=6.0.0` - YAML configuration parsing
- `datasets>=2.0.0` - HuggingFace datasets support
- `openai>=1.0.0` - OpenRouter API client

## Quick Start

### Option 1: Using Configuration File (Recommended)

1. **Configure your setup** in `config.yml`:

```yaml
task:
  name: "CODE_REPAIR"  # or CODE_SYNTHESIS, CODE_SUMMARY, MMLU, custom

model:
  name: "mistralai/mistral-nemo"
  backend: "openrouter"  # or "ollama", "huggingface"
  openrouter:
    api_key: "your-api-key-here"  # or use OPENAI_API_KEY env var

dataset:
  source: "huggingface"
  huggingface:
    name: "Muennighoff/quixbugs"
    split: "train"
  mapping: "quixbugs"
```

2. **Run the generation script**:

```bash
# Sequential processing
python generate.py --config config.yml

# Parallel processing (faster for large benchmarks)
python generate_parallel.py --config config.yml --workers 4
```

### Option 2: Programmatic Usage

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
    model="llama3.1",  # or "openai/gpt-4o" for OpenRouter
    llm_backend="ollama",  # or "openrouter", "huggingface"
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

## Generation Scripts

### `generate.py` - Sequential Processing

Standard script for benchmark transformation with sequential processing.

**Usage:**
```bash
python generate.py [--config config.yml] [--run-id N]
```

**Options:**
- `--config`: Path to configuration file (default: `config.yml`)
- `--run-id`: Run ID for output filename (default: from config)

### `generate_parallel.py` - Parallel Processing

Optimized script for large benchmarks with parallel processing support.

**Usage:**
```bash
python generate_parallel.py [--config config.yml] [--run-id N] [--workers N] [--chunk-size N] [--use-threads] [--no-parallel]
```

**Options:**
- `--config`: Path to configuration file (default: `config.yml`)
- `--run-id`: Run ID for output filename (default: from config)
- `--workers`: Number of parallel workers (default: CPU count)
- `--chunk-size`: Number of samples per chunk (default: auto-calculate)
- `--use-threads`: Use threads instead of processes (for I/O-bound tasks)
- `--no-parallel`: Disable parallel processing

**Example:**
```bash
# Use 8 workers with auto chunk size
python generate_parallel.py --config config.yml --workers 8

# Use threads with custom chunk size
python generate_parallel.py --config config.yml --use-threads --chunk-size 10
```

## Configuration File

The `config.yml` file allows comprehensive configuration of the framework:

### Task Configuration

```yaml
task:
  name: "CODE_REPAIR"  # Predefined: CODE_SYNTHESIS, CODE_REPAIR, CODE_SUMMARY, MMLU
  # Or use "custom" and define below:
  custom:
    description: "Custom task description"
    constraints: ["constraint1", "constraint2"]
    input_space_definition: "Input space definition"
    output_space_definition: "Output space definition"
```

### Model Configuration

```yaml
model:
  name: "mistralai/mistral-nemo"  # Model identifier
  backend: "openrouter"  # "ollama", "huggingface", or "openrouter"
  
  # Ollama settings (if backend is "ollama")
  ollama:
    base_url: "http://localhost:11434"
    timeout: 120
  
  # OpenRouter settings (if backend is "openrouter")
  openrouter:
    api_key: "your-key"  # Optional: can use OPENAI_API_KEY env var
    base_url: "https://openrouter.ai/api/v1"
    timeout: 120
```

### Dataset Configuration

```yaml
dataset:
  source: "huggingface"  # or "local"
  
  # HuggingFace dataset
  huggingface:
    name: "Muennighoff/quixbugs"
    config: null  # Optional config name (e.g., "all" for MMLU)
    split: "train"
    limit: null  # Optional: limit number of samples
  
  # Local dataset
  local:
    path: "path/to/dataset.json"
    structure: "list"  # "list" or "dict"
    samples_key: "samples"  # If structure is "dict"
  
  # Dataset mapping
  mapping: "quixbugs"  # "quixbugs", "mmlu", "mbpp", or "custom"
  
  # Custom mapping (if mapping is "custom")
  custom_mapping:
    input_key: "input"
    output_key: "output"
    metadata_keys: []
  
  benchmark_name: "QuixBugs"
  benchmark_description: "Description of the benchmark"
```

### Orchestrator Configuration

```yaml
orchestrator:
  max_iterations: 3  # Maximum feedback loop iterations
  max_retries_per_sample: 5  # Max retries per sample
  use_per_sample_rules: true  # Generate per-sample rules
```

### Sampling Configuration

```yaml
sampling:
  strategy: "random"  # "random", "diverse", or "stratified"
  sample_size: 5  # Fixed size (null to use ratio)
  sample_ratio: 0.1  # Ratio of benchmark (if sample_size is null)
```

### Parallel Processing Configuration

```yaml
parallel:
  enabled: true  # Enable parallel processing
  workers: null  # Number of workers (null = CPU count)
  chunk_size: null  # Samples per chunk (null = auto)
  use_threads: false  # Use threads instead of processes
```

### Output Configuration

```yaml
output:
  directory: "notebook/generation"
  filename_pattern: "{dataset_name}_{model_name}_FULL_BENCHMARK_{run_id}.json"
  num_outputs: null  # Number of outputs (null = all samples)
  run_id: 0
```

## Predefined Tasks

The framework includes four predefined tasks:

### 1. CODE_SYNTHESIS
Generate Python code from problem descriptions. Modifies problem descriptions while preserving core reasoning.

### 2. CODE_REPAIR
Repair buggy code. Modifies buggy code to introduce different bugs while maintaining the same structure.

### 3. CODE_SUMMARY
Summarize code snippets. Modifies code snippets while preserving the core functionality.

### 4. MMLU
Answer multiple-choice questions. Modifies questions while preserving the reasoning approach.

## Components

### Agents

- **SamplingAgent**: Selects representative samples from benchmark for analysis
- **AnalysisAgent**: Analyzes patterns and enhances task constraints
- **TransformationRuleAgent**: Designs transformation rules
- **TransformationAgent**: Applies transformations to generate modified inputs
- **OutputUpdateAgent**: Determines if outputs should change after transformation
- **ValidationAgent**: Validates generated inputs against constraints

### Models

- **Task**: Defines task constraints and input/output spaces
- **Benchmark**: Container for benchmark samples
- **BenchmarkSample**: Individual benchmark sample with input, output, and metadata
- **ModifiedInput**: Generated modified input with transformation metadata

## LLM Backends

### Ollama (Local)

Use local models via Ollama for privacy and no API costs.

```python
orchestrator = BenchmarkOrchestrator(
    model="llama3.1",
    llm_backend="ollama"
)
```

**Setup:**
1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama3.1`
3. Ensure Ollama is running: `ollama list`

### HuggingFace (via BESSER)

Use HuggingFace models through the BESSER framework.

```python
orchestrator = BenchmarkOrchestrator(
    model="microsoft/phi-2",
    llm_backend="huggingface"
)
```

**Requirements:**
- BESSER framework installed
- HuggingFace model downloaded (automatic on first use)

### OpenRouter (Cloud)

Access multiple cloud models through OpenRouter's unified API.

```python
orchestrator = BenchmarkOrchestrator(
    model="openai/gpt-4o",
    llm_backend="openrouter",
    openrouter_api_key="your-api-key"  # or use OPENAI_API_KEY env var
)
```

**Setup:**
1. Get API key from https://openrouter.ai/
2. Set environment variable: `export OPENAI_API_KEY=your-key`
   - Or provide in config file or code

**Available Models:**
- OpenAI: `openai/gpt-4o`, `openai/gpt-4-turbo`, etc.
- Anthropic: `anthropic/claude-3-opus`, etc.
- Meta: `meta-llama/llama-3.1-70b-instruct`, etc.
- And many more: https://openrouter.ai/models

## Dataset Support

### HuggingFace Datasets

Load datasets directly from HuggingFace:

```yaml
dataset:
  source: "huggingface"
  huggingface:
    name: "Muennighoff/quixbugs"
    split: "train"
  mapping: "quixbugs"
```

**Supported Mappings:**
- `quixbugs`: QuixBugs benchmark format
- `mmlu`: MMLU multiple-choice questions
- `mbpp`: MBPP code generation
- `custom`: Define your own mapping

### Local Datasets

Load from local JSON files:

```yaml
dataset:
  source: "local"
  local:
    path: "data/my_dataset.json"
    structure: "list"  # or "dict"
  mapping: "custom"
  custom_mapping:
    input_key: "input"
    output_key: "output"
```

## Examples

### Example 1: Code Repair with QuixBugs

```yaml
# config.yml
task:
  name: "CODE_REPAIR"

model:
  name: "mistralai/mistral-nemo"
  backend: "openrouter"

dataset:
  source: "huggingface"
  huggingface:
    name: "Muennighoff/quixbugs"
    split: "train"
  mapping: "quixbugs"
```

```bash
python generate.py --config config.yml
```

### Example 2: MMLU Questions

```yaml
task:
  name: "MMLU"

model:
  name: "openai/gpt-4o"
  backend: "openrouter"

dataset:
  source: "huggingface"
  huggingface:
    name: "cais/mmlu"
    config: "all"
    split: "dev"
    limit: 100
  mapping: "mmlu"
```

### Example 3: Custom Task

```yaml
task:
  name: "custom"
  custom:
    description: "Translate English to French"
    constraints:
      - "Preserve meaning"
      - "Maintain formality level"
    input_space_definition: "English sentences"
    output_space_definition: "French translations"
```

## Project Structure

```
BenchFactory_MA/
├── config.yml                 # Configuration file
├── generate.py                # Sequential generation script
├── generate_parallel.py       # Parallel generation script
├── requirements.txt          # Python dependencies
├── README.md                  # This file
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py             # Configuration classes
│   ├── orchestrator.py       # Main orchestrator
│   │
│   ├── agents/               # Agent implementations
│   │   ├── base_agent.py
│   │   ├── sampling_agent.py
│   │   ├── analysis_agent.py
│   │   ├── transformation_rule_agent.py
│   │   ├── transformation_agent.py
│   │   ├── output_update_agent.py
│   │   └── validation_agent.py
│   │
│   ├── models/               # Data models
│   │   ├── task.py
│   │   └── benchmark.py
│   │
│   └── utils/                # Utilities
│       └── besser_integration.py
│
├── notebook/                  # Notebooks and tasks
│   ├── tasks.py              # Predefined tasks
│   └── generation/          # Output directory
│
├── examples/                  # Example scripts
│   └── example_usage.py
│
├── evaluation/               # Evaluation utilities
│   ├── code_utils.py
│   ├── metric.py
│   └── pass_test.py
│
└── scripts/                   # Utility scripts
    └── verify_setup.py
```

## Methodology

The framework implements a cyclical process with feedback loops:

1. **Benchmark** → **Sampling**: Extract representative samples
2. **Task** + **Sampling** → **Analysis**: Enhance constraints and input space definition
3. **Analysis** → **Transformation Rule Design**: Create transformation rules
4. **Transformation Rule** → **Application**: Generate modified inputs
5. **Application** → **Output Update**: Determine if outputs should change
6. **Application** → **Validation**: Check validity and complexity
7. **Validation** → **Analysis** (feedback loop): Refine constraints
8. **Validation** → **Transformation Rule** (feedback loop): Improve rules

## BESSER Integration

The framework integrates with the [BESSER Agentic Framework](https://besser-agentic-framework.readthedocs.io/latest/) for enhanced agent capabilities:

- **Automatic Integration**: Works seamlessly when BESSER is installed
- **Agent Wrapping**: Optional BESSER agent wrapping for state management
- **Multi-Agent Coordination**: Leverage BESSER's coordination features

See `src/utils/besser_integration.py` for advanced BESSER integration options.

## Performance Tips

1. **Use Parallel Processing**: For large benchmarks (>50 samples), use `generate_parallel.py`
2. **Optimize Chunk Size**: Balance between too small (overhead) and too large (memory)
3. **Choose Right Backend**: 
   - Ollama: Best for privacy, no API costs
   - OpenRouter: Best for access to multiple models, faster inference
   - HuggingFace: Best for local fine-tuned models
4. **Adjust Sampling**: Reduce `sample_size` for faster analysis phase
5. **Limit Outputs**: Use `num_outputs` to generate only what you need
