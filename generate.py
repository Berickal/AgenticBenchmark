#!/usr/bin/env python3
#Script to generate benchmark transformations using the Multi-Agent Framework.
#Usage:
#    python generate.py [--config config.yml] [--run-id N]
#This script reads configuration from config.yml (or a custom config file) and
#runs the benchmark transformation pipeline.

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.task import Task
from src.models.benchmark import Benchmark, BenchmarkSample
from src.orchestrator import BenchmarkOrchestrator

# Import predefined tasks
from examples.tasks import (
    CODE_SYNTHESIS_TASK,
    CODE_REPAIR_TASK,
    CODE_SUMMARY_TASK,
    MMLU_TASK,
    TASK_LIST
)

# Task mapping
TASK_MAP = {
    "CODE_SYNTHESIS": CODE_SYNTHESIS_TASK,
    "CODE_REPAIR": CODE_REPAIR_TASK,
    "CODE_SUMMARY": CODE_SUMMARY_TASK,
    "MMLU": MMLU_TASK,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration."""
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    log_file = config.get('logging', {}).get('log_file')
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_task(config: Dict[str, Any]) -> Task:
    """Get task from configuration."""
    task_config = config.get('task', {})
    task_name = task_config.get('name', 'CODE_REPAIR')
    
    if task_name == 'custom':
        custom_task = task_config.get('custom', {})
        return Task(
            description=custom_task.get('description', ''),
            constraints=custom_task.get('constraints', []),
            input_space_definition=custom_task.get('input_space_definition', ''),
            output_space_definition=custom_task.get('output_space_definition', '')
        )
    elif task_name in TASK_MAP:
        return TASK_MAP[task_name]
    else:
        raise ValueError(f"Unknown task name: {task_name}. Available: {list(TASK_MAP.keys())}, 'custom'")


def load_dataset(config: Dict[str, Any]) -> Benchmark:
    """Load dataset from HuggingFace or local file."""
    dataset_config = config.get('dataset', {})
    source = dataset_config.get('source', 'huggingface')
    
    if source == 'huggingface':
        return load_huggingface_dataset(dataset_config)
    elif source == 'local':
        return load_local_dataset(dataset_config)
    else:
        raise ValueError(f"Unknown dataset source: {source}. Available: 'huggingface', 'local'")


def load_huggingface_dataset(dataset_config: Dict[str, Any]) -> Benchmark:
    """Load dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    hf_config = dataset_config.get('huggingface', {})
    dataset_name = hf_config.get('name')
    dataset_config_name = hf_config.get('config')
    split = hf_config.get('split', 'train')
    limit = hf_config.get('limit')
    
    if not dataset_name:
        raise ValueError("HuggingFace dataset name is required")
    
    logging.info(f"Loading HuggingFace dataset: {dataset_name}")
    if dataset_config_name:
        dataset = load_dataset(dataset_name, dataset_config_name, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    # Convert to list for easier processing
    dataset_list = list(dataset)
    
    # Map to BenchmarkSample
    mapping = dataset_config.get('mapping', 'custom')
    samples = map_dataset_samples(dataset_list, mapping, dataset_config)
    
    return Benchmark(
        name=dataset_config.get('benchmark_name', dataset_name),
        description=dataset_config.get('benchmark_description', ''),
        samples=samples
    )


def load_local_dataset(dataset_config: Dict[str, Any]) -> Benchmark:
    """Load dataset from local JSON file."""
    local_config = dataset_config.get('local', {})
    path = local_config.get('path')
    
    if not path:
        raise ValueError("Local dataset path is required")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    logging.info(f"Loading local dataset: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    
    structure = local_config.get('structure', 'list')
    if structure == 'dict':
        samples_key = local_config.get('samples_key', 'samples')
        dataset_list = data.get(samples_key, [])
    else:
        dataset_list = data if isinstance(data, list) else [data]
    
    # Map to BenchmarkSample
    mapping = dataset_config.get('mapping', 'custom')
    samples = map_dataset_samples(dataset_list, mapping, dataset_config)
    
    return Benchmark(
        name=dataset_config.get('benchmark_name', 'Local Dataset'),
        description=dataset_config.get('benchmark_description', ''),
        samples=samples
    )


def map_dataset_samples(dataset_list: List[Dict], mapping: str, dataset_config: Dict[str, Any]) -> List[BenchmarkSample]:
    """Map raw dataset items to BenchmarkSample objects."""
    samples = []
    
    if mapping == 'quixbugs':
        for data in dataset_list:
            samples.append(BenchmarkSample(
                input=data.get('buggy_program', ''),
                output=data.get('solution', ''),
                metadata={
                    'id': data.get('name', ''),
                    'docstring': data.get('docstring', ''),
                    'tests': data.get('tests', '')
                }
            ))
    
    elif mapping == 'mmlu':
        for data in dataset_list:
            choices = data.get('choices', [])
            if len(choices) >= 4:
                choices_dict = {
                    "A": choices[0],
                    "B": choices[1],
                    "C": choices[2],
                    "D": choices[3]
                }
                data_input = f"{data.get('question', '')}\n###Choice\n{choices_dict}"
            else:
                data_input = data.get('question', '')
            
            samples.append(BenchmarkSample(
                input=data_input,
                output=data.get('answer', ''),
                metadata={}
            ))
    
    elif mapping == 'mbpp':
        for data in dataset_list:
            samples.append(BenchmarkSample(
                input=data.get('text', ''),
                output=data.get('code', ''),
                metadata={
                    'id': data.get('task_id', ''),
                    'tests': data.get('test_list', [])
                }
            ))
    
    elif mapping == 'custom':
        custom_mapping = dataset_config.get('custom_mapping', {})
        input_key = custom_mapping.get('input_key', 'input')
        output_key = custom_mapping.get('output_key', 'output')
        metadata_keys = custom_mapping.get('metadata_keys', [])
        
        for data in dataset_list:
            metadata = {}
            for key in metadata_keys:
                if key in data:
                    metadata[key] = data[key]
            
            samples.append(BenchmarkSample(
                input=data.get(input_key, ''),
                output=data.get(output_key, ''),
                metadata=metadata if metadata else None
            ))
    
    else:
        raise ValueError(f"Unknown mapping: {mapping}. Available: 'quixbugs', 'mmlu', 'mbpp', 'custom'")
    
    return samples


def create_orchestrator(config: Dict[str, Any]) -> BenchmarkOrchestrator:
    """Create orchestrator from configuration."""
    model_config = config.get('model', {})
    orchestrator_config = config.get('orchestrator', {})
    
    backend = model_config.get('backend', 'ollama')
    
    # Prepare orchestrator kwargs
    orchestrator_kwargs = {
        'model': model_config.get('name', 'mistral:latest'),
        'max_iterations': orchestrator_config.get('max_iterations', 3),
        'max_retries_per_sample': orchestrator_config.get('max_retries_per_sample', 5),
        'llm_backend': backend,
        'use_per_sample_rules': orchestrator_config.get('use_per_sample_rules', False)
    }
    
    # Add OpenRouter-specific parameters if using OpenRouter backend
    if backend == 'openrouter':
        openrouter_config = model_config.get('openrouter', {})
        orchestrator_kwargs['openrouter_api_key'] = openrouter_config.get('api_key')
        orchestrator_kwargs['openrouter_base_url'] = openrouter_config.get('base_url', 'https://openrouter.ai/api/v1')
    
    orchestrator = BenchmarkOrchestrator(**orchestrator_kwargs)
    
    # Configure sampling agent
    sampling_config = config.get('sampling', {})
    orchestrator.sampling_agent.sampling_strategy = sampling_config.get('strategy', 'random')
    orchestrator.sampling_agent.sample_size = sampling_config.get('sample_size')
    orchestrator.sampling_agent.sample_ratio = sampling_config.get('sample_ratio', 0.1)
    
    return orchestrator


def format_filename(pattern: str, dataset_name: str, model_name: str, task_name: str, 
                   run_id: int, timestamp: Optional[str] = None) -> str:
    """Format output filename using pattern."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Sanitize names for filename
    dataset_name = dataset_name.replace('/', '_').replace(' ', '_')
    model_name = model_name.replace('/', '_').replace(':', '_').replace(' ', '_')
    task_name = task_name.replace(' ', '_')
    
    filename = pattern.format(
        dataset_name=dataset_name,
        model_name=model_name,
        task_name=task_name,
        timestamp=timestamp,
        run_id=run_id
    )
    
    return filename


def save_results(modified_inputs: List, config: Dict[str, Any], dataset_name: str, 
                model_name: str, task_name: str, run_id: int):
    """Save generated results to file."""
    output_config = config.get('output', {})
    output_dir = output_config.get('directory', 'notebook/generation')
    filename_pattern = output_config.get('filename_pattern', '{dataset_name}_{model_name}_FULL_BENCHMARK_{run_id}.json')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Format filename
    filename = format_filename(filename_pattern, dataset_name, model_name, task_name, run_id)
    filepath = os.path.join(output_dir, filename)
    
    # Convert to dict and save
    modified_inputs_dict = [mi.to_dict() for mi in modified_inputs]
    with open(filepath, 'w') as f:
        json.dump(modified_inputs_dict, f, indent=4)
    
    logging.info(f"Results saved to: {filepath}")
    return filepath


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate benchmark transformations')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to configuration file (default: config.yml)')
    parser.add_argument('--run-id', type=int, default=None,
                       help='Run ID for output filename (default: auto-increment)')
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        logging.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Override run_id if provided
    if args.run_id is not None:
        config.setdefault('output', {})['run_id'] = args.run_id
    
    # Get run_id from config
    run_id = config.get('output', {}).get('run_id', 0)
    
    logger.info("=" * 80)
    logger.info("Benchmark Factory Multi-Agent Framework - Generation Script")
    logger.info("=" * 80)
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Run ID: {run_id}")
    
    try:
        # Get task
        logger.info("Loading task...")
        task = get_task(config)
        task_name = config.get('task', {}).get('name', 'UNKNOWN')
        logger.info(f"Task: {task_name}")
        
        # Load dataset
        logger.info("Loading dataset...")
        benchmark = load_dataset(config)
        logger.info(f"Dataset: {benchmark.name}")
        logger.info(f"Number of samples: {len(benchmark.samples)}")
        
        # Create orchestrator
        logger.info("Initializing orchestrator...")
        orchestrator = create_orchestrator(config)
        model_name = config.get('model', {}).get('name', 'unknown')
        logger.info(f"Model: {model_name}")
        logger.info(f"Backend: {config.get('model', {}).get('backend', 'ollama')}")
        
        # Get number of outputs
        num_outputs = config.get('output', {}).get('num_outputs')
        if num_outputs:
            logger.info(f"Target number of outputs: {num_outputs}")
        else:
            logger.info("Will transform all samples in benchmark")
        
        # Run transformation
        logger.info("=" * 80)
        logger.info("Starting benchmark transformation...")
        logger.info("=" * 80)
        
        modified_inputs = orchestrator.transform_benchmark(
            benchmark=benchmark,
            task=task,
            num_outputs=num_outputs
        )
        
        # Get statistics
        stats = orchestrator.get_statistics()
        
        # Save results
        logger.info("=" * 80)
        logger.info("Saving results...")
        logger.info("=" * 80)
        filepath = save_results(
            modified_inputs,
            config,
            benchmark.name,
            model_name,
            task_name,
            run_id
        )
        
        # Print summary
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Dataset: {benchmark.name}")
        logger.info(f"Task: {task_name}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Total samples in benchmark: {len(benchmark.samples)}")
        logger.info(f"Generated valid transformations: {len(modified_inputs)}")
        logger.info(f"Total attempts: {stats['total_attempts']}")
        logger.info(f"Success rate: {stats['success_rate']:.2%}")
        logger.info(f"Output file: {filepath}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

