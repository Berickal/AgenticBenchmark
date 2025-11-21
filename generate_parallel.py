#!/usr/bin/env python3
"""
Parallel processing script for benchmark transformations using the Multi-Agent Framework.

Usage:
    python generate_parallel.py [--config config.yml] [--run-id N] [--workers N] [--chunk-size N]

This script reads configuration from config.yml and runs the benchmark transformation
pipeline with parallel processing capabilities.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.task import Task
from src.models.benchmark import Benchmark, BenchmarkSample, ModifiedInput
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


def setup_logging(config: Dict[str, Any], worker_id: Optional[int] = None):
    """Setup logging configuration."""
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    log_file = config.get('logging', {}).get('log_file')
    
    # Add worker ID to log format if in parallel mode
    if worker_id is not None:
        format_str = f'%(asctime)s - [Worker-{worker_id}] - %(name)s - %(levelname)s - %(message)s'
    else:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        # Append worker ID to log file name if in parallel mode
        if worker_id is not None:
            log_file_base, log_file_ext = os.path.splitext(log_file)
            log_file = f"{log_file_base}_worker{worker_id}{log_file_ext}"
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=format_str,
        handlers=handlers,
        force=True  # Force reconfiguration for multiprocessing
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


def process_chunk(
    chunk_data: Tuple[int, List[BenchmarkSample], Dict[str, Any], Task, Dict[str, Any]]
) -> Tuple[int, List[ModifiedInput]]:
    """
    Process a chunk of samples in parallel.
    
    Args:
        chunk_data: Tuple of (chunk_id, samples, config, task, full_benchmark_info)
        
    Returns:
        Tuple of (chunk_id, list of ModifiedInput objects)
    """
    chunk_id, samples, config, task, full_benchmark_info = chunk_data
    
    # Setup logging for this worker
    setup_logging(config, worker_id=chunk_id)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Worker {chunk_id}: Processing {len(samples)} samples")
        
        # Create a benchmark with just this chunk
        chunk_benchmark = Benchmark(
            name=full_benchmark_info.get('name', 'Chunk'),
            description=full_benchmark_info.get('description', ''),
            samples=samples
        )
        
        # Create orchestrator for this chunk
        orchestrator = create_orchestrator(config)
        
        # Transform the chunk
        modified_inputs = orchestrator.transform_benchmark(
            benchmark=chunk_benchmark,
            task=task,
            num_outputs=len(samples)  # Try to transform all samples in chunk
        )
        
        logger.info(f"Worker {chunk_id}: Generated {len(modified_inputs)} valid transformations")
        return (chunk_id, modified_inputs)
        
    except Exception as e:
        logger.error(f"Worker {chunk_id}: Error processing chunk: {e}", exc_info=True)
        return (chunk_id, [])


def transform_benchmark_parallel(
    benchmark: Benchmark,
    task: Task,
    config: Dict[str, Any],
    num_workers: int = None,
    chunk_size: Optional[int] = None,
    use_threads: bool = False
) -> List[ModifiedInput]:
    """
    Transform benchmark using parallel processing.
    
    Args:
        benchmark: Original benchmark
        task: Task definition
        config: Configuration dictionary
        num_workers: Number of parallel workers (default: CPU count)
        chunk_size: Number of samples per chunk (default: auto-calculate)
        use_threads: Use threads instead of processes (for I/O-bound tasks)
        
    Returns:
        List of valid ModifiedInput objects
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    samples = benchmark.samples
    total_samples = len(samples)
    
    if total_samples == 0:
        logging.warning("No samples to process")
        return []
    
    # Calculate chunk size if not provided
    if chunk_size is None:
        chunk_size = max(1, total_samples // num_workers)
        if chunk_size < 1:
            chunk_size = 1
    
    # Split samples into chunks
    chunks = []
    for i in range(0, total_samples, chunk_size):
        chunk_samples = samples[i:i + chunk_size]
        chunks.append((i // chunk_size, chunk_samples))
    
    logging.info(f"Processing {total_samples} samples in {len(chunks)} chunks with {num_workers} workers")
    logging.info(f"Chunk size: {chunk_size} samples per chunk")
    
    # Prepare chunk data
    full_benchmark_info = {
        'name': benchmark.name,
        'description': benchmark.description
    }
    
    chunk_data_list = [
        (chunk_id, chunk_samples, config, task, full_benchmark_info)
        for chunk_id, chunk_samples in chunks
    ]
    
    # Process chunks in parallel
    all_results = []
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    with executor_class(max_workers=num_workers) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(process_chunk, chunk_data): chunk_data[0]  # chunk_id is first element
            for chunk_data in chunk_data_list
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                result_chunk_id, modified_inputs = future.result()
                all_results.append((result_chunk_id, modified_inputs))
                completed += 1
                logging.info(f"Completed chunk {result_chunk_id + 1}/{len(chunks)} ({completed}/{len(chunks)})")
            except Exception as e:
                logging.error(f"Chunk {chunk_id} generated an exception: {e}", exc_info=True)
    
    # Sort results by chunk_id to maintain order
    all_results.sort(key=lambda x: x[0])
    
    # Combine all results
    combined_results = []
    for chunk_id, modified_inputs in all_results:
        combined_results.extend(modified_inputs)
    
    logging.info(f"Parallel processing complete: {len(combined_results)} total valid transformations")
    return combined_results


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
    parser = argparse.ArgumentParser(description='Generate benchmark transformations with parallel processing')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to configuration file (default: config.yml)')
    parser.add_argument('--run-id', type=int, default=None,
                       help='Run ID for output filename (default: auto-increment)')
    parser.add_argument('--workers', type=int, default=None,
                       help=f'Number of parallel workers (default: CPU count = {mp.cpu_count()})')
    parser.add_argument('--chunk-size', type=int, default=None,
                       help='Number of samples per chunk (default: auto-calculate)')
    parser.add_argument('--use-threads', action='store_true',
                       help='Use threads instead of processes (for I/O-bound tasks)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing (run sequentially)')
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
    
    # Get parallel processing settings from config or args
    parallel_config = config.get('parallel', {})
    num_workers = args.workers or parallel_config.get('workers') or (None if args.no_parallel else mp.cpu_count())
    chunk_size = args.chunk_size or parallel_config.get('chunk_size')
    use_threads = args.use_threads or parallel_config.get('use_threads', False)
    enable_parallel = not args.no_parallel and parallel_config.get('enabled', True)
    
    logger.info("=" * 80)
    logger.info("Benchmark Factory Multi-Agent Framework - Parallel Generation Script")
    logger.info("=" * 80)
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Parallel processing: {'Enabled' if enable_parallel else 'Disabled'}")
    if enable_parallel:
        logger.info(f"Workers: {num_workers}")
        logger.info(f"Chunk size: {chunk_size or 'auto'}")
        logger.info(f"Executor: {'Threads' if use_threads else 'Processes'}")
    
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
        
        if enable_parallel and len(benchmark.samples) > 1:
            # Parallel processing
            model_name = config.get('model', {}).get('name', 'unknown')
            logger.info(f"Model: {model_name}")
            logger.info(f"Backend: {config.get('model', {}).get('backend', 'ollama')}")
            
            modified_inputs = transform_benchmark_parallel(
                benchmark=benchmark,
                task=task,
                config=config,
                num_workers=num_workers,
                chunk_size=chunk_size,
                use_threads=use_threads
            )
            
            # Limit to num_outputs if specified
            if num_outputs and len(modified_inputs) > num_outputs:
                modified_inputs = modified_inputs[:num_outputs]
                logger.info(f"Limited results to {num_outputs} outputs")
        else:
            # Sequential processing (fallback or disabled)
            logger.info("Using sequential processing")
            orchestrator = create_orchestrator(config)
            model_name = config.get('model', {}).get('name', 'unknown')
            logger.info(f"Model: {model_name}")
            logger.info(f"Backend: {config.get('model', {}).get('backend', 'ollama')}")
            
            modified_inputs = orchestrator.transform_benchmark(
                benchmark=benchmark,
                task=task,
                num_outputs=num_outputs
            )
        
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
        logger.info(f"Parallel processing: {'Yes' if enable_parallel else 'No'}")
        if enable_parallel:
            logger.info(f"Workers used: {num_workers}")
        logger.info(f"Output file: {filepath}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

