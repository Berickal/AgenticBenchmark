"""Benchmark Factory Multi-Agent Framework."""
__version__ = "0.1.0"

from src.models.task import Task
from src.models.benchmark import Benchmark, BenchmarkSample, ModifiedInput
from src.orchestrator import BenchmarkOrchestrator, TransformationResult
from src.config import FrameworkConfig, OllamaConfig, OpenRouterConfig

__all__ = [
    "Task",
    "Benchmark",
    "BenchmarkSample",
    "ModifiedInput",
    "BenchmarkOrchestrator",
    "TransformationResult",
    "FrameworkConfig",
    "OllamaConfig",
    "OpenRouterConfig",
]
