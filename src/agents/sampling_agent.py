"""Sampling agent for selecting benchmark samples."""
from typing import List, Optional
import random

from src.models.benchmark import Benchmark, BenchmarkSample
from src.agents.base_agent import BaseAgent


class SamplingAgent(BaseAgent):
    """Agent responsible for sampling benchmark data."""
    
    def __init__(
        self,
        model: str = "llama3.2",
        sampling_strategy: str = "random",
        sample_size: Optional[int] = None,
        sample_ratio: float = 0.1,
        llm_backend: str = "ollama",
        openrouter_api_key: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize sampling agent.
        
        Args:
            model: Model name (Ollama model name, HuggingFace model ID, or OpenRouter model ID)
            sampling_strategy: Strategy for sampling ('random', 'diverse', 'stratified')
            sample_size: Fixed number of samples to extract
            sample_ratio: Ratio of benchmark to sample (if sample_size not provided)
            llm_backend: LLM backend to use ("ollama", "huggingface", or "openrouter")
            openrouter_api_key: OpenRouter API key (for OpenRouter backend)
            openrouter_base_url: OpenRouter API base URL (for OpenRouter backend)
        """
        system_prompt = """You are a sampling agent specialized in selecting representative 
        benchmark samples for transformation. Your goal is to select diverse samples that 
        will help create effective transformation rules."""
        
        super().__init__(
            name="SamplingAgent",
            model=model,
            system_prompt=system_prompt,
            llm_backend=llm_backend,
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=openrouter_base_url
        )
        self.sampling_strategy = sampling_strategy
        self.sample_size = sample_size
        self.sample_ratio = sample_ratio
    
    def process(self, benchmark: Benchmark) -> List[BenchmarkSample]:
        """
        Sample from the benchmark.
        
        Args:
            benchmark: The benchmark to sample from
            
        Returns:
            List of sampled benchmark samples
        """
        if not benchmark.samples:
            return []
        
        if self.sampling_strategy == "random":
            return self._random_sample(benchmark)
        elif self.sampling_strategy == "diverse":
            return self._diverse_sample(benchmark)
        elif self.sampling_strategy == "stratified":
            return self._stratified_sample(benchmark)
        else:
            return self._random_sample(benchmark)
    
    def _random_sample(self, benchmark: Benchmark) -> List[BenchmarkSample]:
        """Random sampling strategy."""
        total_samples = len(benchmark.samples)
        n = self.sample_size or max(1, int(total_samples * self.sample_ratio))
        n = min(n, total_samples)
        # If requesting all samples, return them in order
        if n == total_samples:
            return benchmark.samples[:]
        return random.sample(benchmark.samples, n)
    
    def _diverse_sample(self, benchmark: Benchmark) -> List[BenchmarkSample]:
        """Diverse sampling strategy - tries to select diverse samples."""
        total_samples = len(benchmark.samples)
        n = self.sample_size or max(1, int(total_samples * self.sample_ratio))
        n = min(n, total_samples)
        
        # Simple diversity: select evenly spaced samples
        if n == 1:
            return [random.choice(benchmark.samples)]
        
        step = total_samples // n
        indices = [i * step for i in range(n)]
        return [benchmark.samples[i] for i in indices if i < total_samples]
    
    def _stratified_sample(self, benchmark: Benchmark) -> List[BenchmarkSample]:
        """Stratified sampling - for now, falls back to random."""
        # TODO: Implement proper stratification based on metadata
        return self._random_sample(benchmark)
