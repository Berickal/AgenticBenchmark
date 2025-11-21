"""Configuration for the benchmark transformation framework."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OllamaConfig:
    """Configuration for Ollama integration."""
    model: str = "llama3.1"
    base_url: str = "http://localhost:11434"
    timeout: int = 120


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter integration."""
    model: str = "openai/gpt-4o"
    api_key: Optional[str] = None  # If None, will use OPENAI_API_KEY or OPENROUTER_API_KEY env var
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: int = 120


@dataclass
class FrameworkConfig:
    """Main framework configuration."""
    # Ollama settings
    ollama_config: OllamaConfig = field(default_factory=OllamaConfig)
    
    # OpenRouter settings
    openrouter_config: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    
    # Sampling settings
    sampling_strategy: str = "random"  # random, diverse, stratified
    sample_size: Optional[int] = None
    sample_ratio: float = 0.1
    
    # Transformation settings
    max_iterations: int = 3
    max_retries_per_sample: int = 5
    
    # Validation settings
    min_complexity_score: float = 0.6
    require_complexity_match: bool = True
