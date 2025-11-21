from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

@dataclass
class BenchmarkSample:
    """A single sample from the benchmark."""
    input: Any
    output: Any
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Benchmark:
    """Benchmark dataset container."""
    samples: List[BenchmarkSample]
    name: str
    description: Optional[str] = None

@dataclass
class ModifiedInput:
    """Generated modified input that maintains complexity."""
    original_input: Any
    modified_input: Any
    original_output: Any
    transformation_applied: str
    updated_output: Optional[Any] = None
    complexity_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ModifiedInput to a dictionary for JSON serialization."""
        return asdict(self)
    
    def get_expected_output(self) -> Any:
        """Get the expected output (updated_output if available, otherwise original_output)."""
        return self.updated_output if self.updated_output is not None else self.original_output
