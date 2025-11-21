from dataclasses import dataclass
from typing import List

@dataclass
class Task:
    """Task definition for benchmark transformation."""
    description: str
    constraints: List[str]
    input_space_definition: str
    output_space_definition: str

