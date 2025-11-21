"""Validation agent for checking benchmark validity."""
from typing import Any, Dict, List, Optional, Tuple
import json

from src.models.task import Task
from src.models.benchmark import ModifiedInput
from src.agents.base_agent import BaseAgent


class ValidationAgent(BaseAgent):
    """Agent responsible for validating modified inputs."""
    
    def __init__(
        self,
        model: str = "llama3.1",
        llm_backend: str = "ollama",
        openrouter_api_key: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize validation agent.
        
        Args:
            model: Model name (Ollama model name, HuggingFace model ID, or OpenRouter model ID)
            llm_backend: LLM backend to use ("ollama", "huggingface", or "openrouter")
            openrouter_api_key: OpenRouter API key (for OpenRouter backend)
            openrouter_base_url: OpenRouter API base URL (for OpenRouter backend)
        """
        system_prompt = """You are a validation agent specialized in checking:
        1. Whether inputs satisfy all defined constraints
        2. Whether inputs are within the valid input space
        3. Whether complexity levels are maintained
        
        Provide clear, structured validation results."""
        
        super().__init__(
            name="ValidationAgent",
            model=model,
            system_prompt=system_prompt,
            llm_backend=llm_backend,
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=openrouter_base_url
        )
    
    def process(
        self,
        modified_input: ModifiedInput,
        task: Task,
        analysis: Dict,
        original_sample_output: Optional[Any] = None
    ) -> Dict:
        """
        Validate a modified input.
        
        Args:
            modified_input: The modified input to validate
            task: Task definition
            analysis: Analysis results with constraints
            original_sample_output: Original output for complexity comparison
            
        Returns:
            Dictionary containing:
            - is_valid: Whether input passes all checks
            - constraint_violations: List of violated constraints
            - input_space_valid: Whether input is in valid space
            - complexity_match: Whether complexity matches
            - complexity_score: Computed complexity score
            - validation_details: Detailed validation information
        """
        constraints = analysis.get('enhanced_constraints', task.constraints)
        input_space = analysis.get('enhanced_input_space', task.input_space_definition)
        
        # Check if output was updated
        output_info = ""
        if modified_input.updated_output is not None:
            output_info = f"""
Original Output: {modified_input.original_output}
Updated Output: {modified_input.updated_output}
(Note: Output was updated because the transformation changed the question/problem)
"""
        else:
            output_info = f"""
Expected Output: {original_sample_output if original_sample_output else 'N/A'}
(Output preserved from original)
"""
        
        prompt = f"""
Validate the following modified input:

Original Input: {modified_input.original_input}
Modified Input: {modified_input.modified_input}
Transformation Applied: {modified_input.transformation_applied}
{output_info}
Task Constraints:
{chr(10).join(f'- {c}' for c in constraints)}

Input Space Definition: {input_space}

Task Description: {task.description}

Perform the following checks:
1. Constraint Check: Verify all constraints are satisfied
2. Input Space Check: Verify the input is within the defined input space
3. Complexity Check: Compare complexity between original and modified input
4. Output Consistency: If output was updated, verify the change is justified

Respond in JSON format:
{{
    "is_valid": true/false,
    "constraint_violations": ["violation1", ...],
    "input_space_valid": true/false,
    "complexity_match": true/false,
    "complexity_score": 0.0-1.0,
    "complexity_difference": "description of complexity difference",
    "validation_details": "detailed validation notes"
}}
"""
        
        response = self._call_llm(prompt, temperature=0.2)
        
        try:
            validation_result = self._parse_json_response(response)
        except json.JSONDecodeError:
            # Fallback validation
            validation_result = self._fallback_validation(
                modified_input,
                constraints,
                input_space
            )
        
        return {
            "is_valid": validation_result.get("is_valid", False),
            "constraint_violations": validation_result.get("constraint_violations", []),
            "input_space_valid": validation_result.get("input_space_valid", True),
            "complexity_match": validation_result.get("complexity_match", True),
            "complexity_score": validation_result.get("complexity_score", 0.5),
            "complexity_difference": validation_result.get("complexity_difference", ""),
            "validation_details": validation_result.get("validation_details", response),
            "raw_response": response
        }
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        response = response.strip()
        
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()
        
        return json.loads(response)
    
    def _fallback_validation(
        self,
        modified_input: ModifiedInput,
        constraints: List[str],
        input_space: str
    ) -> Dict:
        """Simple fallback validation."""
        # Basic checks
        constraint_violations = []
        
        # Check if modified input is different from original
        if str(modified_input.modified_input).strip() == str(modified_input.original_input).strip():
            constraint_violations.append("Modified input is identical to original")
        
        # Basic complexity check: compare lengths/structure
        original_str = str(modified_input.original_input)
        modified_str = str(modified_input.modified_input)
        
        complexity_score = 0.5
        if abs(len(original_str) - len(modified_str)) / max(len(original_str), 1) < 0.5:
            complexity_score = 0.8  # Similar length suggests similar complexity
        
        return {
            "is_valid": len(constraint_violations) == 0,
            "constraint_violations": constraint_violations,
            "input_space_valid": True,  # Assume valid if we can't check
            "complexity_match": complexity_score > 0.6,
            "complexity_score": complexity_score,
            "complexity_difference": f"Length difference: {abs(len(original_str) - len(modified_str))}",
            "validation_details": "Fallback validation performed"
        }
