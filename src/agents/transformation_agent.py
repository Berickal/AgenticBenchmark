"""Transformation application agent."""
from typing import Any, Dict, List, Optional
import logging
import json

from src.models.benchmark import BenchmarkSample, ModifiedInput
from src.models.task import Task
from src.agents.base_agent import BaseAgent


class TransformationAgent(BaseAgent):
    """Agent responsible for applying transformations to generate modified inputs."""
    
    def __init__(
        self,
        model: str = "llama3.1",
        llm_backend: str = "ollama",
        openrouter_api_key: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize transformation agent.
        
        Args:
            model: Model name (Ollama model name, HuggingFace model ID, or OpenRouter model ID)
            llm_backend: LLM backend to use ("ollama", "huggingface", or "openrouter")
            openrouter_api_key: OpenRouter API key (for OpenRouter backend)
            openrouter_base_url: OpenRouter API base URL (for OpenRouter backend)
        """
        system_prompt = """You are a transformation application agent. Your role is to:
        1. Apply transformation rules to benchmark inputs
        2. Generate syntactically and semantically different inputs
        3. Maintain complexity levels equivalent to original inputs
        
        Always ensure:
        - Generated inputs are valid within the task input space
        - Complexity is preserved
        - Transformations are applied correctly"""
        
        super().__init__(
            name="TransformationAgent",
            model=model,
            system_prompt=system_prompt,
            llm_backend=llm_backend,
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=openrouter_base_url
        )
    
    def process(
        self,
        sample: BenchmarkSample,
        rule: Dict,
        task: Task,
        analysis: Dict
    ) -> ModifiedInput:
        """
        Apply transformation rule to a sample.
        
        Args:
            sample: Original benchmark sample
            rule: Transformation rule to apply
            task: Task definition
            analysis: Analysis results with constraints
            
        Returns:
            ModifiedInput with transformed input
        """
        constraints = analysis.get('enhanced_constraints', task.constraints)
        input_space = analysis.get('enhanced_input_space', task.input_space_definition)
        output_space = task.output_space_definition
        
        prompt = f"""
Apply the following transformation rule to the input:

Original Input: {sample.input}
Original Output: {sample.output}

Transformation Rule:
- Name: {rule.get('name', 'unknown')}
- Description: {rule.get('description', '')}
- Function: {rule.get('transformation_function', '')}

Task Constraints:
{chr(10).join(f'- {c}' for c in constraints)}

Input Space Definition: {input_space}
Output Space Definition: {output_space}

Generate a modified input that:
1. Is syntactically different from the original
2. Maintains the same complexity level
3. Respects all defined constraints
6. If the transformation legitimately changes the question/problem, it may require a different output

Provide answer in JSON format as follows:
{{
    "modified_input": <the modified input>,
    "output_changed": <true/false>,
    "notes": <any relevant notes about the transformation>
}}
"""
        
        response = self._call_llm(prompt, temperature=0.8)

        try:
            json_response = self._parse_json_response(response)
            modified_input = json_response.get('modified_input', sample.input)
        except (json.JSONDecodeError, KeyError):
            logging.warning("Failed to parse JSON response from LLM, falling back to raw response extraction.")
            modified_input = response.strip()

        
        return ModifiedInput(
            original_input=sample.input,
            modified_input=modified_input,
            original_output=sample.output,
            transformation_applied=rule.get('name', 'unknown'),
            metadata={
                "rule": rule,
                "raw_response": response,
                "notes": json_response.get('notes', '') if 'json_response' in locals() else ''
            }
        )
    
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
    
