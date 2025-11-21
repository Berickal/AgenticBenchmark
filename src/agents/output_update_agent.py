"""Output update agent for determining if outputs should change."""
from typing import Any, Dict, Optional

from src.models.task import Task
from src.models.benchmark import ModifiedInput
from src.agents.base_agent import BaseAgent


class OutputUpdateAgent(BaseAgent):
    """Agent responsible for determining if the output should be updated after transformation."""
    
    def __init__(
        self,
        model: str = "llama3.1",
        llm_backend: str = "ollama",
        openrouter_api_key: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize output update agent.
        
        Args:
            model: Model name (Ollama model name, HuggingFace model ID, or OpenRouter model ID)
            llm_backend: LLM backend to use ("ollama", "huggingface", or "openrouter")
            openrouter_api_key: OpenRouter API key (for OpenRouter backend)
            openrouter_base_url: OpenRouter API base URL (for OpenRouter backend)
        """
        system_prompt = """You are an output update agent. Your role is to:
        1. Analyze transformed inputs to determine if the expected output should change
        2. Determine if the transformation legitimately changes the answer/result
        3. Generate updated outputs when appropriate
        
        You should update the output when:
        - The transformation changes the question/problem in a way that requires a different answer
        - The modified input asks a different question that has a different correct answer
        - The transformation alters the problem such that the original answer is no longer correct
        
        Preserve the original output when:
        - The transformation only changes wording/phrasing but asks the same question
        - The transformation maintains semantic equivalence
        - The correct answer remains the same despite input changes"""
        
        super().__init__(
            name="OutputUpdateAgent",
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
        analysis: Dict
    ) -> Optional[Any]:
        """
        Determine if the output should be updated and return the updated output if needed.
        
        Args:
            modified_input: The modified input with transformation
            task: Task definition
            analysis: Analysis results with constraints
            
        Returns:
            Updated output if the output should change, None if original output should be preserved
        """
        constraints = analysis.get('enhanced_constraints', task.constraints)
        output_space = task.output_space_definition
        
        prompt = f"""
Analyze the following transformation to determine if the expected output should change:

Original Input: {modified_input.original_input}
Original Output: {modified_input.original_output}
Modified Input: {modified_input.modified_input}
Transformation Applied: {modified_input.transformation_applied}

Task Constraints:
{chr(10).join(f'- {c}' for c in constraints)}

Output Space Definition: {output_space}

CRITICAL ANALYSIS:
Carefully compare the original and modified inputs:

1. QUESTION ANALYSIS:
   - Extract the core question from the original input
   - Extract the core question from the modified input
   - Are they asking the SAME question? → Preserve output
   - Are they asking a DIFFERENT question? → Update output

2. ANSWER VALIDITY CHECK:
   - Would the original answer "{modified_input.original_output}" be correct for the modified input?
   - If NO → Output MUST be updated
   - If YES → Output can be preserved

3. SEMANTIC CHANGE DETECTION:
   - Check if key facts/numbers/entities changed:
     * Numbers changed (e.g., "2+2" → "3+3") → UPDATE OUTPUT
     * Entities changed (e.g., "France" → "Germany") → UPDATE OUTPUT  
     * Problem structure changed → UPDATE OUTPUT
   - Only wording/phrasing changed → PRESERVE OUTPUT

Examples for MMLU (multiple choice):
- Original: "What is the capital of France? A) Paris B) London C) Berlin D) Madrid" → Answer: A
- Modified: "Which city serves as France's capital? A) Paris B) London C) Berlin D) Madrid" → PRESERVE (A)
- Modified: "What is the capital of Germany? A) Paris B) London C) Berlin D) Madrid" → UPDATE (C)

- Original: "Find c in Z_3 such that Z_3[x]/(x^2 + c) is a field" → Answer: 1
- Modified: "Find c in Z_3 such that Z_3[x]/(x^2 + c) is a field" (reworded) → PRESERVE (1)
- Modified: "Find d in Z_3 such that Z_3[y]/(y^2 + d) is an integral domain" → UPDATE (different question)

Respond in JSON format:
{{
    "output_changed": true/false,
    "updated_output": "new output value if changed, null if preserved",
    "reason": "clear explanation of why output changed or why it should be preserved"
}}

IMPORTANT: 
- Set "output_changed": true ONLY if the question/problem fundamentally changed
- Set "updated_output" to the actual new output value (not the string "null")
- If preserving, set "output_changed": false and "updated_output": null
"""
        
        response = self._call_llm(prompt, temperature=0.3)
        
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"OutputUpdateAgent response: {response[:300]}")
        
        try:
            result = self._parse_json_response(response)
            output_changed = result.get("output_changed", False)
            updated_output = result.get("updated_output")
            reason = result.get("reason", "")
            
            logger.debug(f"Parsed result - output_changed: {output_changed}, updated_output: {updated_output}")
            
            # Handle different formats of "null" or None
            if updated_output in [None, "null", "None", "", "PRESERVE_OUTPUT", "SAME_OUTPUT"]:
                updated_output = None
            
            if output_changed and updated_output is not None:
                # Store reason in metadata
                if modified_input.metadata is None:
                    modified_input.metadata = {}
                modified_input.metadata["output_update_reason"] = reason
                logger.info(f"OutputUpdateAgent: Output changed from '{modified_input.original_output}' to '{updated_output}'")
                logger.info(f"  Reason: {reason}")
                return updated_output
            
            logger.debug(f"OutputUpdateAgent: Preserving original output '{modified_input.original_output}'")
            return None  # Preserve original output
            
        except Exception as e:
            # Log the error for debugging
            logger.warning(f"Output update agent parsing error: {e}")
            logger.debug(f"Response was: {response[:500]}")
            # If parsing fails, default to preserving original output
            return None
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        import json
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

