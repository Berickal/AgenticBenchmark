"""Analysis agent for benchmark analysis and constraint enhancement."""
from typing import Dict, List, Optional
import json
import logging
from ast import literal_eval

from src.models.task import Task
from src.models.benchmark import BenchmarkSample
from src.agents.base_agent import BaseAgent


class AnalysisAgent(BaseAgent):
    """Agent responsible for analyzing benchmarks and enhancing constraints."""
    
    def __init__(
        self,
        model: str = "llama3.1",
        llm_backend: str = "ollama",
        openrouter_api_key: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize analysis agent.
        
        Args:
            model: Model name (Ollama model name, HuggingFace model ID, or OpenRouter model ID)
            llm_backend: LLM backend to use ("ollama", "huggingface", or "openrouter")
            openrouter_api_key: OpenRouter API key (for OpenRouter backend)
            openrouter_base_url: OpenRouter API base URL (for OpenRouter backend)
        """
        system_prompt = """You are an analysis agent specialized in understanding benchmarks 
        and tasks. Your role is to:
        1. Analyze benchmark samples to understand patterns and structures
        2. Enhance task constraints based on observed patterns
        3. Refine input space definitions to ensure generated inputs are valid
        
        Always provide structured, actionable insights."""
        
        super().__init__(
            name="AnalysisAgent",
            model=model,
            system_prompt=system_prompt,
            llm_backend=llm_backend,
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=openrouter_base_url
        )
    
    def process(
        self,
        task: Task,
        samples: List[BenchmarkSample],
        previous_analysis: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze benchmark samples and enhance task constraints.
        
        Args:
            task: The task definition
            samples: Sampled benchmark samples to analyze
            previous_analysis: Previous analysis results for iterative refinement
            
        Returns:
            Dictionary containing:
            - enhanced_constraints: List of enhanced constraints
            - enhanced_input_space: Enhanced input space definition
            - analysis_insights: Key insights from the analysis
        """
        # Prepare context for LLM
        samples_context = self._prepare_samples_context(samples)
        task_context = f"""
Task Description: {task.description}
Current Constraints: {', '.join(task.constraints)}
Current Input Space Definition: {task.input_space_definition}
Current Output Space Definition: {task.output_space_definition}
"""
        
        previous_context = ""
        if previous_analysis:
            previous_context = f"""
Previous Analysis:
- Enhanced Constraints: {previous_analysis.get('enhanced_constraints', [])}
- Previous Insights: {previous_analysis.get('analysis_insights', '')}
"""
        
        prompt = f"""
Analyze the following benchmark samples and task definition:

{task_context}

{samples_context}

{previous_context}

Based on this analysis, provide:
1. Enhanced constraints that should be applied when generating modified inputs
   - Include constraints about preserving expected outputs
   - Consider input-output relationships
2. Refined input space definition that captures the valid input patterns
   - Consider what input variations are possible while maintaining outputs
3. Key insights about the benchmark structure and patterns
   - Include insights about input-output relationships
   - Note patterns that help preserve outputs during transformation

Respond in JSON format with the following structure:
{{
    "enhanced_constraints": ["constraint1", "constraint2", ...],
    "enhanced_input_space": "detailed input space definition",
    "analysis_insights": "key insights about patterns and structures, including input-output relationships"
}}
"""
        
        response = self._call_llm(prompt, temperature=0.3)
        
        # Parse JSON response
        try:
            analysis_result = self._parse_json_response(response)
        except json.JSONDecodeError:
            # Fallback: extract information manually
            analysis_result = self._extract_analysis_fallback(response, task)

        logging.debug(f"Analysis Agent Response: {response}")
        return {
            "enhanced_constraints": analysis_result.get("enhanced_constraints") + task.constraints if analysis_result.get("enhanced_constraints") else task.constraints,
            "enhanced_input_space": analysis_result.get("enhanced_input_space", task.input_space_definition),
            "analysis_insights": analysis_result.get("analysis_insights", ""),
            "raw_response": response
        }
    
    def _prepare_samples_context(self, samples: List[BenchmarkSample]) -> str:
        """Prepare context string from samples."""
        if not samples:
            return "No samples provided."
        
        context_lines = ["Benchmark Samples:"]
        for i, sample in enumerate(samples[:10]):  # Limit to first 10 for context
            context_lines.append(f"\nSample {i+1}:")
            context_lines.append(f"  Input: {str(sample.input)[:200]}...")
            context_lines.append(f"  Output: {str(sample.output)[:200]}...")
            if sample.metadata:
                context_lines.append(f"  Metadata: {sample.metadata}")
        
        if len(samples) > 10:
            context_lines.append(f"\n... and {len(samples) - 10} more samples")
        
        return "\n".join(context_lines)
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        # Try to extract JSON from response
        response = response.strip()
        
        # Find JSON block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()
        
        return json.loads(response)
    
    def _extract_analysis_fallback(
        self,
        response: str,
        task: Task
    ) -> Dict:
        """Fallback extraction if JSON parsing fails."""
        # Simple fallback - return original task with some enhancements
        return {
            "enhanced_constraints": task.constraints + [
                "Generated inputs must maintain semantic equivalence",
                "Generated inputs must preserve complexity level"
            ],
            "enhanced_input_space": task.input_space_definition,
            "analysis_insights": response[:500]  # First 500 chars as insight
        }
