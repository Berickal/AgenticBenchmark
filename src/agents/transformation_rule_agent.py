"""Transformation rule design agent."""
from typing import Dict, List, Optional
import json

from src.models.task import Task
from src.models.benchmark import BenchmarkSample
from src.agents.base_agent import BaseAgent


class TransformationRuleAgent(BaseAgent):
    """Agent responsible for designing transformation rules."""
    
    def __init__(
        self,
        model: str = "llama3.1",
        llm_backend: str = "ollama",
        openrouter_api_key: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize transformation rule agent.
        
        Args:
            model: Model name (Ollama model name, HuggingFace model ID, or OpenRouter model ID)
            llm_backend: LLM backend to use ("ollama", "huggingface", or "openrouter")
            openrouter_api_key: OpenRouter API key (for OpenRouter backend)
            openrouter_base_url: OpenRouter API base URL (for OpenRouter backend)
        """
        system_prompt = """You are a transformation rule design agent. Your role is to:
        1. Analyze task constraints and input space definitions
        2. Design transformation rules that generate syntactically and semantically 
           different inputs while maintaining complexity of the original inputs
        3. Ensure all transformations respect the input space and constraints
        
        Design rules that are:
        - Applicable within the defined input space
        - Preserve complexity levels
        - Generate diverse syntactic variations
        - Maintain semantic equivalence where appropriate"""
        
        super().__init__(
            name="TransformationRuleAgent",
            model=model,
            system_prompt=system_prompt,
            llm_backend=llm_backend,
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=openrouter_base_url
        )
    
    def process(
        self,
        task: Task,
        analysis: Dict,
        samples: List[BenchmarkSample],
        previous_rules: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Design transformation rules based on analysis.
        
        Args:
            task: Task definition
            analysis: Results from analysis agent
            samples: Benchmark samples for context
            previous_rules: Previously designed rules for iterative refinement
            
        Returns:
            List of transformation rule dictionaries, each containing:
            - name: Rule name
            - description: Rule description
            - applicability: When to apply this rule
            - transformation_function: Description of transformation
        """
        task_context = f"""
Task Description: {task.description}
Input Space Definition: {analysis.get('enhanced_input_space', task.input_space_definition)}
Constraints: {', '.join(analysis.get('enhanced_constraints', task.constraints))}
"""
        
        samples_context = self._prepare_samples_context(samples[:5])  # Few samples for context
        
        previous_rules_context = ""
        if previous_rules:
            previous_rules_context = f"""
Previous Rules (for refinement):
{json.dumps(previous_rules, indent=2)}
"""
        
        prompt = f"""
Design transformation rules for the following task:

{task_context}

Sample inputs and their expected outputs:
{samples_context}

{previous_rules_context}

Design a set of transformation rules that:
1. Are applicable within the defined input space and constraints
2. Generate inputs that are syntactically different but similar in complexity
3. Respect all inputs and outputs constraints

Provide rules in JSON format:
{{
    "rules": [
        {{
            "name": "rule_name",
            "description": "what this rule does",
            "applicability": "when to apply this rule",
            "transformation_function": "how to apply the transformation while preserving output",
            "complexity_preservation": "how complexity is maintained"
        }}
    ]
}}

Generate 3-10 diverse transformation rules.
"""
        
        response = self._call_llm(prompt, temperature=0.7)
        
        try:
            rules_data = self._parse_json_response(response)
            rules = rules_data.get("rules", [])
        except (json.JSONDecodeError, KeyError):
            # Fallback: generate basic rules
            rules = self._generate_fallback_rules(task, analysis)
        
        return rules
    
    def _prepare_samples_context(self, samples: List[BenchmarkSample]) -> str:
        """Prepare context string from samples."""
        if not samples:
            return "No samples provided."
        
        context_lines = []
        for i, sample in enumerate(samples):
            context_lines.append(f"Sample {i+1}:")
            context_lines.append(f"  Input: {str(sample.input)[:150]}")
            context_lines.append(f"  Output: {str(sample.output)[:100]}")
        
        return "\n".join(context_lines)
    
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
    
    def process_single_sample(
        self,
        sample: BenchmarkSample,
        task: Task,
        analysis: Dict,
        general_rules: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate a sample-specific transformation rule for a single sample.
        
        Args:
            sample: The specific sample to generate a rule for
            task: Task definition
            analysis: Results from analysis agent
            general_rules: Optional general rules for reference/inspiration
            
        Returns:
            A single transformation rule dictionary optimized for this sample
        """
        task_context = f"""
Task Description: {task.description}
Input Space Definition: {analysis.get('enhanced_input_space', task.input_space_definition)}
Constraints: {', '.join(analysis.get('enhanced_constraints', task.constraints))}
"""
        
        sample_context = f"""
Sample Input: {sample.input}
Sample Output: {sample.output}
Sample Metadata: {sample.metadata if sample.metadata else 'None'}
"""
        
        general_rules_context = ""
        if general_rules:
            general_rules_context = f"""
General transformation rules (for reference):
{json.dumps(general_rules[:3], indent=2)}
"""
        
        prompt = f"""
Design a SPECIFIC transformation rule tailored for this particular sample:

{task_context}

{sample_context}

{general_rules_context}

CRITICAL: The transformation must preserve the expected output: {sample.output}
The modified input should produce the exact same output as the original.

Analyze this specific sample and design a transformation rule that:
1. Is specifically applicable to THIS sample's structure and content
2. Takes advantage of the unique characteristics of this sample
3. Generates a modified input that is syntactically different but maintains complexity
4. Respects all constraints
5. Is optimized for this sample type
6. MUST preserve the expected output: {sample.output}

Provide a single, sample-specific transformation rule in JSON format:
{{
    "name": "specific_rule_name",
    "description": "what this rule does specifically for this sample",
    "applicability": "when to apply this rule (specific to this sample type)",
    "transformation_function": "detailed description of how to transform THIS specific sample while preserving output {sample.output}",
    "complexity_preservation": "how complexity is maintained for this sample",
    "output_preservation": "how this rule ensures output {sample.output} is preserved",
    "sample_specific_notes": "any specific considerations for this sample"
}}

Generate ONE highly tailored transformation rule.
"""
        
        response = self._call_llm(prompt, temperature=0.8)
        
        try:
            rule_data = self._parse_json_response(response)
            # If response is a list, take first element
            if isinstance(rule_data, list):
                rule = rule_data[0] if rule_data else {}
            # If response has a "rule" key
            elif "rule" in rule_data:
                rule = rule_data["rule"]
            # Otherwise assume the whole response is the rule
            else:
                rule = rule_data
            
            # Ensure required fields exist
            if not rule.get("name"):
                rule["name"] = f"SampleSpecific_{hash(str(sample.input)) % 10000}"
            if not rule.get("transformation_function"):
                rule["transformation_function"] = "Apply sample-specific transformation"
                
        except (json.JSONDecodeError, KeyError):
            # Fallback: generate a basic sample-specific rule
            rule = self._generate_sample_fallback_rule(sample, task, analysis)
        
        return rule
    
    def _generate_fallback_rules(
        self,
        task: Task,
        analysis: Dict
    ) -> List[Dict]:
        """Generate fallback transformation rules."""
        return [
            {
                "name": "syntactic_rephrasing",
                "description": "Rephrase input while maintaining meaning and complexity",
                "applicability": "When input contains natural language",
                "transformation_function": "Rephrase sentences, change word order, use synonyms",
                "complexity_preservation": "Maintain same grammatical complexity and vocabulary level"
            },
            {
                "name": "structural_variation",
                "description": "Vary the structure while keeping logical equivalence",
                "applicability": "When input has structural elements",
                "transformation_function": "Reorganize structure, change ordering, use equivalent structures",
                "complexity_preservation": "Maintain same number of operations and dependencies"
            },
            {
                "name": "notation_change",
                "description": "Change notation or representation format",
                "applicability": "When multiple equivalent notations exist",
                "transformation_function": "Change notation systems, use alternative representations",
                "complexity_preservation": "Maintain same information content and computational complexity"
            }
        ]
    
    def _generate_sample_fallback_rule(
        self,
        sample: BenchmarkSample,
        task: Task,
        analysis: Dict
    ) -> Dict:
        """Generate a fallback rule for a specific sample."""
        sample_str = str(sample.input)[:100]
        return {
            "name": f"SampleSpecific_{hash(sample_str) % 10000}",
            "description": f"Sample-specific transformation for: {sample_str[:50]}...",
            "applicability": "Specific to this sample type",
            "transformation_function": "Apply syntactic variation while preserving meaning and complexity",
            "complexity_preservation": "Maintain same complexity level as original",
            "sample_specific_notes": "Fallback rule generated due to parsing error"
        }
