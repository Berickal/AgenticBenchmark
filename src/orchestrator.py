"""Main orchestrator for the multi-agent benchmark transformation framework."""
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from src.models.task import Task
from src.models.benchmark import Benchmark, BenchmarkSample, ModifiedInput
from src.agents.sampling_agent import SamplingAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.transformation_rule_agent import TransformationRuleAgent
from src.agents.transformation_agent import TransformationAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.output_update_agent import OutputUpdateAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransformationResult:
    """Result of a transformation attempt."""
    modified_input: ModifiedInput
    validation_result: Dict
    is_valid: bool
    iteration: int


class BenchmarkOrchestrator:
    """
    Main orchestrator that coordinates the multi-agent framework.
    
    Implements the methodology with feedback loops:
    1. Sampling -> Analysis -> Transformation Rule Design -> Transformation Application -> Validation
    2. Feedback loops for iterative refinement
    """
    
    def __init__(
        self,
        model: str = "llama3.1",
        max_iterations: int = 3,
        max_retries_per_sample: int = 5,
        llm_backend: str = "ollama",
        use_per_sample_rules: bool = False,
        openrouter_api_key: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize orchestrator.
        
        Args:
            model: Model name to use for all agents (Ollama model name, HuggingFace model ID, or OpenRouter model ID)
            max_iterations: Maximum iterations for feedback loops
            max_retries_per_sample: Maximum retries to generate valid transformation per sample
            llm_backend: LLM backend to use ("ollama", "huggingface", or "openrouter")
            use_per_sample_rules: If True, generate a specific rule for each sample. If False, use general rules.
            openrouter_api_key: OpenRouter API key (for OpenRouter backend, or use OPENAI_API_KEY env var)
            openrouter_base_url: OpenRouter API base URL (default: https://openrouter.ai/api/v1)
        """
        self.model = model
        self.max_iterations = max_iterations
        self.max_retries_per_sample = max_retries_per_sample
        self.llm_backend = llm_backend
        self.use_per_sample_rules = use_per_sample_rules
        
        # Initialize agents with backend-specific parameters
        agent_kwargs = {"model": model, "llm_backend": llm_backend}
        if llm_backend == "openrouter":
            agent_kwargs["openrouter_api_key"] = openrouter_api_key
            agent_kwargs["openrouter_base_url"] = openrouter_base_url
        
        self.sampling_agent = SamplingAgent(**agent_kwargs)
        self.analysis_agent = AnalysisAgent(**agent_kwargs)
        self.transformation_rule_agent = TransformationRuleAgent(**agent_kwargs)
        self.transformation_agent = TransformationAgent(**agent_kwargs)
        self.output_update_agent = OutputUpdateAgent(**agent_kwargs)
        self.validation_agent = ValidationAgent(**agent_kwargs)
        
        # State tracking
        self.current_analysis: Optional[Dict] = None
        self.current_rules: Optional[List[Dict]] = None
        self.transformation_history: List[TransformationResult] = []
    
    def transform_benchmark(
        self,
        benchmark: Benchmark,
        task: Task,
        num_outputs: Optional[int] = None
    ) -> List[ModifiedInput]:
        """
        Transform a benchmark to generate modified inputs.
        
        The workflow:
        1. Sample a representative subset for analysis
        2. Analyze the subset to enhance constraints and input space definition
        3. Transform ALL samples in the benchmark using the enhanced analysis
        
        Args:
            benchmark: Original benchmark
            task: Task definition
            num_outputs: Number of valid modified inputs to generate. If None, will transform
                       all samples in the benchmark.
            
        Returns:
            List of valid ModifiedInput objects
        """
        logger.info(f"Starting benchmark transformation for {benchmark.name}")
        logger.info(f"Total samples in benchmark: {len(benchmark.samples)}")
        
        # Step 1: Sampling for analysis
        logger.info("Step 1: Sampling benchmark for analysis...")
        analysis_samples = self.sampling_agent.process(benchmark)
        logger.info(f"Sampled {len(analysis_samples)} samples for analysis")
        
        if not analysis_samples:
            logger.warning("No samples collected for analysis, returning empty list")
            return []
        
        # Always transform all samples in the benchmark
        transformation_samples = benchmark.samples
        logger.info(f"Will transform all {len(transformation_samples)} samples in benchmark")
        
        # Step 2: Initial Analysis (using sampled samples)
        logger.info("Step 2: Analyzing benchmark using sampled samples...")
        self.current_analysis = self.analysis_agent.process(task, analysis_samples)
        logger.info("Analysis completed")
        
        # Log constraint enhancements
        original_constraints = task.constraints
        enhanced_constraints = self.current_analysis.get('enhanced_constraints', original_constraints)
        original_input_space = task.input_space_definition
        enhanced_input_space = self.current_analysis.get('enhanced_input_space', original_input_space)
        analysis_insights = self.current_analysis.get('analysis_insights', '')
        
        logger.info("=" * 80)
        logger.info("CONSTRAINT ENHANCEMENT RESULTS:")
        logger.info("=" * 80)
        logger.info(f"Original Constraints ({len(original_constraints)}):")
        for i, constraint in enumerate(original_constraints, 1):
            logger.info(f"  {i}. {constraint}")
        
        logger.info(f"\nEnhanced Constraints ({len(enhanced_constraints)}):")
        for i, constraint in enumerate(enhanced_constraints, 1):
            logger.info(f"  {i}. {constraint}")
        
        logger.info(f"\nOriginal Input Space Definition:")
        logger.info(f"  {original_input_space}")
        
        logger.info(f"\nEnhanced Input Space Definition:")
        logger.info(f"  {enhanced_input_space}")
        
        if analysis_insights:
            logger.info(f"\nAnalysis Insights:")
            # Truncate if too long
            insights_str = str(analysis_insights)
            if len(insights_str) > 500:
                logger.info(f"  {insights_str[:500]}...")
            else:
                logger.info(f"  {insights_str}")
        
        logger.info("=" * 80)
        
        # Step 3: Design transformation rules (using sampled samples for context)
        if self.use_per_sample_rules:
            logger.info("Step 3: Will generate per-sample transformation rules during transformation")
            # Still generate general rules for reference/inspiration
            self.current_rules = self.transformation_rule_agent.process(
                task,
                self.current_analysis,
                analysis_samples
            )
            logger.info(f"Generated {len(self.current_rules)} general rules for reference")
        else:
            logger.info("Step 3: Designing general transformation rules...")
            self.current_rules = self.transformation_rule_agent.process(
                task,
                self.current_analysis,
                analysis_samples
            )
            logger.info(f"Designed {len(self.current_rules)} transformation rules")
        
        # Step 4: Apply transformations with validation and feedback loops
        logger.info("Step 4: Applying transformations...")
        valid_results = []
        
        num_outputs = num_outputs or len(transformation_samples)
        
        iteration = 0
        while len(valid_results) < num_outputs and iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            
            # Try to generate valid transformations for each sample
            for sample_idx, sample in enumerate(transformation_samples):
                if len(valid_results) >= num_outputs:
                    break
                
                sample_processed = False
                
                if self.use_per_sample_rules:
                    # Generate a specific rule for this sample
                    logger.info(f"Generating sample-specific rule for sample {sample_idx+1}/{len(transformation_samples)}")
                    sample_rule = self.transformation_rule_agent.process_single_sample(
                        sample,
                        task,
                        self.current_analysis,
                        general_rules=self.current_rules
                    )
                    logger.info(f"Generated rule '{sample_rule.get('name', 'unknown')}' for sample {sample_idx+1}/{len(transformation_samples)}")
                    rules_to_try = [sample_rule]
                else:
                    # Use general rules with rotation
                    rule_start_idx = sample_idx % len(self.current_rules) if self.current_rules else 0
                    rotated_rules = self.current_rules[rule_start_idx:] + self.current_rules[:rule_start_idx]
                    rules_to_try = rotated_rules
                
                # Try each transformation rule
                for rule in rules_to_try:
                    if len(valid_results) >= num_outputs or sample_processed:
                        break
                    
                    retry_count = 0
                    while retry_count < self.max_retries_per_sample:
                        # Apply transformation
                        modified_input = self.transformation_agent.process(
                            sample,
                            rule,
                            task,
                            self.current_analysis
                        )
                        
                        # Check if output should be updated
                        updated_output = self.output_update_agent.process(
                            modified_input,
                            task,
                            self.current_analysis
                        )
                        
                        if updated_output is not None:
                            modified_input.updated_output = updated_output
                            logger.info(f"Output updated for sample {sample_idx+1}: '{sample.output}' -> '{updated_output}'")
                            if modified_input.metadata and "output_update_reason" in modified_input.metadata:
                                logger.info(f"  Reason: {modified_input.metadata['output_update_reason']}")
                        else:
                            logger.debug(f"Output preserved for sample {sample_idx+1}: '{sample.output}'")
                        
                        # Validate (use updated output if available, otherwise original)
                        expected_output = modified_input.get_expected_output()
                        validation_result = self.validation_agent.process(
                            modified_input,
                            task,
                            self.current_analysis,
                            expected_output
                        )
                        
                        result = TransformationResult(
                            modified_input=modified_input,
                            validation_result=validation_result,
                            is_valid=validation_result.get("is_valid", False),
                            iteration=iteration
                        )
                        
                        self.transformation_history.append(result)
                        
                        if result.is_valid:
                            valid_results.append(modified_input)
                            rule_name = rule.get('name', 'unknown')
                            logger.info(f"Generated valid transformation {len(valid_results)}/{num_outputs} for sample {sample_idx+1}/{len(transformation_samples)} using rule '{rule_name}'")
                            sample_processed = True
                            break
                        else:
                            retry_count += 1
                            logger.debug(f"Invalid transformation, retry {retry_count}/{self.max_retries_per_sample}")
            
            # Feedback loop: Refine analysis and rules if needed
            if len(valid_results) < num_outputs and iteration < self.max_iterations:
                logger.info("Refining analysis and rules based on validation feedback...")
                
                # Refine analysis (still using analysis_samples)
                previous_analysis = self.current_analysis
                self.current_analysis = self.analysis_agent.process(
                    task,
                    analysis_samples,
                    previous_analysis=previous_analysis
                )
                
                # Log refined constraints
                refined_constraints = self.current_analysis.get('enhanced_constraints', [])
                refined_input_space = self.current_analysis.get('enhanced_input_space', '')
                logger.info(f"Refined Constraints ({len(refined_constraints)}):")
                for i, constraint in enumerate(refined_constraints, 1):
                    logger.info(f"  {i}. {constraint}")
                # Convert to string and truncate if needed
                refined_input_space_str = str(refined_input_space)
                if len(refined_input_space_str) > 200:
                    logger.info(f"Refined Input Space: {refined_input_space_str[:200]}...")
                else:
                    logger.info(f"Refined Input Space: {refined_input_space_str}")
                
                # Refine rules (still using analysis_samples for context)
                self.current_rules = self.transformation_rule_agent.process(
                    task,
                    self.current_analysis,
                    analysis_samples,
                    previous_rules=self.current_rules
                )
        
        logger.info(f"Transformation complete. Generated {len(valid_results)} valid modified inputs")
        return valid_results
    
    def get_statistics(self) -> Dict:
        """Get statistics about the transformation process."""
        total_attempts = len(self.transformation_history)
        valid_count = sum(1 for r in self.transformation_history if r.is_valid)
        
        return {
            "total_attempts": total_attempts,
            "valid_transformations": valid_count,
            "invalid_transformations": total_attempts - valid_count,
            "success_rate": valid_count / total_attempts if total_attempts > 0 else 0.0,
            "iterations_performed": max((r.iteration for r in self.transformation_history), default=0)
        }

