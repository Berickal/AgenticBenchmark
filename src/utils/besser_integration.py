"""Utilities for integrating with BESSER Agentic Framework."""
from typing import Any, Dict, Optional, List

try:
    from besser.core.agent import Agent
    from besser.core.state import State
    from besser.core.session import Session
    BESSER_AVAILABLE = True
except ImportError:
    BESSER_AVAILABLE = False
    Agent = None
    State = None
    Session = None


class BESSERAgentWrapper:
    """
    Wrapper to integrate custom agents with BESSER framework.
    
    This allows the benchmark transformation agents to work within
    the BESSER agentic framework if needed.
    """
    
    def __init__(self, agent: Any, besser_config: Optional[Dict] = None):
        """
        Initialize BESSER wrapper.
        
        Args:
            agent: The agent instance to wrap
            besser_config: Optional BESSER configuration
        """
        self.agent = agent
        self.besser_config = besser_config or {}
        self.besser_agent: Optional[Agent] = None
        
        if BESSER_AVAILABLE:
            self._create_besser_agent()
    
    def _create_besser_agent(self):
        """Create a BESSER Agent instance wrapping the custom agent."""
        if not BESSER_AVAILABLE:
            return
        
        try:
            # Create a BESSER agent that delegates to our custom agent
            self.besser_agent = Agent(
                name=self.agent.name,
                description=f"Wrapped agent: {self.agent.name}"
            )
            
            # Add a state that processes inputs
            def process_state(session: Session):
                """State that processes input using the wrapped agent."""
                input_data = session.get_data("input_data")
                result = self.agent.process(input_data)
                session.set_data("result", result)
                return result
            
            # Create processing state
            processing_state = State(
                name="process",
                on_enter=process_state
            )
            
            self.besser_agent.add_state(processing_state)
            self.besser_agent.set_initial_state("process")
            
        except Exception as e:
            print(f"Warning: Could not create BESSER agent wrapper: {e}")
            self.besser_agent = None
    
    def process(self, input_data: Any) -> Any:
        """Process input through wrapped agent."""
        return self.agent.process(input_data)
    
    def process_with_besser(self, input_data: Any, session: Optional[Session] = None) -> Any:
        """
        Process input using BESSER framework if available.
        
        Args:
            input_data: Input to process
            session: Optional BESSER session
            
        Returns:
            Processed result
        """
        if BESSER_AVAILABLE and self.besser_agent is not None:
            try:
                if session is None:
                    session = Session()
                session.set_data("input_data", input_data)
                self.besser_agent.run(session)
                return session.get_data("result")
            except Exception as e:
                print(f"Warning: BESSER processing failed: {e}. Falling back to direct processing.")
                return self.agent.process(input_data)
        else:
            return self.agent.process(input_data)
    
    def register_with_besser(self, framework: Any):
        """
        Register this agent with BESSER framework.
        
        Args:
            framework: BESSER framework instance (if using multi-agent features)
        """
        if not BESSER_AVAILABLE:
            raise ImportError("BESSER framework not available. Please install it first.")
        
        # If framework has agent registration, use it
        if hasattr(framework, 'register_agent'):
            framework.register_agent(self.agent.name, self.besser_agent or self)
        elif hasattr(framework, 'add_agent'):
            framework.add_agent(self.besser_agent or self)


def create_besser_orchestrator(
    framework: Any = None,
    model: str = "llama3.1",
    llm_backend: str = "ollama",
    use_besser_agents: bool = True
):
    """
    Create orchestrator integrated with BESSER framework.
    
    Args:
        framework: Optional BESSER framework instance (for multi-agent coordination)
        model: Model name to use
        llm_backend: LLM backend ("ollama" or "huggingface")
        use_besser_agents: Whether to wrap agents with BESSER
        
    Returns:
        Orchestrator configured for BESSER
    """
    from src.orchestrator import BenchmarkOrchestrator
    
    orchestrator = BenchmarkOrchestrator(
        model=model,
        llm_backend=llm_backend
    )
    
    # Wrap agents with BESSER if requested and available
    if use_besser_agents and BESSER_AVAILABLE:
        # Wrap all agents with BESSER wrappers
        orchestrator.sampling_agent = BESSERAgentWrapper(orchestrator.sampling_agent)
        orchestrator.analysis_agent = BESSERAgentWrapper(orchestrator.analysis_agent)
        orchestrator.transformation_rule_agent = BESSERAgentWrapper(orchestrator.transformation_rule_agent)
        orchestrator.transformation_agent = BESSERAgentWrapper(orchestrator.transformation_agent)
        orchestrator.validation_agent = BESSERAgentWrapper(orchestrator.validation_agent)
        
        # Register with framework if provided
        if framework is not None:
            orchestrator.sampling_agent.register_with_besser(framework)
            orchestrator.analysis_agent.register_with_besser(framework)
            orchestrator.transformation_rule_agent.register_with_besser(framework)
            orchestrator.transformation_agent.register_with_besser(framework)
            orchestrator.validation_agent.register_with_besser(framework)
    
    return orchestrator


def get_besser_availability() -> bool:
    """Check if BESSER framework is available."""
    return BESSER_AVAILABLE
