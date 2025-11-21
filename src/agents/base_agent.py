"""Base agent class for BESSER framework integration."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import os

try:
    from besser.core.agent import Agent
    from besser.nlp.llm import LLM
    from besser.nlp.llm_huggingface import LLMHuggingFace
    BESSER_AVAILABLE = True
except ImportError:
    BESSER_AVAILABLE = False
    # Fallback for when BESSER is not available
    Agent = ABC
    LLM = None
    LLMHuggingFace = None

# Fallback to Ollama if BESSER is not available or for Ollama-specific models
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

# OpenRouter support (OpenAI-compatible API)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


class BaseAgent(ABC):
    """Base class for all agents in the framework, integrated with BESSER."""
    
    def __init__(
        self,
        name: str,
        model: str = "llama3.2",
        system_prompt: Optional[str] = None,
        llm_backend: str = "ollama",  # "ollama", "huggingface", or "openrouter"
        ollama_client: Optional[Any] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            model: Model name to use (Ollama model name, HuggingFace model ID, or OpenRouter model ID)
            system_prompt: System prompt for the agent
            llm_backend: Backend to use ("ollama", "huggingface", or "openrouter")
            ollama_client: Optional Ollama client instance (for Ollama backend)
            openrouter_api_key: OpenRouter API key (for OpenRouter backend, or use OPENAI_API_KEY env var)
            openrouter_base_url: OpenRouter API base URL (default: https://openrouter.ai/api/v1)
        """
        self.name = name
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.llm_backend = llm_backend
        
        # Initialize LLM based on backend
        if llm_backend == "huggingface" and BESSER_AVAILABLE:
            try:
                # Use BESSER's HuggingFace LLM
                self.llm = LLMHuggingFace(model=model)
            except Exception as e:
                print(f"Warning: Could not initialize HuggingFace LLM: {e}. Falling back to Ollama.")
                self.llm_backend = "ollama"
                self.llm = None
        else:
            self.llm = None
        
        # Initialize Ollama client if needed
        if self.llm_backend == "ollama":
            if OLLAMA_AVAILABLE:
                self.client = ollama_client or ollama
            else:
                raise RuntimeError("Ollama is not available. Please install it with: pip install ollama")
        
        # Initialize OpenRouter client if needed
        if self.llm_backend == "openrouter":
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI library is not available. Please install it with: pip install openai")
            
            # Get API key from parameter or environment variable
            api_key = openrouter_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OpenRouter API key is required. Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable, "
                    "or pass openrouter_api_key parameter."
                )
            
            # Initialize OpenAI client with OpenRouter settings
            self.openai_client = openai.OpenAI(
                api_key=api_key,
                base_url=openrouter_base_url
            )
        
    def _default_system_prompt(self) -> str:
        """Default system prompt for the agent."""
        return f"You are {self.name}, a specialized agent in the benchmark transformation framework."
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return output."""
        pass
    
    def _call_llm(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Call LLM with the given prompt using BESSER framework or Ollama.
        
        Args:
            prompt: User prompt
            context: Optional context to include
            temperature: Sampling temperature
            
        Returns:
            LLM response text
        """
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {context}\n\n{prompt}"
        
        # Use BESSER LLM if available
        if self.llm is not None and BESSER_AVAILABLE:
            try:
                # BESSER LLM interface - try different possible API signatures
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": full_prompt}
                ]
                
                # Try chat() method first
                if hasattr(self.llm, 'chat'):
                    response = self.llm.chat(messages=messages, temperature=temperature)
                # Try generate() method as alternative
                elif hasattr(self.llm, 'generate'):
                    response = self.llm.generate(prompt=full_prompt, system_prompt=self.system_prompt, temperature=temperature)
                else:
                    raise AttributeError("LLM does not have chat or generate method")
                
                if isinstance(response, str):
                    return response
                # Handle structured responses - try to extract text
                if hasattr(response, 'text'):
                    return response.text
                if hasattr(response, 'content'):
                    return response.content
                if hasattr(response, 'message'):
                    return str(response.message)
                # Handle dictionary responses
                if isinstance(response, dict):
                    return response.get('text', response.get('content', response.get('message', str(response))))
                # Handle structured objects
                return str(response)
            except Exception as e:
                print(f"Warning: BESSER LLM call failed: {e}. Falling back to Ollama.")
        
        # Fallback to Ollama
        if self.llm_backend == "ollama" and OLLAMA_AVAILABLE:
            try:
                # Use chat API for better compatibility
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    options={
                        "temperature": temperature,
                    }
                )
                return response.get("message", {}).get("content", "")
            except AttributeError:
                # Fallback to generate API if chat doesn't exist
                try:
                    response = self.client.generate(
                        model=self.model,
                        prompt=full_prompt,
                        system=self.system_prompt,
                        options={
                            "temperature": temperature,
                        }
                    )
                    return response.get("response", "")
                except Exception as e:
                    raise RuntimeError(f"Error calling Ollama model {self.model}: {e}")
            except Exception as e:
                raise RuntimeError(f"Error calling Ollama model {self.model}: {e}")
        
        # OpenRouter backend
        elif self.llm_backend == "openrouter" and OPENAI_AVAILABLE:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(f"Error calling OpenRouter model {self.model}: {e}")
        
        else:
            raise RuntimeError(f"No LLM backend available. Please install BESSER, Ollama, or OpenAI (for OpenRouter).")
    
    def _call_llm_stream(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Stream response from LLM.
        
        Args:
            prompt: User prompt
            context: Optional context to include
            temperature: Sampling temperature
            
        Yields:
            Response chunks
        """
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {context}\n\n{prompt}"
        
        # Use BESSER LLM streaming if available
        if self.llm is not None and BESSER_AVAILABLE:
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": full_prompt}
                ]
                
                # Try different streaming method names
                if hasattr(self.llm, 'chat_stream'):
                    stream = self.llm.chat_stream(messages=messages, temperature=temperature)
                elif hasattr(self.llm, 'stream'):
                    stream = self.llm.stream(prompt=full_prompt, system_prompt=self.system_prompt, temperature=temperature)
                elif hasattr(self.llm, 'generate_stream'):
                    stream = self.llm.generate_stream(prompt=full_prompt, system_prompt=self.system_prompt, temperature=temperature)
                else:
                    raise AttributeError("LLM does not have streaming method")
                
                for chunk in stream:
                    if isinstance(chunk, str):
                        yield chunk
                    elif isinstance(chunk, dict):
                        # Extract text from dictionary chunks
                        yield chunk.get('text', chunk.get('content', chunk.get('delta', str(chunk))))
                    elif hasattr(chunk, 'text'):
                        yield chunk.text
                    elif hasattr(chunk, 'content'):
                        yield chunk.content
                    else:
                        yield str(chunk)
                return
            except Exception as e:
                print(f"Warning: BESSER LLM streaming failed: {e}. Falling back to Ollama.")
        
        # Fallback to Ollama streaming
        if self.llm_backend == "ollama" and OLLAMA_AVAILABLE:
            try:
                # Use chat API for streaming
                stream = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    options={
                        "temperature": temperature,
                    },
                    stream=True
                )
                for chunk in stream:
                    if "message" in chunk:
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    elif "response" in chunk:
                        yield chunk["response"]
            except AttributeError:
                # Fallback to generate API
                try:
                    stream = self.client.generate(
                        model=self.model,
                        prompt=full_prompt,
                        system=self.system_prompt,
                        options={
                            "temperature": temperature,
                        },
                        stream=True
                    )
                    for chunk in stream:
                        if "response" in chunk:
                            yield chunk["response"]
                except Exception as e:
                    raise RuntimeError(f"Error calling Ollama model {self.model}: {e}")
            except Exception as e:
                raise RuntimeError(f"Error calling Ollama model {self.model}: {e}")
        
        # OpenRouter streaming
        elif self.llm_backend == "openrouter" and OPENAI_AVAILABLE:
            try:
                stream = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=temperature,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                raise RuntimeError(f"Error calling OpenRouter model {self.model}: {e}")
        
        else:
            raise RuntimeError(f"No LLM backend available for streaming.")
