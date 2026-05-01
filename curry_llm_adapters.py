"""
Curry LLM Adapters: Integration with language model APIs.
Provides deterministic, versioned inference recording.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMAdapter(ABC):
    """Abstract base class for LLM service adapters."""
    
    def __init__(self, curry_db):
        """Initialize adapter with Curry database reference."""
        self.db = curry_db
    
    @abstractmethod
    def infer(
        self,
        model_name: str,
        model_version: int,
        prompt: str,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run inference and return output."""
        pass
    
    @abstractmethod
    def infer_and_record(
        self,
        model_name: str,
        model_version: int,
        prompt: str,
        seed: int = 42,
    ) -> str:
        """Run inference, record to Curry, return inference_id."""
        pass


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API (GPT models)."""
    
    def __init__(self, curry_db, api_key: Optional[str] = None):
        """Initialize OpenAI adapter."""
        super().__init__(curry_db)
        self.api_key = api_key
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    def infer(
        self,
        model_name: str,
        model_version: int,
        prompt: str,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run OpenAI inference with locked parameters."""
        # Get model config from Curry
        model_config = self.db.get_model(model_name, model_version)
        
        start_time = time.time()
        
        try:
            # Call OpenAI API with exact parameters from model version
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=model_config["temperature"],
                top_p=model_config["top_p"],
                max_tokens=model_config["max_tokens"],
                seed=seed,  # Deterministic seed (OpenAI supports this for reproducibility)
            )
            
            output_text = response.choices[0].message.content
            duration_ms = int((time.time() - start_time) * 1000)
            usage = getattr(response, "usage", None)
            request_id = getattr(response, "id", None)
            
            return {
                "success": True,
                "output": output_text,
                "duration_ms": duration_ms,
                "tokens_used": usage.total_tokens if usage else None,
                "input_tokens_count": usage.prompt_tokens if usage else None,
                "output_tokens_count": usage.completion_tokens if usage else None,
                "request_id": request_id,
                "model_response": response,
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "provider_error_type": e.__class__.__name__,
                "duration_ms": int((time.time() - start_time) * 1000),
            }
    
    def infer_and_record(
        self,
        model_name: str,
        model_version: int,
        prompt: str,
        seed: int = 42,
    ) -> str:
        """Run inference and record to Curry database."""
        result = self.infer(model_name, model_version, prompt, seed)
        
        if not result["success"]:
            raise RuntimeError(f"Inference failed: {result['error']}")
        
        # Record inference with full provenance
        inference_id = self.db.record_inference(
            model_name=model_name,
            model_version=model_version,
            input_tokens=prompt,  # In production, would be actual tokens
            output_tokens=result["output"].encode('utf-8'),
            seed=seed,
            temperature_used=None,
            top_p_used=None,
            duration_ms=result["duration_ms"],
            metadata={
                "tokens_used": result["tokens_used"],
                "input_tokens_count": result.get("input_tokens_count"),
                "output_tokens_count": result.get("output_tokens_count"),
                "api_provider": "openai",
                "request_id": result.get("request_id"),
                "prompt_text": prompt,
            },
        )
        
        return inference_id


class ClaudeAdapter(LLMAdapter):
    """Adapter for Anthropic Claude API."""
    
    def __init__(self, curry_db, api_key: Optional[str] = None):
        """Initialize Claude adapter."""
        super().__init__(curry_db)
        self.api_key = api_key
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
    
    def infer(
        self,
        model_name: str,
        model_version: int,
        prompt: str,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run Claude inference with locked parameters."""
        model_config = self.db.get_model(model_name, model_version)
        
        start_time = time.time()
        
        try:
            # Call Claude API
            response = self.client.messages.create(
                model=model_name,
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
                top_p=model_config["top_p"],
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            
            output_text = response.content[0].text
            duration_ms = int((time.time() - start_time) * 1000)
            usage = getattr(response, "usage", None)
            request_id = getattr(response, "id", None)
            
            return {
                "success": True,
                "output": output_text,
                "duration_ms": duration_ms,
                "tokens_used": usage.output_tokens if usage else None,
                "input_tokens_count": usage.input_tokens if usage else None,
                "output_tokens_count": usage.output_tokens if usage else None,
                "request_id": request_id,
                "model_response": response,
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "provider_error_type": e.__class__.__name__,
                "duration_ms": int((time.time() - start_time) * 1000),
            }
    
    def infer_and_record(
        self,
        model_name: str,
        model_version: int,
        prompt: str,
        seed: int = 42,
    ) -> str:
        """Run inference and record to Curry database."""
        result = self.infer(model_name, model_version, prompt, seed)
        
        if not result["success"]:
            raise RuntimeError(f"Inference failed: {result['error']}")
        
        # Record inference
        inference_id = self.db.record_inference(
            model_name=model_name,
            model_version=model_version,
            input_tokens=prompt,
            output_tokens=result["output"].encode('utf-8'),
            seed=seed,
            temperature_used=None,
            top_p_used=None,
            duration_ms=result["duration_ms"],
            metadata={
                "tokens_used": result["tokens_used"],
                "input_tokens_count": result.get("input_tokens_count"),
                "output_tokens_count": result.get("output_tokens_count"),
                "api_provider": "anthropic",
                "request_id": result.get("request_id"),
                "prompt_text": prompt,
            },
        )
        
        return inference_id


class LocalModelAdapter(LLMAdapter):
    """Adapter for local models (Ollama, llama.cpp, etc.)."""
    
    def __init__(
        self,
        curry_db,
        base_url: str = "http://localhost:11434",
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.0,
        timeout_seconds: float = 30.0,
    ):
        """Initialize local model adapter."""
        super().__init__(curry_db)
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.timeout_seconds = timeout_seconds
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")
    
    def infer(
        self,
        model_name: str,
        model_version: int,
        prompt: str,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run local model inference."""
        model_config = self.db.get_model(model_name, model_version)
        
        start_time = time.time()

        attempts = max(1, self.max_retries + 1)
        for attempt_idx in range(attempts):
            try:
                # Call local model via Ollama or similar
                response = self.requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "temperature": model_config["temperature"],
                        "top_p": model_config["top_p"],
                        "num_predict": model_config["max_tokens"],
                        "seed": seed,
                    },
                    stream=False,
                    timeout=self.timeout_seconds,
                )

                if response.status_code == 200:
                    result = response.json()
                    output_text = result.get("response", "")
                    duration_ms = int((time.time() - start_time) * 1000)
                    request_id = (
                        response.headers.get("x-request-id")
                        or response.headers.get("request-id")
                        or result.get("id")
                    )

                    return {
                        "success": True,
                        "output": output_text,
                        "duration_ms": duration_ms,
                        "tokens_used": None,
                        "input_tokens_count": result.get("prompt_eval_count"),
                        "output_tokens_count": result.get("eval_count"),
                        "request_id": request_id,
                        "model_response": result,
                        "attempts": attempt_idx + 1,
                    }

                # Retry server-side errors only.
                if response.status_code >= 500 and attempt_idx < attempts - 1:
                    if self.retry_backoff_seconds > 0:
                        time.sleep(self.retry_backoff_seconds)
                    continue

                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "attempts": attempt_idx + 1,
                }

            except Exception as e:
                if attempt_idx < attempts - 1:
                    if self.retry_backoff_seconds > 0:
                        time.sleep(self.retry_backoff_seconds)
                    continue

                return {
                    "success": False,
                    "error": str(e),
                    "provider_error_type": e.__class__.__name__,
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "attempts": attempt_idx + 1,
                }
    
    def infer_and_record(
        self,
        model_name: str,
        model_version: int,
        prompt: str,
        seed: int = 42,
    ) -> str:
        """Run inference and record to Curry database."""
        result = self.infer(model_name, model_version, prompt, seed)
        
        if not result["success"]:
            raise RuntimeError(f"Inference failed: {result['error']}")
        
        inference_id = self.db.record_inference(
            model_name=model_name,
            model_version=model_version,
            input_tokens=prompt,
            output_tokens=result["output"].encode('utf-8'),
            seed=seed,
            temperature_used=None,
            top_p_used=None,
            duration_ms=result["duration_ms"],
            metadata={
                "api_provider": "local",
                "base_url": self.base_url,
                "request_id": result.get("request_id"),
                "input_tokens_count": result.get("input_tokens_count"),
                "output_tokens_count": result.get("output_tokens_count"),
                "prompt_text": prompt,
            },
        )
        
        return inference_id


# Factory function
def get_adapter(adapter_type: str, curry_db, **kwargs) -> LLMAdapter:
    """Get an LLM adapter by type."""
    adapters = {
        "openai": OpenAIAdapter,
        "claude": ClaudeAdapter,
        "local": LocalModelAdapter,
    }
    
    if adapter_type not in adapters:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    return adapters[adapter_type](curry_db, **kwargs)
