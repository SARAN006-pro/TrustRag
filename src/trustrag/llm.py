"""Multi-provider LLM interface supporting local and cloud backends.

Supported providers:
- ollama (default local, no external key needed)
- openai (OpenAI GPT-4o / GPT-4o-mini, etc.)
- groq (Groq Llama-3.3-70b, etc.)
- openrouter (OpenRouter models)
- openai_compatible (Any custom endpoint like vLLM, LMStudio, LocalAI)
- anthropic (Claude 3.5 Sonnet, Claude 3 Haiku)
- gemini (Google Gemini 1.5 Flash / Pro, Gemini 2.0 Flash)

Adheres strictly to security requirements:
- Ephemeral user keys are passed directly to HTTP requests and never saved to disk or logs.
- Server-managed credentials fallback to environment variables.
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, Any

import httpx

from trustrag.config import CONFIG

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are TrustRAG, an assistant for Indian government welfare schemes.\n"
    "Answer the user's question directly and concisely in complete sentences, strictly and only using the provided numbered context passages below.\n"
    "Every statement you make must be directly supported by the context.\n"
    "Cite source passages using their exact [chunk_id] tag (for example: [pm_kisan.txt#0]).\n"
    "Structure your response with clear headings (e.g., Eligibility, Benefits, Required Documents, Application Steps, Missing Information) only when supported by the context.\n"
    "If the context does not contain enough information to answer the question, reply with exactly:\n"
    "Not found in the provided documents.\n"
    "Do not extrapolate, assume, or invent facts or eligibility rules."
)

ABSTENTION_ANSWER = "Not found in the provided documents."


class LLMError(Exception):
    """Base exception for LLM provider errors."""
    def __init__(self, message: str, status_code: int = 503, provider: str = "unknown"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.provider = provider


class LLMAuthError(LLMError):
    """Raised when provider credentials are invalid or missing."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, status_code=401, provider=provider)


class LLMRateLimitError(LLMError):
    """Raised when provider rate limits are exceeded."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, status_code=429, provider=provider)


class BaseLLMProvider(ABC):
    """Abstract interface for LLM backends."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, timeout_s: float = 60.0):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """Synchronously generate an answer."""
        pass

    @abstractmethod
    async def astream(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AsyncIterator[str]:
        """Asynchronously stream answer tokens."""
        pass


class OllamaProvider(BaseLLMProvider):
    """Local Ollama provider."""

    def __init__(self, model: str = CONFIG.llm_model, api_key: str | None = None, base_url: str | None = None, timeout_s: float = CONFIG.llm_timeout_s):
        url = (base_url or os.getenv("OLLAMA_BASE_URL") or CONFIG.ollama_url).rstrip("/")
        super().__init__(model=model, api_key=None, base_url=url, timeout_s=timeout_s)

    def _build_payload(self, prompt: str, system_prompt: str, stream: bool = False) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0.0,
                "num_predict": 128,
                "num_ctx": 1536,
            },
            "stream": stream,
        }

    def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        payload = self._build_payload(prompt, system_prompt, stream=False)
        url = f"{self.base_url}/api/chat"
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 404:
                    raise LLMError(f"Model '{self.model}' not found in local Ollama.", status_code=404, provider="ollama")
                if resp.status_code != 200:
                    raise LLMError(f"Ollama returned HTTP {resp.status_code}: {resp.text}", status_code=503, provider="ollama")
                data = resp.json()
                return data.get("message", {}).get("content", "").strip()
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(f"Cannot connect to local Ollama at {self.base_url}. Ensure Ollama is running.", status_code=503, provider="ollama") from exc
        except httpx.ReadTimeout as exc:
            raise LLMError(f"Ollama generation timed out after {self.timeout_s}s", status_code=504, provider="ollama") from exc
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama error: {exc}", status_code=503, provider="ollama") from exc

    async def astream(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AsyncIterator[str]:
        payload = self._build_payload(prompt, system_prompt, stream=True)
        url = f"{self.base_url}/api/chat"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code == 404:
                        raise LLMError(f"Model '{self.model}' not found in local Ollama.", status_code=404, provider="ollama")
                    if resp.status_code != 200:
                        err_body = await resp.aread()
                        raise LLMError(f"Ollama error {resp.status_code}: {err_body.decode('utf-8', errors='ignore')}", status_code=503, provider="ollama")
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk_data = json.loads(line)
                            token = chunk_data.get("message", {}).get("content", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(f"Cannot connect to local Ollama at {self.base_url}.", status_code=503, provider="ollama") from exc
        except httpx.ReadTimeout as exc:
            raise LLMError(f"Ollama stream timed out after {self.timeout_s}s", status_code=504, provider="ollama") from exc


class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider for OpenAI and OpenAI-compatible services (Groq, OpenRouter, vLLM, etc.)."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        provider_name: str = "openai",
        timeout_s: float = 60.0,
    ):
        super().__init__(model=model, api_key=api_key, base_url=base_url.rstrip("/"), timeout_s=timeout_s)
        self.provider_name = provider_name

    def _get_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise LLMAuthError(f"API key missing for provider '{self.provider_name}'. Please supply your key.", provider=self.provider_name)
        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
        }
        if self.provider_name == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/trustrag"
            headers["X-Title"] = "TrustRAG Scheme Assistant"
        return headers

    def _build_payload(self, prompt: str, system_prompt: str, stream: bool = False) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "stream": stream,
        }

    def _handle_error_status(self, status_code: int, response_text: str) -> None:
        if status_code in (401, 403):
            raise LLMAuthError(f"Invalid API key for {self.provider_name}: {response_text}", provider=self.provider_name)
        if status_code == 429:
            raise LLMRateLimitError(f"Rate limit exceeded for {self.provider_name}: {response_text}", provider=self.provider_name)
        if status_code == 404:
            raise LLMError(f"Model '{self.model}' not found on {self.provider_name}.", status_code=404, provider=self.provider_name)
        raise LLMError(f"{self.provider_name} returned HTTP {status_code}: {response_text}", status_code=status_code, provider=self.provider_name)

    def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        headers = self._get_headers()
        payload = self._build_payload(prompt, system_prompt, stream=False)
        url = f"{self.base_url}/chat/completions"

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    self._handle_error_status(resp.status_code, resp.text)
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                return ""
        except (LLMAuthError, LLMRateLimitError, LLMError):
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(f"Connection failure to {self.provider_name} at {self.base_url}: {exc}", status_code=503, provider=self.provider_name) from exc
        except httpx.ReadTimeout as exc:
            raise LLMError(f"{self.provider_name} request timed out after {self.timeout_s}s", status_code=504, provider=self.provider_name) from exc
        except Exception as exc:
            raise LLMError(f"Unexpected error calling {self.provider_name}: {exc}", status_code=500, provider=self.provider_name) from exc

    async def astream(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AsyncIterator[str]:
        headers = self._get_headers()
        payload = self._build_payload(prompt, system_prompt, stream=True)
        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        err_body = await resp.aread()
                        self._handle_error_status(resp.status_code, err_body.decode("utf-8", errors="ignore"))

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
        except (LLMAuthError, LLMRateLimitError, LLMError):
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(f"Connection error to {self.provider_name}: {exc}", status_code=503, provider=self.provider_name) from exc
        except httpx.ReadTimeout as exc:
            raise LLMError(f"{self.provider_name} stream timed out after {self.timeout_s}s", status_code=504, provider=self.provider_name) from exc


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider using Messages API."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: str | None = None, base_url: str | None = None, timeout_s: float = 60.0):
        url = (base_url or "https://api.anthropic.com/v1").rstrip("/")
        super().__init__(model=model, api_key=api_key, base_url=url, timeout_s=timeout_s)

    def _get_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise LLMAuthError("Anthropic API key is required. Please supply your x-api-key.", provider="anthropic")
        return {
            "x-api-key": self.api_key.strip(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _build_payload(self, prompt: str, system_prompt: str, stream: bool = False) -> dict[str, Any]:
        return {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": stream,
        }

    def _handle_error_status(self, status_code: int, response_text: str) -> None:
        if status_code in (401, 403):
            raise LLMAuthError(f"Anthropic authentication failed: {response_text}", provider="anthropic")
        if status_code == 429:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {response_text}", provider="anthropic")
        raise LLMError(f"Anthropic API returned {status_code}: {response_text}", status_code=status_code, provider="anthropic")

    def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        headers = self._get_headers()
        payload = self._build_payload(prompt, system_prompt, stream=False)
        url = f"{self.base_url}/messages"

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    self._handle_error_status(resp.status_code, resp.text)
                data = resp.json()
                content = data.get("content", [])
                if content and isinstance(content, list):
                    return "".join(c.get("text", "") for c in content if c.get("type") == "text").strip()
                return ""
        except (LLMAuthError, LLMRateLimitError, LLMError):
            raise
        except Exception as exc:
            raise LLMError(f"Anthropic error: {exc}", status_code=503, provider="anthropic") from exc

    async def astream(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AsyncIterator[str]:
        headers = self._get_headers()
        payload = self._build_payload(prompt, system_prompt, stream=True)
        url = f"{self.base_url}/messages"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        err_body = await resp.aread()
                        self._handle_error_status(resp.status_code, err_body.decode("utf-8", errors="ignore"))

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        try:
                            event = json.loads(data_str)
                            event_type = event.get("type")
                            if event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                            elif event_type == "message_stop":
                                break
                        except json.JSONDecodeError:
                            continue
        except (LLMAuthError, LLMRateLimitError, LLMError):
            raise
        except Exception as exc:
            raise LLMError(f"Anthropic stream error: {exc}", status_code=503, provider="anthropic") from exc


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider via Google AI Studio REST API."""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None, base_url: str | None = None, timeout_s: float = 60.0):
        url = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        super().__init__(model=model, api_key=api_key, base_url=url, timeout_s=timeout_s)

    def _get_key(self) -> str:
        if not self.api_key:
            raise LLMAuthError("Google Gemini API key is required. Please supply your API key.", provider="gemini")
        return self.api_key.strip()

    def _build_payload(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 512,
            },
        }

    def _handle_error_status(self, status_code: int, response_text: str) -> None:
        if status_code in (400, 401, 403):
            raise LLMAuthError(f"Gemini authentication / request error: {response_text}", provider="gemini")
        if status_code == 429:
            raise LLMRateLimitError(f"Gemini rate limit exceeded: {response_text}", provider="gemini")
        raise LLMError(f"Gemini API returned {status_code}: {response_text}", status_code=status_code, provider="gemini")

    def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        key = self._get_key()
        payload = self._build_payload(prompt, system_prompt)
        url = f"{self.base_url}/models/{self.model}:generateContent?key={key}"

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(url, json=payload)
                if resp.status_code != 200:
                    self._handle_error_status(resp.status_code, resp.text)
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    return "".join(p.get("text", "") for p in parts).strip()
                return ""
        except (LLMAuthError, LLMRateLimitError, LLMError):
            raise
        except Exception as exc:
            raise LLMError(f"Gemini error: {exc}", status_code=503, provider="gemini") from exc

    async def astream(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> AsyncIterator[str]:
        key = self._get_key()
        payload = self._build_payload(prompt, system_prompt)
        url = f"{self.base_url}/models/{self.model}:streamGenerateContent?alt=sse&key={key}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        err_body = await resp.aread()
                        self._handle_error_status(resp.status_code, err_body.decode("utf-8", errors="ignore"))

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        try:
                            chunk = json.loads(data_str)
                            candidates = chunk.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for p in parts:
                                    t = p.get("text")
                                    if t:
                                        yield t
                        except json.JSONDecodeError:
                            continue
        except (LLMAuthError, LLMRateLimitError, LLMError):
            raise
        except Exception as exc:
            raise LLMError(f"Gemini stream error: {exc}", status_code=503, provider="gemini") from exc


PROVIDER_DEFAULTS = {
    "ollama": {
        "model": CONFIG.llm_model,
        "base_url": CONFIG.ollama_url,
        "env_key": None,
        "label": "Ollama (Local)",
    },
    "openai": {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "label": "OpenAI",
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "label": "Groq Cloud",
    },
    "openrouter": {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "label": "OpenRouter",
    },
    "anthropic": {
        "model": "claude-3-5-haiku-20241022",
        "base_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
        "label": "Anthropic Claude",
    },
    "gemini": {
        "model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "env_key": "GEMINI_API_KEY",
        "label": "Google Gemini",
    },
    "openai_compatible": {
        "model": "default",
        "base_url": "http://localhost:8000/v1",
        "env_key": "CUSTOM_LLM_API_KEY",
        "label": "Custom OpenAI-Compatible",
    },
}


def get_llm_provider(
    provider_name: str = "ollama",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_s: float | None = None,
) -> BaseLLMProvider:
    """Factory creating an LLMProvider instance with precedence: explicit params -> env -> defaults."""
    provider_key = (provider_name or "ollama").lower().strip()
    if provider_key not in PROVIDER_DEFAULTS:
        raise LLMError(f"Unsupported provider '{provider_name}'. Supported: {list(PROVIDER_DEFAULTS.keys())}", status_code=400, provider=provider_name)

    spec = PROVIDER_DEFAULTS[provider_key]
    selected_model = model or spec["model"]
    resolved_base_url = base_url or spec["base_url"]
    env_var = spec.get("env_key")
    selected_api_key = api_key or (os.getenv(env_var) if env_var else None)
    effective_timeout = timeout_s or float(os.getenv("TRUSTRAG_LLM_TIMEOUT", "60"))

    if provider_key == "ollama":
        return OllamaProvider(
            model=selected_model,
            base_url=resolved_base_url,
            timeout_s=effective_timeout,
        )
    elif provider_key in ("openai", "groq", "openrouter", "openai_compatible"):
        return OpenAICompatibleProvider(
            model=selected_model,
            api_key=selected_api_key,
            base_url=resolved_base_url or "https://api.openai.com/v1",
            provider_name=provider_key,
            timeout_s=effective_timeout,
        )
    elif provider_key == "anthropic":
        return AnthropicProvider(
            model=selected_model,
            api_key=selected_api_key,
            base_url=resolved_base_url,
            timeout_s=effective_timeout,
        )
    elif provider_key == "gemini":
        return GeminiProvider(
            model=selected_model,
            api_key=selected_api_key,
            base_url=resolved_base_url,
            timeout_s=effective_timeout,
        )
    else:
        raise LLMError(f"Provider '{provider_key}' handler not implemented.", status_code=400, provider=provider_key)


def get_available_providers_info() -> list[dict[str, Any]]:
    """Return public metadata on supported providers, default models, and server credential availability."""
    info = []
    for key, spec in PROVIDER_DEFAULTS.items():
        env_key = spec.get("env_key")
        has_server_key = bool(env_key and os.getenv(env_key))
        info.append({
            "id": key,
            "label": spec["label"],
            "default_model": spec["model"],
            "base_url": spec["base_url"],
            "requires_key": key != "ollama",
            "has_server_key": has_server_key,
        })
    return info
