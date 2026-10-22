"""LLMProvider interface with Ollama, OpenAI-compatible, and privacy enforcement.

V2 requirements:
- LLMProvider interface: generate(messages, **opts) -> {text, usage} and stream(messages, **opts).
- Implement OllamaProvider (is_hosted = False).
- Implement OpenAICompatibleProvider (is_hosted = True, reads credentials from env only, never logged or returned).
- Privacy rule: private workspace refuses hosted providers unless ALLOW_HOSTED_FOR_PRIVATE=true is set.
"""
from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, Union

import httpx

from trustrag.config import CONFIG
from trustrag.db import get_workspace

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
        clean_msg = sanitize_secret(message)
        super().__init__(clean_msg)
        self.message = clean_msg
        self.status_code = status_code
        self.provider = provider


class LLMAuthError(LLMError):
    """Raised when provider credentials are invalid or missing."""
    def __init__(self, message: str = "Authentication failed", provider: str = "unknown"):
        super().__init__(message, status_code=401, provider=provider)


class LLMRateLimitError(LLMError):
    """Raised when provider rate limits are exceeded."""
    def __init__(self, message: str = "Rate limit exceeded", provider: str = "unknown"):
        super().__init__(message, status_code=429, provider=provider)


class PrivacyViolationError(Exception):
    """Raised when a private workspace attempts to use a hosted provider without permission."""
    pass


def sanitize_secret(text: str) -> str:
    """Mask potential secret tokens and API keys from logs or error messages."""
    if not text:
        return text
    sanitized = re.sub(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{8,}", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    sanitized = re.sub(r"(sk-[a-zA-Z0-9_\-\.]{6,})", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"(gsk_[a-zA-Z0-9_\-\.]{6,})", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"(AIzaSy[a-zA-Z0-9_\-\.]{6,})", "[REDACTED_API_KEY]", sanitized)
    return sanitized


def normalize_messages(messages: Union[list[dict[str, str]], str]) -> list[dict[str, str]]:
    """Convert string prompt or message list into standard OpenAI message format."""
    if isinstance(messages, str):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": messages.strip()},
        ]
    return messages


class LLMProvider(ABC):
    """Abstract interface for LLM backends."""
    name: str = "base"
    is_hosted: bool = False

    @abstractmethod
    def generate(self, messages: Union[list[dict[str, str]], str], **opts: Any) -> dict[str, Any]:
        """Generate completion.
        
        Returns:
            {"text": str, "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}}
        """
        pass

    @abstractmethod
    def stream(self, messages: Union[list[dict[str, str]], str], **opts: Any) -> Iterator[str]:
        """Synchronously stream answer tokens."""
        pass

    @abstractmethod
    async def astream(self, messages: Union[list[dict[str, str]], str], **opts: Any) -> AsyncIterator[str]:
        """Asynchronously stream answer tokens."""
        pass


class OllamaProvider(LLMProvider):
    """Local Ollama provider (unhosted / on-premise)."""
    name: str = "ollama"
    is_hosted: bool = False

    def __init__(
        self,
        model: str = CONFIG.llm_model,
        base_url: str | None = None,
        timeout_s: float = CONFIG.llm_timeout_s,
    ):
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or CONFIG.ollama_url).rstrip("/")
        self.timeout_s = timeout_s

    def _build_payload(self, messages: Union[list[dict[str, str]], str], stream: bool = False, **opts: Any) -> dict[str, Any]:
        norm_messages = normalize_messages(messages)
        options = {
            "temperature": opts.get("temperature", 0.0),
            "num_predict": opts.get("num_predict", 128),
            "num_ctx": opts.get("num_ctx", 1536),
        }
        return {
            "model": opts.get("model", self.model),
            "messages": norm_messages,
            "options": options,
            "stream": stream,
        }

    def generate(self, messages: Union[list[dict[str, str]], str], **opts: Any) -> dict[str, Any]:
        payload = self._build_payload(messages, stream=False, **opts)
        url = f"{self.base_url}/api/chat"
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 404:
                    raise LLMError(f"Model '{self.model}' not found in local Ollama.", status_code=404, provider=self.name)
                if resp.status_code != 200:
                    raise LLMError(f"Ollama returned HTTP {resp.status_code}", status_code=503, provider=self.name)
                data = resp.json()
                text = data.get("message", {}).get("content", "").strip()
                prompt_eval = data.get("prompt_eval_count", 0)
                eval_count = data.get("eval_count", 0)
                usage = {
                    "prompt_tokens": prompt_eval,
                    "completion_tokens": eval_count,
                    "total_tokens": prompt_eval + eval_count,
                }
                return {"text": text, "usage": usage}
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(f"Cannot connect to local Ollama at {self.base_url}.", status_code=503, provider=self.name) from exc
        except httpx.ReadTimeout as exc:
            raise LLMError(f"Ollama generation timed out after {self.timeout_s}s", status_code=504, provider=self.name) from exc
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama error: {exc}", status_code=503, provider=self.name) from exc

    def stream(self, messages: Union[list[dict[str, str]], str], **opts: Any) -> Iterator[str]:
        payload = self._build_payload(messages, stream=True, **opts)
        url = f"{self.base_url}/api/chat"
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        raise LLMError(f"Ollama error {resp.status_code}", status_code=503, provider=self.name)
                    for line in resp.iter_lines():
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
            raise LLMError(f"Cannot connect to local Ollama at {self.base_url}.", status_code=503, provider=self.name) from exc
        except Exception as exc:
            raise LLMError(f"Ollama stream error: {exc}", status_code=503, provider=self.name) from exc

    async def astream(self, messages: Union[list[dict[str, str]], str], **opts: Any) -> AsyncIterator[str]:
        payload = self._build_payload(messages, stream=True, **opts)
        url = f"{self.base_url}/api/chat"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        raise LLMError(f"Ollama error {resp.status_code}", status_code=503, provider=self.name)
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
            raise LLMError(f"Cannot connect to local Ollama at {self.base_url}.", status_code=503, provider=self.name) from exc
        except Exception as exc:
            raise LLMError(f"Ollama stream error: {exc}", status_code=503, provider=self.name) from exc


class OpenAICompatibleProvider(LLMProvider):
    """Hosted OpenAI-compatible provider.
    
    Security:
    - base_url and API key are read from environment variables or explicit config.
    - Keys are never logged and never included in error messages.
    """
    name: str = "openai_compatible"
    is_hosted: bool = True

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        provider_name: str = "openai_compatible",
        timeout_s: float = 60.0,
    ):
        self.provider_name = provider_name
        self.name = provider_name
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout_s = timeout_s

    def _get_headers(self) -> dict[str, str]:
        if not self._api_key:
            raise LLMAuthError(f"API key missing for provider '{self.provider_name}'. Set OPENAI_API_KEY.", provider=self.name)
        return {
            "Authorization": f"Bearer {self._api_key.strip()}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: Union[list[dict[str, str]], str], stream: bool = False, **opts: Any) -> dict[str, Any]:
        norm_messages = normalize_messages(messages)
        return {
            "model": opts.get("model", self.model),
            "messages": norm_messages,
            "temperature": opts.get("temperature", 0.0),
            "max_tokens": opts.get("max_tokens", 512),
            "stream": stream,
        }

    def _handle_error_status(self, status_code: int) -> None:
        if status_code in (401, 403):
            raise LLMAuthError(f"Authentication failed for {self.provider_name}.", provider=self.name)
        if status_code == 429:
            raise LLMRateLimitError(f"Rate limit exceeded on {self.provider_name}.", provider=self.name)
        raise LLMError(f"{self.provider_name} returned HTTP {status_code}", status_code=status_code, provider=self.name)

    def generate(self, messages: Union[list[dict[str, str]], str], **opts: Any) -> dict[str, Any]:
        headers = self._get_headers()
        payload = self._build_payload(messages, stream=False, **opts)
        url = f"{self.base_url}/chat/completions"

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    self._handle_error_status(resp.status_code)
                data = resp.json()
                choices = data.get("choices", [])
                text = choices[0].get("message", {}).get("content", "").strip() if choices else ""
                usage = data.get("usage", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                })
                return {"text": text, "usage": usage}
        except (LLMAuthError, LLMRateLimitError, LLMError):
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(f"Connection failure to {self.name} at {self.base_url}", status_code=503, provider=self.name) from exc
        except Exception as exc:
            raise LLMError(f"Unexpected error calling {self.name}: {sanitize_secret(str(exc))}", status_code=500, provider=self.name) from exc

    def stream(self, messages: Union[list[dict[str, str]], str], **opts: Any) -> Iterator[str]:
        headers = self._get_headers()
        payload = self._build_payload(messages, stream=True, **opts)
        url = f"{self.base_url}/chat/completions"

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        self._handle_error_status(resp.status_code)
                    for line in resp.iter_lines():
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
        except Exception as exc:
            raise LLMError(f"Stream failure: {sanitize_secret(str(exc))}", status_code=503, provider=self.name) from exc

    async def astream(self, messages: Union[list[dict[str, str]], str], **opts: Any) -> AsyncIterator[str]:
        headers = self._get_headers()
        payload = self._build_payload(messages, stream=True, **opts)
        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        self._handle_error_status(resp.status_code)
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
        except Exception as exc:
            raise LLMError(f"Stream failure: {sanitize_secret(str(exc))}", status_code=503, provider=self.name) from exc


class AnthropicProvider(OpenAICompatibleProvider):
    """Anthropic Claude provider alias."""
    def __init__(self, model: str = "claude-3-5-haiku-20241022", api_key: str | None = None, base_url: str | None = None, timeout_s: float = 60.0):
        super().__init__(model=model, base_url=base_url or "https://api.anthropic.com/v1", api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), provider_name="anthropic", timeout_s=timeout_s)


class GeminiProvider(OpenAICompatibleProvider):
    """Google Gemini provider alias."""
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None, base_url: str | None = None, timeout_s: float = 60.0):
        super().__init__(model=model, base_url=base_url or "https://generativelanguage.googleapis.com/v1beta", api_key=api_key or os.getenv("GEMINI_API_KEY"), provider_name="gemini", timeout_s=timeout_s)


def check_workspace_privacy(workspace_id: str, provider: LLMProvider) -> None:
    """Enforce privacy rule:
    A workspace with privacy='private' must refuse hosted providers
    unless ALLOW_HOSTED_FOR_PRIVATE=true is set.
    """
    ws = get_workspace(workspace_id)
    if not ws:
        return

    privacy = ws.get("privacy", "shared")
    if privacy == "private" and provider.is_hosted:
        allow_hosted = os.getenv("ALLOW_HOSTED_FOR_PRIVATE", "false").strip().lower() in ("true", "1")
        if not allow_hosted:
            raise PrivacyViolationError(
                f"Workspace '{workspace_id}' is configured with privacy='private' and refuses hosted provider '{provider.name}'. "
                "Set ALLOW_HOSTED_FOR_PRIVATE=true to allow hosted inference."
            )


def get_llm_provider(
    provider_name: str = "ollama",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_s: float | None = None,
) -> LLMProvider:
    """Factory creating an LLMProvider instance."""
    p_name = (provider_name or "ollama").lower().strip()
    timeout = timeout_s or float(os.getenv("TRUSTRAG_LLM_TIMEOUT", "60"))

    if p_name in ("ollama", "local"):
        return OllamaProvider(
            model=model or CONFIG.llm_model,
            base_url=base_url,
            timeout_s=timeout,
        )
    elif p_name in ("openai", "openai_compatible", "groq", "openrouter"):
        return OpenAICompatibleProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
            provider_name=p_name,
            timeout_s=timeout,
        )
    elif p_name == "anthropic":
        return AnthropicProvider(
            model=model or "claude-3-5-haiku-20241022",
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout,
        )
    elif p_name == "gemini":
        return GeminiProvider(
            model=model or "gemini-2.0-flash",
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout,
        )
    else:
        raise LLMError(f"Unsupported provider '{provider_name}'. Supported: ['ollama', 'openai', 'groq', 'anthropic', 'gemini', 'openai_compatible']", status_code=400, provider=p_name)


def get_available_providers_info() -> list[dict[str, Any]]:
    """Return public metadata on supported providers."""
    return [
        {
            "id": "ollama",
            "label": "Ollama (Local)",
            "default_model": CONFIG.llm_model,
            "base_url": CONFIG.ollama_url,
            "requires_key": False,
            "is_hosted": False,
            "has_server_key": True,
        },
        {
            "id": "openai",
            "label": "OpenAI",
            "default_model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "requires_key": True,
            "is_hosted": True,
            "has_server_key": bool(os.getenv("OPENAI_API_KEY")),
        },
        {
            "id": "groq",
            "label": "Groq Cloud",
            "default_model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1",
            "requires_key": True,
            "is_hosted": True,
            "has_server_key": bool(os.getenv("GROQ_API_KEY")),
        },
        {
            "id": "anthropic",
            "label": "Anthropic Claude",
            "default_model": "claude-3-5-haiku-20241022",
            "base_url": "https://api.anthropic.com/v1",
            "requires_key": True,
            "is_hosted": True,
            "has_server_key": bool(os.getenv("ANTHROPIC_API_KEY")),
        },
        {
            "id": "gemini",
            "label": "Google Gemini",
            "default_model": "gemini-2.0-flash",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "requires_key": True,
            "is_hosted": True,
            "has_server_key": bool(os.getenv("GEMINI_API_KEY")),
        },
        {
            "id": "openai_compatible",
            "label": "OpenAI-Compatible (Hosted)",
            "default_model": "gpt-4o-mini",
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "requires_key": True,
            "is_hosted": True,
            "has_server_key": bool(os.getenv("OPENAI_API_KEY")),
        },
    ]
