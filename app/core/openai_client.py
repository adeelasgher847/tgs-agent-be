"""
Shared OpenAI client factory — BYO LLM support for on-premise deployments.

When LLM_PROVIDER=azure_openai, builds an AzureOpenAI/AsyncAzureOpenAI client
(OPENAI_BASE_URL is the Azure resource endpoint, OPENAI_API_VERSION is required).
Otherwise builds a plain OpenAI/AsyncOpenAI client, optionally pointed at a
custom OPENAI_BASE_URL (Ollama/vLLM/LiteLLM or any OpenAI-compatible endpoint).

Every call site that previously did `OpenAI(api_key=...)` / `AsyncOpenAI(api_key=...)`
should go through get_openai_client()/get_async_openai_client() instead, so a single
on-premise customer can redirect all OpenAI-shaped traffic with two env vars.
"""

from __future__ import annotations

from typing import Optional, Union

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from app.core.config import settings


def get_openai_client(api_key: Optional[str] = None) -> Union[OpenAI, AzureOpenAI]:
    key = api_key or settings.OPENAI_API_KEY
    if settings.LLM_PROVIDER == "azure_openai":
        return AzureOpenAI(
            api_key=key,
            azure_endpoint=settings.OPENAI_BASE_URL,
            api_version=settings.OPENAI_API_VERSION,
        )
    return OpenAI(api_key=key, base_url=settings.OPENAI_BASE_URL or None)


def get_async_openai_client(api_key: Optional[str] = None) -> Union[AsyncOpenAI, AsyncAzureOpenAI]:
    key = api_key or settings.OPENAI_API_KEY
    if settings.LLM_PROVIDER == "azure_openai":
        return AsyncAzureOpenAI(
            api_key=key,
            azure_endpoint=settings.OPENAI_BASE_URL,
            api_version=settings.OPENAI_API_VERSION,
        )
    return AsyncOpenAI(api_key=key, base_url=settings.OPENAI_BASE_URL or None)
