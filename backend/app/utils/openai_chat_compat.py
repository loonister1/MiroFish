"""
OpenAI Chat Completions compatibility helpers.

This module keeps existing behavior for legacy models/providers while
gracefully adapting request parameters for GPT-5 family models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def is_gpt5_family(model: Optional[str]) -> bool:
    """Return True when model belongs to GPT-5 family aliases/snapshots."""
    if not model:
        return False
    return model.strip().lower().startswith("gpt-5")


def create_chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a chat completion with model-specific request parameters.

    Compatibility strategy:
    - For GPT-5 family, avoid sending temperature by default.
    - For token limit, use `max_completion_tokens` on GPT-5, `max_tokens` otherwise.
    - Preserve the legacy request shape for every non-GPT-5 model/provider.
    - Propagate provider errors unchanged instead of guessing from message text.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if response_format is not None:
        kwargs["response_format"] = response_format

    gpt5_family = is_gpt5_family(model)

    if temperature is not None and not gpt5_family:
        kwargs["temperature"] = temperature

    if max_tokens is not None:
        if gpt5_family:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

    return client.chat.completions.create(**kwargs)


GEMINI_SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def preserve_gemini_thought_signatures(model: Any) -> Any:
    """
    Re-attach Gemini thought signatures that camel drops from replayed tool calls.

    Gemini 3+ rejects a request whose earlier assistant tool call lacks the
    `extra_content.google.thought_signature` it returned. camel rebuilds those
    messages without it, so wrap the camel model's clients to remember each
    signature by tool call id and put it back. Non-Gemini models are untouched.
    """
    if "generativelanguage.googleapis.com" not in str(getattr(model, "_url", "") or ""):
        return model

    # shortcut: grows by one entry per tool call for the process lifetime; fine for one simulation run.
    signatures: Dict[str, str] = {}

    def remember(response: Any) -> Any:
        for choice in getattr(response, "choices", None) or []:
            for call in getattr(choice.message, "tool_calls", None) or []:
                extra = (getattr(call, "model_extra", None) or {}).get("extra_content") or {}
                signature = (extra.get("google") or {}).get("thought_signature")
                if signature:
                    signatures[call.id] = signature
        return response

    def restore(messages: Any) -> None:
        for message in messages or []:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                signature = signatures.get(call.get("id"), GEMINI_SKIP_THOUGHT_SIGNATURE)
                call.setdefault("extra_content", {"google": {"thought_signature": signature}})

    sync_create = model._client.chat.completions.create
    async_create = model._async_client.chat.completions.create

    def create(**kwargs: Any) -> Any:
        restore(kwargs.get("messages"))
        return remember(sync_create(**kwargs))

    async def acreate(**kwargs: Any) -> Any:
        restore(kwargs.get("messages"))
        return remember(await async_create(**kwargs))

    model._client.chat.completions.create = create
    model._async_client.chat.completions.create = acreate
    return model


def extract_chat_completion_text(response: Any) -> str:
    """Extract plain text from chat completion response across SDK content shapes."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_obj = item.get("text")
                if isinstance(text_obj, dict):
                    text_obj = text_obj.get("value")
                if isinstance(text_obj, str):
                    chunks.append(text_obj)
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
                continue

            text_obj = getattr(item, "text", None)
            if isinstance(text_obj, dict):
                text_obj = text_obj.get("value")
            if isinstance(text_obj, str):
                chunks.append(text_obj)
                continue

            content_obj = getattr(item, "content", None)
            if isinstance(content_obj, str):
                chunks.append(content_obj)

        return "".join(chunks).strip()

    return str(content or "")
