import asyncio
from types import SimpleNamespace

import pytest

from app.utils.openai_chat_compat import (
    create_chat_completion,
    extract_chat_completion_text,
    is_gpt5_family,
    preserve_gemini_thought_signatures,
)


class CompletionRecorder:
    def __init__(self, result=None, error=None):
        self.result = result or object()
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def client_for(recorder):
    return SimpleNamespace(chat=SimpleNamespace(completions=recorder))


def test_gpt5_uses_completion_token_limit_without_temperature():
    recorder = CompletionRecorder()
    messages = [{"role": "user", "content": "hello"}]

    result = create_chat_completion(
        client_for(recorder),
        model="gpt-5-2025-08-07",
        messages=messages,
        temperature=0.2,
        max_tokens=123,
        response_format={"type": "json_object"},
    )

    assert result is recorder.result
    assert recorder.calls == [
        {
            "model": "gpt-5-2025-08-07",
            "messages": messages,
            "max_completion_tokens": 123,
            "response_format": {"type": "json_object"},
        }
    ]


def test_legacy_model_preserves_original_request_shape():
    recorder = CompletionRecorder()
    messages = [{"role": "user", "content": "hello"}]

    create_chat_completion(
        client_for(recorder),
        model="third-party-chat-model",
        messages=messages,
        temperature=0.7,
        max_tokens=456,
        response_format={"type": "json_object"},
    )

    assert recorder.calls == [
        {
            "model": "third-party-chat-model",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 456,
            "response_format": {"type": "json_object"},
        }
    ]


def test_provider_error_is_propagated_without_guessing_or_retrying():
    provider_error = RuntimeError("unsupported max_tokens due to a server outage")
    recorder = CompletionRecorder(error=provider_error)

    with pytest.raises(RuntimeError) as captured:
        create_chat_completion(
            client_for(recorder),
            model="legacy-model",
            messages=[],
            max_tokens=10,
        )

    assert captured.value is provider_error
    assert len(recorder.calls) == 1


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5", True),
        (" GPT-5.1-mini ", True),
        ("gpt-4.1", False),
        ("my-gpt-5-proxy", False),
        (None, False),
    ],
)
def test_gpt5_family_detection(model, expected):
    assert is_gpt5_family(model) is expected


def test_extracts_text_from_supported_content_shapes():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"text": {"value": "first"}},
                        {"content": " second"},
                        SimpleNamespace(text=" third"),
                    ]
                )
            )
        ]
    )

    assert extract_chat_completion_text(response) == "first second third"
    assert extract_chat_completion_text(SimpleNamespace(choices=[])) == ""


def gemini_tool_response(*calls):
    """calls: (id, signature-or-None) pairs, shaped like the OpenAI SDK objects."""
    tool_calls = [
        SimpleNamespace(
            id=call_id,
            model_extra={"extra_content": {"google": {"thought_signature": sig}}} if sig else {},
        )
        for call_id, sig in calls
    ]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls))])


def replayed_tool_calls(*call_ids):
    return [
        {"role": "user", "content": "like posts"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "like_post", "arguments": "{}"}}
            for call_id in call_ids
        ]},
        {"role": "tool", "tool_call_id": call_ids[0], "content": "ok"},
    ]


class AsyncCompletionRecorder(CompletionRecorder):
    async def create(self, **kwargs):
        return CompletionRecorder.create(self, **kwargs)


def fake_camel_model(url, sync_recorder, async_recorder):
    return SimpleNamespace(_url=url, _client=client_for(sync_recorder), _async_client=client_for(async_recorder))


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_gemini_signatures_are_replayed_on_later_tool_call_requests():
    sync_recorder = CompletionRecorder(result=gemini_tool_response(("call_1", "SIG"), ("call_2", None)))
    async_recorder = AsyncCompletionRecorder(result=gemini_tool_response())
    model = preserve_gemini_thought_signatures(fake_camel_model(GEMINI_URL, sync_recorder, async_recorder))

    model._client.chat.completions.create(model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}])
    asyncio.run(model._async_client.chat.completions.create(
        model="gemini-3.8-flash", messages=replayed_tool_calls("call_1", "call_2")
    ))

    replayed = async_recorder.calls[0]["messages"][1]["tool_calls"]
    assert replayed[0]["extra_content"] == {"google": {"thought_signature": "SIG"}}
    # Parallel calls after the first carry no signature from Gemini; use its documented skip value.
    assert replayed[1]["extra_content"] == {"google": {"thought_signature": "skip_thought_signature_validator"}}


def test_non_gemini_models_are_left_untouched():
    sync_recorder = CompletionRecorder()
    async_recorder = AsyncCompletionRecorder()
    model = fake_camel_model("https://api.openai.com/v1", sync_recorder, async_recorder)

    assert preserve_gemini_thought_signatures(model) is model
    model._client.chat.completions.create(model="gpt-4o-mini", messages=replayed_tool_calls("call_1"))

    assert "extra_content" not in sync_recorder.calls[0]["messages"][1]["tool_calls"][0]
