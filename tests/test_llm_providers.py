"""Tests for packages.llm -- LLM provider abstraction."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from packages.common.exceptions import ProviderError
from packages.llm import (
    BaseLLMProvider,
    LLMResponse,
    OllamaProvider,
    OpenRouterProvider,
    ProviderFactory,
    ProviderType,
)

# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------


class TestLLMResponse:
    def test_frozen(self) -> None:
        r = LLMResponse(text="hello", model="m1")
        with pytest.raises(AttributeError):
            r.text = "changed"  # type: ignore[misc]

    def test_total_tokens(self) -> None:
        r = LLMResponse(text="x", model="m", prompt_tokens=10, completion_tokens=20)
        assert r.total_tokens == 30

    def test_total_tokens_none(self) -> None:
        r = LLMResponse(text="x", model="m")
        assert r.total_tokens is None

    def test_defaults(self) -> None:
        r = LLMResponse(text="hi", model="m1")
        assert r.prompt_tokens is None
        assert r.completion_tokens is None
        assert r.finish_reason is None
        assert r.raw == {}


# ---------------------------------------------------------------------------
# BaseLLMProvider ABC
# ---------------------------------------------------------------------------


class TestBaseLLMProviderABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseLLMProvider()  # type: ignore[abstract]

    def test_subclass_must_implement(self) -> None:
        class Incomplete(BaseLLMProvider):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# BaseLLMProvider.generate_json
# ---------------------------------------------------------------------------


class _StubProvider(BaseLLMProvider):
    """Minimal concrete provider for testing base class methods."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    async def generate(
        self,
        model: str,
        prompt: str,  # noqa: ARG002
        **kw: Any,  # noqa: ARG002
    ) -> LLMResponse:
        return LLMResponse(text=self._response_text, model=model)

    async def check_connection(self) -> bool:
        return True

    async def list_models(self) -> list[str]:
        return []


class TestGenerateJson:
    async def test_generate_json_success(self) -> None:
        result = await _StubProvider('{"key": "value"}').generate_json("m", "give json")
        assert result == {"key": "value"}

    async def test_generate_json_invalid(self) -> None:
        with pytest.raises(ProviderError, match="invalid JSON"):
            await _StubProvider("not json").generate_json("m", "give json")

    async def test_generate_json_non_object(self) -> None:
        with pytest.raises(ProviderError, match="Expected JSON object"):
            await _StubProvider("[1,2,3]").generate_json("m", "give json")


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------


_DUMMY_REQUEST = httpx.Request("POST", "http://test")


def _mock_ollama_response(text: str = "hello", model: str = "qwen3:8b") -> dict[str, Any]:
    return {
        "response": text,
        "model": model,
        "prompt_eval_count": 10,
        "eval_count": 5,
        "done_reason": "stop",
        "done": True,
    }


class TestOllamaProvider:
    async def test_generate(self) -> None:
        mock_resp = httpx.Response(200, request=_DUMMY_REQUEST, json=_mock_ollama_response())
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp
        ):
            async with OllamaProvider() as p:
                result = await p.generate("qwen3:8b", "hi")
        assert result.text == "hello"
        assert result.model == "qwen3:8b"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5

    async def test_generate_json_mode(self) -> None:
        resp_data = _mock_ollama_response(text='{"a":1}')
        mock_resp = httpx.Response(200, request=_DUMMY_REQUEST, json=resp_data)

        async def capture_post(_url: str, **kwargs: Any) -> httpx.Response:
            assert kwargs["json"]["format"] == "json"
            return mock_resp

        with patch.object(httpx.AsyncClient, "post", side_effect=capture_post):
            async with OllamaProvider() as p:
                result = await p.generate("m", "q", json_mode=True)
        assert result.text == '{"a":1}'

    async def test_generate_with_system(self) -> None:
        mock_resp = httpx.Response(200, request=_DUMMY_REQUEST, json=_mock_ollama_response())

        async def capture_post(_url: str, **kwargs: Any) -> httpx.Response:
            assert kwargs["json"]["system"] == "be helpful"
            return mock_resp

        with patch.object(httpx.AsyncClient, "post", side_effect=capture_post):
            async with OllamaProvider() as p:
                await p.generate("m", "q", system="be helpful")

    async def test_generate_http_error(self) -> None:
        mock_resp = httpx.Response(
            500, text="server error", request=httpx.Request("POST", "http://x")
        )
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp
        ):
            async with OllamaProvider() as p:
                with pytest.raises(ProviderError, match="Ollama HTTP 500"):
                    await p.generate("m", "q")

    async def test_generate_request_error(self) -> None:
        with patch.object(
            httpx.AsyncClient,
            "post",
            side_effect=httpx.ConnectError("refused"),
        ):
            async with OllamaProvider() as p:
                with pytest.raises(ProviderError, match="request failed"):
                    await p.generate("m", "q")

    async def test_check_connection_ok(self) -> None:
        mock_resp = httpx.Response(200, request=_DUMMY_REQUEST, json={"models": []})
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
            async with OllamaProvider() as p:
                assert await p.check_connection() is True

    async def test_check_connection_fail(self) -> None:
        with patch.object(httpx.AsyncClient, "get", side_effect=httpx.ConnectError("refused")):
            async with OllamaProvider() as p:
                assert await p.check_connection() is False

    async def test_list_models(self) -> None:
        data = {"models": [{"name": "m1"}, {"name": "m2"}]}
        mock_resp = httpx.Response(200, request=_DUMMY_REQUEST, json=data)
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
            async with OllamaProvider() as p:
                models = await p.list_models()
        assert models == ["m1", "m2"]

    async def test_list_models_error(self) -> None:
        with patch.object(httpx.AsyncClient, "get", side_effect=httpx.ConnectError("x")):
            async with OllamaProvider() as p:
                assert await p.list_models() == []

    def test_repr(self) -> None:
        p = OllamaProvider(base_url="http://host:1234")
        assert "http://host:1234" in repr(p)


# ---------------------------------------------------------------------------
# OpenRouterProvider
# ---------------------------------------------------------------------------


def _mock_openrouter_response(
    text: str = "hello",
    model: str = "qwen/qwen3-max",
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"content": text, "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "model": model,
        "usage": {"prompt_tokens": 15, "completion_tokens": 8},
    }


class TestOpenRouterProvider:
    def test_requires_api_key(self) -> None:
        import os

        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            with pytest.raises(ProviderError, match="API key required"):
                OpenRouterProvider()

    async def test_generate(self) -> None:
        mock_resp = httpx.Response(200, request=_DUMMY_REQUEST, json=_mock_openrouter_response())
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp
        ):
            async with OpenRouterProvider(api_key="sk-test") as p:
                result = await p.generate("qwen/qwen3-max", "hi")
        assert result.text == "hello"
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 8
        assert result.finish_reason == "stop"

    async def test_generate_with_json_mode(self) -> None:
        resp_data = _mock_openrouter_response(text='{"a":1}')
        mock_resp = httpx.Response(200, request=_DUMMY_REQUEST, json=resp_data)

        async def capture_post(_url: str, **kwargs: Any) -> httpx.Response:
            assert kwargs["json"]["response_format"] == {"type": "json_object"}
            return mock_resp

        with patch.object(httpx.AsyncClient, "post", side_effect=capture_post):
            async with OpenRouterProvider(api_key="sk-test") as p:
                await p.generate("m", "q", json_mode=True)

    async def test_generate_with_json_schema(self) -> None:
        resp_data = _mock_openrouter_response(text='{"name":"x"}')
        mock_resp = httpx.Response(200, request=_DUMMY_REQUEST, json=resp_data)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        async def capture_post(_url: str, **kwargs: Any) -> httpx.Response:
            rf = kwargs["json"]["response_format"]
            assert rf["type"] == "json_schema"
            assert rf["json_schema"]["schema"] == schema
            return mock_resp

        with patch.object(httpx.AsyncClient, "post", side_effect=capture_post):
            async with OpenRouterProvider(api_key="sk-test") as p:
                await p.generate("m", "q", json_schema=schema)

    async def test_generate_empty_choices(self) -> None:
        mock_resp = httpx.Response(200, request=_DUMMY_REQUEST, json={"choices": []})
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp
        ):
            async with OpenRouterProvider(api_key="sk-test") as p:
                with pytest.raises(ProviderError, match="empty choices"):
                    await p.generate("m", "q")

    async def test_generate_http_error(self) -> None:
        mock_resp = httpx.Response(
            400, text="bad request", request=httpx.Request("POST", "http://x")
        )
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp
        ):
            async with OpenRouterProvider(api_key="sk-test") as p:
                with pytest.raises(ProviderError, match="OpenRouter HTTP 400"):
                    await p.generate("m", "q")

    async def test_generate_retries_on_429(self) -> None:
        retry_resp = httpx.Response(
            429, text="rate limited", request=httpx.Request("POST", "http://x")
        )
        ok_resp = httpx.Response(200, request=_DUMMY_REQUEST, json=_mock_openrouter_response())
        call_count = 0

        async def mock_post(_url: str, **_kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return retry_resp
            return ok_resp

        with (
            patch.object(httpx.AsyncClient, "post", side_effect=mock_post),
            patch("packages.llm.openrouter.asyncio.sleep", new_callable=AsyncMock),
        ):
            async with OpenRouterProvider(api_key="sk-test") as p:
                result = await p.generate("m", "q")
        assert result.text == "hello"
        assert call_count == 2

    async def test_check_connection_ok(self) -> None:
        mock_resp = httpx.Response(200, request=_DUMMY_REQUEST, json={"data": []})
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
            async with OpenRouterProvider(api_key="sk-test") as p:
                assert await p.check_connection() is True

    async def test_check_connection_fail(self) -> None:
        with patch.object(httpx.AsyncClient, "get", side_effect=httpx.ConnectError("x")):
            async with OpenRouterProvider(api_key="sk-test") as p:
                assert await p.check_connection() is False

    async def test_list_models(self) -> None:
        data = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        mock_resp = httpx.Response(200, request=_DUMMY_REQUEST, json=data)
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
            async with OpenRouterProvider(api_key="sk-test") as p:
                models = await p.list_models()
        assert models == ["model-a", "model-b"]

    def test_repr(self) -> None:
        p = OpenRouterProvider(api_key="sk-test", base_url="https://custom.api/v1")
        assert "https://custom.api/v1" in repr(p)


# ---------------------------------------------------------------------------
# ProviderFactory
# ---------------------------------------------------------------------------


class TestProviderFactory:
    def test_create_ollama(self) -> None:
        provider = ProviderFactory.create(ProviderType.OLLAMA)
        assert isinstance(provider, OllamaProvider)

    def test_create_ollama_string(self) -> None:
        provider = ProviderFactory.create("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_create_openrouter(self) -> None:
        provider = ProviderFactory.create(ProviderType.OPENROUTER, api_key="sk-test")
        assert isinstance(provider, OpenRouterProvider)

    def test_create_unsupported(self) -> None:
        with pytest.raises(ProviderError, match="Unsupported provider"):
            ProviderFactory.create("nonexistent")

    def test_provider_type_enum(self) -> None:
        assert ProviderType.OLLAMA == "ollama"
        assert ProviderType.OPENROUTER == "openrouter"


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_package_imports(self) -> None:
        from packages.llm import (
            BaseLLMProvider as _B,
        )
        from packages.llm import (
            LLMResponse as _R,
        )
        from packages.llm import (
            OllamaProvider as _O,
        )
        from packages.llm import (
            OpenRouterProvider as _OR,
        )
        from packages.llm import (
            ProviderFactory as _F,
        )
        from packages.llm import (
            ProviderType as _T,
        )

        assert _B is not None
        assert _R is not None
        assert _O is not None
        assert _OR is not None
        assert _F is not None
        assert _T is not None
