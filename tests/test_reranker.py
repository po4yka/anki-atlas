"""Tests for packages/search/reranker.py."""

from __future__ import annotations

import builtins
from typing import Any
from unittest.mock import MagicMock

import pytest

from packages.common.exceptions import ConfigurationError
from packages.search.reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    async def test_rerank_empty_documents(self) -> None:
        reranker = CrossEncoderReranker()
        result = await reranker.rerank("query", [])
        assert result == []

    async def test_rerank_returns_scores(self) -> None:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8, 0.5]
        reranker._model = mock_model

        docs = [(1, "first doc"), (2, "second doc")]
        result = await reranker.rerank("query", docs)

        assert result == [(1, 0.8), (2, 0.5)]

    async def test_rerank_builds_correct_pairs(self) -> None:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7]
        reranker._model = mock_model

        docs = [(10, "alpha"), (20, "beta")]
        await reranker.rerank("my query", docs)

        call_args = mock_model.predict.call_args
        pairs = call_args[0][0]
        assert pairs == [("my query", "alpha"), ("my query", "beta")]

    def test_predict_scores_empty(self) -> None:
        reranker = CrossEncoderReranker()
        assert reranker._predict_scores([]) == []

    def test_predict_scores_with_tolist(self) -> None:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_array = MagicMock()
        mock_array.tolist.return_value = [0.6, 0.3]
        mock_model.predict.return_value = mock_array
        reranker._model = mock_model

        result = reranker._predict_scores([("q", "a"), ("q", "b")])
        assert result == [0.6, 0.3]
        mock_array.tolist.assert_called_once()

    def test_predict_scores_plain_list(self) -> None:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock(spec=[])  # no tolist attribute
        mock_model.predict = MagicMock(return_value=[0.4, 0.2])
        reranker._model = mock_model

        result = reranker._predict_scores([("q", "a"), ("q", "b")])
        assert result == [0.4, 0.2]

    def test_get_model_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reranker = CrossEncoderReranker()
        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "sentence_transformers":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ConfigurationError, match="sentence-transformers"):
            reranker._get_model()
