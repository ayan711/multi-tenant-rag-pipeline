"""
Tests for app/embedder.py — Task 2.3: Local Vector Model Initialization.

All tests mock SentenceTransformer so the suite runs without downloading
model weights (~80 MB). The mock returns a realistic fake instance whose
get_sentence_embedding_dimension() yields 384, matching all-MiniLM-L6-v2.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.embedder import LocalEmbedder


@pytest.fixture
def mock_st():
    """Patch SentenceTransformer at the point of import in app.embedder."""
    with patch("app.embedder.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


class TestLocalEmbedderInit:

    def test_default_model_name_constant(self):
        assert LocalEmbedder.MODEL_NAME == "all-MiniLM-L6-v2"

    def test_loads_default_model_on_instantiation(self, mock_st):
        mock_cls, _ = mock_st
        LocalEmbedder()
        mock_cls.assert_called_once_with(LocalEmbedder.MODEL_NAME)

    def test_loads_custom_model_name(self, mock_st):
        mock_cls, _ = mock_st
        LocalEmbedder("custom-model")
        mock_cls.assert_called_once_with("custom-model")

    def test_model_name_attribute_set_to_default(self, mock_st):
        emb = LocalEmbedder()
        assert emb.model_name == LocalEmbedder.MODEL_NAME

    def test_model_name_attribute_set_to_custom(self, mock_st):
        emb = LocalEmbedder("custom-model")
        assert emb.model_name == "custom-model"

    def test_embedding_dim_read_from_model(self, mock_st):
        _, mock_instance = mock_st
        emb = LocalEmbedder()
        assert emb.embedding_dim == 384
        mock_instance.get_sentence_embedding_dimension.assert_called_once()

    def test_model_instance_stored_on_self(self, mock_st):
        _, mock_instance = mock_st
        emb = LocalEmbedder()
        assert emb._model is mock_instance

    def test_sentence_transformer_called_exactly_once_per_instance(self, mock_st):
        mock_cls, _ = mock_st
        LocalEmbedder()
        LocalEmbedder()
        assert mock_cls.call_count == 2


class TestEmbed:

    def test_returns_list_of_float(self, mock_st):
        import numpy as np
        _, mock_instance = mock_st
        mock_instance.encode.return_value = np.zeros(384)
        result = LocalEmbedder().embed("hello")
        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(v, float) for v in result)

    def test_calls_encode_with_correct_args(self, mock_st):
        import numpy as np
        _, mock_instance = mock_st
        mock_instance.encode.return_value = np.zeros(384)
        LocalEmbedder().embed("test sentence")
        mock_instance.encode.assert_called_once_with("test sentence", show_progress_bar=False)


class TestEmbedBatch:

    def test_returns_list_of_lists(self, mock_st):
        import numpy as np
        _, mock_instance = mock_st
        mock_instance.encode.return_value = np.zeros((3, 384))
        result = LocalEmbedder().embed_batch(["a", "b", "c"])
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(v, list) for v in result)
        assert all(len(v) == 384 for v in result)

    def test_calls_encode_with_batch_size_and_no_progress_bar(self, mock_st):
        import numpy as np
        _, mock_instance = mock_st
        mock_instance.encode.return_value = np.zeros((2, 384))
        LocalEmbedder().embed_batch(["x", "y"])
        mock_instance.encode.assert_called_once_with(["x", "y"], batch_size=32, show_progress_bar=False)

    def test_empty_input_returns_empty_list(self, mock_st):
        import numpy as np
        _, mock_instance = mock_st
        mock_instance.encode.return_value = np.zeros((0, 384))
        result = LocalEmbedder().embed_batch([])
        assert result == []
