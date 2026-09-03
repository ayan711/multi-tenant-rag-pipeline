from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """Wraps sentence-transformers to produce 384-dim embeddings locally."""

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        # SentenceTransformer downloads once then caches weights to ~/.cache/huggingface/
        self._model = SentenceTransformer(model_name)
        # Read dim from the model rather than hardcoding — stays correct if the model changes.
        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """Embed a single string and return a plain Python list of floats."""
        # suppress_progress_bar avoids tqdm noise in server logs
        vector = self._model.encode(text, show_progress_bar=False)
        # ChromaDB expects list[float], not a numpy array
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings in one forward pass; returns list of vectors."""
        # batch_size=32: amortises tokenisation overhead on CPU without exhausting RAM
        vectors = self._model.encode(texts, batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in vectors]
