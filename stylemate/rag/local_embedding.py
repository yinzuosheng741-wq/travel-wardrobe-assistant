"""Optional CPU BGE embedding with explicit, local model assets."""
from pathlib import Path
from threading import RLock


class LocalBGEEmbedding:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise ValueError("local_embedding_model_missing")
        self.model = "bge-small-zh-v1.5:" + str(self.path.resolve())
        self.lock = RLock()
        self.encoder = None

    def _encode(self, texts):
        with self.lock:
            if self.encoder is None:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(str(self.path), device="cpu", local_files_only=True)
            return self.encoder.encode(texts, normalize_embeddings=True).tolist()

    def embed_documents(self, texts):
        return self._encode(texts)

    def embed_query(self, text):
        return self._encode(["为这个句子生成表示以用于检索相关文章：" + text])[0]
