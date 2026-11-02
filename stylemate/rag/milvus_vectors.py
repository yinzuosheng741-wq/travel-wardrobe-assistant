"""Milvus Lite 3.x native adapter; no local RPC server or WSL required.

Preserves our scoped collection protocol so evidence governance stays above
the retrieval engine. COSINE hits are similarity, converted to ascending distance.
"""
from __future__ import annotations

import atexit
import json
import re
import tempfile
from pathlib import Path
from threading import RLock

from stylemate.rag.portable_vectors import _vector

_persistent_clients = {}
_clients_lock = RLock()


def persistent_client(path):
    key = str(Path(path).resolve())
    with _clients_lock:
        client = _persistent_clients.get(key)
        if client is None or client.closed:
            client = MilvusVectorClient(path)
            _persistent_clients[key] = client
        return client


def filter_expression(where):
    clauses = []
    for key, value in (where or {}).items():
        if key == "$and":
            if not isinstance(value, list) or not value:
                raise ValueError("invalid_filter")
            clauses.append("(" + " and ".join(filter_expression(item) for item in value) + ")")
        else:
            if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", key):
                raise ValueError("invalid_filter_key")
            field = 'metadata[' + json.dumps(key) + ']'
            if isinstance(value, dict):
                if set(value) != {"$in"} or not isinstance(value["$in"], list):
                    raise ValueError("unsupported_filter")
                clauses.append("(" + " or ".join(field + " == " + json.dumps(item, ensure_ascii=False)
                                                 for item in value["$in"]) + ")" if value["$in"] else 'id == ""')
            else:
                clauses.append(field + " == " + json.dumps(value, ensure_ascii=False))
    return " and ".join(clauses)


class MilvusVectorClient:
    backend = "milvus"

    def __init__(self, path=None):
        from milvus_lite import MilvusLite
        self.temporary = tempfile.TemporaryDirectory(prefix="wardrobe-milvus-", ignore_cleanup_errors=True) if path is None else None
        self.path = Path(path or self.temporary.name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.db = MilvusLite(str(self.path))
        self.closed = False
        atexit.register(self.close)

    def get_or_create_collection(self, name, metadata=None):
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", name):
            raise ValueError("invalid_collection_name")
        return MilvusCollection(self, name)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.db.close()
            if self.temporary:
                self.temporary.cleanup()


class MilvusCollection:
    def __init__(self, client, name):
        self.client, self.name = client, name

    def _collection(self, dimension=None):
        from milvus_lite import CollectionSchema, DataType, FieldSchema
        if self.name not in self.client.db.list_collections():
            if dimension is None:
                return None
            schema = CollectionSchema([
                FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=256),
                FieldSchema("content", DataType.VARCHAR, max_length=65535),
                FieldSchema("metadata", DataType.JSON),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dimension),
            ])
            return self.client.db.create_collection(self.name, schema)
        return self.client.db.get_collection(self.name)

    def count(self):
        with self.client.lock:
            collection = self._collection()
            return len(collection.query(output_fields=["id"])) if collection else 0

    def upsert(self, *, ids, embeddings, documents, metadatas=None):
        metadatas = metadatas if metadatas is not None else [{} for _ in ids]
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)) or len(set(ids)) != len(ids):
            raise ValueError("invalid_vector_batch")
        vectors = [_vector(vector) for vector in embeddings]
        if not vectors:
            return
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("vector_dimension_mismatch")
        with self.client.lock:
            collection = self._collection(dimension)
            collection.upsert([{"id": key, "content": content, "metadata": meta, "vector": vector}
                               for key, content, meta, vector in zip(ids, documents, metadatas, vectors)])

    def get(self, *, ids=None, where=None, include=None):
        expression = filter_expression(where)
        if ids is not None:
            extra = "id in " + json.dumps(ids)
            expression = "(" + expression + ") and " + extra if expression else extra
        with self.client.lock:
            collection = self._collection()
            rows = collection.query(expr=expression or 'id != ""', output_fields=["id", "content", "metadata"]) if collection else []
        return {"ids": [row["id"] for row in rows], "documents": [row["content"] for row in rows],
                "metadatas": [_metadata(row["metadata"]) for row in rows]}

    def delete(self, *, ids=None, where=None):
        if ids is None and where is None:
            raise ValueError("delete_filter_required")
        keys = self.get(ids=ids, where=where)["ids"]
        if keys:
            with self.client.lock:
                self._collection().delete(pks=keys)

    def query(self, *, query_embeddings, n_results, where=None, include=None):
        output = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        with self.client.lock:
            collection = self._collection()
            if collection and self.count():
                results = collection.search([_vector(vector) for vector in query_embeddings], top_k=n_results,
                                            metric_type="COSINE", expr=filter_expression(where) or None,
                                            output_fields=["content", "metadata"])
            else:
                results = [[] for _ in query_embeddings]
            for hits in results:
                output["ids"].append([hit["id"] for hit in hits])
                output["documents"].append([hit["entity"]["content"] for hit in hits])
                output["metadatas"].append([_metadata(hit["entity"]["metadata"]) for hit in hits])
                output["distances"].append([1.0 - float(hit["distance"]) for hit in hits])
        return output


def _metadata(value):
    return json.loads(value) if isinstance(value, str) else value
