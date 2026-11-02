"""Local cross-conversation facts, with owner-scoped lexical retrieval.

No network, remote embeddings, cloud identity, or background upload queue.
Confirmed preferences and user statements retain different provenance.
"""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from stylemate.rag.retriever import _tokenize


class LocalMemory:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.lock = path, RLock()
        self.status = "local_ready"
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, slot TEXT NOT NULL,
                content TEXT NOT NULL, provenance TEXT NOT NULL, source_run TEXT NOT NULL,
                updated_at TEXT NOT NULL, UNIQUE(owner,slot));
            CREATE INDEX IF NOT EXISTS memories_owner_updated ON memories(owner,updated_at);
            CREATE TABLE IF NOT EXISTS suppressed (owner TEXT NOT NULL, slot TEXT NOT NULL,
                content TEXT NOT NULL, PRIMARY KEY(owner,slot));
        """)
        self.db.commit()

    def enqueue(self, owner, run, profile, facts):
        values = []
        for key, value in profile.items():
            if key in {"fit_preference", "style_preference", "color_preference", "scene_preference"} and value:
                values.append(("profile:" + key, key + "：" + str(value)[:160], "confirmed_preference"))
        for constraint in facts.constraints:
            if constraint.strip():
                values.append(("constraint:" + constraint.strip(), "用户曾表达的条件：" + constraint[:400], "user_statement"))
        for scene in facts.scenes:
            if scene in {"通勤", "旅行", "日常", "约会", "运动", "面试", "聚会"}:
                values.append(("scene:" + scene, "用户曾咨询的服饰场景：" + scene, "user_statement"))
        stamp = datetime.now(timezone.utc).isoformat()
        with self.lock, self.db:
            # Profile removals also remove the corresponding remembered field.
            keep = {slot for slot, _, _ in values if slot.startswith("profile:")}
            for (slot,) in self.db.execute("SELECT slot FROM memories WHERE owner=? AND slot LIKE 'profile:%'", (owner,)).fetchall():
                if slot not in keep:
                    self.db.execute("DELETE FROM memories WHERE owner=? AND slot=?", (owner, slot))
            for slot, content, provenance in values:
                suppressed = self.db.execute("SELECT content FROM suppressed WHERE owner=? AND slot=?", (owner, slot)).fetchone()
                if suppressed and suppressed[0] == content:
                    continue
                self.db.execute("DELETE FROM suppressed WHERE owner=? AND slot=?", (owner, slot))
                self.db.execute("""INSERT INTO memories VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(owner,slot) DO UPDATE SET content=excluded.content,
                    provenance=excluded.provenance,source_run=excluded.source_run,updated_at=excluded.updated_at""",
                                (str(uuid.uuid4()), owner, slot, content, provenance, run, stamp))
            self.db.execute("""DELETE FROM memories WHERE owner=? AND id NOT IN
                (SELECT id FROM memories WHERE owner=? ORDER BY updated_at DESC LIMIT 500)""", (owner, owner))

    def list(self, owner):
        with self.lock:
            rows = self.db.execute("SELECT id,content,provenance,source_run,updated_at FROM memories WHERE owner=? ORDER BY updated_at DESC,id", (owner,)).fetchall()
        return [{"id": key, "memory": text, "provenance": provenance, "source_run": run, "updated_at": stamp}
                for key, text, provenance, run, stamp in rows]

    def search(self, owner, query):
        values = self.list(owner)
        tokens = set(_tokenize(query))
        if not values:
            return []
        documents = [Counter(_tokenize(item["memory"])) for item in values]
        frequencies = Counter(token for document in documents for token in document)
        ranked = []
        for item, document in zip(values, documents):
            score = sum(math.log(1 + (len(values) + 1) / (frequencies[token] + 1)) * min(document[token], 2)
                        for token in tokens if token in document)
            if score > 0 or item["provenance"] == "confirmed_preference":
                ranked.append((score, item))
        ranked.sort(key=lambda pair: (pair[0], pair[1]["updated_at"]), reverse=True)
        return [{"memory": item["memory"], "source": "personal_memory", "provenance": item["provenance"],
                 "source_run": item["source_run"]} for _, item in ranked[:3]]

    def delete(self, owner, memory_id):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO suppressed SELECT owner,slot,content FROM memories WHERE owner=? AND id=?", (owner, memory_id))
            result = self.db.execute("DELETE FROM memories WHERE owner=? AND id=?", (owner, memory_id))
            if result.rowcount != 1:
                raise ValueError("memory_identity_denied")

    def clear(self, owner):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO suppressed SELECT owner,slot,content FROM memories WHERE owner=?", (owner,))
            self.db.execute("DELETE FROM memories WHERE owner=?", (owner,))

    def snapshot(self):
        return {"status": self.status, "provider": "local_sqlite", "queue": {}}

    def export(self, owner):
        return json.dumps(self.list(owner), ensure_ascii=False, indent=2)
