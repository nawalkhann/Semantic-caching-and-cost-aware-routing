"""Per-request cost/latency log (SQLite) and aggregate summaries."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass
class RequestRecord:
    ts: float
    query: str
    route: str  # "cache" | model name
    cache_hit: bool
    similarity: float | None
    complexity: float | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    embed_ms: float
    retrieve_ms: float
    cache_ms: float
    llm_ms: float
    correct: bool | None = None
    run: str = "default"


_COLUMNS = [f.name for f in fields(RequestRecord)]
_TYPES = {"ts": "REAL", "query": "TEXT", "route": "TEXT", "cache_hit": "INTEGER",
          "similarity": "REAL", "complexity": "REAL", "input_tokens": "INTEGER",
          "output_tokens": "INTEGER", "cost_usd": "REAL", "latency_ms": "REAL",
          "embed_ms": "REAL", "retrieve_ms": "REAL", "cache_ms": "REAL", "llm_ms": "REAL",
          "correct": "INTEGER", "run": "TEXT"}


class MetricsStore:
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        cols = ", ".join(f"{c} {_TYPES[c]}" for c in _COLUMNS)
        with self._lock:
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS requests "
                               f"(id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_run ON requests(run, id)")
            self._conn.commit()

    def log(self, rec: RequestRecord) -> None:
        row = asdict(rec)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO requests ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
                [row[c] for c in _COLUMNS])
            self._conn.commit()

    def runs(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._conn.execute(
                "SELECT run FROM requests GROUP BY run ORDER BY MIN(id)")]

    def records(self, run: str | None = None, limit: int | None = None) -> list[dict]:
        sql = f"SELECT id, {', '.join(_COLUMNS)} FROM requests"
        args: list = []
        if run is not None:
            sql += " WHERE run = ?"
            args.append(run)
        sql += " ORDER BY id"
        if limit:
            sql = f"SELECT * FROM ({sql.replace('ORDER BY id', 'ORDER BY id DESC')} LIMIT ?) ORDER BY id"
            args.append(limit)
        with self._lock:
            cur = self._conn.execute(sql, args)
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"requests": 0}
    lat = np.array([r["latency_ms"] for r in records])
    cost = np.array([r["cost_usd"] for r in records])
    graded = [r["correct"] for r in records if r["correct"] is not None]
    by_route: dict[str, dict] = {}
    for r in records:
        b = by_route.setdefault(r["route"], {"requests": 0, "cost_usd": 0.0, "latency_ms": []})
        b["requests"] += 1
        b["cost_usd"] += r["cost_usd"]
        b["latency_ms"].append(r["latency_ms"])
    for b in by_route.values():
        l = b.pop("latency_ms")
        b["share"] = b["requests"] / len(records)
        b["p50_latency_ms"] = float(np.percentile(l, 50))
        b["p95_latency_ms"] = float(np.percentile(l, 95))
    return {
        "requests": len(records),
        "total_cost_usd": float(cost.sum()),
        "cost_per_1k_requests_usd": float(cost.mean() * 1000),
        "p50_latency_ms": float(np.percentile(lat, 50)),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
        "mean_latency_ms": float(lat.mean()),
        "cache_hit_rate": sum(1 for r in records if r["cache_hit"]) / len(records),
        "accuracy": (sum(graded) / len(graded)) if graded else None,
        "by_route": by_route,
    }
