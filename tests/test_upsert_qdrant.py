"""Unit tests for ``embed.upsert_qdrant`` (Qdrant collection + upsert, Task 2.3).

These tests run **fully offline**: they use an **embedded on-disk Qdrant**
(``QdrantClient(path=...)`` under a temp dir) — no server, no Docker, and
**no OpenAI call** (dummy vectors only, cost rule §(i)). Each test gets its own
isolated collection + storage path via fixtures, so nothing leaks between tests
or into ``artifacts/``.

Run from the project root::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_upsert_qdrant.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
from qdrant_client import QdrantClient, models

import config
from embed import upsert_qdrant as U


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def local_client(tmp_path: Path) -> Iterator[QdrantClient]:
    """An embedded, isolated Qdrant client backed by a temp directory."""
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        yield client
    finally:
        client.close()


@pytest.fixture()
def isolated_collection(monkeypatch: pytest.MonkeyPatch) -> str:
    """Point ``config.COLLECTION`` at a unique test collection name."""
    name = "lawbot_test_upsert"
    monkeypatch.setattr(config, "COLLECTION", name)
    return name


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _chunk(i: int, *, doc_type: str = "law", jurisdiction: str = "국가") -> dict:
    cid = f"LAW:{i:06d}:법률#제{i + 1}조#0"
    return {
        "chunk_id": cid,
        "doc_id": f"LAW:{i:06d}:법률",
        "parent_id": f"LAW:{i:06d}:법률",
        "text": f"[테스트법 제{i + 1}조]\n조문 본문 {i}.",
        "payload": {
            "doc_type": doc_type,
            "title": "테스트법",
            "jurisdiction": jurisdiction,
            "law_kind": "법률",
            "article_no": f"제{i + 1}조",
            "effective_from": "2026-01-01",
            "trust_grade": "A",
            "license": config.DEFAULT_LICENSE,
            "parent_id": f"LAW:{i:06d}:법률",
        },
    }


def _unit_vector(seed: int) -> list[float]:
    vec = [0.0] * config.EMBED_DIM
    vec[seed % config.EMBED_DIM] = 1.0
    return vec


# --------------------------------------------------------------------------- #
# point_id determinism                                                          #
# --------------------------------------------------------------------------- #
def test_point_id_is_deterministic_uuid() -> None:
    cid = "LAW:014565:법률#제4조#0"
    a = U.point_id(cid)
    b = U.point_id(cid)
    assert a == b
    # Valid UUID string, and distinct ids for distinct chunk_ids.
    import uuid

    uuid.UUID(a)  # raises if malformed
    assert U.point_id("OTHER#x#0") != a


# --------------------------------------------------------------------------- #
# ensure_collection                                                             #
# --------------------------------------------------------------------------- #
def test_ensure_collection_creates_with_right_dim(
    local_client: QdrantClient, isolated_collection: str
) -> None:
    U.ensure_collection(local_client)
    assert local_client.collection_exists(isolated_collection)
    info = local_client.get_collection(isolated_collection)
    assert U._existing_vector_size(info) == config.EMBED_DIM


def test_ensure_collection_is_idempotent(
    local_client: QdrantClient, isolated_collection: str
) -> None:
    U.ensure_collection(local_client)
    # Second call must not raise nor wipe the collection.
    U.ensure_collection(local_client)
    assert local_client.collection_exists(isolated_collection)


def test_ensure_collection_rejects_dim_mismatch(
    local_client: QdrantClient, isolated_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create a collection at a deliberately wrong dimension, then assert
    # ensure_collection refuses to reuse it for a different EMBED_DIM.
    local_client.create_collection(
        collection_name=isolated_collection,
        vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
    )
    monkeypatch.setattr(config, "EMBED_DIM", 1536)
    with pytest.raises(ValueError):
        U.ensure_collection(local_client)


# --------------------------------------------------------------------------- #
# upsert_all                                                                    #
# --------------------------------------------------------------------------- #
def test_upsert_all_joins_and_counts(
    local_client: QdrantClient, isolated_collection: str, tmp_path: Path
) -> None:
    chunks = [_chunk(i) for i in range(3)]
    embs = [{"chunk_id": c["chunk_id"], "vector": _unit_vector(i)} for i, c in enumerate(chunks)]
    cp, ep = tmp_path / "c.jsonl", tmp_path / "e.jsonl"
    _write_jsonl(cp, chunks)
    _write_jsonl(ep, embs)

    n = U.upsert_all(local_client, chunks_path=cp, embeddings_path=ep, ensure=True)
    assert n == 3
    assert local_client.count(isolated_collection, exact=True).count == 3

    # Payload carries the join key + filter keys + text.
    pid = U.point_id(chunks[0]["chunk_id"])
    got = local_client.retrieve(isolated_collection, ids=[pid], with_payload=True)[0]
    assert got.payload["chunk_id"] == chunks[0]["chunk_id"]
    assert got.payload["doc_type"] == "law"
    assert got.payload["parent_id"] == chunks[0]["parent_id"]
    assert "text" in got.payload


def test_upsert_skips_chunks_without_embeddings(
    local_client: QdrantClient, isolated_collection: str, tmp_path: Path
) -> None:
    chunks = [_chunk(i) for i in range(3)]
    # Only the first two chunks have embeddings.
    embs = [{"chunk_id": chunks[i]["chunk_id"], "vector": _unit_vector(i)} for i in range(2)]
    cp, ep = tmp_path / "c.jsonl", tmp_path / "e.jsonl"
    _write_jsonl(cp, chunks)
    _write_jsonl(ep, embs)

    n = U.upsert_all(local_client, chunks_path=cp, embeddings_path=ep, ensure=True)
    assert n == 2
    assert local_client.count(isolated_collection, exact=True).count == 2


def test_upsert_is_idempotent_on_reingest(
    local_client: QdrantClient, isolated_collection: str, tmp_path: Path
) -> None:
    chunks = [_chunk(i) for i in range(4)]
    embs = [{"chunk_id": c["chunk_id"], "vector": _unit_vector(i)} for i, c in enumerate(chunks)]
    cp, ep = tmp_path / "c.jsonl", tmp_path / "e.jsonl"
    _write_jsonl(cp, chunks)
    _write_jsonl(ep, embs)

    U.upsert_all(local_client, chunks_path=cp, embeddings_path=ep, ensure=True)
    U.upsert_all(local_client, chunks_path=cp, embeddings_path=ep, ensure=True)
    # Re-ingest overwrites in place (UUID5 point ids) — no duplicates.
    assert local_client.count(isolated_collection, exact=True).count == 4


def test_upsert_rejects_wrong_dim_vector(
    local_client: QdrantClient, isolated_collection: str, tmp_path: Path
) -> None:
    chunks = [_chunk(0)]
    embs = [{"chunk_id": chunks[0]["chunk_id"], "vector": [0.1, 0.2, 0.3]}]  # wrong dim
    cp, ep = tmp_path / "c.jsonl", tmp_path / "e.jsonl"
    _write_jsonl(cp, chunks)
    _write_jsonl(ep, embs)
    with pytest.raises(ValueError):
        U.upsert_all(local_client, chunks_path=cp, embeddings_path=ep, ensure=True)


def test_upsert_missing_artifact_raises(
    local_client: QdrantClient, isolated_collection: str, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        U.upsert_all(
            local_client,
            chunks_path=tmp_path / "nope.jsonl",
            embeddings_path=tmp_path / "also_nope.jsonl",
            ensure=True,
        )


# --------------------------------------------------------------------------- #
# filtered search (payload filter keys)                                         #
# --------------------------------------------------------------------------- #
def test_payload_filter_narrows_results(
    local_client: QdrantClient, isolated_collection: str, tmp_path: Path
) -> None:
    chunks = [
        _chunk(0, doc_type="law"),
        _chunk(1, doc_type="ordinance", jurisdiction="전라남도"),
        _chunk(2, doc_type="law"),
    ]
    embs = [{"chunk_id": c["chunk_id"], "vector": _unit_vector(i)} for i, c in enumerate(chunks)]
    cp, ep = tmp_path / "c.jsonl", tmp_path / "e.jsonl"
    _write_jsonl(cp, chunks)
    _write_jsonl(ep, embs)
    U.upsert_all(local_client, chunks_path=cp, embeddings_path=ep, ensure=True)

    rows, _ = local_client.scroll(
        collection_name=isolated_collection,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="law"))]
        ),
        limit=100,
        with_payload=True,
    )
    assert {r.payload["doc_type"] for r in rows} == {"law"}
    assert len(rows) == 2


# --------------------------------------------------------------------------- #
# self_test (the 5-point DoD smoke), run against the isolated collection        #
# --------------------------------------------------------------------------- #
def test_self_test_passes(
    isolated_collection: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the local-path fallback into a temp dir (no server, no artifacts/).
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant_selftest"))
    monkeypatch.delenv("LAWBOT_QDRANT_REQUIRE_SERVER", raising=False)
    # Make the server probe always fail so we deterministically use local path.
    monkeypatch.setattr(U, "_server_reachable", lambda: False)

    report = U.self_test()
    assert report["ok"] is True
    assert report["is_server"] is False
    assert report["count_after_upsert"] >= 5
    assert report["top_hit_chunk_id"] == "__dummy__#0"


# --------------------------------------------------------------------------- #
# get_client: require-server guard                                              #
# --------------------------------------------------------------------------- #
def test_require_server_forbids_local_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(U, "_server_reachable", lambda: False)
    monkeypatch.setenv("LAWBOT_QDRANT_REQUIRE_SERVER", "1")
    with pytest.raises(RuntimeError):
        U.get_client()
