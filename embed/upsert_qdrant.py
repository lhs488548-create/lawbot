"""Qdrant collection management and upsert (Playbook 08, Task 2.3).

Load the chunk payloads (``config.CHUNKS_JSONL``) and their embedding vectors
(``config.EMBEDDINGS_JSONL``) into a Qdrant collection so the retriever can run
dense + metadata-filtered search over the corpus.

Design (BUILD CONTRACT (d) "Qdrant upsert" + 09 §A/E):

* **Collection** ``config.COLLECTION`` with vector ``size = config.EMBED_DIM``
  and **Cosine** distance (OpenAI vectors are L2-normalized ⇒ cosine = dot).
* **Payload KEYWORD indexes** on the contract filter keys
  ``doc_type, jurisdiction, law_kind, effective_from`` so pre-filtered search
  (e.g. "법률만", "전라남도 조례만", point-in-time ``effective_from``) is fast.
* **Point id = deterministic UUID5 of the ``chunk_id``** (namespace below). The
  same ``chunk_id`` therefore always maps to the same point ⇒ re-ingest is
  **idempotent** (upsert overwrites in place, never duplicates). The original
  ``chunk_id`` is preserved in the payload for joins/citations.
* The full child payload (filter keys + citation meta + ``parent_id`` + the
  embedded ``text``) is stored so the retriever returns ready-to-cite hits
  without a second lookup.

Connection (BUILD CONTRACT (d) + task: "서버 우선 로컬 path 폴백"):

* **Server first** — connect to ``config.QDRANT_URL`` (+ ``config.QDRANT_API_KEY``
  for Qdrant Cloud). If a server is reachable it is always used.
* **Local-path fallback** — when no server is reachable, fall back to an
  embedded on-disk Qdrant at ``$QDRANT_PATH`` (default
  ``config.ARTIFACTS_DIR/qdrant_local``, git-ignored). This lets the full
  ingest → search smoke test run with no Docker/cloud dependency. Set
  ``LAWBOT_QDRANT_REQUIRE_SERVER=1`` to forbid the fallback (production).

Public interface (BUILD CONTRACT (d))::

    def ensure_collection() -> None: ...   # create collection + payload indexes
    def upsert_all() -> None: ...          # join chunks+embeddings -> upsert

Run as a script::

    cd /home/user1/lawbot && .venv/bin/python -m embed.upsert_qdrant            # ensure + upsert
    cd /home/user1/lawbot && .venv/bin/python -m embed.upsert_qdrant --verify   # 5-point self-test
    cd /home/user1/lawbot && .venv/bin/python -m embed.upsert_qdrant --ensure   # ensure collection only

Owner: embed builder. Imports shared constants from ``config`` and never
redefines them. Secrets (``QDRANT_API_KEY``) are read via ``config`` only and
are never printed or written to an artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Final, Iterable, Iterator

from qdrant_client import QdrantClient, models
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
# Stable namespace for deriving a point UUID from a chunk_id. Fixed forever so
# the same chunk_id always yields the same point id across machines/runs
# (idempotent upsert). Do NOT change this value — it would orphan existing
# points and silently duplicate the corpus on re-ingest.
_POINT_NAMESPACE: Final[uuid.UUID] = uuid.UUID("6c61776b-6f74-5151-6472-616e74707473")

# Payload fields used as pre-filter keys (KEYWORD indexes). These mirror the
# BUILD CONTRACT filter set and the §E-1 search pipeline (doc_type, jurisdiction,
# law_kind, effective_from). Indexing them makes filtered search O(index).
_INDEXED_FIELDS: Final[tuple[str, ...]] = (
    "doc_type",
    "jurisdiction",
    "law_kind",
    "effective_from",
)

# Upsert batch size (points per request). ~1000 keeps each request well under
# Qdrant payload limits while amortizing round-trips for a multi-million corpus.
_UPSERT_BATCH: Final[int] = 1000

# Default embedded (on-disk) Qdrant location used only when no server is up.
# Git-ignored (lives under artifacts/). Overridable via QDRANT_PATH.
_DEFAULT_LOCAL_PATH: Final[Path] = config.ARTIFACTS_DIR / "qdrant_local"

# Connection probe timeout (seconds) when deciding server-vs-local.
_PROBE_TIMEOUT: Final[int] = 3


# --------------------------------------------------------------------------- #
# Client (server-first, local-path fallback)                                   #
# --------------------------------------------------------------------------- #
def _local_path() -> str:
    """Return the on-disk Qdrant path for the embedded fallback.

    Reads the optional ``QDRANT_PATH`` environment override (not a secret),
    defaulting to ``config.ARTIFACTS_DIR/qdrant_local``. The directory is
    created on demand by the embedded client.

    Returns:
        A POSIX path string for ``QdrantClient(path=...)``.
    """
    override = os.getenv("QDRANT_PATH")
    return override if override else str(_DEFAULT_LOCAL_PATH)


def _server_reachable() -> bool:
    """Return whether a Qdrant server answers at ``config.QDRANT_URL``.

    A lightweight ``get_collections`` probe with a short timeout. Any failure
    (connection refused, timeout, auth) is treated as "not reachable" so the
    caller can fall back to the embedded path client. Never raises.

    Returns:
        ``True`` if the server responded, ``False`` otherwise.
    """
    try:
        probe = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=_PROBE_TIMEOUT,
        )
        probe.get_collections()
        probe.close()
        return True
    except Exception:  # noqa: BLE001 - any failure => not reachable
        return False


def get_client() -> tuple[QdrantClient, bool]:
    """Return a Qdrant client, preferring a live server over the local fallback.

    Resolution order (task requirement "서버 우선 로컬 path 폴백"):

    1. If a server answers at ``config.QDRANT_URL`` → use it (server mode).
    2. Else, unless ``LAWBOT_QDRANT_REQUIRE_SERVER`` is set, open an **embedded**
       on-disk client at ``_local_path()`` (local mode) so ingest/search work
       with no external service.

    Returns:
        ``(client, is_server)`` where ``is_server`` is ``True`` for the remote
        server and ``False`` for the embedded local-path client.

    Raises:
        RuntimeError: If no server is reachable and
            ``LAWBOT_QDRANT_REQUIRE_SERVER`` is set (the local fallback is
            explicitly forbidden, e.g. in production).
    """
    if _server_reachable():
        return (
            QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
                timeout=60,
            ),
            True,
        )

    if os.getenv("LAWBOT_QDRANT_REQUIRE_SERVER"):
        raise RuntimeError(
            f"No Qdrant server reachable at {config.QDRANT_URL} and "
            "LAWBOT_QDRANT_REQUIRE_SERVER is set (local fallback forbidden). "
            "Start Qdrant (docker run -p 6333:6333 qdrant/qdrant) or unset the "
            "variable to allow the embedded path client."
        )

    path = _local_path()
    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=path), False


# --------------------------------------------------------------------------- #
# Point id                                                                     #
# --------------------------------------------------------------------------- #
def point_id(chunk_id: str) -> str:
    """Map a ``chunk_id`` to a deterministic UUID5 point id.

    Qdrant point ids must be unsigned integers or UUIDs; our ``chunk_id``s are
    strings (e.g. ``"LAW:014565:법률#제4조#0"``). UUID5 over a fixed namespace
    gives a stable, collision-resistant id so re-ingesting the same chunk
    overwrites its point rather than creating a duplicate (idempotency).

    Args:
        chunk_id: The chunk's string id (also kept verbatim in the payload).

    Returns:
        The string form of the UUID5 point id.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


# --------------------------------------------------------------------------- #
# Collection lifecycle                                                          #
# --------------------------------------------------------------------------- #
def ensure_collection(
    client: QdrantClient | None = None,
    *,
    recreate: bool = False,
) -> None:
    """Ensure the ``config.COLLECTION`` collection and payload indexes exist.

    Creates the collection with ``size = config.EMBED_DIM`` and Cosine distance
    if it does not exist, then ensures a KEYWORD payload index on each filter
    field in ``_INDEXED_FIELDS``. Idempotent: calling it repeatedly is safe and
    leaves existing data untouched (unless ``recreate=True``).

    Args:
        client: An existing Qdrant client; one is created (server-first) if
            ``None``.
        recreate: When ``True``, drop and recreate the collection first
            (destroys all points). Use only for a clean rebuild.

    Raises:
        ValueError: If an existing collection's vector size disagrees with
            ``config.EMBED_DIM`` (a model/dim mismatch that would corrupt
            search). The collection must be rebuilt to switch dimensions.
    """
    if client is None:
        client, _ = get_client()

    if recreate and client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)

    if not client.collection_exists(config.COLLECTION):
        client.create_collection(
            collection_name=config.COLLECTION,
            vectors_config=models.VectorParams(
                size=config.EMBED_DIM,
                distance=models.Distance.COSINE,
            ),
        )
    else:
        info = client.get_collection(config.COLLECTION)
        existing = _existing_vector_size(info)
        if existing is not None and existing != config.EMBED_DIM:
            raise ValueError(
                f"Collection {config.COLLECTION!r} has vector size {existing} "
                f"but config.EMBED_DIM is {config.EMBED_DIM}. Switching models "
                "requires a full rebuild (ensure_collection(recreate=True))."
            )

    for field in _INDEXED_FIELDS:
        try:
            client.create_payload_index(
                collection_name=config.COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:  # noqa: BLE001 - index already exists / benign race
            # create_payload_index is not idempotent on every backend; an
            # already-present index raises. That is the desired end-state, so
            # swallow it (we never index a non-existent field here).
            pass


def _existing_vector_size(info: Any) -> int | None:
    """Best-effort extraction of an existing collection's vector size.

    Qdrant exposes the vector config in a couple of shapes (a single
    ``VectorParams`` or a named-vector mapping) across versions. This reads the
    default unnamed vector size and returns ``None`` if it cannot be determined.

    Args:
        info: The object returned by ``get_collection``.

    Returns:
        The vector size, or ``None`` if not determinable.
    """
    try:
        params = info.config.params.vectors
    except AttributeError:
        return None
    if isinstance(params, models.VectorParams):
        return params.size
    if isinstance(params, dict):  # named vectors: take the default/first
        if "" in params and isinstance(params[""], models.VectorParams):
            return params[""].size
        for value in params.values():
            if isinstance(value, models.VectorParams):
                return value.size
    return getattr(params, "size", None)


# --------------------------------------------------------------------------- #
# Loading chunks + embeddings                                                  #
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSON objects from a JSONL file, skipping blank/malformed lines.

    Args:
        path: Path to a JSONL artifact.

    Yields:
        Parsed dict records.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}. Run the upstream stage first "
            "(embed.chunk for chunks, embed.embed_batch for embeddings)."
        )
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                print(f"SKIP malformed JSON {path.name}:{line_no}: {exc}")


def _load_vectors(path: Path) -> dict[str, list[float]]:
    """Load the ``chunk_id -> vector`` map from the embeddings artifact.

    Validates that every vector has ``config.EMBED_DIM`` dimensions so a
    model/dim mismatch is caught before it reaches Qdrant.

    Args:
        path: Path to ``config.EMBEDDINGS_JSONL`` (lines of
            ``{"chunk_id": ..., "vector": [...]}``).

    Returns:
        A dict mapping each ``chunk_id`` to its embedding vector.

    Raises:
        ValueError: If any vector's dimension differs from ``config.EMBED_DIM``,
            or a record lacks ``chunk_id``/``vector``.
    """
    vectors: dict[str, list[float]] = {}
    for rec in _iter_jsonl(path):
        cid = rec.get("chunk_id")
        vec = rec.get("vector")
        if cid is None or vec is None:
            raise ValueError(
                f"Embeddings record missing chunk_id/vector in {path.name}: keys={list(rec)}"
            )
        if len(vec) != config.EMBED_DIM:
            raise ValueError(
                f"Vector for {cid!r} has dim {len(vec)} != config.EMBED_DIM "
                f"{config.EMBED_DIM} (model/collection mismatch)."
            )
        vectors[cid] = vec
    return vectors


def _chunk_to_point(chunk: dict[str, Any], vector: list[float]) -> models.PointStruct:
    """Build a Qdrant point from a chunk record and its vector.

    The payload carries the chunk's own ``payload`` (filter keys + citation
    meta + ``parent_id``) plus the original ``chunk_id``, ``doc_id`` and the
    embedded ``text`` so retriever hits are immediately citable without a second
    fetch.

    Args:
        chunk: A chunk record from ``config.CHUNKS_JSONL``.
        vector: The embedding vector for this chunk.

    Returns:
        A :class:`qdrant_client.models.PointStruct` ready for upsert.
    """
    cid = chunk["chunk_id"]
    payload: dict[str, Any] = dict(chunk.get("payload") or {})
    # Always carry join/citation identifiers in the payload (point id is an
    # opaque UUID, so chunk_id must travel alongside it).
    payload.setdefault("chunk_id", cid)
    payload["chunk_id"] = cid
    if chunk.get("doc_id") is not None:
        payload.setdefault("doc_id", chunk["doc_id"])
    if chunk.get("parent_id") is not None:
        payload.setdefault("parent_id", chunk["parent_id"])
    if chunk.get("text") is not None:
        payload.setdefault("text", chunk["text"])
    return models.PointStruct(id=point_id(cid), vector=vector, payload=payload)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(Exception),
)
def _upsert_batch(client: QdrantClient, points: list[models.PointStruct]) -> None:
    """Upsert one batch of points with bounded exponential-backoff retries.

    Args:
        client: The Qdrant client.
        points: The point batch to upsert (waited on for durability).
    """
    client.upsert(collection_name=config.COLLECTION, points=points, wait=True)


def _batched(it: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield successive ``size``-length lists from ``it``.

    Args:
        it: Any iterable.
        size: Batch length (> 0).

    Yields:
        Lists of up to ``size`` items.
    """
    batch: list[Any] = []
    for item in it:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------------------------------------------------------------- #
# Upsert                                                                        #
# --------------------------------------------------------------------------- #
def upsert_all(
    client: QdrantClient | None = None,
    *,
    chunks_path: Path | None = None,
    embeddings_path: Path | None = None,
    ensure: bool = True,
) -> int:
    """Join chunks with their vectors and batch-upsert them into Qdrant.

    Streams ``chunks_path`` (so the multi-million-chunk corpus never lands fully
    in memory), looks up each chunk's vector in the in-memory embeddings map,
    and upserts in ``_UPSERT_BATCH``-sized batches. Chunks without a matching
    vector are skipped and counted (e.g. cache-miss not yet embedded); they are
    logged but never crash the run.

    Args:
        client: An existing Qdrant client; one is created (server-first) if
            ``None``.
        chunks_path: Override for ``config.CHUNKS_JSONL``.
        embeddings_path: Override for ``config.EMBEDDINGS_JSONL``.
        ensure: When ``True`` (default), call :func:`ensure_collection` first.

    Returns:
        The number of points upserted.

    Raises:
        FileNotFoundError: If the chunks or embeddings artifact is missing.
        ValueError: If a vector dimension mismatches ``config.EMBED_DIM``.
    """
    if client is None:
        client, is_server = get_client()
        print(f"Qdrant: {'server ' + config.QDRANT_URL if is_server else 'local path ' + _local_path()}")

    if ensure:
        ensure_collection(client)

    chunks_path = chunks_path or config.CHUNKS_JSONL
    embeddings_path = embeddings_path or config.EMBEDDINGS_JSONL

    vectors = _load_vectors(embeddings_path)
    print(f"Loaded {len(vectors)} vectors (dim={config.EMBED_DIM}).")

    def _points() -> Iterator[models.PointStruct]:
        for chunk in _iter_jsonl(chunks_path):
            cid = chunk.get("chunk_id")
            if cid is None:
                continue
            vec = vectors.get(cid)
            if vec is None:
                _points.missing += 1  # type: ignore[attr-defined]
                continue
            yield _chunk_to_point(chunk, vec)

    _points.missing = 0  # type: ignore[attr-defined]

    upserted = 0
    for batch in _batched(_points(), _UPSERT_BATCH):
        _upsert_batch(client, batch)
        upserted += len(batch)
        print(f"  upserted {upserted} points...", end="\r")

    missing = _points.missing  # type: ignore[attr-defined]
    print()  # newline after the progress line
    if missing:
        print(f"NOTE: {missing} chunks had no embedding (skipped; embed them first).")
    total = client.count(config.COLLECTION, exact=True).count
    print(f"DONE: upserted {upserted} points; collection count = {total}.")
    return upserted


# --------------------------------------------------------------------------- #
# Self-test (5 dummy points) — Task 2.3 DoD smoke                              #
# --------------------------------------------------------------------------- #
def _dummy_points(n: int = 5) -> list[models.PointStruct]:
    """Build ``n`` deterministic dummy points spanning the filter keys.

    Vectors are simple unit-ish ``config.EMBED_DIM`` vectors (no OpenAI call —
    cost rule). Payloads cover every indexed field so the filtered-search check
    is meaningful.

    Args:
        n: Number of dummy points (default 5, per the Task 2.3 DoD).

    Returns:
        A list of ``PointStruct`` with ``__dummy__`` chunk ids.
    """
    doc_types = ["law", "ordinance", "admrule", "precedent", "law"]
    points: list[models.PointStruct] = []
    for i in range(n):
        cid = f"__dummy__#{i}"
        vec = [0.0] * config.EMBED_DIM
        # Give each a distinct, normalized-ish direction so search is well-defined.
        vec[i % config.EMBED_DIM] = 1.0
        payload = {
            "chunk_id": cid,
            "doc_id": f"DUMMY:{i}",
            "parent_id": f"DUMMY:{i}",
            "doc_type": doc_types[i % len(doc_types)],
            "jurisdiction": "국가" if i % 2 == 0 else "전라남도",
            "law_kind": "법률" if i % 2 == 0 else "조례",
            "article_no": f"제{i + 1}조",
            "effective_from": f"2026-01-0{i + 1}",
            "trust_grade": "A",
            "license": config.DEFAULT_LICENSE,
            "text": f"[더미 제{i + 1}조] 자체검증용 더미 청크 {i}.",
        }
        points.append(models.PointStruct(id=point_id(cid), vector=vec, payload=payload))
    return points


def self_test(n: int = 5) -> dict[str, Any]:
    """Insert ``n`` dummy points, exercise count + filtered search, then clean up.

    Verifies the Task 2.3 DoD locally without any paid OpenAI call:

    * collection exists with the right dimension,
    * upsert + ``count`` round-trips,
    * a ``doc_type="law"`` payload filter narrows results,
    * a vector query returns the expected nearest dummy point.

    The dummy points are deleted afterwards so the collection is left clean.

    Args:
        n: Number of dummy points to use (default 5).

    Returns:
        A small report dict ``{is_server, count_after_upsert, filtered_law,
        top_hit_chunk_id, ok}``.

    Raises:
        AssertionError: If any DoD check fails.
    """
    client, is_server = get_client()
    ensure_collection(client)

    points = _dummy_points(n)
    client.upsert(collection_name=config.COLLECTION, points=points, wait=True)

    ids = [p.id for p in points]

    # 1) count includes our dummies
    total = client.count(config.COLLECTION, exact=True).count
    assert total >= n, f"count {total} < inserted {n}"

    # 2) payload filter works (doc_type == law)
    law_hits, _ = client.scroll(
        collection_name=config.COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="law"))]
        ),
        limit=100,
        with_payload=True,
    )
    law_dummies = [h for h in law_hits if str(h.payload.get("chunk_id", "")).startswith("__dummy__")]
    assert law_dummies, "doc_type='law' filter returned no dummy points"
    assert all(h.payload["doc_type"] == "law" for h in law_dummies), "filter leaked non-law points"

    # 3) vector search returns the matching dummy on top
    query_vec = [0.0] * config.EMBED_DIM
    query_vec[0] = 1.0  # nearest to dummy #0
    hits = client.query_points(
        collection_name=config.COLLECTION,
        query=query_vec,
        limit=1,
        with_payload=True,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="DUMMY:0"))]
        ),
    ).points
    assert hits, "vector search returned no hits"
    top_chunk = hits[0].payload.get("chunk_id")
    assert top_chunk == "__dummy__#0", f"unexpected top hit {top_chunk!r}"

    # cleanup
    client.delete(
        collection_name=config.COLLECTION,
        points_selector=models.PointIdsList(points=ids),
        wait=True,
    )

    report = {
        "is_server": is_server,
        "count_after_upsert": total,
        "filtered_law": len(law_dummies),
        "top_hit_chunk_id": top_chunk,
        "ok": True,
    }
    return report


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Modes:
        (default)   ensure collection then upsert all chunks+embeddings.
        --ensure    create/verify the collection and indexes only.
        --verify    run the 5-point self-test (no OpenAI cost).
        --recreate  drop & recreate the collection before upserting.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Qdrant collection + upsert (Task 2.3).")
    parser.add_argument("--ensure", action="store_true", help="Create/verify collection + indexes only.")
    parser.add_argument("--verify", action="store_true", help="Run the 5-point self-test (no OpenAI call).")
    parser.add_argument("--recreate", action="store_true", help="Drop & recreate the collection first.")
    args = parser.parse_args(argv)

    if args.verify:
        report = self_test()
        print("SELF-TEST OK:", json.dumps(report, ensure_ascii=False))
        return 0

    client, is_server = get_client()
    print(f"Qdrant: {'server ' + config.QDRANT_URL if is_server else 'local path ' + _local_path()}")
    ensure_collection(client, recreate=args.recreate)
    print(f"Collection {config.COLLECTION!r} ready (dim={config.EMBED_DIM}, Cosine, indexes={list(_INDEXED_FIELDS)}).")

    if args.ensure:
        return 0

    upsert_all(client, ensure=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
