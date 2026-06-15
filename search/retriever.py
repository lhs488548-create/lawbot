"""Dense retriever over the Qdrant ``lawbot`` collection (Playbook 08 Task 3.1; 09 §E-1).

Pipeline (09 §E-1): metadata pre-filter (``doc_type`` / ``jurisdiction`` /
``law_kind`` / ``parent_id``, **plus an ``as_of_date`` point-in-time current-law
filter** on ``effective_from``) → embed the query with the *same* model used at
index time (:data:`config.EMBED_MODEL`) → dense Qdrant similarity search top-K →
return clean :class:`Hit` dataclasses. A child hit can then be promoted to its
**parent** full text via :func:`get_parent` (parent-promotion for ``ask`` /
``source-pack``).

Design contract (see ``_BUILD_CONTRACT.md`` §(d) → Retriever, 09 alignment):

* ``embed_query(query) -> list[float]`` — one vector of length
  :data:`config.EMBED_DIM` (same space as the indexed documents).
* ``search(query, k=config.DEFAULT_TOP_K, flt=None, as_of_date=None) -> list[Hit]``
  where ``flt`` is a flat ``{payload_key: value}`` mapping AND-ed into a Qdrant
  filter, and ``as_of_date`` (ISO ``YYYY-MM-DD``) restricts results to rows whose
  ``effective_from <= as_of_date`` (point-in-time current-law lookup).
* ``get_parent(parent_id) -> dict | None`` — load a parent record (full text)
  from :data:`config.PARENTS_JSONL` for parent-promotion.
* Each :class:`Hit` exposes ``.id``, ``.score``, ``.payload`` (carrying ``text``
  and the standard payload keys incl. ``parent_id``) and convenience accessors.
* OpenAI and Qdrant network calls are wrapped with ``tenacity`` retry.

This module owns only retrieval; it never calls a generation model. The point id
and payload layout it consumes are produced by ``embed/upsert_qdrant.py`` (point
id = deterministic UUID5 of ``chunk_id``; ``chunk_id`` and ``parent_id`` kept in
payload). It is resilient to payloads written before the 09 revision: when
``parent_id`` is absent it is derived from the ``chunk_id`` via
:func:`ingest.schema.parent_id_of`.

Run a free, offline self-check against an in-memory Qdrant collection (no OpenAI,
no Docker, no cloud)::

    cd /home/user1/lawbot && .venv/bin/python -m search.retriever --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Final, Mapping

from openai import OpenAI
from qdrant_client import QdrantClient, models
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config
from ingest.schema import parent_id_of

# --------------------------------------------------------------------------- #
# Module-level clients                                                         #
# --------------------------------------------------------------------------- #
# Both clients are cheap to hold open and are safe to reuse across requests
# (the OpenAI SDK and qdrant-client are thread-safe for read traffic). They are
# created lazily so that importing this module never opens a socket or requires
# a reachable Qdrant — important for unit tests and for the API process startup
# order. Use the accessor functions below rather than touching these directly.
_openai_client: OpenAI | None = None
_qdrant_client: QdrantClient | None = None

# Payload keys that callers are allowed to filter on with ``flt``. These mirror
# the KEYWORD payload indexes created by ``embed/upsert_qdrant.py``; filtering on
# an unindexed key would silently degrade to a slow scan, so we reject unknowns.
# ``parent_id`` is included so callers can scope a search to one document's
# children (09 §B-1 parent/child). ``as_of_date`` is NOT a member here — it is a
# dedicated range parameter on ``effective_from`` (see :func:`search`).
ALLOWED_FILTER_KEYS: Final[frozenset[str]] = frozenset(
    {"doc_type", "jurisdiction", "law_kind", "effective_from", "parent_id"}
)

# ISO calendar date (YYYY-MM-DD) — the accepted ``as_of_date`` / ``effective_from``
# shape. We keep validation strict (no times, no offsets) so the point-in-time
# filter is unambiguous and matches the stored ``effective_from`` format.
_ISO_DATE_RE: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Exceptions worth retrying. We keep this broad (any Exception) because both the
# OpenAI SDK and qdrant-client raise their own connection/timeout error types
# and we want transient network blips to be retried regardless of which layer
# raised them. Programming errors surface after the attempts are exhausted.
_RETRYABLE = retry_if_exception_type(Exception)

# Shared retry policy: 3 attempts, exponential backoff capped at ~6s. Kept short
# so a request-path failure fails fast rather than hanging an API worker.
_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=6),
    retry=_RETRYABLE,
)


def get_openai_client() -> OpenAI:
    """Return the lazily-initialized shared OpenAI client.

    The API key is read from :mod:`config` (which loads ``.env``); it is never
    passed around in logs or printed here.
    """
    global _openai_client
    if _openai_client is None:
        # Pass the key explicitly from config so behavior does not depend on the
        # SDK's own env lookup, while still never printing it.
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def get_qdrant_client() -> QdrantClient:
    """Return the lazily-initialized shared Qdrant client.

    Resolution follows ``embed.upsert_qdrant.get_client()``: a live server at
    :data:`config.QDRANT_URL` is preferred, otherwise it falls back to the
    embedded on-disk Qdrant (``artifacts/qdrant_local``) so the demo/local
    index built without Docker is served. Set ``LAWBOT_QDRANT_REQUIRE_SERVER=1``
    to forbid the local fallback in production.
    """
    global _qdrant_client
    if _qdrant_client is None:
        from embed import upsert_qdrant

        _qdrant_client, _ = upsert_qdrant.get_client()
    return _qdrant_client


def set_clients(
    *,
    openai_client: OpenAI | None = None,
    qdrant_client: QdrantClient | None = None,
) -> None:
    """Inject pre-built clients (tests, or an in-memory Qdrant for self-checks).

    Args:
        openai_client: Replacement OpenAI client, or ``None`` to leave as-is.
        qdrant_client: Replacement Qdrant client, or ``None`` to leave as-is.
    """
    global _openai_client, _qdrant_client
    if openai_client is not None:
        _openai_client = openai_client
    if qdrant_client is not None:
        _qdrant_client = qdrant_client


# --------------------------------------------------------------------------- #
# Result type                                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Hit:
    """A single retrieved child chunk.

    Attributes:
        id: The Qdrant point id (deterministic UUID5 of the chunk id). Stable
            across re-ingests, so it is a valid citation ``source_id``.
        score: Cosine similarity in ``[-1, 1]`` (higher is more relevant).
        payload: The stored payload. Always contains ``text`` and the standard
            filter keys (``doc_type``, ``jurisdiction``, ``law_kind``,
            ``effective_from``) plus ``title``, ``article_no``, ``source_url``,
            ``trust_grade``, ``chunk_id`` and ``parent_id`` when present.
    """

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)

    # -- Convenience accessors (read-only views into the payload) ----------- #
    @property
    def text(self) -> str:
        """The embedded chunk text (header-prefixed body), or ``""``."""
        return str(self.payload.get("text", ""))

    @property
    def chunk_id(self) -> str | None:
        """The original ``chunk_id`` (for joins/debugging), if stored."""
        cid = self.payload.get("chunk_id")
        return None if cid is None else str(cid)

    @property
    def parent_id(self) -> str | None:
        """The owning document (parent) id, for parent-promotion (09 §B-1).

        Prefers the explicit ``parent_id`` payload key; falls back to deriving it
        from the ``chunk_id`` (``{parent_id}#{article_no}#{part_idx}``) so the
        accessor still works for payloads written before the 09 revision added
        the key. Returns ``None`` only when neither is available.
        """
        pid = self.payload.get("parent_id")
        if pid:
            return str(pid)
        cid = self.payload.get("chunk_id")
        return parent_id_of(str(cid)) if cid else None

    @property
    def doc_id(self) -> str | None:
        """The owning document id (alias of :attr:`parent_id` for child chunks)."""
        return self.parent_id

    @property
    def doc_type(self) -> str | None:
        """The document kind (``law``/``ordinance``/``admrule``/``precedent``)."""
        dt = self.payload.get("doc_type")
        return None if dt is None else str(dt)

    @property
    def title(self) -> str | None:
        """Law name / case name, if stored."""
        t = self.payload.get("title")
        return None if t is None else str(t)

    @property
    def article_no(self) -> str | None:
        """Article ("제4조") or precedent section name, if stored."""
        a = self.payload.get("article_no")
        return None if a is None else str(a)

    @property
    def effective_from(self) -> str | None:
        """Enforcement/decision date (ISO-ish string), if stored."""
        e = self.payload.get("effective_from")
        return None if e in (None, "") else str(e)

    @property
    def source_url(self) -> str | None:
        """Canonical source URL, if stored."""
        u = self.payload.get("source_url")
        return None if u is None else str(u)

    @property
    def trust_grade(self) -> str:
        """Trust grade: ``"A"`` (text present) or ``"B"`` (metadata only)."""
        return str(self.payload.get("trust_grade", "A"))

    def location(self) -> str:
        """Human-readable in-document location for citations.

        Returns the article/section label when present, else an empty string.
        """
        return self.article_no or ""


# --------------------------------------------------------------------------- #
# Filter construction                                                         #
# --------------------------------------------------------------------------- #
def _validate_iso_date(value: str, *, param: str) -> str:
    """Validate and normalize an ISO ``YYYY-MM-DD`` date string.

    Args:
        value: The candidate date string.
        param: Parameter name, used only in error messages.

    Returns:
        The stripped date string.

    Raises:
        ValueError: If ``value`` is not a ``YYYY-MM-DD`` calendar date.
    """
    s = (value or "").strip()
    if not _ISO_DATE_RE.match(s):
        raise ValueError(
            f"{param} must be an ISO calendar date 'YYYY-MM-DD'; got {value!r}."
        )
    return s


def build_filter(
    flt: Mapping[str, Any] | None,
    as_of_date: str | None = None,
) -> models.Filter | None:
    """Translate a flat ``{key: value}`` mapping (+ ``as_of_date``) into a Qdrant ``Filter``.

    All conditions are AND-ed together (Qdrant ``must``). ``flt`` values may be a
    single scalar (exact match) or a list/tuple/set of scalars (match-any),
    enabling e.g. ``{"law_kind": ["법률", "시행령"]}``. When ``as_of_date`` is
    given, an additional ``effective_from <= as_of_date`` range condition is
    AND-ed in (point-in-time current-law lookup, 09 §E-1): rows whose
    ``effective_from`` is missing/blank or later than ``as_of_date`` are excluded.

    Args:
        flt: The requested metadata filter, or ``None``/empty for none.
        as_of_date: Optional ISO ``YYYY-MM-DD`` "as of" date.

    Returns:
        A :class:`qdrant_client.models.Filter`, or ``None`` when neither ``flt``
        nor ``as_of_date`` is supplied.

    Raises:
        ValueError: If a key is not in :data:`ALLOWED_FILTER_KEYS`, a value is
            empty/``None``, or ``as_of_date`` is not a valid ISO date.
    """
    conditions: list[models.FieldCondition] = []

    for key, value in (flt or {}).items():
        if key not in ALLOWED_FILTER_KEYS:
            raise ValueError(
                f"Unsupported filter key {key!r}. "
                f"Allowed (indexed) keys: {sorted(ALLOWED_FILTER_KEYS)}."
            )
        if value is None:
            raise ValueError(f"Filter value for {key!r} must not be None.")

        if isinstance(value, (list, tuple, set)):
            members = [str(v) for v in value if v is not None and str(v) != ""]
            if not members:
                raise ValueError(f"Filter value list for {key!r} is empty.")
            match: models.Match = models.MatchAny(any=members)
        else:
            sval = str(value)
            if sval == "":
                raise ValueError(f"Filter value for {key!r} must not be empty.")
            match = models.MatchValue(value=sval)

        conditions.append(models.FieldCondition(key=key, match=match))

    if as_of_date is not None:
        iso = _validate_iso_date(as_of_date, param="as_of_date")
        # DatetimeRange parses the ISO string and keeps rows with
        # effective_from <= as_of_date. Rows with a missing/blank/unparseable
        # effective_from are excluded — the safe default for a current-law query
        # (we do not assume an undated row is in force at an arbitrary instant).
        conditions.append(
            models.FieldCondition(
                key="effective_from",
                range=models.DatetimeRange(lte=iso),
            )
        )

    return models.Filter(must=conditions) if conditions else None


# --------------------------------------------------------------------------- #
# Query-embedding LRU cache (09 §E-4.2 — "동일/반복 질문은 임베딩 재호출 0")      #
# --------------------------------------------------------------------------- #
# A small, thread-safe LRU keyed by SHA-256 of the (model, normalized-text) pair
# so repeated/identical queries skip the OpenAI round-trip entirely. The key
# folds in EMBED_MODEL so a model change cannot serve a stale-space vector. The
# cache stores tuples (immutable) and hands out fresh lists to callers so a
# caller mutating the result can never corrupt a cached entry. Bounded size keeps
# memory flat under unbounded distinct queries (real-time serving).
_QCACHE_MAX: Final[int] = 2048
_qcache: "OrderedDict[str, tuple[float, ...]]" = OrderedDict()
_qcache_lock = threading.Lock()


def _qcache_key(text: str) -> str:
    """Stable cache key for a normalized query text under the current model."""
    h = hashlib.sha256()
    h.update(config.EMBED_MODEL.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def reset_query_cache() -> None:
    """Clear the query-embedding LRU (tests / after a model switch)."""
    with _qcache_lock:
        _qcache.clear()


@_retry
def _embed_query_uncached(text: str) -> list[float]:
    """Call the embedding model for an already-normalized, non-empty ``text``."""
    response = get_openai_client().embeddings.create(
        model=config.EMBED_MODEL,
        input=text,
    )
    vector = response.data[0].embedding
    if len(vector) != config.EMBED_DIM:
        raise ValueError(
            f"Embedding dimensionality {len(vector)} != config.EMBED_DIM "
            f"{config.EMBED_DIM}. Check EMBED_MODEL/EMBED_DIM and the Qdrant "
            f"collection vector size are consistent."
        )
    return list(vector)


def embed_query(query: str) -> list[float]:
    """Embed a query string with the index-time embedding model (LRU-cached).

    Uses :data:`config.EMBED_MODEL` so that query and document vectors live in
    the same space. Identical queries are served from a bounded, thread-safe LRU
    (09 §E-4.2) so repeated questions cost no OpenAI round-trip; misses are
    wrapped with ``tenacity`` retry for transient network errors.

    Args:
        query: The user's question. Must be non-empty after stripping.

    Returns:
        The embedding vector; its length is validated to equal
        :data:`config.EMBED_DIM`. A fresh list is returned each call so callers
        may mutate it without affecting the cache.

    Raises:
        ValueError: If ``query`` is blank, or the returned vector has an
            unexpected dimensionality (model/collection mismatch).
    """
    text = (query or "").strip()
    if not text:
        raise ValueError("query must be a non-empty string")

    key = _qcache_key(text)
    with _qcache_lock:
        cached = _qcache.get(key)
        if cached is not None:
            _qcache.move_to_end(key)  # mark most-recently-used
            return list(cached)

    vector = _embed_query_uncached(text)

    with _qcache_lock:
        _qcache[key] = tuple(vector)
        _qcache.move_to_end(key)
        while len(_qcache) > _QCACHE_MAX:
            _qcache.popitem(last=False)  # evict least-recently-used
    return list(vector)


# --------------------------------------------------------------------------- #
# Search                                                                       #
# --------------------------------------------------------------------------- #
@_retry
def _qdrant_query(
    vector: list[float],
    k: int,
    qfilter: models.Filter | None,
) -> list[models.ScoredPoint]:
    """Run the Qdrant similarity query (retried on transient errors)."""
    response = get_qdrant_client().query_points(
        collection_name=config.COLLECTION,
        query=vector,
        query_filter=qfilter,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )
    return response.points


def _post_filter_as_of(
    points: list[models.ScoredPoint], as_of_date: str
) -> list[models.ScoredPoint]:
    """Client-side fallback for the ``effective_from <= as_of_date`` filter.

    Used only when the server rejects a server-side ``DatetimeRange`` (e.g. the
    ``effective_from`` field is indexed as KEYWORD rather than DATETIME on a
    given deployment). Because ``effective_from`` is stored as an ISO
    ``YYYY-MM-DD`` string, lexical ``<=`` equals chronological ``<=``. Rows with
    a missing/blank/non-ISO ``effective_from`` are excluded, matching the
    server-side semantics.
    """
    kept: list[models.ScoredPoint] = []
    for p in points:
        ef = str((p.payload or {}).get("effective_from") or "").strip()
        if _ISO_DATE_RE.match(ef) and ef <= as_of_date:
            kept.append(p)
    return kept


def search(
    query: str,
    k: int = config.DEFAULT_TOP_K,
    flt: Mapping[str, Any] | None = None,
    as_of_date: str | None = None,
) -> list[Hit]:
    """Retrieve the top-``k`` child chunks most relevant to ``query``.

    Args:
        query: The user's question (non-empty).
        k: Maximum number of hits to return; coerced to ``>= 1``.
        flt: Optional flat metadata pre-filter, AND-ed into the search. Keys
            must be in :data:`ALLOWED_FILTER_KEYS`; values are scalars (exact)
            or sequences (match-any). Examples: ``{"doc_type": "law"}``,
            ``{"jurisdiction": "전라남도"}``,
            ``{"doc_type": "law", "law_kind": ["법률", "시행령"]}``,
            ``{"parent_id": "LAW:014565:법률"}``.
        as_of_date: Optional ISO ``YYYY-MM-DD``. When given, only rows whose
            ``effective_from <= as_of_date`` are returned (point-in-time
            current-law lookup, 09 §E-1).

    Returns:
        A list of :class:`Hit`, ordered by descending similarity score. Empty
        if nothing matches the (optional) filter / as-of constraint.

    Raises:
        ValueError: For a blank query, an invalid filter (see
            :func:`build_filter`), or a malformed ``as_of_date``.
    """
    if not (query or "").strip():
        raise ValueError("query must be a non-empty string")
    k = max(1, int(k))

    qfilter = build_filter(flt, as_of_date=as_of_date)
    vector = embed_query(query)

    try:
        points = _qdrant_query(vector, k, qfilter)
    except Exception:
        # If the server rejected the as_of_date DatetimeRange (e.g. the
        # effective_from field is not indexed as DATETIME on this deployment),
        # retry without the range and apply the date cut client-side. Over-fetch
        # so the post-filter can still return up to k results. Any other failure
        # (after tenacity retries) re-raises as before.
        if as_of_date is None:
            raise
        base_filter = build_filter(flt, as_of_date=None)
        points = _qdrant_query(vector, max(k * 5, k), base_filter)
        points = _post_filter_as_of(points, as_of_date)[:k]

    return [
        Hit(
            id=str(p.id),
            score=float(p.score) if p.score is not None else 0.0,
            payload=dict(p.payload or {}),
        )
        for p in points
    ]


# --------------------------------------------------------------------------- #
# Parent promotion (09 §B-1 / §E-1)                                            #
# --------------------------------------------------------------------------- #
# Lazy, thread-safe in-memory index of parent records keyed by ``parent_id``.
# Built once from ``config.PARENTS_JSONL`` on first use and reused thereafter
# (the file is the build-time materialization of the parent = whole-law /
# whole-precedent texts). ``None`` means "not yet loaded"; an empty dict is a
# valid loaded state (e.g. the artifact does not exist yet).
_parents_index: dict[str, dict[str, Any]] | None = None
_parents_lock = threading.Lock()


def _load_parents_index() -> dict[str, dict[str, Any]]:
    """Build (once) and return the ``parent_id -> parent record`` index.

    Streams :data:`config.PARENTS_JSONL` once and caches the result. Missing
    file ⇒ empty index (parent-promotion simply returns ``None`` until the
    build stage materializes parents). Malformed lines are skipped, never fatal.
    """
    global _parents_index
    if _parents_index is not None:
        return _parents_index
    with _parents_lock:
        if _parents_index is not None:  # re-check under lock
            return _parents_index
        index: dict[str, dict[str, Any]] = {}
        path = config.PARENTS_JSONL
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Never let one bad line break parent-promotion.
                        continue
                    pid = rec.get("parent_id") or rec.get("doc_id")
                    if pid:
                        index[str(pid)] = rec
        _parents_index = index
        return _parents_index


def reset_parents_cache() -> None:
    """Drop the cached parents index (tests / after a rebuild of ``PARENTS_JSONL``)."""
    global _parents_index
    with _parents_lock:
        _parents_index = None


def get_parent(parent_id: str) -> dict[str, Any] | None:
    """Return the parent (full-text) record for ``parent_id``, or ``None``.

    Powers parent-promotion (09 §B-1 / §E-1): a child :class:`Hit` is promoted to
    its parent's full text for ``ask`` / ``source-pack``. The lookup reads
    :data:`config.PARENTS_JSONL` (built by the chunking stage) and is cached in
    memory after the first call.

    Args:
        parent_id: A document/parent id, e.g. ``"LAW:014565:법률"`` or
            ``"PREC:424370"`` — typically ``hit.parent_id``.

    Returns:
        The parent record dict (``parent_id``, ``doc_type``, ``title``,
        ``full_text``, common meta …), or ``None`` if unknown / not yet built.
    """
    pid = (parent_id or "").strip()
    if not pid:
        return None
    return _load_parents_index().get(pid)


# --------------------------------------------------------------------------- #
# Self-test (offline): exercise filter + search + parent wiring on dummy data #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Validate filtering, as_of_date, ranking and parent-promotion offline.

    Embeddings are *faked* (no OpenAI call) so the test is free and offline; the
    Qdrant search path, payload round-trip, metadata + as_of_date filtering,
    :class:`Hit` construction, and :func:`get_parent` are exercised end to end.
    Returns a process exit code.
    """
    import tempfile
    import uuid
    from pathlib import Path

    dim = 4  # tiny vectors for the dummy collection
    # search() always queries config.COLLECTION, so the dummy collection must
    # use that same name.
    collection = config.COLLECTION

    # Spin up an in-memory Qdrant (no Docker, no network).
    qc = QdrantClient(location=":memory:")
    qc.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=dim, distance=models.Distance.COSINE
        ),
    )
    # Payload indexes are a no-op on the in-memory backend (it filters by scan);
    # creating them only emits a warning, so skip them here. The real KEYWORD
    # indexes are created by embed/upsert_qdrant.py on the server.

    # Four documents along orthogonal-ish axes so we can predict ranking, with a
    # spread of effective_from dates to exercise the as_of_date filter. One law
    # row omits parent_id from its payload so we also test the chunk_id-derived
    # parent_id fallback.
    fixtures = [
        {
            "vec": [1.0, 0.0, 0.0, 0.0],
            "chunk_id": "LAW:000001:법률#제4조#0",
            "payload": {
                "text": "[민법 제4조 성년] 사람은 19세로 성년에 이르게 된다.",
                "doc_type": "law",
                "title": "민법",
                "jurisdiction": "국가",
                "law_kind": "법률",
                "article_no": "제4조",
                "effective_from": "2013-07-01",
                "source_url": "https://law.go.kr/민법",
                "trust_grade": "A",
                # parent_id intentionally omitted -> tests fallback derivation.
            },
        },
        {
            "vec": [0.0, 1.0, 0.0, 0.0],
            "chunk_id": "ORD:전라남도:2200001#제2조#0",
            "payload": {
                "text": "[전라남도 ○○ 조례 제2조] 정의 규정.",
                "doc_type": "ordinance",
                "title": "전라남도 ○○ 조례",
                "jurisdiction": "전라남도",
                "law_kind": "조례",
                "article_no": "제2조",
                "effective_from": "2022-01-01",
                "source_url": "https://law.go.kr/ord",
                "trust_grade": "A",
                "parent_id": "ORD:전라남도:2200001",
            },
        },
        {
            "vec": [0.0, 0.0, 1.0, 0.0],
            "chunk_id": "PREC:424370#판결요지#0",
            "payload": {
                "text": "[대법원 2020다12345 판결요지] 손해배상 책임 인정.",
                "doc_type": "precedent",
                "title": "손해배상청구",
                "jurisdiction": "대법원",
                "law_kind": "민사",
                "article_no": "판결요지",
                "effective_from": "2020-05-14",
                "source_url": "https://law.go.kr/prec",
                "trust_grade": "A",
                "parent_id": "PREC:424370",
            },
        },
        {
            # A "future" law (effective after our as_of_date) on the same axis as
            # the민법 row but slightly off, so an as_of_date cut must drop it.
            "vec": [0.95, 0.05, 0.0, 0.0],
            "chunk_id": "LAW:000099:법률#제1조#0",
            "payload": {
                "text": "[미래법 제1조] 2030년 시행 예정 조문.",
                "doc_type": "law",
                "title": "미래법",
                "jurisdiction": "국가",
                "law_kind": "법률",
                "article_no": "제1조",
                "effective_from": "2030-01-01",
                "source_url": "https://law.go.kr/future",
                "trust_grade": "A",
                "parent_id": "LAW:000099:법률",
            },
        },
    ]
    qc.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, fx["chunk_id"])),
                vector=fx["vec"],
                payload={"chunk_id": fx["chunk_id"], **fx["payload"]},
            )
            for fx in fixtures
        ],
    )

    # Point the module at this in-memory client and stub embedding so search()
    # runs without any OpenAI call. The stub returns the 민법-axis vector, so an
    # unfiltered search must rank the 민법 chunk first.
    set_clients(qdrant_client=qc)
    global embed_query  # noqa: PLW0603 - intentional monkeypatch for selftest
    _real_embed = embed_query

    def _fake_embed(_query: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    embed_query = _fake_embed  # type: ignore[assignment]

    # A throwaway PARENTS_JSONL so get_parent has something to load.
    _real_parents = config.PARENTS_JSONL
    tmpdir = Path(tempfile.mkdtemp(prefix="lawbot_selftest_"))
    parents_path = tmpdir / "parents.jsonl"
    parents_path.write_text(
        json.dumps(
            {
                "parent_id": "LAW:000001:법률",
                "doc_type": "law",
                "title": "민법",
                "law_kind": "법률",
                "jurisdiction": "국가",
                "effective_from": "2013-07-01",
                "source_url": "https://law.go.kr/민법",
                "license": "공공저작물",
                "trust_grade": "A",
                "full_text": "[민법] 제4조(성년) 사람은 19세로 성년에 이르게 된다. …",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config.PARENTS_JSONL = parents_path  # type: ignore[misc]
    reset_parents_cache()

    try:
        failures: list[str] = []

        # 1) Unfiltered: 민법 chunk ranks first; Hit accessors populated; the
        #    parent_id fallback (payload key absent) derives the doc_id.
        hits = search("성년 나이", k=4)
        if not hits:
            failures.append("unfiltered search returned no hits")
        else:
            top = hits[0]
            if top.doc_type != "law":
                failures.append(f"top hit doc_type {top.doc_type!r} != 'law'")
            if not top.text:
                failures.append("top hit has empty text")
            if top.title != "민법":
                failures.append(f"top hit title {top.title!r} != '민법'")
            if top.location() != "제4조":
                failures.append(f"top hit location {top.location()!r} != '제4조'")
            if top.parent_id != "LAW:000001:법률":
                failures.append(
                    f"top hit parent_id {top.parent_id!r} != 'LAW:000001:법률' "
                    f"(fallback derivation failed)"
                )
            if not (-1.0001 <= top.score <= 1.0001):
                failures.append(f"score out of range: {top.score}")

        # 2) doc_type filter: only ordinance comes back.
        ord_hits = search("아무 질의", k=4, flt={"doc_type": "ordinance"})
        if [h.doc_type for h in ord_hits] != ["ordinance"]:
            failures.append(
                f"doc_type=ordinance filter returned {[h.doc_type for h in ord_hits]}"
            )

        # 3) jurisdiction filter that matches nothing -> empty list.
        none_hits = search("아무 질의", k=4, flt={"jurisdiction": "서울특별시"})
        if none_hits:
            failures.append(f"non-matching jurisdiction returned {len(none_hits)} hits")

        # 4) match-any filter spanning two doc types.
        multi = search(
            "아무 질의", k=4, flt={"doc_type": ["ordinance", "precedent"]}
        )
        if {h.doc_type for h in multi} != {"ordinance", "precedent"}:
            failures.append(
                f"match-any doc_type returned {[h.doc_type for h in multi]}"
            )

        # 5) parent_id filter scopes to one document's children.
        scoped = search("아무 질의", k=4, flt={"parent_id": "PREC:424370"})
        if [h.parent_id for h in scoped] != ["PREC:424370"]:
            failures.append(
                f"parent_id filter returned {[h.parent_id for h in scoped]}"
            )

        # 6) as_of_date: 2025-01-01 keeps 민법(2013) but DROPS 미래법(2030).
        as_of = search("성년 나이", k=4, as_of_date="2025-01-01")
        titles = {h.title for h in as_of}
        if "미래법" in titles:
            failures.append("as_of_date 2025 wrongly kept future law (2030)")
        if "민법" not in titles:
            failures.append("as_of_date 2025 wrongly dropped 민법 (2013)")

        # 7) as_of_date before everything -> empty.
        empty_asof = search("성년 나이", k=4, as_of_date="2000-01-01")
        if empty_asof:
            failures.append(
                f"as_of_date 2000 should drop all rows, got {len(empty_asof)}"
            )

        # 8) as_of_date combined with a metadata filter.
        combo = search(
            "성년 나이", k=4, flt={"doc_type": "law"}, as_of_date="2025-01-01"
        )
        if {h.title for h in combo} != {"민법"}:
            failures.append(
                f"doc_type=law + as_of_date 2025 returned {[h.title for h in combo]}"
            )

        # 9) get_parent promotes a child hit to its parent full text.
        parent = get_parent("LAW:000001:법률")
        if parent is None:
            failures.append("get_parent('LAW:000001:법률') returned None")
        elif "full_text" not in parent or "민법" not in parent.get("title", ""):
            failures.append(f"get_parent returned unexpected record: {parent!r}")
        if get_parent("LAW:does-not-exist") is not None:
            failures.append("get_parent for unknown id should be None")

        # 10) invalid filter key is rejected.
        try:
            build_filter({"unknown_key": "x"})
        except ValueError:
            pass
        else:
            failures.append("invalid filter key was not rejected")

        # 11) malformed as_of_date is rejected.
        try:
            search("질의", as_of_date="2025/01/01")
        except ValueError:
            pass
        else:
            failures.append("malformed as_of_date was not rejected")

        # 12) blank query is rejected.
        try:
            search("   ")
        except ValueError:
            pass
        else:
            failures.append("blank query was not rejected")

        if failures:
            print("SELFTEST FAILED:")
            for f in failures:
                print("  -", f)
            return 1
        print(
            "SELFTEST PASSED: 12 checks (ranking, metadata/parent_id filters, "
            "match-any, as_of_date point-in-time, parent-promotion, validation)."
        )
        return 0
    finally:
        embed_query = _real_embed  # type: ignore[assignment]
        config.PARENTS_JSONL = _real_parents  # type: ignore[misc]
        reset_parents_cache()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search.retriever",
        description="Dense retriever over the Qdrant lawbot collection.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run an offline self-check against an in-memory Qdrant collection.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="If given (and not --selftest), embed+search the live collection.",
    )
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument(
        "--doc-type",
        help="Optional doc_type filter (law|ordinance|admrule|precedent).",
    )
    parser.add_argument(
        "--as-of-date",
        help="Optional ISO YYYY-MM-DD point-in-time filter (effective_from <= date).",
    )
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.query:
        print("Provide a QUERY or use --selftest.", file=sys.stderr)
        return 2
    flt = {"doc_type": args.doc_type} if args.doc_type else None
    for i, hit in enumerate(
        search(args.query, k=args.k, flt=flt, as_of_date=args.as_of_date), start=1
    ):
        print(
            f"[{i}] score={hit.score:.4f} {hit.doc_type} "
            f"{hit.title} {hit.article_no} "
            f"(eff {hit.effective_from or '-'}) parent={hit.parent_id} id={hit.id}"
        )
        print(f"    {hit.text[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
