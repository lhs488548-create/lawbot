"""Dense retriever over the **FAISS** ``의료관련`` index (Playbook 08 Task 3.1; 09 §E-1).

Pipeline (09 §E-1, FAISS build): embed the query with the *same* model used at
index time (:data:`config.EMBED_MODEL`, ``dimensions=512``) → **L2-normalize** →
FAISS ``IndexFlatIP`` brute-force top-``(k*over)`` (inner product == cosine on
normalized vectors) → Python **post-filter** (metadata ``doc_type`` /
``jurisdiction`` / ``law_kind`` / ``effective_from`` / ``parent_id`` **plus an
``as_of_date`` point-in-time current-law filter** on ``effective_from``) → keep
the top-``k`` → return clean :class:`Hit` dataclasses. A child hit can then be
promoted to its **parent** full text via :func:`get_parent` (parent-promotion for
``ask`` / ``source-pack``).

This module is the **only** retrieval surface upstream code (``search/{rag,verify,
ad_review,source_pack,statutes}``, the API) calls. Its public shape is frozen by
``docs/_FAISS_BUILD_CONTRACT.md`` §3 and is preserved verbatim across the
Qdrant→FAISS switch:

* ``embed_query(query) -> list[float]`` — one **L2-normalized** vector of length
  :data:`config.EMBED_DIM` (same space as the indexed documents).
* ``search(query, k=config.DEFAULT_TOP_K, flt=None, as_of_date=None) -> list[Hit]``
  where ``flt`` is a flat ``{payload_key: value}`` mapping AND-ed into a Python
  post-filter, and ``as_of_date`` (ISO ``YYYY-MM-DD``) restricts results to rows
  whose ``effective_from <= as_of_date`` (point-in-time current-law lookup).
* ``get_parent(parent_id) -> dict | None`` — load a parent record (full text)
  from :data:`config.PARENTS_JSONL` for parent-promotion (unchanged).
* Each :class:`Hit` exposes ``.id``, ``.score``, ``.payload`` (carrying ``text``
  and the standard payload keys incl. ``parent_id``) and convenience accessors.
* OpenAI embedding calls reuse :mod:`embed.embed_client` (``dimensions=512``,
  ``tenacity`` retry, content-hash cache); the FAISS index/metadata are produced
  by :mod:`embed.faiss_index` from the canonical ``chunks_with_vectors.jsonl``
  (``embed.faiss_index.build_index``); ``load_index()`` is consumed here.

This module owns only retrieval; it never calls a generation model. The vector
store has been migrated from Qdrant to a self-contained FAISS index
(``IndexFlatIP`` over L2-normalized 512-d vectors); Qdrant/Redis are removed. It
is resilient to payloads written before the 09 revision: when ``parent_id`` is
absent it is derived from the ``chunk_id`` via :func:`ingest.schema.parent_id_of`.

Run a free, offline self-check against an in-memory dummy index (no OpenAI, no
faiss-server, no Docker, no cloud)::

    cd /home/user1/lawbot && .venv/bin/python -m search.retriever --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Final, Mapping, Sequence

import config
from ingest.schema import parent_id_of

# --------------------------------------------------------------------------- #
# Payload keys that callers are allowed to filter on with ``flt``. These mirror #
# the standard payload layout written into ``chunks_with_vectors.jsonl`` /      #
# ``FAISS_META`` (see _FAISS_BUILD_CONTRACT.md §1). Filtering on an unknown key  #
# is a programming error and is rejected so callers cannot silently get an       #
# always-empty result. ``parent_id`` is included so callers can scope a search   #
# to one document's children (09 §B-1 parent/child). ``as_of_date`` is NOT a     #
# member here — it is a dedicated range parameter on ``effective_from`` (see     #
# :func:`search`).                                                               #
# --------------------------------------------------------------------------- #
ALLOWED_FILTER_KEYS: Final[frozenset[str]] = frozenset(
    {"doc_type", "jurisdiction", "law_kind", "effective_from", "parent_id"}
)

# ISO calendar date (YYYY-MM-DD) — the accepted ``as_of_date`` / ``effective_from``
# shape. We keep validation strict (no times, no offsets) so the point-in-time
# filter is unambiguous and matches the stored ``effective_from`` format.
_ISO_DATE_RE: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Over-fetch multiplier: when a post-filter (``flt`` and/or ``as_of_date``) is in
# effect, we ask FAISS for ``k * _OVERFETCH`` candidates so the Python filter can
# still return up to ``k`` survivors after dropping non-matching rows. With no
# post-filter the top-``k`` from FAISS is already final, so we fetch exactly ``k``
# (over-fetch factor 1). 5x mirrors the Qdrant-era client-side as_of fallback.
_OVERFETCH: Final[int] = 5


# --------------------------------------------------------------------------- #
# Result type                                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Hit:
    """A single retrieved child chunk.

    Attributes:
        id: The stable chunk identity (the canonical ``chunk_id``). Stable across
            re-ingests, so it is a valid citation ``source_id``.
        score: Cosine similarity in ``[-1, 1]`` (higher is more relevant). Equal
            to the FAISS inner product because both query and document vectors are
            L2-normalized.
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
# FAISS index (process-wide, lazily loaded once)                              #
# --------------------------------------------------------------------------- #
# The FAISS index and its row-aligned metadata are loaded exactly once per
# process via ``embed.faiss_index.load_index`` and reused across all requests
# (the IndexFlatIP search path is read-only and thread-safe). Loading is lazy so
# importing this module never touches the filesystem or requires faiss/a built
# index — important for unit tests and API process startup order. ``None`` means
# "not yet loaded"; use :func:`get_index` rather than touching these directly.
#
# Layout: ``_index`` exposes a FAISS ``.search(matrix, n) -> (scores, ids)``
# (numpy float32 matrix in, ``(scores[Q,n], ids[Q,n])`` out; ``id == -1`` pads a
# short result). ``_metas`` is row-aligned: FAISS row id ``i`` ↔ ``_metas[i]``
# ``{chunk_id, doc_id, parent_id, text, payload}``.
_index: Any | None = None
_metas: list[dict[str, Any]] | None = None
_index_lock = threading.Lock()


class _DiskMetas:
    """Row-aligned meta accessor that reads ``meta.jsonl`` from DISK on demand.

    For the full corpus the meta sidecar is ~3 GB; holding it as an in-RAM list
    (the medical path) would OOM a small box. This builds only a compact per-row
    byte-offset table at startup (``array('q')``, ~10 MB for 1.24M rows) and
    seeks the requested line per access. The search path only needs meta for the
    over-fetched candidate rows (~``k * _OVERFETCH``) per query, so the extra
    disk reads are a handful per query. It exposes the same ``len(metas)`` /
    ``metas[i]`` interface the search path already uses, so no other retriever
    logic changes. Each ``__getitem__`` returns a fresh dict (safe to mutate).
    """

    def __init__(self, path) -> None:
        import array

        self._path = str(path)
        offsets = array.array("q")
        with open(self._path, "rb") as fh:
            pos = fh.tell()
            line = fh.readline()
            while line:
                offsets.append(pos)
                pos = fh.tell()
                line = fh.readline()
        self._offsets = offsets
        self._fh = open(self._path, "rb")
        self._read_lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, i: int) -> dict[str, Any]:
        with self._read_lock:
            self._fh.seek(self._offsets[i])
            raw = self._fh.readline()
        return json.loads(raw)


def get_index() -> tuple[Any, list[dict[str, Any]]]:
    """Return the lazily-loaded, process-cached ``(faiss_index, metas)`` pair.

    Loads via :func:`embed.faiss_index.load_index` on first use (reading
    :data:`config.FAISS_INDEX` + :data:`config.FAISS_META`) and caches the result
    for the lifetime of the process. ``metas`` is row-aligned to the index, i.e.
    ``index.ntotal == len(metas)`` and FAISS row id ``i`` maps to ``metas[i]``.

    Returns:
        A tuple ``(index, metas)`` where ``index`` exposes the FAISS
        ``search(matrix, n) -> (scores, ids)`` API and ``metas`` is a list of
        per-row metadata dicts.

    Raises:
        Whatever :func:`embed.faiss_index.load_index` raises when the index has
        not been built yet (a clear "build first" error).
    """
    global _index, _metas
    if _index is not None and _metas is not None:
        return _index, _metas
    with _index_lock:
        if _index is not None and _metas is not None:  # re-check under lock
            return _index, _metas
        # Full corpus: read the flat index into RAM (~2.5GB) but keep the ~3GB
        # meta on DISK (offset lookup) so a small box does not OOM. Selected via
        # config.ACTIVE_INDEX == "full" (env LAWBOT_INDEX=full).
        if config.ACTIVE_INDEX == "full":
            import faiss  # lazy: importing this module never requires faiss

            index = faiss.read_index(str(config.FULL_FAISS_INDEX))
            metas = _DiskMetas(config.FULL_FAISS_META)
            if index.ntotal != len(metas):
                raise RuntimeError(
                    f"Corrupt full index: ntotal={index.ntotal} != "
                    f"meta rows={len(metas)} ({config.FULL_FAISS_META})."
                )
            _index, _metas = index, metas
            return _index, _metas
        # Medical (default): small index + in-RAM metas via the builder's loader.
        # Imported lazily so importing ``search.retriever`` never requires faiss
        # or a built index (the builder owns embed/faiss_index.py).
        from embed.faiss_index import load_index

        index, metas = load_index()
        _index, _metas = index, list(metas)
        return _index, _metas


def set_index(
    index: Any | None = None,
    metas: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Inject a pre-built ``(index, metas)`` pair (tests / offline self-checks).

    Replaces the process-cached FAISS index and its row-aligned metadata so the
    search path can run without ``embed.faiss_index.load_index`` (e.g. against an
    in-memory dummy index in :func:`_selftest`). Passing ``None`` for either
    argument leaves that side of the cache unchanged.

    Args:
        index: A FAISS-compatible index exposing ``search(matrix, n)``.
        metas: Row-aligned metadata list (``metas[i]`` ↔ index row ``i``).
    """
    global _index, _metas
    with _index_lock:
        if index is not None:
            _index = index
        if metas is not None:
            _metas = list(metas)


def reset_index_cache() -> None:
    """Drop the cached FAISS index/metadata (tests / after a rebuild)."""
    global _index, _metas
    with _index_lock:
        _index = None
        _metas = None


# --------------------------------------------------------------------------- #
# Filter construction (Python post-filter; same semantics as the Qdrant era)   #
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
) -> Callable[[Mapping[str, Any]], bool] | None:
    """Translate a flat ``{key: value}`` mapping (+ ``as_of_date``) into a predicate.

    The returned callable takes a row **payload** and returns ``True`` when the
    row satisfies *all* conditions (AND-ed). ``flt`` values may be a single
    scalar (exact match) or a list/tuple/set of scalars (match-any), enabling e.g.
    ``{"law_kind": ["법률", "시행령"]}``. When ``as_of_date`` is given, an
    additional ``effective_from <= as_of_date`` condition is AND-ed in
    (point-in-time current-law lookup, 09 §E-1): rows whose ``effective_from`` is
    missing/blank or later than ``as_of_date`` are excluded. Comparison is lexical
    on the ISO ``YYYY-MM-DD`` string, which equals chronological order.

    This is the FAISS-era replacement for the previous Qdrant ``Filter`` builder:
    validation semantics (allowed keys, empty-value rejection, ISO-date
    validation, and the resulting ``ValueError`` cases) are **identical**; only
    the returned object changed from a server-side ``Filter`` to an in-process
    predicate applied as a Python post-filter (FAISS has no native metadata
    filtering).

    Args:
        flt: The requested metadata filter, or ``None``/empty for none.
        as_of_date: Optional ISO ``YYYY-MM-DD`` "as of" date.

    Returns:
        A predicate ``payload -> bool``, or ``None`` when neither ``flt`` nor
        ``as_of_date`` is supplied (i.e. no filtering needed).

    Raises:
        ValueError: If a key is not in :data:`ALLOWED_FILTER_KEYS`, a value is
            empty/``None``, or ``as_of_date`` is not a valid ISO date.
    """
    # Normalize ``flt`` into a list of (key, accepted-values set) up front so the
    # validation (and its ValueErrors) happen eagerly at build time, exactly like
    # the Qdrant builder did — not lazily on the first row.
    scalar_conditions: list[tuple[str, frozenset[str]]] = []
    for key, value in (flt or {}).items():
        if key not in ALLOWED_FILTER_KEYS:
            raise ValueError(
                f"Unsupported filter key {key!r}. "
                f"Allowed keys: {sorted(ALLOWED_FILTER_KEYS)}."
            )
        if value is None:
            raise ValueError(f"Filter value for {key!r} must not be None.")

        if isinstance(value, (list, tuple, set)):
            members = {str(v) for v in value if v is not None and str(v) != ""}
            if not members:
                raise ValueError(f"Filter value list for {key!r} is empty.")
        else:
            sval = str(value)
            if sval == "":
                raise ValueError(f"Filter value for {key!r} must not be empty.")
            members = {sval}
        scalar_conditions.append((key, frozenset(members)))

    iso_as_of: str | None = None
    if as_of_date is not None:
        iso_as_of = _validate_iso_date(as_of_date, param="as_of_date")

    if not scalar_conditions and iso_as_of is None:
        return None

    def _predicate(payload: Mapping[str, Any]) -> bool:
        for key, accepted in scalar_conditions:
            pv = payload.get(key)
            if pv is None or str(pv) not in accepted:
                return False
        if iso_as_of is not None:
            ef = str(payload.get("effective_from") or "").strip()
            # Missing/blank/non-ISO effective_from is excluded — the safe default
            # for a current-law query (an undated row is not assumed in force).
            if not _ISO_DATE_RE.match(ef) or ef > iso_as_of:
                return False
        return True

    return _predicate


# --------------------------------------------------------------------------- #
# Query-embedding LRU cache (09 §E-4.2 — "동일/반복 질문은 임베딩 재호출 0")      #
# --------------------------------------------------------------------------- #
# A small, thread-safe LRU keyed by SHA-256 of the (model, normalized-text) pair
# so repeated/identical queries skip the OpenAI round-trip entirely. The key
# folds in EMBED_MODEL so a model change cannot serve a stale-space vector. The
# cache stores the *normalized* (unit-length) vector as an immutable tuple and
# hands out fresh lists to callers so a caller mutating the result can never
# corrupt a cached entry. Bounded size keeps memory flat under unbounded distinct
# queries (real-time serving).
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


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    """Return ``vector`` scaled to unit L2 norm (so inner product == cosine).

    FAISS ``IndexFlatIP`` ranks by inner product; ranking equals cosine only when
    both the indexed vectors (normalized at build time, contract §1) and the query
    are unit-length. A zero/degenerate vector is returned unchanged (its norm is
    0); this never happens for a real embedding but keeps the function total.

    Args:
        vector: The raw embedding.

    Returns:
        A new list, the L2-normalized vector.
    """
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if norm == 0.0:
        return [float(x) for x in vector]
    return [float(x) / norm for x in vector]


def _embed_query_uncached(text: str) -> list[float]:
    """Embed an already-normalized, non-empty ``text`` → L2-normalized 512d vector.

    Delegates the OpenAI call to :func:`embed.embed_client.embed_texts`, which
    enforces ``dimensions=config.EMBED_DIMENSIONS`` (=512), validates the returned
    dimensionality, and wraps the request in ``tenacity`` retry. The result is
    then L2-normalized here so the FAISS inner-product search equals cosine
    similarity.

    Args:
        text: A non-empty query string.

    Returns:
        The L2-normalized embedding vector of length :data:`config.EMBED_DIM`.

    Raises:
        ValueError: If the returned vector has an unexpected dimensionality
            (model/index mismatch) — surfaced from ``embed_client`` or re-checked
            here.
    """
    from embed.embed_client import embed_texts

    vector = embed_texts([text])[0]
    if len(vector) != config.EMBED_DIM:
        raise ValueError(
            f"Embedding dimensionality {len(vector)} != config.EMBED_DIM "
            f"{config.EMBED_DIM}. Check EMBED_MODEL/EMBED_DIM (dimensions=512) and "
            f"the FAISS index vector size are consistent."
        )
    return _l2_normalize(vector)


def embed_query(query: str) -> list[float]:
    """Embed a query string with the index-time embedding model (LRU-cached).

    Uses :data:`config.EMBED_MODEL` at ``dimensions=512`` (via
    :mod:`embed.embed_client`) so that query and document vectors live in the same
    space, then **L2-normalizes** the result so the FAISS ``IndexFlatIP`` inner
    product equals cosine similarity. Identical queries are served from a bounded,
    thread-safe LRU (09 §E-4.2) so repeated questions cost no OpenAI round-trip;
    misses are wrapped with ``tenacity`` retry for transient network errors.

    Args:
        query: The user's question. Must be non-empty after stripping.

    Returns:
        The L2-normalized embedding vector; its length is validated to equal
        :data:`config.EMBED_DIM`. A fresh list is returned each call so callers
        may mutate it without affecting the cache.

    Raises:
        ValueError: If ``query`` is blank, or the returned vector has an
            unexpected dimensionality (model/index mismatch).
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
def _faiss_search(vector: list[float], n: int) -> list[tuple[int, float]]:
    """Run the FAISS top-``n`` inner-product query → list of ``(row_id, score)``.

    Builds a ``(1, EMBED_DIM)`` float32 query matrix, calls the index's
    ``search`` (FAISS API: ``(scores, ids)`` arrays), and flattens the single
    query row into ``(row_id, score)`` pairs in descending-score order. FAISS pads
    a short result with ``id == -1``; those padding entries are dropped.

    Args:
        vector: The L2-normalized query vector (length :data:`config.EMBED_DIM`).
        n: Number of candidates to fetch (the over-fetched ``k * over``).

    Returns:
        Up to ``n`` ``(row_id, score)`` pairs, best first.
    """
    import numpy as np

    index, _ = get_index()
    q = np.asarray([vector], dtype="float32")
    scores, ids = index.search(q, n)
    out: list[tuple[int, float]] = []
    for row_id, score in zip(ids[0].tolist(), scores[0].tolist()):
        if row_id < 0:  # FAISS padding for fewer-than-n results
            continue
        out.append((int(row_id), float(score)))
    return out


def _row_to_hit(row_id: int, score: float, metas: list[dict[str, Any]]) -> Hit:
    """Assemble a :class:`Hit` from a FAISS row id + its row-aligned metadata.

    The :class:`Hit` payload preserves the historical key layout (so every
    accessor keeps working): the stored ``payload`` is copied and the identity
    keys ``text``/``chunk_id``/``doc_id``/``parent_id`` from the meta record are
    merged in (the meta record carries them alongside ``payload`` per contract
    §1). ``Hit.id`` is the canonical ``chunk_id``.
    """
    meta = metas[row_id]
    payload = dict(meta.get("payload") or {})
    # Surface the identity/text fields into the payload so Hit accessors (.text,
    # .chunk_id, .parent_id, .doc_id) read them even if the builder kept them at
    # the meta top level rather than inside payload.
    if "text" in meta and "text" not in payload:
        payload["text"] = meta["text"]
    for k in ("chunk_id", "doc_id", "parent_id"):
        if meta.get(k) is not None and payload.get(k) is None:
            payload[k] = meta[k]
    hit_id = str(meta.get("chunk_id") or payload.get("chunk_id") or row_id)
    return Hit(id=hit_id, score=score, payload=payload)


# --------------------------------------------------------------------------- #
# BM25 sidecar (SQLite FTS5 unicode61) for hybrid retrieval                    #
# --------------------------------------------------------------------------- #
_bm25_conn_obj: Any | None = None
_bm25_conn_lock = threading.Lock()


def _bm25_conn() -> Any:
    """Process-cached read-only SQLite connection to the BM25 FTS5 sidecar."""
    global _bm25_conn_obj
    if _bm25_conn_obj is None:
        import sqlite3

        _bm25_conn_obj = sqlite3.connect(
            f"file:{config.BM25_DB}?mode=ro", uri=True, check_same_thread=False
        )
    return _bm25_conn_obj


# Korean question/filler words that, as broad prefix tokens, flood BM25 with
# noise (e.g. "내용*" matches every "판례내용"); dropped from the BM25 query so
# only discriminative terms (law names, articles, content nouns) remain.
_BM25_STOPWORDS: frozenset[str] = frozenset({
    "내용", "알려줘", "알려주세요", "알려줄래", "뭐야", "무엇", "무엇인가", "무슨",
    "어떻게", "어떤", "어떠한", "경우", "관련", "대해", "대한", "대하여", "좀",
    "해줘", "하나요", "한가요", "인가", "인가요", "되나요", "될까요", "되는",
    "받아요", "받나요", "받을", "얼마나", "며칠", "있어", "있나요", "있는",
    "어디", "어디서", "누가", "언제", "왜", "그것", "이것", "저것", "그리고",
    "또는", "정도", "등의", "그리고", "싶어", "싶은데", "알고", "관하여", "관한",
    "규정", "어떻해", "어떡해", "할까요", "하는", "되어", "위한", "위해",
})


def _bm25_search(query: str, n: int) -> list[int]:
    """Return up to ``n`` FAISS row ids ranked by BM25 (best first), or ``[]``.

    Builds an FTS5 ``OR`` of PREFIX tokens (``term*``) so Korean inflection /
    compounds match (e.g. ``처벌*`` hits "처벌한다", ``사기*`` hits "사기죄로",
    ``민법*`` hits "민법"). Common question/filler words are dropped (see
    :data:`_BM25_STOPWORDS`) so they don't flood results via broad prefix match.
    The ``ttl`` column (law name + article) is weighted far above ``txt`` so an
    exact citation like "민법 제4조" surfaces its target. The FTS5 ``rowid``
    equals the FAISS row id, so the ids map straight back to ``metas[row]``.
    """
    if not config.BM25_DB.exists():
        return []
    terms = [t for t in re.split(r"\s+", (query or "").strip()) if t]
    prefixes: list[str] = []
    for t in terms:
        # Keep only word chars (Korean/Latin/digits); strip ALL punctuation —
        # a stray "?"/"!" etc. is an FTS5 syntax error that would void the query.
        t = re.sub(r"[^0-9A-Za-z가-힣]", "", t)
        if len(t) >= 2 and t not in _BM25_STOPWORDS:
            prefixes.append(f"{t}*")
    if not prefixes:
        return []
    match = " OR ".join(prefixes)
    try:
        with _bm25_conn_lock:
            cur = _bm25_conn().execute(
                "SELECT rowid FROM bm25 WHERE bm25 MATCH ? "
                "ORDER BY bm25(bm25, 5.0, 1.0) LIMIT ?",
                (match, n),
            )
            return [int(r[0]) for r in cur.fetchall()]
    except Exception:
        return []


# Colloquial → legal-term expansion (query normalization). Keys are surface forms
# users actually type; values are the statutory vocabulary the corpus uses. The
# legal term is APPENDED (not replaced) so the original signal is preserved while
# the embedding + BM25 also see the legal word. Only well-established equivalences.
_QUERY_SYNONYMS: dict[str, str] = {
    "월급": "임금", "봉급": "임금", "급여": "임금", "월급여": "임금",
    "성인": "성년",
    "집주인": "임대인", "세입자": "임차인", "세 든": "임차", "세든": "임차",
    "전세": "임대차", "월세": "임대차",
    "잘렸": "해고", "잘림": "해고", "잘리": "해고", "짤렸": "해고",
    "짤림": "해고", "해고당": "해고",
    "빌린 돈": "채무", "빌린돈": "채무", "빚": "채무",
    "보이스피싱": "전기통신금융사기",
    "갑질": "직장 내 괴롭힘",
    "음주 운전": "음주운전",
}


_REWRITE_SYS = (
    "너는 한국 법률 검색 보조기다. 사용자의 구어체 법률 질문을, 법령 검색에 적합한 "
    "핵심 법률 용어·관련 법령명·조문 주제어로 한 줄로 변환하라. 설명·문장 없이 키워드만. "
    "**그 쟁점을 직접 규율하는 핵심 조문 번호(제N조, 가지번호 포함 예: 제839조의2)를 "
    "아는 경우 반드시 맨 앞에 포함**하라(여러 법이 관련되면 가장 직접적인 근거 법령·조문을 "
    "먼저). 불확실할 때만 생략. "
    "예) '월급 떼였는데 어떻게 받아요?' -> '근로기준법 제43조 임금 체불 임금 지급' / "
    "'집주인이 보증금 안 줘요' -> '주택임대차보호법 제3조의2 보증금 반환 우선변제권' / "
    "'사기 치면 처벌 얼마나?' -> '형법 제347조 사기 기망 처벌' / "
    "'성인은 몇 살부터?' -> '민법 제4조 성년 19세' / "
    "'이혼하면 남편 명의 집도 절반?' -> '민법 제839조의2 재산분할청구권 기여도' / "
    "'집 팔았는데 양도세 안 내도 되나?' -> '소득세법 제89조 1세대1주택 비과세 양도소득' / "
    "'상속 빚이 더 많으면?' -> '민법 제1019조 상속포기 한정승인'."
)


@lru_cache(maxsize=4096)
def _llm_rewrite(query: str) -> str:
    """LLM 기반 구어→법률용어 재작성(캐시). 실패 시 빈 문자열(원질의 사용)."""
    try:
        from embed.embed_client import _client

        resp = _client().chat.completions.create(
            model=config.GEN_MODEL,
            messages=[
                {"role": "system", "content": _REWRITE_SYS},
                {"role": "user", "content": query},
            ],
            # Keyword rewrite is a light task; use the lighter rewrite effort
            # (default "minimal") so this pre-retrieval call stays cheap and fast.
            **config.reasoning_effort_kwargs(config.GEN_MODEL, config.GEN_REWRITE_EFFORT),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _normalize_query(query: str) -> str:
    """Append legal-term equivalents for any colloquial words in the query.

    Expansion (not replacement): "월급을 안 주는데" -> "월급을 안 주는데 임금",
    so both dense and BM25 see the statutory vocabulary while the original phrasing
    is kept. Closes the colloquial↔statute gap (free; no LLM call).
    """
    q = query or ""
    extra: list[str] = []
    for collo, legal in _QUERY_SYNONYMS.items():
        if collo in q and legal not in q and legal not in extra:
            extra.append(legal)
    return (q + " " + " ".join(extra)) if extra else q


_LAW_NAME_SUFFIXES = ("법", "법률", "령", "규칙")


def _is_citation_like(query: str) -> bool:
    """True if the query carries an exact-citation / law-name signal.

    Triggers BM25 fusion only for "민법 제4조" / "도로교통법 ..." style queries,
    where lexical matching is precise; pure colloquial keyword queries skip BM25
    (it would flood with keyword-dense precedents and hurt dense's hits).
    """
    if re.search(r"제\s?\d+\s?조", query):
        return True
    for t in re.split(r"\s+", (query or "").strip()):
        t = re.sub(r"[^0-9A-Za-z가-힣]", "", t)
        if len(t) >= 2 and t.endswith(_LAW_NAME_SUFFIXES):
            return True
    return False


def _hybrid_search(
    query: str,
    vector: list[float],
    index: Any,
    metas: Any,
    k: int,
    predicate: Callable[[Mapping[str, Any]], bool] | None,
) -> list[Hit]:
    """Fuse dense (FAISS cosine) and BM25 ranks via RRF, then post-filter to k.

    Ordering uses RRF over both rank lists; the returned ``Hit.score`` is the
    TRUE dense cosine (looked up from the dense pass, or reconstructed from the
    index for BM25-only rows) so the downstream MIN_RETRIEVAL_SCORE gate keeps
    working on dense confidence — not the tiny RRF value (bug #2).
    """
    import numpy as np

    n = max(config.HYBRID_CANDIDATES, k)
    dense = _faiss_search(vector, n)  # [(row, cosine)], best-first
    # TARGETED BM25: only fuse BM25 when the query carries a citation / law-name
    # signal (제N조, or a token ending in 법/법률/령/규칙). For pure colloquial
    # keyword queries BM25 floods with keyword-dense precedents and *degrades*
    # dense's correct hits (measured), so we skip it there. This makes hybrid a
    # strict add for exact-citation lookups without hurting the rest.
    bm = _bm25_search(query, n) if _is_citation_like(query) else []
    # Statute preference for law-name queries: precedents (case names dense in the
    # queried keywords) often out-rank the governing article in BM25. Reorder so
    # doc_type=="law" rows take the top BM25 ranks → higher RRF → the statute
    # article surfaces. Reads doc_type for the (<= n) BM25 candidates only.
    if bm:
        law_rows: list[int] = []
        other_rows: list[int] = []
        for r in bm:
            dt = None
            if 0 <= r < len(metas):
                dt = (metas[r].get("payload") or {}).get("doc_type")
            (law_rows if dt == "law" else other_rows).append(r)
        bm = law_rows + other_rows
    k0 = config.RRF_K0
    w_bm = config.RRF_W_BM25
    rrf: dict[int, float] = {}
    dense_cos: dict[int, float] = {}
    for rank, (row, cos) in enumerate(dense):
        if row < 0:
            continue
        rrf[row] = rrf.get(row, 0.0) + 1.0 / (k0 + rank + 1)  # dense weight = 1.0
        dense_cos[row] = cos
    for rank, row in enumerate(bm):
        rrf[row] = rrf.get(row, 0.0) + w_bm / (k0 + rank + 1)  # BM25 boosts only
    ordered = sorted(rrf, key=lambda r: rrf[r], reverse=True)

    qv = np.asarray(vector, dtype=np.float32)
    hits: list[Hit] = []
    for row in ordered:
        if row < 0 or row >= len(metas):
            continue
        if row in dense_cos:
            score = dense_cos[row]
        else:
            try:
                rec = index.reconstruct(int(row))
                score = float(np.dot(qv, np.asarray(rec, dtype=np.float32)))
            except Exception:
                score = 0.0
        hit = _row_to_hit(row, score, metas)
        if predicate is not None and not predicate(hit.payload):
            continue
        hits.append(hit)
        if len(hits) >= k:
            break
    return hits


def search(
    query: str,
    k: int = config.DEFAULT_TOP_K,
    flt: Mapping[str, Any] | None = None,
    as_of_date: str | None = None,
) -> list[Hit]:
    """Retrieve the top-``k`` child chunks most relevant to ``query``.

    Embeds the query (``dimensions=512``, L2-normalized), runs a FAISS
    ``IndexFlatIP`` inner-product search (== cosine), then applies the ``flt`` /
    ``as_of_date`` constraints as an in-process Python **post-filter** and returns
    the surviving top-``k`` hits (FAISS has no native metadata filtering). When a
    post-filter is active the search over-fetches ``k * 5`` candidates so the
    filter can still yield up to ``k`` results.

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

    # Build (and validate) the post-filter predicate eagerly so a bad filter key
    # or as_of_date raises ValueError before any embedding/search work.
    predicate = build_filter(flt, as_of_date=as_of_date)
    # Query understanding (colloquial → legal vocab) applies to BOTH dense
    # embedding and BM25; original phrasing is preserved (appended). LLM rewrite
    # is the strong lever; dictionary normalization is the free fallback.
    q_search = query
    if config.QUERY_REWRITE:
        rw = _llm_rewrite(query)
        if rw:
            q_search = f"{query} {rw}"
    elif config.QUERY_NORM:
        q_search = _normalize_query(query)
    vector = embed_query(q_search)

    index, metas = get_index()

    # Hybrid (BM25 + dense via RRF) when enabled and the BM25 sidecar exists.
    # Fixes dense's weak spots (colloquial phrasing, exact law-name/article).
    # Gate hybrid on the FULL index: the BM25 sidecar's rowids are aligned to the
    # full-index meta order, so using it against a different (e.g. medical) index
    # would map BM25 hits to the wrong rows (audit: HIGH). Only fuse when both match.
    if (
        config.HYBRID_SEARCH
        and config.ACTIVE_INDEX == "full"
        and config.BM25_DB.exists()
    ):
        return _hybrid_search(q_search, vector, index, metas, k, predicate)

    # Over-fetch when a post-filter is active so survivors can still reach k; with
    # no filter the FAISS top-k is already final. Never request more than the
    # index holds (FAISS pads with -1 otherwise, which we drop anyway).
    fetch = k * _OVERFETCH if predicate is not None else k
    fetch = min(max(fetch, k), len(metas)) if metas else fetch
    candidates = _faiss_search(vector, max(fetch, 1))

    hits: list[Hit] = []
    for row_id, score in candidates:
        if row_id >= len(metas):  # defensive: index/meta misalignment
            continue
        hit = _row_to_hit(row_id, score, metas)
        if predicate is not None and not predicate(hit.payload):
            continue
        hits.append(hit)
        if len(hits) >= k:
            break
    return hits


# --------------------------------------------------------------------------- #
# Parent promotion (09 §B-1 / §E-1)                                            #
# --------------------------------------------------------------------------- #
# Lazy, thread-safe in-memory index of parent records keyed by ``parent_id``.
# Built once from ``config.PARENTS_JSONL`` on first use and reused thereafter
# (the file is the build-time materialization of the parent = whole-law /
# whole-precedent texts). ``None`` means "not yet loaded"; an empty dict is a
# valid loaded state (e.g. the artifact does not exist yet). UNCHANGED across the
# Qdrant→FAISS switch — parent-promotion never touched the vector store.
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
class _DummyFlatIP:
    """Minimal in-memory stand-in for ``faiss.IndexFlatIP`` (offline self-test).

    Implements only the surface :func:`_faiss_search` uses: ``ntotal`` and
    ``search(matrix, n) -> (scores, ids)`` with FAISS semantics (descending inner
    product, ``id == -1`` padding when fewer than ``n`` rows exist). This lets the
    self-test run with **no faiss dependency and no OpenAI call**, while the
    production path uses the real ``faiss.IndexFlatIP`` via
    ``embed.faiss_index.load_index`` — both honor the same ``.search`` contract,
    so :func:`search` is exercised identically.
    """

    def __init__(self, vectors: list[list[float]]) -> None:
        import numpy as np

        # Store L2-normalized rows so inner product == cosine, mirroring the real
        # build (vectors are normalized before ``index.add``).
        self._mat = np.asarray(
            [_l2_normalize(v) for v in vectors], dtype="float32"
        )
        self.ntotal = int(self._mat.shape[0])

    def search(self, q, n):  # noqa: ANN001 - mimics faiss signature
        import numpy as np

        q = np.asarray(q, dtype="float32")
        sims = q @ self._mat.T  # (Q, ntotal) inner products
        n = int(n)
        scores_out = np.full((q.shape[0], n), -np.inf, dtype="float32")
        ids_out = np.full((q.shape[0], n), -1, dtype="int64")
        for r in range(q.shape[0]):
            order = np.argsort(-sims[r])[:n]  # descending, top-n
            for c, idx in enumerate(order):
                scores_out[r, c] = sims[r, idx]
                ids_out[r, c] = idx
        return scores_out, ids_out


def _selftest() -> int:
    """Validate filtering, as_of_date, ranking and parent-promotion offline.

    Both the FAISS index (a pure-numpy :class:`_DummyFlatIP`) and the query
    embedding are *faked* (no faiss install needed, no OpenAI call) so the test is
    free and offline; the FAISS search path, row→meta→:class:`Hit` round-trip,
    metadata + as_of_date post-filtering, over-fetch, and :func:`get_parent` are
    exercised end to end (the same 12 checks the Qdrant-era self-test ran).

    Returns a process exit code (0 pass, 1 fail).
    """
    import tempfile
    from pathlib import Path

    # Four documents along orthogonal-ish axes so we can predict ranking, with a
    # spread of effective_from dates to exercise the as_of_date filter. One law
    # row omits parent_id from its payload so we also test the chunk_id-derived
    # parent_id fallback. Dummy vectors are tiny (4-d) — dimensionality is
    # irrelevant to the FAISS search path being exercised, and embed_query is
    # stubbed so config.EMBED_DIM (512) is never asserted here.
    fixtures = [
        {
            "vec": [1.0, 0.0, 0.0, 0.0],
            "chunk_id": "LAW:000001:법률#제4조#0",
            "doc_id": "LAW:000001:법률",
            # parent_id intentionally omitted at meta level AND payload level ->
            # tests the chunk_id-derived parent_id fallback.
            "text": "[민법 제4조 성년] 사람은 19세로 성년에 이르게 된다.",
            "payload": {
                "doc_type": "law",
                "title": "민법",
                "jurisdiction": "국가",
                "law_kind": "법률",
                "article_no": "제4조",
                "effective_from": "2013-07-01",
                "source_url": "https://law.go.kr/민법",
                "trust_grade": "A",
            },
        },
        {
            "vec": [0.0, 1.0, 0.0, 0.0],
            "chunk_id": "ORD:전라남도:2200001#제2조#0",
            "doc_id": "ORD:전라남도:2200001",
            "parent_id": "ORD:전라남도:2200001",
            "text": "[전라남도 ○○ 조례 제2조] 정의 규정.",
            "payload": {
                "doc_type": "ordinance",
                "title": "전라남도 ○○ 조례",
                "jurisdiction": "전라남도",
                "law_kind": "조례",
                "article_no": "제2조",
                "effective_from": "2022-01-01",
                "source_url": "https://law.go.kr/ord",
                "trust_grade": "A",
            },
        },
        {
            "vec": [0.0, 0.0, 1.0, 0.0],
            "chunk_id": "PREC:424370#판결요지#0",
            "doc_id": "PREC:424370",
            "parent_id": "PREC:424370",
            "text": "[대법원 2020다12345 판결요지] 손해배상 책임 인정.",
            "payload": {
                "doc_type": "precedent",
                "title": "손해배상청구",
                "jurisdiction": "대법원",
                "law_kind": "민사",
                "article_no": "판결요지",
                "effective_from": "2020-05-14",
                "source_url": "https://law.go.kr/prec",
                "trust_grade": "A",
            },
        },
        {
            # A "future" law (effective after our as_of_date) on the same axis as
            # the 민법 row but slightly off, so an as_of_date cut must drop it.
            "vec": [0.95, 0.05, 0.0, 0.0],
            "chunk_id": "LAW:000099:법률#제1조#0",
            "doc_id": "LAW:000099:법률",
            "parent_id": "LAW:000099:법률",
            "text": "[미래법 제1조] 2030년 시행 예정 조문.",
            "payload": {
                "doc_type": "law",
                "title": "미래법",
                "jurisdiction": "국가",
                "law_kind": "법률",
                "article_no": "제1조",
                "effective_from": "2030-01-01",
                "source_url": "https://law.go.kr/future",
                "trust_grade": "A",
            },
        },
    ]

    # Build the in-memory dummy index + row-aligned metas (row i ↔ metas[i]),
    # mirroring embed.faiss_index.load_index()'s return shape.
    index = _DummyFlatIP([fx["vec"] for fx in fixtures])
    metas = [
        {
            "chunk_id": fx["chunk_id"],
            "doc_id": fx["doc_id"],
            "parent_id": fx.get("parent_id"),
            "text": fx["text"],
            "payload": dict(fx["payload"]),
        }
        for fx in fixtures
    ]
    set_index(index=index, metas=metas)

    # Stub embedding so search() runs without any OpenAI call. The stub returns
    # the 민법-axis vector, so an unfiltered search must rank the 민법 chunk first.
    global embed_query  # noqa: PLW0603 - intentional monkeypatch for selftest
    _real_embed = embed_query

    def _fake_embed(_query: str) -> list[float]:
        return _l2_normalize([1.0, 0.0, 0.0, 0.0])

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
        #    parent_id fallback (payload + meta key absent) derives the doc_id.
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
            if top.chunk_id != "LAW:000001:법률#제4조#0":
                failures.append(f"top hit chunk_id {top.chunk_id!r} unexpected")
            if top.id != "LAW:000001:법률#제4조#0":
                failures.append(f"top hit id {top.id!r} != chunk_id")
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
            "SELFTEST PASSED: 12 checks (FAISS ranking, metadata/parent_id "
            "post-filters, match-any, as_of_date point-in-time, over-fetch, "
            "parent-promotion, validation)."
        )
        return 0
    finally:
        embed_query = _real_embed  # type: ignore[assignment]
        config.PARENTS_JSONL = _real_parents  # type: ignore[misc]
        reset_parents_cache()
        reset_index_cache()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search.retriever",
        description="Dense retriever over the FAISS 의료관련 index.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run an offline self-check against an in-memory dummy FAISS index.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="If given (and not --selftest), embed+search the live FAISS index.",
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
