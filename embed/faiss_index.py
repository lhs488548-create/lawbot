"""FAISS index builder + sqlite-vec exporter for the medical sub-corpus.

This module is the **store layer** of the medical build. It consumes the
canonical, store-agnostic artifact ``config.CHUNKS_VEC_JSONL`` (one JSON object
per line: ``{chunk_id, doc_id, parent_id, text, payload, vector[512]}``) and
materializes two interchangeable vector stores from that single source:

* a ``faiss.IndexFlatIP(512)`` over **L2-normalized** vectors (inner product ==
  cosine similarity) plus a row-aligned ``meta.jsonl`` sidecar, and
* a self-contained SQLite database (``export_sqlite_vec``) shaped so that
  sqlite-vec / HMS can read it, with a portable ``float32`` BLOB fallback for
  environments where the ``vec0`` extension is unavailable.

Invariants (``docs/_FAISS_BUILD_CONTRACT.md`` §0, §2):

* dimension is fixed at ``config.EMBED_DIM`` (== 512); every vector is asserted
  to that length before use.
* vectors are L2-normalized at load time (``IndexFlatIP`` inner product ==
  cosine only for unit vectors); inputs are normalized again defensively even if
  the producer already normalized them.
* FAISS row ``i`` corresponds to input line ``i``; ``meta.jsonl`` line ``i``
  carries the matching metadata so ``retriever`` can map ``row -> meta -> Hit``.
* every function is **idempotent**: re-running with the same ``src`` yields the
  same outputs; existing files are replaced atomically.
* **no embedding / OpenAI calls** happen here — vectors already live in ``src``.

Public interface (BUILD CONTRACT §2)::

    def build_index(src: pathlib.Path = config.CHUNKS_VEC_JSONL) -> None: ...
    def load_index() -> tuple["faiss.Index", list[dict]]: ...
    def export_sqlite_vec(out: pathlib.Path) -> None: ...

Owner: FAISS index builder. Imports shared constants from ``config`` only.

Offline self-test (no network, no OpenAI — dummy vectors)::

    cd /home/user1/lawbot && .venv/bin/python -m embed.faiss_index --selftest
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import config

try:
    import faiss
except ModuleNotFoundError as exc:  # pragma: no cover - install guard
    raise ModuleNotFoundError(
        "faiss-cpu is required for embed.faiss_index. Install it into the WSL "
        "venv:  cd /home/user1/lawbot && ~/.local/bin/uv pip install "
        "--python .venv/bin/python faiss-cpu"
    ) from exc


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
_DIM: int = config.EMBED_DIM  # 512, code-pinned invariant.
# Metadata keys copied verbatim into meta.jsonl / sqlite (everything except the
# bulky vector, which lives in the FAISS index / vec table).
_META_KEYS: tuple[str, ...] = ("chunk_id", "doc_id", "parent_id", "text", "payload")


# --------------------------------------------------------------------------- #
# Low-level helpers                                                            #
# --------------------------------------------------------------------------- #
def _iter_records(src: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file, skipping blank lines.

    Args:
        src: Path to the canonical ``chunks_with_vectors.jsonl``.

    Yields:
        One decoded record dict per non-empty line, in file order.

    Raises:
        FileNotFoundError: If ``src`` does not exist (with a build hint).
        ValueError: If a line is not valid JSON (line number is reported).
    """
    if not src.exists():
        raise FileNotFoundError(
            f"Canonical vectors file not found: {src}. Build it first "
            f"(embed/medical_corpus -> embed/chunk -> embed/embed_client) so "
            f"that {config.CHUNKS_VEC_JSONL.name} exists."
        )
    with src.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"{src}:{lineno}: invalid JSON in vectors file: {exc}"
                ) from exc


def _as_unit_vector(record: dict[str, Any], where: str) -> np.ndarray:
    """Return the record's vector as an L2-normalized float32 array.

    Args:
        record: A canonical chunk record carrying a ``vector`` list.
        where: Human-readable locator (e.g. ``"row 7"``) for error messages.

    Returns:
        A ``(512,)`` float32 array with ``‖v‖₂ == 1`` (zero vectors are passed
        through unchanged to avoid div-by-zero; they simply never match).

    Raises:
        ValueError: If the vector is missing or not exactly ``config.EMBED_DIM``.
    """
    vec = record.get("vector")
    if vec is None:
        raise ValueError(f"{where}: record has no 'vector' field.")
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape != (_DIM,):
        raise ValueError(
            f"{where}: vector dimension {arr.shape} != ({_DIM},). "
            f"This build pins EMBED_DIM={_DIM}."
        )
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / norm
    return arr


def _meta_of(record: dict[str, Any], where: str) -> dict[str, Any]:
    """Project a canonical record down to its vector-free metadata.

    Args:
        record: A canonical chunk record.
        where: Locator for error messages.

    Returns:
        A dict with exactly ``{chunk_id, doc_id, parent_id, text, payload}``.

    Raises:
        ValueError: If ``chunk_id`` is missing/empty.
    """
    chunk_id = record.get("chunk_id")
    if not chunk_id:
        raise ValueError(f"{where}: record has empty/missing 'chunk_id'.")
    return {key: record.get(key) for key in _META_KEYS}


def _atomic_write_bytes(path: Path, writer) -> None:
    """Write to ``path`` atomically via a temp file in the same directory.

    Args:
        path: Final destination path.
        writer: Callable invoked with the temp file's string path; it must fully
            produce the file contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(str(tmp_path))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def build_index(src: Path = config.CHUNKS_VEC_JSONL) -> None:
    """Build ``config.FAISS_INDEX`` + ``config.FAISS_META`` from ``src``.

    Streams the canonical JSONL once, L2-normalizes each 512-d vector, and adds
    it to a fresh ``faiss.IndexFlatIP(512)`` in file order. A row-aligned
    ``meta.jsonl`` (FAISS row ``i`` <-> input line ``i``) holds the vector-free
    metadata so the retriever can recover ``Hit`` records from search row ids.

    Args:
        src: Canonical ``chunks_with_vectors.jsonl``. Defaults to
            ``config.CHUNKS_VEC_JSONL``.

    Raises:
        FileNotFoundError: If ``src`` is absent.
        ValueError: On wrong dimension, missing vector/chunk_id, or a duplicate
            ``chunk_id`` (the canonical artifact requires globally-unique ids).

    Notes:
        Idempotent: identical ``src`` yields identical outputs; both files are
        replaced atomically. No OpenAI/network calls (vectors are already in
        ``src``). ``index.ntotal == line count == len(meta.jsonl)``.
    """
    index = faiss.IndexFlatIP(_DIM)
    seen_ids: set[str] = set()
    metas: list[dict[str, Any]] = []
    # Accumulate vectors then add in one batched call (faster + same result).
    vectors: list[np.ndarray] = []

    for row, record in enumerate(_iter_records(src)):
        where = f"{src}: row {row}"
        meta = _meta_of(record, where)
        chunk_id = meta["chunk_id"]
        if chunk_id in seen_ids:
            raise ValueError(f"{where}: duplicate chunk_id {chunk_id!r}.")
        seen_ids.add(chunk_id)
        vectors.append(_as_unit_vector(record, where))
        metas.append(meta)

    if vectors:
        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        # Defense in depth: ensure unit norm at the matrix level too.
        faiss.normalize_L2(matrix)
        index.add(matrix)

    assert index.ntotal == len(metas), (
        f"FAISS ntotal {index.ntotal} != meta count {len(metas)} (row alignment broken)."
    )

    config.FAISS_DIR.mkdir(parents=True, exist_ok=True)

    # Atomic FAISS index write.
    _atomic_write_bytes(config.FAISS_INDEX, lambda p: faiss.write_index(index, p))

    # Atomic, row-aligned meta.jsonl write.
    def _write_meta(tmp_path: str) -> None:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            for meta in metas:
                fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

    _atomic_write_bytes(config.FAISS_META, _write_meta)


def load_index() -> tuple["faiss.Index", list[dict]]:
    """Load the FAISS index and its row-aligned metadata.

    Returns:
        ``(index, metas)`` where ``index`` is the loaded
        ``faiss.IndexFlatIP(512)`` and ``metas`` is the list of meta dicts in
        FAISS row order. Invariant: ``index.ntotal == len(metas)``.

    Raises:
        FileNotFoundError: If either ``config.FAISS_INDEX`` or
            ``config.FAISS_META`` is missing (run ``build_index`` first).
        RuntimeError: If the loaded index and metadata disagree on row count.
    """
    if not config.FAISS_INDEX.exists() or not config.FAISS_META.exists():
        raise FileNotFoundError(
            f"FAISS store not built: expected {config.FAISS_INDEX} and "
            f"{config.FAISS_META}. Run embed.faiss_index.build_index() first."
        )
    index = faiss.read_index(str(config.FAISS_INDEX))
    metas: list[dict] = []
    with config.FAISS_META.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                metas.append(json.loads(line))
    if index.ntotal != len(metas):
        raise RuntimeError(
            f"Corrupt FAISS store: index.ntotal={index.ntotal} but "
            f"meta rows={len(metas)}. Rebuild with build_index()."
        )
    return index, metas


def export_sqlite_vec(out: Path) -> None:
    """Export the canonical vectors to a sqlite-vec / HMS-readable SQLite DB.

    Reads ``config.CHUNKS_VEC_JSONL`` and writes ``out`` with two tables:

    * ``chunks(chunk_id PK, doc_id, parent_id, text, payload JSON, embedding
      BLOB)`` — the ``embedding`` BLOB holds the L2-normalized vector as 512
      little-endian ``float32`` values, a portable fallback readable without the
      sqlite-vec extension.
    * ``vec_chunks`` — a sqlite-vec ``vec0`` virtual table
      (``embedding float[512]``) whose ``rowid`` matches ``chunks.rowid``, when
      the ``vec0`` extension is loadable. Skipped (with the BLOB column still
      populated) when the extension is unavailable.

    Args:
        out: Destination SQLite file path.

    Raises:
        FileNotFoundError: If ``config.CHUNKS_VEC_JSONL`` is absent.
        ValueError: On wrong dimension / missing fields / duplicate chunk_id.

    Notes:
        Idempotent: tables are dropped and recreated, and the file is written to
        a temp path then atomically moved into place. Vectors are stored
        normalized. No OpenAI/network calls.
    """
    src = config.CHUNKS_VEC_JSONL

    def _build_db(tmp_db_path: str) -> None:
        conn = sqlite3.connect(tmp_db_path)
        try:
            vec_available = _try_load_sqlite_vec(conn)
            conn.execute("DROP TABLE IF EXISTS chunks")
            conn.execute(
                "CREATE TABLE chunks ("
                "  chunk_id  TEXT PRIMARY KEY,"
                "  doc_id    TEXT,"
                "  parent_id TEXT,"
                "  text      TEXT,"
                "  payload   TEXT,"  # JSON string
                "  embedding BLOB"  # 512 float32 LE
                ")"
            )
            if vec_available:
                conn.execute("DROP TABLE IF EXISTS vec_chunks")
                conn.execute(
                    f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
                    f"  embedding float[{_DIM}]"
                    f")"
                )

            seen_ids: set[str] = set()
            for row, record in enumerate(_iter_records(src)):
                where = f"{src}: row {row}"
                meta = _meta_of(record, where)
                chunk_id = meta["chunk_id"]
                if chunk_id in seen_ids:
                    raise ValueError(f"{where}: duplicate chunk_id {chunk_id!r}.")
                seen_ids.add(chunk_id)
                unit = _as_unit_vector(record, where)
                blob = struct.pack(f"<{_DIM}f", *unit.tolist())
                cur = conn.execute(
                    "INSERT INTO chunks "
                    "(chunk_id, doc_id, parent_id, text, payload, embedding) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        chunk_id,
                        meta["doc_id"],
                        meta["parent_id"],
                        meta["text"],
                        json.dumps(meta["payload"], ensure_ascii=False),
                        blob,
                    ),
                )
                if vec_available:
                    conn.execute(
                        "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                        (cur.lastrowid, blob),
                    )
            conn.commit()
        finally:
            conn.close()

    _atomic_write_bytes(out, _build_db)


def _try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Best-effort load of the sqlite-vec extension.

    Args:
        conn: An open SQLite connection.

    Returns:
        ``True`` if the ``vec0`` virtual-table module is usable, else ``False``
        (the caller still writes the portable ``embedding`` BLOB column).
    """
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.OperationalError):
        return False
    try:
        import sqlite_vec  # type: ignore

        sqlite_vec.load(conn)
    except Exception:  # pragma: no cover - optional dependency / build
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass
        return False
    try:
        conn.enable_load_extension(False)
    except Exception:
        pass
    return True


# --------------------------------------------------------------------------- #
# Offline self-test (no network, no OpenAI)                                    #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Build -> load -> search -> export round-trip on dummy 512-d vectors.

    Writes a synthetic canonical JSONL to a temp dir, overrides the config
    output paths to that dir, then verifies: row alignment, top-1 retrieval
    exactness for each query equal to a stored vector, cosine score ~1.0, and a
    populated sqlite export (both the BLOB column and, if available, vec_chunks).

    Returns:
        Process exit code (0 = all checks passed).
    """
    rng = np.random.default_rng(20260616)
    n = 32
    # Distinct random unit vectors so each is its own nearest neighbor.
    raw = rng.standard_normal((n, _DIM)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory(prefix="faiss_selftest_") as td:
        tdp = Path(td)
        src = tdp / "chunks_with_vectors.jsonl"
        with src.open("w", encoding="utf-8") as fh:
            for i in range(n):
                rec = {
                    "chunk_id": f"DUMMY:{i:04d}#0",
                    "doc_id": f"DUMMY:{i:04d}",
                    "parent_id": f"DUMMY:{i:04d}",
                    "text": f"dummy chunk number {i}",
                    "payload": {
                        "doc_type": "law",
                        "title": f"더미법 {i}",
                        "article_no": f"제{i}조",
                        "effective_from": "2024-01-01",
                        "source_url": f"https://example.test/{i}",
                        "trust_grade": "A",
                    },
                    "vector": raw[i].tolist(),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Redirect config output paths into the temp dir (selftest isolation).
        orig = (config.FAISS_DIR, config.FAISS_INDEX, config.FAISS_META, config.CHUNKS_VEC_JSONL)
        config.FAISS_DIR = tdp / "faiss"  # type: ignore[misc]
        config.FAISS_INDEX = config.FAISS_DIR / "index.faiss"  # type: ignore[misc]
        config.FAISS_META = config.FAISS_DIR / "meta.jsonl"  # type: ignore[misc]
        config.CHUNKS_VEC_JSONL = src  # type: ignore[misc]

        failures: list[str] = []

        def check(cond: bool, msg: str) -> None:
            print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
            if not cond:
                failures.append(msg)

        try:
            build_index(src)
            check(config.FAISS_INDEX.exists(), "index.faiss written")
            check(config.FAISS_META.exists(), "meta.jsonl written")

            index, metas = load_index()
            check(index.ntotal == n, f"ntotal == {n} (got {index.ntotal})")
            check(len(metas) == n, f"meta rows == {n} (got {len(metas)})")
            check(
                all(metas[i]["chunk_id"] == f"DUMMY:{i:04d}#0" for i in range(n)),
                "meta row alignment matches input order",
            )

            # Re-normalize queries (mirrors retriever's embed_query contract).
            q = raw.copy()
            faiss.normalize_L2(q)
            scores, ids = index.search(q, 1)
            top1_ok = all(int(ids[i][0]) == i for i in range(n))
            check(top1_ok, "top-1 retrieval returns the query's own row")
            check(
                bool(np.all(scores[:, 0] > 0.999)),
                f"top-1 cosine ~1.0 (min={float(scores[:, 0].min()):.4f})",
            )

            # Idempotency: rebuild and confirm identical ntotal + meta.
            build_index(src)
            index2, metas2 = load_index()
            check(
                index2.ntotal == n and metas2 == metas,
                "rebuild is idempotent (same ntotal + meta)",
            )

            # sqlite-vec export.
            db_path = tdp / "med.sqlite"
            export_sqlite_vec(db_path)
            check(db_path.exists(), "sqlite export file written")
            conn = sqlite3.connect(str(db_path))
            try:
                (cnt,) = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
                check(cnt == n, f"chunks table has {n} rows (got {cnt})")
                blob = conn.execute(
                    "SELECT embedding FROM chunks WHERE chunk_id = ?",
                    ("DUMMY:0000#0",),
                ).fetchone()[0]
                vals = struct.unpack(f"<{_DIM}f", blob)
                recovered = np.asarray(vals, dtype=np.float32)
                check(len(vals) == _DIM, f"embedding BLOB has {_DIM} floats")
                check(
                    float(np.linalg.norm(recovered)) > 0.999,
                    "recovered BLOB vector is unit-normalized",
                )
                check(
                    float(np.dot(recovered, raw[0])) > 0.999,
                    "recovered BLOB vector matches input vector 0",
                )
                has_vec = bool(
                    conn.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name='vec_chunks'"
                    ).fetchone()
                )
                print(
                    f"  [INFO] vec_chunks (sqlite-vec) "
                    f"{'present' if has_vec else 'absent (BLOB fallback used)'}"
                )
            finally:
                conn.close()
        finally:
            (
                config.FAISS_DIR,
                config.FAISS_INDEX,
                config.FAISS_META,
                config.CHUNKS_VEC_JSONL,
            ) = orig  # type: ignore[misc]

    if failures:
        print(f"\nSELFTEST FAILED: {len(failures)} check(s) failed.")
        return 1
    print("\nSELFTEST OK: all checks passed (offline, no OpenAI calls).")
    return 0


def _main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    print(__doc__)
    print(
        "Usage: python -m embed.faiss_index --selftest\n"
        "  build_index(src=config.CHUNKS_VEC_JSONL) -> None\n"
        "  load_index() -> (faiss.Index, list[dict])\n"
        "  export_sqlite_vec(out: Path) -> None"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
