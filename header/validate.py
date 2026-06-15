"""Header governance validator (09 §D-3 · BUILD CONTRACT (d)) ★.

"관리·감독 = 강제 스키마 + 검증기 + 테스트." This module is the *검증기*: it
asserts that every produced chunk obeys :mod:`header.schema`. It is run in the
ingest pipeline **and** in ``pytest`` so a header regression fails the build —
i.e. continuous supervision rather than a one-off check.

A chunk (the record written to ``artifacts/chunks.jsonl``) is valid when:

* **(a) L1 required fields** — the first header line contains the doc_type's
  required identifiers (:data:`header.schema.HEADER_REQUIRED`: law name /
  identifier, article/section number, effective/decision date) by substring.
* **(b) L2 context line** — a non-empty second header line exists.
* **(c) payload completeness** — every key in
  :data:`header.schema.PAYLOAD_REQUIRED` is present, the four filter keys are
  non-null, and ``kind`` / ``trust_grade`` / ``doc_type`` are in their closed
  value sets.

Public interface (BUILD CONTRACT (d))::

    def validate_chunk(chunk: dict) -> list[str]: ...   # [] if valid
    def validate_file(path) -> dict: ...                # {n, n_bad, errors[...]}

Run as a script over the built chunks to print a missing-field report::

    cd /home/user1/lawbot && .venv/bin/python -m header.validate            # config.CHUNKS_JSONL
    cd /home/user1/lawbot && .venv/bin/python -m header.validate <path> [N] # sample first N

Owner: header builder. Imports the schema (single rule definition) and config
(artifact paths) only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import config
from header.schema import (
    DOC_TYPES,
    FILTER_KEY_NULLABLE,
    HEADER_REQUIRED,
    KINDS,
    N_HEADER_LINES,
    PAYLOAD_FILTER_KEYS,
    PAYLOAD_REQUIRED,
    TRUST_GRADES,
)


def _split_header(text: str) -> tuple[str, str, str]:
    """Split a chunk's embed ``text`` into (L1, L2, body).

    The layout is ``<L1>\\n<L2>\\n<body...>`` (09 §D-1). Splits on at most
    :data:`header.schema.N_HEADER_LINES` leading newlines so a body containing
    newlines is preserved intact.

    Args:
        text: The chunk's embedded text.

    Returns:
        ``(l1, l2, body)``; missing trailing parts are returned as ``""``.
    """
    parts = (text or "").split("\n", N_HEADER_LINES)
    l1 = parts[0] if len(parts) > 0 else ""
    l2 = parts[1] if len(parts) > 1 else ""
    body = parts[2] if len(parts) > 2 else ""
    return l1, l2, body


def _l1_tokens(doc_type: str, payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Return the (field-name, expected-substring) pairs the L1 line must carry.

    For each required field of ``doc_type`` (:data:`HEADER_REQUIRED`), map it to
    the literal value the L1 string should contain. Effective dates may be
    rendered via 시행/선고 prefixes, so the raw value is what we look for.

    Args:
        doc_type: The chunk's doc_type.
        payload: The chunk payload (source of the expected values).

    Returns:
        A list of ``(field, expected_substring)`` to assert by ``in``. Fields
        whose payload value is empty are reported separately as missing data and
        skipped here.
    """
    pairs: list[tuple[str, str]] = []
    for field in HEADER_REQUIRED.get(doc_type, []):
        value = payload.get(field)
        if value is None or str(value).strip() == "":
            # Missing underlying data — reported as a payload error, not an L1
            # substring error (avoids double-counting one root cause).
            continue
        pairs.append((field, str(value).strip()))
    return pairs


def validate_chunk(chunk: dict[str, Any]) -> list[str]:
    """Validate one chunk against the header schema; return error strings.

    Args:
        chunk: A chunk record (``{chunk_id, doc_id, parent_id, text,
            content_hash, payload}``) as written to ``config.CHUNKS_JSONL``.

    Returns:
        An empty list if the chunk is valid, otherwise a list of human-readable
        error strings (one per violation). The chunk_id is *not* prefixed here;
        :func:`validate_file` attaches it in the report.
    """
    errors: list[str] = []

    if not isinstance(chunk, dict):
        return ["chunk is not a JSON object"]

    text = chunk.get("text")
    if not isinstance(text, str) or not text.strip():
        errors.append("empty or missing 'text'")

    payload = chunk.get("payload")
    if not isinstance(payload, dict):
        return errors + ["missing or non-dict 'payload'"]

    doc_type = payload.get("doc_type")

    # (c) payload completeness ------------------------------------------------ #
    for key in PAYLOAD_REQUIRED:
        if key not in payload:
            errors.append(f"payload missing required key '{key}'")
    # Filter keys must always be *present*; their value must be non-null EXCEPT
    # where the doc_type legitimately lacks that datum (FILTER_KEY_NULLABLE) —
    # e.g. a precedent with no 선고일자 (effective_from). The key still exists, so
    # downstream filtering can rely on it (a null effective_from = no as_of lower
    # bound). The key must be present regardless.
    nullable = FILTER_KEY_NULLABLE.get(str(doc_type), frozenset())
    for key in PAYLOAD_FILTER_KEYS:
        if key not in payload:
            errors.append(f"payload filter key '{key}' is missing")
            continue
        if key in nullable:
            continue
        value = payload.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"payload filter key '{key}' is empty/None")

    if doc_type not in DOC_TYPES:
        errors.append(f"invalid doc_type {doc_type!r} (allowed: {DOC_TYPES})")
    kind = payload.get("kind")
    if kind not in KINDS:
        errors.append(f"invalid kind {kind!r} (allowed: {KINDS})")
    grade = payload.get("trust_grade")
    if grade not in TRUST_GRADES:
        errors.append(f"invalid trust_grade {grade!r} (allowed: {TRUST_GRADES})")

    # Linkage sanity: parent_id must equal the chunk's doc_id (09 §B-1).
    if payload.get("parent_id") and chunk.get("doc_id"):
        if payload["parent_id"] != chunk["doc_id"]:
            errors.append(
                f"parent_id {payload['parent_id']!r} != doc_id "
                f"{chunk['doc_id']!r}"
            )

    # Header-line structure ---------------------------------------------------- #
    if isinstance(text, str):
        l1, l2, _ = _split_header(text)

        # (a) L1 required fields present as substrings.
        if not l1.strip():
            errors.append("missing L1 인용헤더 (first line empty)")
        elif doc_type in DOC_TYPES:
            for field, expected in _l1_tokens(str(doc_type), payload):
                if expected not in l1:
                    errors.append(
                        f"L1 missing required field '{field}': "
                        f"{expected!r} not in L1 header"
                    )

        # (b) non-empty L2 context line.
        if not l2.strip():
            errors.append("missing L2 맥락헤더 (second line empty)")

    return errors


def _iter_chunks(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream ``(line_no, chunk)`` from a chunks JSONL, skipping bad JSON.

    Args:
        path: Path to a chunks JSONL artifact.

    Yields:
        ``(line_no, chunk_dict)`` pairs. Malformed JSON lines are surfaced as a
        synthetic chunk ``{"__bad_json__": ...}`` so the report counts them.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, {"__bad_json__": str(exc)}


def validate_file(
    path: Path | str | None = None,
    *,
    sample: int | None = None,
    max_errors: int = 200,
) -> dict[str, Any]:
    """Validate the chunks in a JSONL file and return a governance report.

    Streams the file (safe for the multi-million-line full corpus). Designed to
    be called from the ingest pipeline and from ``pytest`` (where ``sample`` caps
    the work). A non-zero ``n_bad`` should **fail the build** (09 §D-3).

    Args:
        path: Chunks JSONL path. Defaults to ``config.CHUNKS_JSONL``.
        sample: If given, validate only the first ``sample`` chunks (for fast CI
            gates over a representative slice).
        max_errors: Cap on the number of detailed error records retained in the
            report (counts are always exact; details are truncated to keep the
            report small).

    Returns:
        A report dict::

            {
              "path": "<path>",
              "n": <chunks checked>,
              "n_ok": <valid chunks>,
              "n_bad": <invalid chunks>,
              "by_doc_type": {"law": {"n": .., "n_bad": ..}, ...},
              "error_counts": {"<error text>": <count>, ...},
              "errors": [{"line": .., "chunk_id": .., "errors": [...]}, ...],
            }
    """
    path = Path(path) if path is not None else config.CHUNKS_JSONL
    n = n_bad = 0
    by_doc_type: dict[str, dict[str, int]] = {}
    error_counts: dict[str, int] = {}
    errors: list[dict[str, Any]] = []

    if not path.exists():
        return {
            "path": str(path),
            "n": 0,
            "n_ok": 0,
            "n_bad": 0,
            "by_doc_type": {},
            "error_counts": {},
            "errors": [],
            "note": f"chunks file not found: {path}",
        }

    for line_no, chunk in _iter_chunks(path):
        if sample is not None and n >= sample:
            break
        n += 1
        if "__bad_json__" in chunk:
            chunk_errors = [f"malformed JSON: {chunk['__bad_json__']}"]
            dt = "?"
        else:
            chunk_errors = validate_chunk(chunk)
            dt = str((chunk.get("payload") or {}).get("doc_type", "?"))

        bucket = by_doc_type.setdefault(dt, {"n": 0, "n_bad": 0})
        bucket["n"] += 1

        if chunk_errors:
            n_bad += 1
            bucket["n_bad"] += 1
            for e in chunk_errors:
                # Normalize variable bits out of the key so counts aggregate.
                key = e.split(":")[0]
                error_counts[key] = error_counts.get(key, 0) + 1
            if len(errors) < max_errors:
                errors.append(
                    {
                        "line": line_no,
                        "chunk_id": chunk.get("chunk_id"),
                        "errors": chunk_errors,
                    }
                )

    return {
        "path": str(path),
        "n": n,
        "n_ok": n - n_bad,
        "n_bad": n_bad,
        "by_doc_type": by_doc_type,
        "error_counts": error_counts,
        "errors": errors,
    }


def _main(argv: list[str]) -> int:
    """CLI entry: validate a chunks file and print a human-readable report.

    Args:
        argv: ``[path?, sample?]``. ``path`` defaults to ``config.CHUNKS_JSONL``;
            ``sample`` (int) limits how many chunks to check.

    Returns:
        Process exit code: ``0`` if all checked chunks pass, ``1`` otherwise.
    """
    path = Path(argv[0]) if len(argv) >= 1 else config.CHUNKS_JSONL
    sample = int(argv[1]) if len(argv) >= 2 else None
    report = validate_file(path, sample=sample)
    print(f"header.validate report for {report['path']}")
    if report.get("note"):
        print(f"  note: {report['note']}")
    print(f"  checked={report['n']}  ok={report['n_ok']}  bad={report['n_bad']}")
    for dt, b in sorted(report["by_doc_type"].items()):
        print(f"    {dt:>10}: {b['n']} chunks, {b['n_bad']} bad")
    if report["error_counts"]:
        print("  error breakdown:")
        for key, cnt in sorted(
            report["error_counts"].items(), key=lambda kv: -kv[1]
        ):
            print(f"    {cnt:>7}  {key}")
    if report["errors"]:
        print(f"  first {len(report['errors'])} failing chunks:")
        for rec in report["errors"][:10]:
            print(f"    L{rec['line']} {rec['chunk_id']}: {rec['errors']}")
    return 0 if report["n_bad"] == 0 else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))


__all__ = ["validate_chunk", "validate_file"]
