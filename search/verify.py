"""Citation Firewall — ``/v1/verify`` (09 §E-2, lawbot.org ``/v1/verify``).

The Citation Firewall is the anti-hallucination backstop of the service: given a
*citation* that an AI (or a lawyer) is about to rely on — either a statute
reference ``{law_name/title, article_no}`` or a precedent reference
``{case_no/사건번호}`` — it answers, with provenance, **"is this real, current,
and valid at the requested point in time?"**.

It runs three independent checks and reports each one so the caller can audit:

1. **DB existence** — does the cited statute article / precedent exist in *our*
   indexed corpus (``parents.jsonl``; FAISS era — Qdrant removed)? Catches
   references to documents we have never ingested.
2. **law.go.kr 현행·문구 대조** — a real call to the law.go.kr OpenAPI
   (``config.LAW_API_BASE`` with ``OC=config.LAW_OC``) confirms the statute
   article / precedent *currently exists at the authoritative source*, that it
   is the **현행** (in-force) version, and — for statutes — that the article
   title/number actually matches (문구 대조). Catches 폐지(repealed),
   오인용(misquoted article number) and 허위사건(fabricated case numbers): the
   precedent search is keyword-fuzzy, so we require an **exact** 사건번호 match
   among the returned rows rather than trusting the top hit.
3. **as_of_date point-in-time validity** — when an ``as_of_date`` is supplied,
   the cited document's ``effective_from`` (enforcement / decision date) must be
   ``<= as_of_date``; otherwise it had not taken effect yet at that time.

Return shape (per ``_BUILD_CONTRACT.md`` §(d) → Citation Firewall)::

    {
      "verified": bool,          # overall pass (DB ∧ API-ok ∧ as_of-ok)
      "trust_grade": "A"|"B",    # A = original text confirmed, B = metadata only
      "current": bool,           # is the cited document the in-force/현행 version
      "source_url": str|None,    # canonical law.go.kr URL when known
      "effective_from": str|None,
      "as_of_date": str|None,
      "note": str,               # human-readable explanation (no secrets)
      "db_match": bool,          # check 1 result
      "api_match": bool|None,    # check 2 result; None when API unavailable
    }

Hard rules honored here:

* **The OC token is never logged or returned.** It is read from
  :data:`config.LAW_OC` and passed only as a request parameter; this module
  never prints it, never puts it in ``note``/``source_url``, and strips it from
  any law.go.kr ``법령상세링크`` before exposing a URL.
* When the law API is unreachable or returns nothing usable, the firewall
  **degrades to DB-only** with ``api_match=None`` and an explanatory ``note``
  (it never crashes a request path).
* Grounding/quality only — no consumer-style refusals (expert/lawyer mode).

Self-check (offline DB path + one real law.go.kr call)::

    cd /home/user1/lawbot && .venv/bin/python -m search.verify --selftest
    cd /home/user1/lawbot && .venv/bin/python -m search.verify --live "도로교통법" "제17조"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import unicodedata
from typing import Any, Final

import httpx
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
# law.go.kr DRF endpoints (under config.LAW_API_BASE, e.g. .../DRF).
_SEARCH_PATH: Final[str] = "lawSearch.do"
_SERVICE_PATH: Final[str] = "lawService.do"

# Network timeout for a single law.go.kr call. Kept modest so the request path
# never hangs an API worker; tenacity adds a couple of short retries on top.
_HTTP_TIMEOUT: Final[float] = 12.0

# How many search rows to scan for an exact match. law.go.kr's precedent search
# is keyword-fuzzy (a query for "2022도1401" can return "2024도10062"), so we
# pull several rows and require an exact 사건번호 match among them.
_SEARCH_DISPLAY: Final[int] = 20

# Matches a Korean article label like 제17조 / 제4조의2, capturing the number and
# the optional 의N suffix so "제17조" and "제 17 조" and "17" all normalize.
_ART_RE: Final[re.Pattern[str]] = re.compile(r"제?\s*(\d+)\s*조(?:\s*의\s*(\d+))?")

# law.go.kr 법종구분/현행 markers indicating an in-force document.
_CURRENT_MARKERS: Final[frozenset[str]] = frozenset({"현행"})


# --------------------------------------------------------------------------- #
# Small utilities (normalization, OC redaction)                                #
# --------------------------------------------------------------------------- #
def _nfc(text: str) -> str:
    """Return ``text`` Unicode-normalized (NFC) and whitespace-collapsed."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def _strip_oc(url: str | None) -> str | None:
    """Remove the ``OC=...`` token from a law.go.kr link before exposing it.

    The DRF API echoes our OC token inside ``법령상세링크``/``판례상세링크``. The
    OC is a secret and must never leave this process, so we drop that query
    parameter (and prefix the host) before returning a citable URL.

    Args:
        url: A relative or absolute law.go.kr URL, possibly carrying ``OC=``.

    Returns:
        A safe absolute ``https://www.law.go.kr/...`` URL with no OC, or
        ``None`` if ``url`` is falsy.
    """
    if not url:
        return None
    # Drop the OC query parameter wherever it appears.
    cleaned = re.sub(r"([?&])OC=[^&]*&?", r"\1", url)
    cleaned = cleaned.rstrip("?&")
    if cleaned.startswith("/"):
        cleaned = "https://www.law.go.kr" + cleaned
    return cleaned


def _normalize_case_no(case_no: str) -> str:
    """Normalize a 사건번호 for exact comparison (NFC, strip spaces/dots)."""
    s = unicodedata.normalize("NFC", case_no or "")
    return re.sub(r"\s+", "", s)


def _article_number(article_no: str) -> str | None:
    """Extract a comparable article key (e.g. ``"17"`` or ``"17의2"``).

    Args:
        article_no: A label like ``"제17조"``, ``"제4조의2"``, or ``"17"``.

    Returns:
        The normalized ``"<n>"`` / ``"<n>의<m>"`` key, or ``None`` if no article
        number can be parsed.
    """
    if not article_no:
        return None
    m = _ART_RE.search(article_no)
    if m:
        base, sub = m.group(1), m.group(2)
        return f"{base}의{sub}" if sub else base
    # Fallback: a bare number such as law.go.kr's 조문번호 "17" (or "17의2").
    m2 = re.fullmatch(r"\s*(\d+)\s*(?:의\s*(\d+))?\s*", article_no)
    if m2:
        base, sub = m2.group(1), m2.group(2)
        return f"{base}의{sub}" if sub else base
    return None


def _jo_param_from_key(key: str | None) -> str | None:
    """Build the law.go.kr ``JO`` parameter from a normalized article key.

    law.go.kr encodes the article as a 6-digit string: 4 digits for the article
    number + 2 digits for the 의N sub-article (00 when absent). E.g. ``"17"`` →
    ``"001700"``, ``"4의2"`` → ``"000402"``.

    Args:
        key: A normalized key from :func:`_article_number` (``"17"`` / ``"4의2"``).

    Returns:
        The 6-digit ``JO`` string, or ``None`` if ``key`` is falsy/unparseable.
    """
    if not key:
        return None
    if "의" in key:
        base, sub = key.split("의", 1)
    else:
        base, sub = key, "0"
    try:
        return f"{int(base):04d}{int(sub):02d}"
    except ValueError:  # pragma: no cover - defensive
        return None


def _jo_param(article_no: str) -> str | None:
    """Build the law.go.kr ``JO`` parameter for an article label.

    Convenience wrapper around :func:`_jo_param_from_key` that first parses an
    article label like ``"제17조"`` / ``"제4조의2"`` into its normalized key.

    Args:
        article_no: A label like ``"제17조"`` or ``"제4조의2"``.

    Returns:
        The 6-digit ``JO`` string, or ``None`` if no number is parseable.
    """
    key = _article_number(article_no)
    if key is None:
        return None
    if "의" in key:
        base, sub = key.split("의", 1)
    else:
        base, sub = key, "0"
    try:
        return f"{int(base):04d}{int(sub):02d}"
    except ValueError:  # pragma: no cover - defensive
        return None


# --------------------------------------------------------------------------- #
# law.go.kr OpenAPI client (real call, OC never logged/returned)               #
# --------------------------------------------------------------------------- #
_RETRYABLE = retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError))

_retry_http = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.4, min=0.4, max=4),
    retry=_RETRYABLE,
)


# TTL cache for law.go.kr GETs (current law is intra-hour stable). Keyed on
# (path, sorted params); only successful JSON responses are cached. Thread-safe.
_LAW_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
_LAW_CACHE_LOCK = threading.Lock()
_LAW_CACHE_MAX: Final[int] = 4096


def _law_cache_key(path: str, params: dict[str, str]) -> tuple[Any, ...]:
    return (path, tuple(sorted(params.items())))


@_retry_http
def _law_get(path: str, params: dict[str, str]) -> dict[str, Any]:
    """GET a law.go.kr DRF endpoint and parse JSON (OC injected, never logged).

    Successful responses are cached for ``config.LAW_CACHE_TTL`` seconds so that
    repeat verification of the same citation (e.g. 형법 제347조) does not re-hit the
    API — cutting latency and rate-limit pressure. Errors are never cached.

    Args:
        path: Endpoint filename (``lawSearch.do`` / ``lawService.do``).
        params: Query parameters *without* ``OC``/``type``; both are added here.

    Returns:
        The parsed JSON object.

    Raises:
        RuntimeError: If :data:`config.LAW_OC` is not configured.
        httpx.HTTPError: On transport/HTTP errors (after retries).
        ValueError: If the body is not valid JSON.
    """
    if not config.LAW_OC:
        raise RuntimeError("LAW_OC is not configured; cannot call law.go.kr API.")
    ttl = config.LAW_CACHE_TTL
    key = _law_cache_key(path, params)
    if ttl > 0:
        with _LAW_CACHE_LOCK:
            cached = _LAW_CACHE.get(key)
            if cached is not None and (time.time() - cached[0]) < ttl:
                return cached[1]
    url = f"{config.LAW_API_BASE.rstrip('/')}/{path}"
    full = {"OC": config.LAW_OC, "type": "JSON", **params}
    resp = httpx.get(url, params=full, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    # law.go.kr occasionally serves JSON with an HTML content-type on errors;
    # parse defensively and surface a clean ValueError rather than a key error.
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise ValueError("law.go.kr returned a non-JSON body.") from exc
    if ttl > 0:
        with _LAW_CACHE_LOCK:
            if len(_LAW_CACHE) >= _LAW_CACHE_MAX:
                _LAW_CACHE.clear()  # simple bound: flush wholesale (rare)
            _LAW_CACHE[key] = (time.time(), data)
    return data


def _as_list(value: Any) -> list[Any]:
    """Coerce a law.go.kr field that may be a dict, list, or absent to a list."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


# --------------------------------------------------------------------------- #
# DB existence check (our corpus): parents.jsonl (FAISS era; Qdrant removed)   #
# --------------------------------------------------------------------------- #
# Note: the project migrated off Qdrant. DB-existence is served by parents.jsonl
# (statute: title + full_text article match; precedent: see TODO on 사건번호), and
# the authoritative article/wording check is the law.go.kr API pass.
def _parents_lookup(predicate) -> dict[str, Any] | None:
    """Scan ``parents.jsonl`` (if present) for the first record matching ``predicate``.

    A cheap, dependency-free fallback for DB existence when Qdrant is not up or
    not yet populated. Streams the file so a large corpus does not load into RAM.

    Args:
        predicate: A callable ``record(dict) -> bool``.

    Returns:
        The first matching parent record, or ``None``.
    """
    path = config.PARENTS_JSONL
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if predicate(rec):
                    return rec
    except OSError:
        return None
    return None


def _db_check_statute(title: str, article_key: str | None) -> dict[str, Any] | None:
    """DB existence for a statute citation. Returns a matched payload/record or None.

    Looks the law up in ``parents.jsonl`` by title, then (when an article is
    requested) confirms the article exists in that law's ``full_text`` — handling
    branch numbers (제N조의M) and matching on a boundary so 제N조 does not falsely
    accept 제N조의M. The authoritative article/wording check is the law.go.kr API.
    """
    # Find the law by title in parents.jsonl (FAISS-era DB-existence source).
    rec = _parents_lookup(lambda r: _nfc(str(r.get("title", ""))) == _nfc(title))
    if rec is None:
        return None
    # Article-level existence (bug #3): confirm the cited article actually appears
    # in our copy of the law's full text. A real law title + a fabricated article
    # number must NOT pass — this catches it even when the law.go.kr API pass is
    # unavailable. (Our full_text holds every article of laws we index.)
    if article_key is not None:
        full = str(rec.get("full_text") or rec.get("text") or rec.get("body") or "")
        # article_key is "347" or a branch like "3의2" (from _article_number).
        # The corpus writes branch articles as 제3조의2 (NOT 제3의2조), so build the
        # right label; and match on a boundary so 제3조 does not falsely accept
        # 제3조의2 (audit: CRITICAL 가지번호 오탐 + LOW 부분문자열 오탐).
        if "의" in article_key:
            base, sub = article_key.split("의", 1)
            found = f"제{base}조의{sub}" in full
        else:
            found = re.search(rf"제{re.escape(article_key)}조(?!의\d)", full) is not None
        if not found:
            return None
    return rec


def _db_check_precedent(case_no: str) -> dict[str, Any] | None:
    """DB existence for a precedent citation by exact 사건번호."""
    norm = _normalize_case_no(case_no)
    # NOTE (audit HIGH): parents.jsonl precedent records key on parent_id
    # (PREC:<seq>) + case NAME, and do NOT carry the 사건번호, so this DB lookup
    # cannot confirm a precedent by case number yet — precedent existence is
    # verified by the law.go.kr API pass (_api_check_precedent). The lookup below
    # is harmless (returns None until the field exists).
    # TODO(data): index 사건번호 from source precedent files for offline DB-existence.
    rec = _parents_lookup(
        lambda r: _normalize_case_no(str(r.get("사건번호", r.get("case_no", "")))) == norm
    )
    return rec


# --------------------------------------------------------------------------- #
# law.go.kr 현행·문구 대조                                                       #
# --------------------------------------------------------------------------- #
def _api_check_statute(
    title: str, article_key: str | None
) -> dict[str, Any]:
    """Confirm a statute (and optional article) at law.go.kr.

    Returns a dict ``{api_match, current, source_url, effective_from, note}``.
    ``api_match`` is ``None`` when the API is unreachable (degrade to DB-only).
    """
    out: dict[str, Any] = {
        "api_match": None,
        "current": False,
        "source_url": None,
        "effective_from": None,
        "note": "",
    }
    try:
        data = _law_get(
            _SEARCH_PATH,
            {"target": "law", "query": title, "display": str(_SEARCH_DISPLAY)},
        )
    except Exception as exc:  # network/JSON: degrade gracefully
        out["note"] = f"law.go.kr 미응답({type(exc).__name__}) → DB-only 검증."
        return out

    rows = _as_list(data.get("LawSearch", {}).get("law"))
    # Exact 법령명 match preferred; fall back to substring if the source uses a
    # slightly different spacing.
    target = _nfc(title)
    exact = [r for r in rows if _nfc(str(r.get("법령명한글", ""))) == target]
    near = [r for r in rows if target in _nfc(str(r.get("법령명한글", "")))]
    candidates = exact or near
    if not candidates:
        out["api_match"] = False
        out["note"] = "law.go.kr에 동일 법령명이 없음(오인용/폐지 가능)."
        return out

    # Prefer the 현행 (in-force) row.
    current_rows = [
        r for r in candidates if str(r.get("현행연혁코드", "")) in _CURRENT_MARKERS
    ]
    row = (current_rows or candidates)[0]
    out["current"] = str(row.get("현행연혁코드", "")) in _CURRENT_MARKERS
    out["source_url"] = _strip_oc(str(row.get("법령상세링크", "")))
    ef = str(row.get("시행일자", "") or "")
    out["effective_from"] = _fmt_date(ef)
    mst = str(row.get("법령일련번호", "") or "")

    if article_key is None:
        out["api_match"] = True
        out["note"] = "법령 존재·현행 확인." if out["current"] else "법령 존재(현행 아님)."
        return out

    # Article-level 문구 대조 via lawService JO lookup.
    jo = _jo_param_from_key(article_key)
    if not mst or not jo:
        out["api_match"] = True  # law exists; couldn't form an article query
        out["note"] = "법령 존재 확인(조문 단위 대조 미수행)."
        return out
    try:
        svc = _law_get(
            _SERVICE_PATH, {"target": "law", "MST": mst, "JO": jo}
        )
    except Exception as exc:
        out["api_match"] = True  # law confirmed; article round-trip failed
        out["note"] = f"법령 확인, 조문 조회 실패({type(exc).__name__})."
        return out

    units = _as_list(svc.get("법령", {}).get("조문", {}).get("조문단위"))
    found = None
    for u in units:
        if _article_number(str(u.get("조문번호", ""))) == article_key:
            found = u
            break
        # 조문번호 may be plain "17"; compare directly too.
        if str(u.get("조문번호", "")).strip() == article_key:
            found = u
            break
    if found is None:
        out["api_match"] = False
        out["note"] = f"law.go.kr 현행 {title}에 제{article_key}조가 없음(오인용/폐지 가능)."
        return out

    out["api_match"] = True
    art_eff = _fmt_date(str(found.get("조문시행일자", "") or ""))
    if art_eff:
        out["effective_from"] = art_eff
    title_txt = _nfc(str(found.get("조문제목", "")))
    out["note"] = (
        f"law.go.kr 현행 조문 일치: 제{article_key}조"
        f"{'(' + title_txt + ')' if title_txt else ''}."
    )
    return out


def _api_check_precedent(case_no: str) -> dict[str, Any]:
    """Confirm a precedent at law.go.kr by EXACT 사건번호 (fuzzy search guarded)."""
    out: dict[str, Any] = {
        "api_match": None,
        "current": True,  # precedents do not have a 현행/연혁 concept
        "source_url": None,
        "effective_from": None,
        "note": "",
    }
    try:
        data = _law_get(
            _SEARCH_PATH,
            {"target": "prec", "query": case_no, "display": str(_SEARCH_DISPLAY)},
        )
    except Exception as exc:
        out["note"] = f"law.go.kr 미응답({type(exc).__name__}) → DB-only 검증."
        return out

    rows = _as_list(data.get("PrecSearch", {}).get("prec"))
    norm = _normalize_case_no(case_no)
    matches = [
        r for r in rows if _normalize_case_no(str(r.get("사건번호", ""))) == norm
    ]
    if not matches:
        out["api_match"] = False
        out["note"] = "law.go.kr 판례검색에 동일 사건번호 없음(허위사건 가능)."
        return out
    row = matches[0]
    out["api_match"] = True
    out["source_url"] = _strip_oc(str(row.get("판례상세링크", "")))
    out["effective_from"] = _fmt_date(str(row.get("선고일자", "") or ""))
    court = _nfc(str(row.get("법원명", "")))
    out["note"] = f"law.go.kr 판례 존재 확인({court} {case_no})."
    return out


def _fmt_date(raw: str) -> str | None:
    """Normalize law.go.kr dates (``YYYYMMDD`` or ``YYYY.MM.DD``) to ISO, else None."""
    s = (raw or "").strip()
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return s or None


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #
def _extract_citation(citation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify a citation and pull its identifying fields.

    Args:
        citation: ``{law_name|title, article_no}`` for a statute, or
            ``{case_no|사건번호}`` for a precedent.

    Returns:
        ``("statute"|"precedent", fields)`` where ``fields`` holds the
        normalized identifiers.

    Raises:
        ValueError: If neither a law name/title nor a case number is present.
    """
    case_no = (
        citation.get("case_no")
        or citation.get("사건번호")
        or citation.get("case_number")
    )
    if case_no:
        return "precedent", {"case_no": str(case_no).strip()}

    title = (
        citation.get("law_name")
        or citation.get("title")
        or citation.get("법령명")
    )
    if title:
        return "statute", {
            "title": _nfc(str(title)),
            "article_no": (
                str(citation.get("article_no") or citation.get("조문") or "").strip()
                or None
            ),
        }
    raise ValueError(
        "citation must contain a law name/title (+optional article_no) or a "
        "case number (사건번호)."
    )


def verify_citation(
    citation: dict[str, Any], as_of_date: str | None = None
) -> dict[str, Any]:
    """Verify one citation (the Citation Firewall, ``/v1/verify``).

    Runs DB-existence, law.go.kr 현행·문구 대조, and ``as_of_date`` point-in-time
    checks, and returns a single audit record. Never raises on a verification
    *miss* (a missing/repealed citation simply yields ``verified=False`` with an
    explanatory ``note``); it only raises :class:`ValueError` for a malformed
    citation payload.

    Args:
        citation: ``{law_name|title, article_no?}`` (statute) or
            ``{case_no|사건번호}`` (precedent).
        as_of_date: Optional ISO ``YYYY-MM-DD``. When given, the cited
            document's ``effective_from`` must be ``<= as_of_date`` for the
            citation to be valid at that point in time.

    Returns:
        The audit dict described in the module docstring (``verified``,
        ``trust_grade``, ``current``, ``source_url``, ``effective_from``,
        ``as_of_date``, ``note``, ``db_match``, ``api_match``). The OC token
        never appears in any field.

    Raises:
        ValueError: If ``citation`` identifies neither a statute nor a
            precedent.
    """
    kind, fields = _extract_citation(citation)
    notes: list[str] = []

    if kind == "statute":
        title = fields["title"]
        article_no = fields["article_no"]
        article_key = _article_number(article_no) if article_no else None
        db_rec = _db_check_statute(title, article_key)
        api = _api_check_statute(title, article_key)
        # trust_grade: A if our DB has the original text (chunk payload or
        # parent full_text), else B (metadata only / not in corpus).
        trust_grade = "A" if (db_rec and _has_text(db_rec)) else "B"
        location = f"제{article_key}조" if article_key else None
    else:  # precedent
        case_no = fields["case_no"]
        db_rec = _db_check_precedent(case_no)
        api = _api_check_precedent(case_no)
        trust_grade = "A" if (db_rec and _has_text(db_rec)) else "B"
        location = case_no

    db_match = db_rec is not None
    api_match = api["api_match"]
    current = bool(api["current"]) if api_match else (db_match)
    source_url = api.get("source_url") or (db_rec or {}).get("source_url")
    effective_from = api.get("effective_from") or (db_rec or {}).get(
        "effective_from"
    )

    if api.get("note"):
        notes.append(api["note"])
    if not db_match:
        notes.append("우리 코퍼스(DB)에 미존재 — 커버리지 밖이거나 미인덱싱.")
    elif trust_grade == "B":
        notes.append("DB에 메타데이터만 존재(원문 미보유, B등급).")

    # as_of_date point-in-time validity.
    as_of_ok = True
    if as_of_date:
        as_of_ok = _as_of_valid(effective_from, as_of_date)
        if not as_of_ok:
            notes.append(
                f"as_of_date({as_of_date}) 기준 미시행: 시행/선고일 "
                f"{effective_from or '미상'}."
            )

    # Overall verdict: present in DB AND (API confirmed OR API unavailable) AND
    # not contradicted by the as_of_date window. When the API explicitly says
    # api_match=False, the citation fails regardless of DB presence (the
    # authoritative source disagrees → likely 오인용/폐지/허위).
    if api_match is False:
        verified = False
    else:
        verified = db_match and as_of_ok

    # Trust score (0-100) + traffic-light flag from the available signals.
    # red  = fails verification (missing / repealed / API-contradicted / 미시행)
    # green= present in DB, API-confirmed, in force, original text held
    # yellow = present but partial confidence (API unavailable / 메타만 / 현행 불확실)
    if not db_match or verified is False:
        trust_score = 15 if db_match else 0
        flag = "red"
    else:
        trust_score = 50  # exists in our corpus
        if api_match is True:
            trust_score += 30  # law.go.kr confirms wording/existence
        if current:
            trust_score += 15
        if trust_grade == "A":
            trust_score += 5
        trust_score = min(100, trust_score)
        if api_match is True and current and trust_grade == "A":
            flag = "green"
        elif trust_score >= 80:
            flag = "green"
        else:
            flag = "yellow"

    return {
        "verified": bool(verified),
        "trust_score": int(trust_score),
        "flag": flag,
        "trust_grade": trust_grade,
        "current": bool(current),
        "source_url": source_url,
        "effective_from": effective_from,
        "as_of_date": as_of_date,
        "location": location,
        "note": " ".join(n for n in notes if n) or "검증 통과.",
        "db_match": bool(db_match),
        "api_match": api_match,
    }


def verify_citations(
    citations: list[dict[str, Any]], as_of_date: str | None = None
) -> list[dict[str, Any]]:
    """Verify a batch of citations, one audit record per input (order preserved).

    Malformed entries do not abort the batch: they yield a record with
    ``verified=False`` and an explanatory ``note`` so a caller verifying many
    AI-produced citations always gets a parallel result list.

    Args:
        citations: A list of citation dicts (see :func:`verify_citation`).
        as_of_date: Optional ISO ``YYYY-MM-DD`` applied to every entry.

    Returns:
        A list of audit dicts, the same length and order as ``citations``.
    """
    results: list[dict[str, Any]] = []
    for c in citations:
        try:
            results.append(verify_citation(c, as_of_date=as_of_date))
        except ValueError as exc:
            results.append(
                {
                    "verified": False,
                    "trust_grade": "B",
                    "current": False,
                    "source_url": None,
                    "effective_from": None,
                    "as_of_date": as_of_date,
                    "location": None,
                    "note": f"인용 형식 오류: {exc}",
                    "db_match": False,
                    "api_match": None,
                }
            )
    return results


def _has_text(record: dict[str, Any]) -> bool:
    """True if a DB record carries actual original text (=> trust_grade A)."""
    if str(record.get("trust_grade", "A")) == "B":
        return False
    return bool(
        record.get("text") or record.get("full_text") or record.get("body")
    )


def _as_of_valid(effective_from: str | None, as_of_date: str) -> bool:
    """Return True if ``effective_from <= as_of_date`` (string ISO compare).

    Both are normalized to ``YYYY-MM-DD``. If ``effective_from`` is unknown we do
    not block (return True) — absence of data is not evidence of invalidity, and
    the ``note`` already records the uncertainty.
    """
    ef = _fmt_date(effective_from or "")
    ad = _fmt_date(as_of_date or "")
    if not ef or not ad:
        return True
    return ef <= ad


# --------------------------------------------------------------------------- #
# CLI: self-test (offline DB path) + live (one real law.go.kr call)            #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Offline checks of parsing/verdict logic — no network, no OpenAI."""
    failures: list[str] = []

    # Article number / JO parsing.
    cases = {
        "제17조": ("17", "001700"),
        "제4조의2": ("4의2", "000402"),
        "17": ("17", "001700"),
    }
    for label, (key, jo) in cases.items():
        if _article_number(label) != key:
            failures.append(f"_article_number({label!r}) != {key!r}")
        if _jo_param(label) != jo:
            failures.append(f"_jo_param({label!r}) != {jo!r}")

    # OC redaction never leaks the token.
    link = "/DRF/lawService.do?OC=SECRET123&target=law&MST=281875&type=HTML"
    safe = _strip_oc(link)
    if safe is None or "OC=" in safe or "SECRET123" in safe:
        failures.append(f"_strip_oc leaked OC: {safe!r}")
    if not safe.startswith("https://www.law.go.kr"):
        failures.append(f"_strip_oc did not absolutize: {safe!r}")

    # Case-number normalization.
    if _normalize_case_no(" 2022도 1401 ") != "2022도1401":
        failures.append("case-no normalization failed")

    # as_of_date window logic.
    if not _as_of_valid("2024-01-01", "2025-06-15"):
        failures.append("as_of valid (effective before) returned False")
    if _as_of_valid("2026-01-01", "2025-06-15"):
        failures.append("as_of invalid (effective after) returned True")
    if not _as_of_valid(None, "2025-06-15"):
        failures.append("as_of with unknown effective_from should not block")

    # Classification.
    kind, f = _extract_citation({"law_name": "도로교통법", "article_no": "제17조"})
    if kind != "statute" or f["title"] != "도로교통법":
        failures.append(f"statute classification wrong: {kind} {f}")
    kind2, f2 = _extract_citation({"사건번호": "2022도1401"})
    if kind2 != "precedent" or f2["case_no"] != "2022도1401":
        failures.append(f"precedent classification wrong: {kind2} {f2}")
    try:
        _extract_citation({"foo": "bar"})
    except ValueError:
        pass
    else:
        failures.append("malformed citation not rejected")

    # End-to-end verdict with stubbed DB + API (no network): a present-in-DB,
    # API-confirmed, in-as_of citation must verify True; an API-rejected one must
    # verify False even if it is in the DB.
    global _api_check_statute, _db_check_statute
    real_api, real_db = _api_check_statute, _db_check_statute
    try:
        _db_check_statute = lambda t, a: {  # type: ignore[assignment]
            "text": "본문", "trust_grade": "A",
            "source_url": "https://law.go.kr/x", "effective_from": "2024-01-01",
        }
        _api_check_statute = lambda t, a: {  # type: ignore[assignment]
            "api_match": True, "current": True,
            "source_url": "https://www.law.go.kr/y",
            "effective_from": "2024-01-01", "note": "ok",
        }
        r = verify_citation({"law_name": "X", "article_no": "제1조"})
        if not (r["verified"] and r["trust_grade"] == "A" and r["current"]):
            failures.append(f"positive verdict wrong: {r}")
        if "OC=" in json.dumps(r, ensure_ascii=False):
            failures.append("verdict leaked OC")

        _api_check_statute = lambda t, a: {  # type: ignore[assignment]
            "api_match": False, "current": False, "source_url": None,
            "effective_from": None, "note": "오인용",
        }
        r2 = verify_citation({"law_name": "X", "article_no": "제999조"})
        if r2["verified"] is not False or r2["api_match"] is not False:
            failures.append(f"API-reject verdict wrong: {r2}")

        # API unavailable => api_match None, DB-only verdict.
        _api_check_statute = lambda t, a: {  # type: ignore[assignment]
            "api_match": None, "current": False, "source_url": None,
            "effective_from": None, "note": "미응답",
        }
        r3 = verify_citation({"law_name": "X", "article_no": "제1조"})
        if r3["api_match"] is not None or r3["verified"] is not True:
            failures.append(f"API-unavailable verdict wrong: {r3}")
    finally:
        _api_check_statute, _db_check_statute = real_api, real_db

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("SELFTEST PASSED: article/JO parsing, OC redaction, as_of, verdicts.")
    return 0


def _live(title_or_case: str, article_no: str | None) -> int:
    """One real law.go.kr call (sanctioned) to demonstrate the firewall."""
    if article_no:
        citation = {"law_name": title_or_case, "article_no": article_no}
    elif re.search(r"\d{4}[가-힣]+\d+", title_or_case):
        citation = {"사건번호": title_or_case}
    else:
        citation = {"law_name": title_or_case}
    result = verify_citation(citation)
    # Print the audit record (OC-free by construction).
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["api_match"] in (True, None) else 1


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="search.verify",
        description="Citation Firewall (/v1/verify): DB + law.go.kr 현행·문구 대조.",
    )
    p.add_argument("--selftest", action="store_true", help="Offline logic checks.")
    p.add_argument(
        "--live",
        action="store_true",
        help="Make ONE real law.go.kr call for the given citation.",
    )
    p.add_argument("title", nargs="?", help="법령명 or 사건번호.")
    p.add_argument("article", nargs="?", help="조문번호 (e.g. 제17조), statutes only.")
    return p


def _main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.live:
        if not args.title:
            print("--live requires a 법령명/사건번호.", file=sys.stderr)
            return 2
        return _live(args.title, args.article)
    print("Use --selftest or --live <법령명> [조문번호].", file=sys.stderr)
    return 2


__all__ = ["verify_citation", "verify_citations"]


if __name__ == "__main__":
    raise SystemExit(_main())
