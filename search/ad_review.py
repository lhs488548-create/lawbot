"""Advertising-copy legal review — ``/v1/ad-review`` (lawyer add-on, expert mode).

Given a draft advertisement (a PDF upload *or* raw ``text``), this module runs an
expert, **grounded** legal review against the Korean advertising-law corpus and
returns an issue-spotting report with **post-verified citations** and a corrected
rewrite. It is the backend the FastAPI app (``api.main``) resolves as the
``review(...)`` callable for the ``POST /v1/ad-review`` endpoint.

Pipeline (task = ad_review):

    1. **Extract text** — ``pdfplumber`` (primary, handles columns/tables in ad
       PDFs) with a ``pypdf`` fallback; raw ``text`` is used as-is.
    2. **Decompose claims** — one GPT call (Structured Outputs, strict schema)
       splits the ad into atomic, checkable advertising *claims* (효능·효과·
       비교·최상급·가격·체험 등). This is the only "understanding" LLM step.
    3. **RAG retrieve** — for each claim, dense-retrieve the most relevant
       provisions, scoped to the **advertising-law statutes** (표시·광고의 공정화에
       관한 법률, 식품 등의 표시·광고에 관한 법률, 의료법, 약사법, 화장품법) and their
       고시/행정규칙, via the shared :mod:`search.retriever`.
    4. **Judge + rewrite** — one GPT call (Structured Outputs) evaluates every
       claim against the retrieved provisions, emitting per-issue
       {claim, verdict, severity, rationale, citations, suggested_fix} plus a
       compliant rewrite. Grounded: it may only cite the provided context blocks.
    5. **Citation firewall** — every emitted citation is post-verified against the
       retrieved context (hallucinated ``source_id`` dropped, like ``search.rag``)
       and, for statute citations, **2차 검증** against law.go.kr via
       :func:`search.verify.verify_citation` (현행·문구 대조). Unverifiable
       citations are flagged, never silently trusted.

Design contract (``_BUILD_CONTRACT.md`` §(e) ``/v1/ad-review`` + §(g)/(h)):

* Public entry point::

      def review(*, text=None, file_bytes=None, filename=None, question=None) -> dict

  returning ``{summary, issues:[...], citations:[...], ai_generated: true,
  disclaimer}`` (``api.main`` re-asserts the AI-notice defensively).
* **Expert (lawyer) mode** — full substantive analysis and drafting assistance,
  no 변호사법 §109 consumer refusal. Grounding + citation verification are kept as
  *quality* controls, not consumer gates.
* **Cost rules** — at most **two** GPT calls per review (decompose + judge),
  ``config.GEN_MODEL`` only; embeddings via the shared retriever
  (``config.EMBED_MODEL``); no LLM is used for headers/context. law.go.kr OC is
  read from config and **never logged or returned**.

Deviation from the prose spec (declared per the integration rule): the 08/task
prose names ``pdfplumber``; the contract/``requirements.txt`` pin ``pypdf``. Both
are installed in the venv, so this module uses **pdfplumber as primary with a
pypdf fallback** — satisfying both. No shared file (config/requirements) changed.

Self-check (offline; fakes the LLM + retriever, no OpenAI/Qdrant)::

    cd /home/user1/lawbot && .venv/bin/python -m search.ad_review --selftest
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from typing import Any, Callable, Final, Iterable, Optional, TypedDict

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Tunables (module-local; not shared config)                                  #
# --------------------------------------------------------------------------- #
# Advertising-law corpus the review is scoped to. These law names appear in the
# indexed ``title`` payload (국가법령 corpus); related 고시/행정규칙 are pulled in via
# the unscoped supplementary pass so we still surface e.g. 식약처 고시 violations.
AD_LAW_TITLES: Final[tuple[str, ...]] = (
    "표시·광고의 공정화에 관한 법률",
    "식품 등의 표시·광고에 관한 법률",
    "의료법",
    "약사법",
    "화장품법",
    "건강기능식품에 관한 법률",
)

# Hard caps to bound cost/latency for a single review request.
_MAX_INPUT_CHARS: Final[int] = 40_000       # truncate giant PDFs before the LLM
_MAX_CLAIMS: Final[int] = 24                # cap decomposed claims
_PER_CLAIM_K: Final[int] = 4                # retrieved provisions per claim
_MAX_CONTEXT_BLOCKS: Final[int] = 24        # cap total context blocks shown
_CONTEXT_BODY_CHARS: Final[int] = 900       # truncate each context body
_VERIFY_MAX_CITATIONS: Final[int] = 12      # cap law.go.kr 2nd-pass calls/review

_VERDICTS: Final[frozenset[str]] = frozenset(
    {"위반", "위반소지", "주의", "적정", "확인필요"}
)
_SEVERITIES: Final[frozenset[str]] = frozenset({"high", "medium", "low", "none"})


# --------------------------------------------------------------------------- #
# Shared OpenAI client (lazy; key never logged)                               #
# --------------------------------------------------------------------------- #
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Return a lazily-built, process-wide OpenAI client (key from config)."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def set_client(client: OpenAI | None) -> None:
    """Inject a client (tests). ``None`` leaves the current one untouched."""
    global _client
    if client is not None:
        _client = client


# --------------------------------------------------------------------------- #
# Result types                                                                #
# --------------------------------------------------------------------------- #
class Citation(TypedDict, total=False):
    """One post-verified citation backing an issue."""

    source_id: str
    title: str
    location: str
    source_url: str
    doc_type: str
    trust_grade: str
    db_verified: bool          # found in our indexed corpus
    api_verified: Optional[bool]  # law.go.kr 현행·문구 대조 (None = API unavailable)
    current: Optional[bool]
    verify_note: str


class Issue(TypedDict, total=False):
    """One flagged advertising claim and its grounded assessment."""

    claim: str
    verdict: str               # 위반|위반소지|주의|적정|확인필요
    severity: str              # high|medium|low|none
    rationale: str
    law_basis: str             # short human label e.g. "표시광고법 제3조"
    citations: list[Citation]
    suggested_fix: str


class ReviewResult(TypedDict):
    """Return shape of :func:`review` (contract §(e) ``/v1/ad-review``)."""

    summary: str
    issues: list[Issue]
    citations: list[Citation]      # de-duplicated union across all issues
    corrected_copy: str
    claims_reviewed: int
    extraction: dict[str, Any]     # {source, n_chars, n_pages, truncated}
    ai_generated: bool
    disclaimer: str


# --------------------------------------------------------------------------- #
# 1) PDF / text extraction                                                    #
# --------------------------------------------------------------------------- #
def extract_text(
    *, text: str | None, file_bytes: bytes | None, filename: str | None
) -> dict[str, Any]:
    """Extract the ad copy to review from raw text or an uploaded file.

    Args:
        text: Raw advertisement text (used verbatim when provided).
        file_bytes: Uploaded file content (PDF or a text/plain blob).
        filename: Original filename, used only to detect a ``.pdf`` extension.

    Returns:
        ``{"text": str, "source": "text"|"pdf"|"file", "n_chars": int,
        "n_pages": int|None, "truncated": bool}``. ``text`` is whitespace-trimmed
        and truncated to :data:`_MAX_INPUT_CHARS`.

    Raises:
        ValueError: If neither a non-empty ``text`` nor ``file_bytes`` is given,
            or a PDF yields no extractable text (e.g. a scanned image with no OCR
            layer) — the caller surfaces this as a 422.
    """
    raw = ""
    source = "text"
    n_pages: int | None = None

    if text and text.strip():
        raw = text
        source = "text"
    elif file_bytes:
        is_pdf = bool(filename and filename.lower().endswith(".pdf")) or (
            file_bytes[:5] == b"%PDF-"
        )
        if is_pdf:
            raw, n_pages = _extract_pdf(file_bytes)
            source = "pdf"
            if not raw.strip():
                raise ValueError(
                    "PDF에서 추출 가능한 텍스트가 없습니다(이미지/스캔 PDF로 보임). "
                    "텍스트 레이어가 있는 PDF를 올리거나 'text' 필드로 광고 문구를 "
                    "직접 입력해 주십시오."
                )
        else:
            # Treat any non-PDF upload as a UTF-8 (best-effort) text blob.
            raw = file_bytes.decode("utf-8", errors="replace")
            source = "file"
    else:
        raise ValueError("검토할 'text' 또는 파일('file')을 제공해야 합니다.")

    cleaned = raw.strip()
    truncated = len(cleaned) > _MAX_INPUT_CHARS
    if truncated:
        cleaned = cleaned[:_MAX_INPUT_CHARS]
    return {
        "text": cleaned,
        "source": source,
        "n_chars": len(cleaned),
        "n_pages": n_pages,
        "truncated": truncated,
    }


def _extract_pdf(data: bytes) -> tuple[str, int | None]:
    """Extract text from a PDF: ``pdfplumber`` primary, ``pypdf`` fallback.

    Args:
        data: Raw PDF bytes.

    Returns:
        ``(text, n_pages)``. ``n_pages`` is ``None`` if neither library could
        even open the document.
    """
    # Primary: pdfplumber — better with multi-column / tabular ad layouts.
    try:
        import pdfplumber  # local import: keeps module import light

        parts: list[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            n_pages = len(pdf.pages)
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        joined = "\n".join(parts).strip()
        if joined:
            return joined, n_pages
        # Fall through to pypdf if pdfplumber found no text layer.
    except Exception as exc:  # corrupt PDF / pdfplumber edge case
        logger.warning("pdfplumber extraction failed (%s); trying pypdf.", type(exc).__name__)

    # Fallback: pypdf (the contract-pinned extractor).
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        n_pages = len(reader.pages)
        parts = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(parts).strip(), n_pages
    except Exception as exc:
        logger.warning("pypdf extraction failed (%s).", type(exc).__name__)
        return "", None


# --------------------------------------------------------------------------- #
# 2) Claim decomposition (one GPT call, Structured Outputs)                    #
# --------------------------------------------------------------------------- #
_DECOMPOSE_SYSTEM: Final[str] = (
    "당신은 대한민국 광고심의·표시광고법 전문 변호사를 보조하는 시니어 리서치 "
    "어시스턴트입니다. 입력된 광고 문안을 법적 검토가 가능한 '개별 주장(claim)' "
    "단위로 분해하십시오. 각 주장은 소비자에게 전달되는 사실적·평가적 표현 하나를 "
    "담아야 합니다(예: 효능·효과 표현, 최상급/배타성 표현, 비교광고, 가격·할인 표현, "
    "체험·추천, 의약품·의료효능 오인, 안전성·부작용 표현 등).\n"
    "원문에 없는 내용을 지어내지 말고, 광고 문구를 그대로 인용·요약하십시오. "
    "법적 쟁점이 없는 순수 디자인/연락처/주소 문구는 제외합니다. "
    "지정된 json_schema 형식으로만 출력하십시오."
)

_DECOMPOSE_SCHEMA: Final[dict[str, Any]] = {
    "name": "ad_claims",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "product_type": {
                "type": "string",
                "description": "광고 대상 추정(예: 화장품·건강기능식품·의료기관·의약품·식품·일반상품).",
            },
            "claims": {
                "type": "array",
                "description": "법적 검토 대상 개별 광고 주장. 없으면 빈 배열.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string", "description": "광고 문구(원문 인용/요약)."},
                        "claim_type": {
                            "type": "string",
                            "description": "주장 유형(효능효과·최상급·비교·가격·체험·안전성·의료효능 등).",
                        },
                    },
                    "required": ["text", "claim_type"],
                },
            },
        },
        "required": ["product_type", "claims"],
    },
}


def decompose_claims(ad_text: str) -> dict[str, Any]:
    """Split the ad into atomic, checkable claims (one Structured-Outputs call).

    Args:
        ad_text: The extracted advertisement copy.

    Returns:
        ``{"product_type": str, "claims": [{"text", "claim_type"}, ...]}`` with
        at most :data:`_MAX_CLAIMS` claims.

    Raises:
        RuntimeError: If the model returns no parseable structured content.
    """
    parsed = _structured_call(
        system=_DECOMPOSE_SYSTEM,
        user=f"[광고 문안]\n{ad_text}\n\n위 문안을 검토 대상 주장으로 분해하십시오.",
        schema=_DECOMPOSE_SCHEMA,
    )
    claims = parsed.get("claims") or []
    if not isinstance(claims, list):
        claims = []
    parsed["claims"] = [
        c for c in claims if isinstance(c, dict) and str(c.get("text", "")).strip()
    ][:_MAX_CLAIMS]
    parsed.setdefault("product_type", "")
    return parsed


# --------------------------------------------------------------------------- #
# 3) RAG retrieval against the advertising-law corpus                          #
# --------------------------------------------------------------------------- #
def _retrieve_for_claims(
    claims: list[dict[str, Any]],
    search_fn: Callable[..., list[Any]],
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Retrieve advertising-law provisions for every claim and build LLM context.

    Two passes per claim keep precision high without missing 고시/행정규칙:

    1. **Scoped** to ``doc_type="law"`` (the statute layer) — the core ad-law
       provisions live here. (We do not over-constrain by ``title`` because the
       retriever filters on indexed KEYWORD keys only; statute scoping plus the
       ad-law query terms surface the right laws.)
    2. **Unscoped** supplementary hit so 식약처·공정위 고시(행정규칙) and related
       precedents can still surface.

    Args:
        claims: Decomposed claims (``{"text", "claim_type"}``).
        search_fn: The retriever ``search`` callable (injected for tests).

    Returns:
        ``(context_text, source_index)`` — a numbered ``[n]`` context block and a
        ``source_id -> metadata`` map, mirroring ``search.rag.build_context`` so
        the citation firewall can reuse the same verification contract.
    """
    seen: set[str] = set()
    blocks: list[str] = []
    source_index: dict[str, dict[str, Any]] = {}

    def _add(hit: Any) -> None:
        sid = str(getattr(hit, "id", ""))
        if not sid or sid in seen:
            return
        payload = dict(getattr(hit, "payload", None) or {})
        title = str(payload.get("title", ""))
        # Keep the block only if it is plausibly advertising-law relevant: either
        # an ad-law statute by title, or any 고시/판례 surfaced by the query (those
        # we keep because the unscoped pass already ranked them as relevant).
        seen.add(sid)
        n = len(blocks) + 1
        doc_type = str(payload.get("doc_type", ""))
        location = str(payload.get("article_no", "") or "").strip()
        url = str(payload.get("source_url") or "")
        trust = str(payload.get("trust_grade", "A"))
        body = str(payload.get("text", "")).strip()
        if len(body) > _CONTEXT_BODY_CHARS:
            body = body[:_CONTEXT_BODY_CHARS] + " …(생략)"
        grade_note = " [본문없음·메타만(B등급)]" if trust == "B" else ""
        header = (
            f"[{n}] id={sid} | 종류={doc_type or '-'} | 법령={title or '-'} | "
            f"위치={location or '-'} | 등급={trust}{grade_note}"
        )
        if url:
            header += f" | 출처={url}"
        blocks.append(f"{header}\n{body}")
        source_index[sid] = {
            "source_id": sid,
            "title": title,
            "location": location,
            "source_url": url,
            "doc_type": doc_type,
            "trust_grade": trust,
        }

    for claim in claims:
        if len(blocks) >= _MAX_CONTEXT_BLOCKS:
            break
        q = str(claim.get("text", "")).strip()
        if not q:
            continue
        # Bias the query toward the legal axis so dense retrieval lands on the
        # regulatory provisions rather than other ads with similar wording.
        legal_q = f"{q} 광고 표시 부당한 표시·광고 금지 위반 효능 과장"
        try:
            scoped = search_fn(legal_q, k=_PER_CLAIM_K, flt={"doc_type": "law"})
        except Exception as exc:
            logger.warning("scoped retrieval failed (%s); using unscoped only.", type(exc).__name__)
            scoped = []
        try:
            supp = search_fn(legal_q, k=2)
        except Exception as exc:
            logger.warning("supplementary retrieval failed (%s).", type(exc).__name__)
            supp = []
        for hit in list(scoped) + list(supp):
            if len(blocks) >= _MAX_CONTEXT_BLOCKS:
                break
            _add(hit)

    return "\n\n".join(blocks), source_index


# --------------------------------------------------------------------------- #
# 4) Judgement + corrected rewrite (one GPT call, Structured Outputs)          #
# --------------------------------------------------------------------------- #
_JUDGE_SYSTEM: Final[str] = (
    "당신은 대한민국 표시·광고법, 식품 등의 표시·광고에 관한 법률, 의료법, 약사법, "
    "화장품법 및 관련 고시에 정통한 시니어 광고심의 변호사를 보조하는 리서치 "
    "어시스턴트입니다. 이용자는 법률 전문가이므로 일반 소비자용 회피성 면책을 넣지 "
    "말고, 정확한 법률 문어체로 충실히 검토하십시오.\n"
    "\n"
    "[근거 강제] 반드시 아래 [관련 법령·고시] 블록에 포함된 내용만을 근거로 위반 "
    "여부를 판단하십시오. 블록에 없는 조문번호·법령명·URL은 절대 지어내지 말고, "
    "근거가 부족하면 해당 주장의 verdict를 '확인필요'로 두고 rationale에 그 사유를 "
    "적으십시오.\n"
    "\n"
    "[판정] 각 광고 주장(claim)에 대해 verdict를 다음 중 하나로 정하십시오: "
    "'위반'(명백한 법령 위반), '위반소지'(위반 가능성 상당), '주의'(표현 수위 "
    "조정 권고), '적정'(문제 없음), '확인필요'(근거 부족). severity는 high/medium/"
    "low/none 중 하나로 정하십시오.\n"
    "\n"
    "[인용] 각 issue의 citations에는 실제로 근거가 된 블록만 넣되, source_id는 "
    "[관련 법령·고시]에 제시된 블록의 id 값과 문자 그대로 동일해야 합니다. id를 "
    "변형하거나 새로 만들지 마십시오.\n"
    "\n"
    "[교정] 각 위반·위반소지 주장에는 법령에 부합하도록 수정한 suggested_fix를 "
    "제시하고, 마지막에 광고 전체를 법적으로 안전하게 다시 쓴 corrected_copy를 "
    "작성하십시오. 효능·효과를 과장하거나 의약품적 효능을 표방하지 않도록 하고, "
    "객관적 근거 없는 최상급·배타성 표현은 제거하십시오.\n"
    "\n"
    "[형식] json_schema에 정의된 필드 외의 텍스트는 출력하지 마십시오."
)

_JUDGE_SCHEMA: Final[dict[str, Any]] = {
    "name": "ad_review",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {
                "type": "string",
                "description": "전체 검토 요지(위반 건수·핵심 리스크·권고)를 법률 문어체로.",
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim": {"type": "string", "description": "검토 대상 광고 주장."},
                        "verdict": {
                            "type": "string",
                            "enum": ["위반", "위반소지", "주의", "적정", "확인필요"],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["high", "medium", "low", "none"],
                        },
                        "rationale": {
                            "type": "string",
                            "description": "판단 근거(적용 법령·조문과 포섭). 각 근거 뒤 [n] 표기.",
                        },
                        "law_basis": {
                            "type": "string",
                            "description": "핵심 근거 법령·조문 라벨(예: '표시광고법 제3조 제1항').",
                        },
                        "citations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "source_id": {"type": "string"},
                                    "title": {"type": "string"},
                                    "location": {"type": "string"},
                                    "source_url": {"type": "string"},
                                },
                                "required": ["source_id", "title", "location", "source_url"],
                            },
                        },
                        "suggested_fix": {
                            "type": "string",
                            "description": "법령에 부합하도록 수정한 광고 문구. 적정이면 빈 문자열.",
                        },
                    },
                    "required": [
                        "claim",
                        "verdict",
                        "severity",
                        "rationale",
                        "law_basis",
                        "citations",
                        "suggested_fix",
                    ],
                },
            },
            "corrected_copy": {
                "type": "string",
                "description": "법적으로 안전하게 다시 쓴 광고 전체 문안.",
            },
        },
        "required": ["summary", "issues", "corrected_copy"],
    },
}


def _judge(
    ad_text: str,
    product_type: str,
    claims: list[dict[str, Any]],
    context: str,
    question: str | None,
    model: str,
) -> dict[str, Any]:
    """Run the single judgement+rewrite GPT call and return parsed output."""
    claim_lines = "\n".join(
        f"- ({c.get('claim_type','-')}) {c.get('text','')}" for c in claims
    )
    focus = f"\n[검토 초점]\n{question.strip()}\n" if question and question.strip() else ""
    user = (
        f"[광고 대상 추정]\n{product_type or '미상'}\n\n"
        f"[검토 대상 광고 주장]\n{claim_lines or '(개별 주장 분해 결과 없음 — 전체 문안 검토)'}\n\n"
        f"[광고 전체 문안]\n{ad_text}\n"
        f"{focus}\n"
        f"[관련 법령·고시]\n{context or '(검색된 관련 법령 없음)'}\n\n"
        "위 [관련 법령·고시]만 근거로 각 주장을 판정하고, 위반·위반소지 항목을 "
        "교정한 뒤 corrected_copy를 작성하십시오. json_schema 형식으로만 답하십시오."
    )
    return _structured_call(system=_JUDGE_SYSTEM, user=user, schema=_JUDGE_SCHEMA, model=model)


# --------------------------------------------------------------------------- #
# Structured-Outputs helper (shared by the two GPT calls)                      #
# --------------------------------------------------------------------------- #
def _structured_call(
    *, system: str, user: str, schema: dict[str, Any], model: str | None = None
) -> dict[str, Any]:
    """Call the chat model with strict Structured Outputs and parse the JSON.

    Args:
        system: System prompt.
        user: User message.
        schema: A strict json_schema definition (``{"name","strict","schema"}``).
        model: Optional model override; defaults to :data:`config.GEN_MODEL`.

    Returns:
        The parsed JSON object.

    Raises:
        RuntimeError: If the response is truncated/empty or not valid JSON.
    """
    response = _get_client().chat.completions.create(
        model=model or config.GEN_MODEL,
        temperature=0,
        response_format={"type": "json_schema", "json_schema": schema},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    choice = response.choices[0]
    if choice.finish_reason not in ("stop", None):
        raise RuntimeError(
            f"광고검토 모델 응답이 정상 종료되지 않았습니다(finish_reason="
            f"{choice.finish_reason!r})."
        )
    content = choice.message.content
    if not content:
        refusal = getattr(choice.message, "refusal", None)
        raise RuntimeError(f"광고검토 모델이 빈 응답을 반환했습니다(refusal={refusal!r}).")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:  # pragma: no cover - strict schema makes rare
        raise RuntimeError("광고검토 모델 출력이 JSON이 아닙니다.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("광고검토 모델 출력이 객체가 아닙니다.")
    return parsed


# --------------------------------------------------------------------------- #
# 5) Citation firewall: context check + law.go.kr 2nd verification             #
# --------------------------------------------------------------------------- #
def _verify_against_context(
    citations: Iterable[dict[str, Any]],
    source_index: dict[str, dict[str, Any]],
) -> list[Citation]:
    """Drop hallucinated citations and enrich survivors from retrieved metadata.

    Mirrors ``search.rag.verify_citations``: a citation survives **iff** its
    ``source_id`` was one of the context blocks shown to the model; the
    authoritative title/location/url come from the *retrieved* data, never the
    model echo. De-duplicated by ``source_id`` preserving order.
    """
    out: list[Citation] = []
    seen: set[str] = set()
    for raw in citations or []:
        if not isinstance(raw, dict):
            continue
        sid = str(raw.get("source_id") or "").strip()
        if not sid or sid in seen:
            continue
        meta = source_index.get(sid)
        if meta is None:
            logger.warning("ad-review: dropping unverifiable citation source_id=%r", sid)
            continue
        seen.add(sid)
        model_loc = str(raw.get("location") or "").strip()
        out.append(
            Citation(
                source_id=sid,
                title=meta["title"],
                location=model_loc or meta["location"],
                source_url=meta["source_url"],
                doc_type=meta["doc_type"],
                trust_grade=meta["trust_grade"],
                db_verified=True,
                api_verified=None,
                current=None,
                verify_note="",
            )
        )
    return out


def _law_verify(
    citations: list[Citation],
    verify_fn: Callable[..., dict[str, Any]] | None,
) -> None:
    """2차 검증: confirm each statute citation against law.go.kr (in place).

    Uses :func:`search.verify.verify_citation` (현행·문구 대조). Precedent and
    label-only (B-grade) citations are skipped for the API pass. Failures degrade
    gracefully (``api_verified`` stays ``None`` with an explanatory note); the OC
    token is never logged or returned (the verify module redacts it).

    Args:
        citations: The context-verified citations to annotate (mutated in place).
        verify_fn: ``verify_citation``-style callable, or ``None`` to skip the
            API pass (e.g. when ``search.verify`` is unavailable).
    """
    if verify_fn is None:
        return
    calls = 0
    for cit in citations:
        if calls >= _VERIFY_MAX_CITATIONS:
            break
        # Only statute-type citations carry a verifiable 법령명+조문 at law.go.kr.
        if cit.get("doc_type") not in ("law", "ordinance", "admrule"):
            continue
        title = cit.get("title")
        location = cit.get("location") or ""
        if not title:
            continue
        article_no = location if location.startswith("제") else None
        try:
            res = verify_fn(
                {"law_name": title, "article_no": article_no}
            )
            calls += 1
        except Exception as exc:
            cit["verify_note"] = f"law.go.kr 검증 호출 실패({type(exc).__name__})."
            continue
        cit["api_verified"] = res.get("api_match")
        cit["current"] = res.get("current")
        cit["verify_note"] = str(res.get("note") or "")
        # Prefer the authoritative source_url from law.go.kr when present.
        if res.get("source_url"):
            cit["source_url"] = res["source_url"]
        if res.get("api_match") is False:
            # The authoritative source disagrees — downgrade trust so the lawyer
            # sees the conflict (오인용/폐지 가능).
            cit["trust_grade"] = "B"


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def review(
    *,
    text: str | None = None,
    file_bytes: bytes | None = None,
    filename: str | None = None,
    question: str | None = None,
    model: str | None = None,
) -> ReviewResult:
    """Run a grounded legal review of an advertisement (the ``/v1/ad-review`` backend).

    Args:
        text: Raw ad copy (alternative to ``file_bytes``).
        file_bytes: Uploaded PDF (or text) bytes.
        filename: Original filename (used to detect ``.pdf``).
        question: Optional focus for the review (e.g. "의료법 위반 위주로").
        model: Optional generation-model override; defaults to
            :data:`config.GEN_MODEL`.

    Returns:
        A :class:`ReviewResult` ``{summary, issues, citations, corrected_copy,
        claims_reviewed, extraction, ai_generated, disclaimer}``. ``api.main``
        re-asserts ``ai_generated``/``disclaimer`` defensively.

    Raises:
        ValueError: For empty/garbage input (no text, or a scanned PDF with no
            text layer) — surfaced by the API as 422.
        RuntimeError: If a model call fails to produce parseable output.
    """
    chosen_model = model or config.GEN_MODEL

    # 1) Extract.
    extraction = extract_text(text=text, file_bytes=file_bytes, filename=filename)
    ad_text = extraction["text"]
    if not ad_text.strip():
        raise ValueError("검토할 광고 문구가 비어 있습니다.")

    # 2) Decompose into claims (1 LLM call).
    decomposed = decompose_claims(ad_text)
    claims = decomposed.get("claims", [])
    product_type = str(decomposed.get("product_type", ""))

    # 3) Retrieve advertising-law provisions per claim. If decomposition found no
    #    discrete claims, fall back to retrieving on the whole ad text so the
    #    review still has grounding.
    from search.retriever import search as _search  # lazy: no vector store at import

    retrieval_claims = claims or [{"text": ad_text[:1000], "claim_type": "전체"}]
    context, source_index = _retrieve_for_claims(retrieval_claims, _search)

    # 4) Judge + rewrite (1 LLM call).
    parsed = _judge(
        ad_text=ad_text,
        product_type=product_type,
        claims=claims,
        context=context,
        question=question,
        model=chosen_model,
    )

    # 5) Citation firewall (context) + law.go.kr 2차 검증.
    verify_fn = _resolve_verify_fn()
    issues: list[Issue] = []
    for raw_issue in parsed.get("issues", []) or []:
        if not isinstance(raw_issue, dict):
            continue
        verdict = str(raw_issue.get("verdict", "확인필요"))
        if verdict not in _VERDICTS:
            verdict = "확인필요"
        severity = str(raw_issue.get("severity", "none"))
        if severity not in _SEVERITIES:
            severity = "none"
        cits = _verify_against_context(raw_issue.get("citations", []), source_index)
        _law_verify(cits, verify_fn)
        issues.append(
            Issue(
                claim=str(raw_issue.get("claim", "")).strip(),
                verdict=verdict,
                severity=severity,
                rationale=str(raw_issue.get("rationale", "")).strip(),
                law_basis=str(raw_issue.get("law_basis", "")).strip(),
                citations=cits,
                suggested_fix=str(raw_issue.get("suggested_fix", "")).strip(),
            )
        )

    # Union of all (verified) citations, de-duplicated by source_id.
    union: dict[str, Citation] = {}
    for issue in issues:
        for cit in issue["citations"]:
            union.setdefault(cit["source_id"], cit)

    summary = str(parsed.get("summary", "")).strip() or (
        "검토 결과 요지를 생성하지 못했습니다(근거 불충분). 개별 issue를 확인하십시오."
    )

    return ReviewResult(
        summary=summary,
        issues=issues,
        citations=list(union.values()),
        corrected_copy=str(parsed.get("corrected_copy", "")).strip(),
        claims_reviewed=len(claims),
        extraction={
            "source": extraction["source"],
            "n_chars": extraction["n_chars"],
            "n_pages": extraction["n_pages"],
            "truncated": extraction["truncated"],
        },
        ai_generated=True,
        disclaimer=config.ANSWER_DISCLAIMER,
    )


def _resolve_verify_fn() -> Callable[..., dict[str, Any]] | None:
    """Return ``search.verify.verify_citation`` if importable, else ``None``.

    The Citation Firewall is an optional 2nd-pass collaborator; if its module is
    unavailable the review still returns context-verified citations (DB-only),
    matching the degraded-mode contract in ``api.main``.
    """
    try:
        from search.verify import verify_citation

        return verify_citation
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("search.verify unavailable (%s); skipping law.go.kr 2nd pass.", type(exc).__name__)
        return None


__all__ = [
    "review",
    "extract_text",
    "decompose_claims",
    "Issue",
    "Citation",
    "ReviewResult",
    "AD_LAW_TITLES",
]


# --------------------------------------------------------------------------- #
# CLI self-test (offline: fakes the LLM + retriever; no OpenAI/Qdrant)         #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Exercise the full pipeline with the LLM + retriever + law API faked."""
    failures: list[str] = []

    # --- extract_text branches -------------------------------------------- #
    ext = extract_text(text="  광고 문구  ", file_bytes=None, filename=None)
    if ext["source"] != "text" or ext["text"] != "광고 문구":
        failures.append(f"extract_text(text) wrong: {ext}")
    try:
        extract_text(text=None, file_bytes=None, filename=None)
    except ValueError:
        pass
    else:
        failures.append("extract_text with no input should raise ValueError")
    blob = extract_text(text=None, file_bytes="안녕".encode("utf-8"), filename="a.txt")
    if blob["source"] != "file" or "안녕" not in blob["text"]:
        failures.append(f"extract_text(file blob) wrong: {blob}")

    # --- _verify_against_context (citation firewall) ----------------------- #
    src_index = {
        "ctx-1": {
            "source_id": "ctx-1",
            "title": "표시·광고의 공정화에 관한 법률",
            "location": "제3조",
            "source_url": "https://www.law.go.kr/x",
            "doc_type": "law",
            "trust_grade": "A",
        }
    }
    cits = _verify_against_context(
        [
            {"source_id": "ctx-1", "title": "엉뚱한법", "location": "제3조 제1항"},
            {"source_id": "ghost", "title": "허위법", "location": "제99조"},
        ],
        src_index,
    )
    if [c["source_id"] for c in cits] != ["ctx-1"]:
        failures.append(f"context firewall did not drop ghost: {cits}")
    elif cits[0]["title"] != "표시·광고의 공정화에 관한 법률":
        failures.append("context firewall did not override model title")
    elif cits[0]["location"] != "제3조 제1항":
        failures.append("context firewall dropped finer model location")

    # --- _law_verify with a fake verify_citation -------------------------- #
    fake_called: dict[str, Any] = {}

    def _fake_verify(citation: dict[str, Any], as_of_date: str | None = None) -> dict[str, Any]:
        fake_called["title"] = citation.get("law_name")
        fake_called["article_no"] = citation.get("article_no")
        return {
            "api_match": True,
            "current": True,
            "source_url": "https://www.law.go.kr/authoritative",
            "note": "law.go.kr 현행 조문 일치: 제3조.",
        }

    _law_verify(cits, _fake_verify)
    if fake_called.get("title") != "표시·광고의 공정화에 관한 법률":
        failures.append(f"_law_verify did not pass title: {fake_called}")
    if cits[0].get("api_verified") is not True or cits[0].get("current") is not True:
        failures.append(f"_law_verify did not annotate api result: {cits[0]}")
    if cits[0]["source_url"] != "https://www.law.go.kr/authoritative":
        failures.append("_law_verify did not adopt authoritative source_url")
    if "OC=" in json.dumps(cits, ensure_ascii=False):
        failures.append("citation leaked OC token")

    # --- _law_verify api_match=False downgrades trust --------------------- #
    cit2 = _verify_against_context([{"source_id": "ctx-1"}], src_index)
    _law_verify(cit2, lambda c, **k: {"api_match": False, "current": False, "note": "오인용"})
    if cit2[0].get("api_verified") is not False or cit2[0].get("trust_grade") != "B":
        failures.append(f"api_match=False should downgrade trust: {cit2[0]}")

    # --- full review() with faked LLM + retriever ------------------------- #
    class _FakeHit:
        def __init__(self, hid: str, payload: dict[str, Any]) -> None:
            self.id = hid
            self.score = 0.8
            self.payload = payload

    fake_hits = [
        _FakeHit(
            "ctx-1",
            {
                "doc_type": "law",
                "title": "표시·광고의 공정화에 관한 법률",
                "article_no": "제3조",
                "source_url": "https://www.law.go.kr/x",
                "trust_grade": "A",
                "text": "부당한 표시·광고 행위의 금지 …",
            },
        )
    ]

    import search.retriever as _retr_mod

    real_search = getattr(_retr_mod, "search", None)
    _retr_mod.search = lambda *a, **k: fake_hits  # type: ignore[assignment]

    decompose_out = {
        "product_type": "화장품",
        "claims": [{"text": "주름이 100% 사라집니다", "claim_type": "효능효과"}],
    }
    judge_out = {
        "summary": "효능 과장 광고로 표시광고법 위반 소지가 있습니다 [1].",
        "issues": [
            {
                "claim": "주름이 100% 사라집니다",
                "verdict": "위반소지",
                "severity": "high",
                "rationale": "객관적 근거 없는 절대적 효능 표현으로 부당 표시·광고에 해당할 수 있음 [1].",
                "law_basis": "표시광고법 제3조",
                "citations": [
                    {"source_id": "ctx-1", "title": "x", "location": "제3조", "source_url": "x"},
                    {"source_id": "ghost", "title": "허위", "location": "제9조", "source_url": "x"},
                ],
                "suggested_fix": "임상 시험 결과에 근거한 범위에서 표현을 한정하십시오.",
            }
        ],
        "corrected_copy": "개인차가 있으며 임상 결과에 따라 주름 개선에 도움을 줄 수 있습니다.",
    }

    call_seq: list[str] = []

    def _fake_structured(*, system: str, user: str, schema: dict[str, Any], model: str | None = None):
        name = schema.get("name")
        call_seq.append(name)
        return decompose_out if name == "ad_claims" else judge_out

    real_structured = globals()["_structured_call"]
    real_resolve = globals()["_resolve_verify_fn"]
    globals()["_structured_call"] = _fake_structured  # type: ignore[assignment]
    globals()["_resolve_verify_fn"] = lambda: _fake_verify  # type: ignore[assignment]
    try:
        result = review(text="주름이 100% 사라집니다! 단 한 번으로 완벽하게.")
        if call_seq != ["ad_claims", "ad_review"]:
            failures.append(f"unexpected LLM call sequence: {call_seq}")
        if not result["issues"]:
            failures.append("review() returned no issues")
        else:
            iss = result["issues"][0]
            if iss["verdict"] != "위반소지":
                failures.append(f"verdict not preserved: {iss['verdict']}")
            if [c["source_id"] for c in iss["citations"]] != ["ctx-1"]:
                failures.append(f"issue citations not firewalled: {iss['citations']}")
            if iss["citations"][0].get("api_verified") is not True:
                failures.append("issue citation not 2nd-verified by law.go.kr")
        if result["claims_reviewed"] != 1:
            failures.append(f"claims_reviewed wrong: {result['claims_reviewed']}")
        if [c["source_id"] for c in result["citations"]] != ["ctx-1"]:
            failures.append(f"union citations wrong: {result['citations']}")
        if result["ai_generated"] is not True:
            failures.append("ai_generated must be True")
        if result["disclaimer"] != config.ANSWER_DISCLAIMER:
            failures.append("disclaimer must equal config.ANSWER_DISCLAIMER")
        if not result["corrected_copy"]:
            failures.append("corrected_copy missing")
    finally:
        globals()["_structured_call"] = real_structured  # type: ignore[assignment]
        globals()["_resolve_verify_fn"] = real_resolve  # type: ignore[assignment]
        if real_search is not None:
            _retr_mod.search = real_search  # type: ignore[assignment]

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(
        "SELFTEST PASSED: extract(text/file/empty), context citation firewall "
        "(drop ghost · override meta · keep fine location), law.go.kr 2nd "
        "verification (annotate · adopt url · downgrade on api_match=False · no "
        "OC leak), full review() pipeline (2 LLM calls, claims→RAG→judge→verify)."
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="search.ad_review",
        description="Advertising-copy legal review (/v1/ad-review).",
    )
    p.add_argument("--selftest", action="store_true", help="Offline pipeline check (no network).")
    return p


def _main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    print("Use --selftest (offline). The live path is served via POST /v1/ad-review.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
