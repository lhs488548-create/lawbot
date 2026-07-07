"""RAG answer pipeline (Playbook 08, Task 3.2) — expert mode.

This module turns a lawyer's natural-language question into a grounded,
citation-bearing answer:

    retrieve top-K  ->  build numbered [n] context  ->  GPT (Structured Outputs,
    strict json_schema)  ->  citation post-verification (drop any citation whose
    ``source_id`` is not one of the retrieved hits)  ->  attach AI/provenance
    notice.

Design principles (see ``_BUILD_CONTRACT.md`` sections (d), (g), (h)):

* **Expert mode, not a consumer guard.** The audience is practicing lawyers, so
  we do *not* refuse to analyse or inject 변호사법 §109 "consult a lawyer"
  disclaimers. We provide full professional analysis.
* **Grounding is mandatory for quality.** The model answers *only* from the
  retrieved originals; when retrieval is insufficient it must say "확인 필요"
  rather than fabricate.
* **Citations are post-verified.** Any citation the model emits whose
  ``source_id`` does not correspond to a retrieved context block is dropped.
  This is an always-on anti-hallucination control.

Owner: ``rag`` builder. Consumes the Contracts-owned ``config`` and the
``search.retriever`` interface; never modifies shared files.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any, Callable, Iterable, Iterator, Optional, TypedDict

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

# A single shared OpenAI client. The key is read from the environment by the
# SDK (populated from ``.env`` via ``config``); it is never logged or printed.
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Return a lazily-initialised, process-wide OpenAI client.

    Lazy construction keeps module import side-effect free (importing this
    module must not require network or a valid key), which matters for unit
    tests of the pure-Python citation verifier.

    Returns:
        A cached :class:`openai.OpenAI` instance.
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


# --------------------------------------------------------------------------- #
# Result types                                                                #
# --------------------------------------------------------------------------- #
class Citation(TypedDict, total=False):
    """One verified citation attached to an answer.

    Attributes:
        source_id: The retrieved context block id this citation points at; must
            equal one of the hit ids shown to the model (post-verified).
        title: Law / case / rule name.
        location: Human-readable pin-cite, e.g. ``"제4조"`` or ``"판결요지"``.
        source_url: Canonical source URL when available.
        doc_type: One of law/ordinance/admrule/precedent.
        trust_grade: "A" (full text) or "B" (metadata only).
    """

    source_id: str
    title: str
    location: str
    source_url: str
    doc_type: str
    trust_grade: str
    effective_from: str  # ISO date the cited provision takes effect ("" if unknown)
    status: str  # 현행 | 시행예정(effective_from이 미래) | 미상 — 미래조문 투명표기
    trust_flag: str  # green | yellow — 즉시 산출 신뢰 신호등(인용 카드 색)
    trust_note: str  # 신호 사유(사용자 표시용 짧은 설명)


class AskResult(TypedDict):
    """The structured result returned by :func:`ask`.

    See ``_BUILD_CONTRACT.md`` section (d) for the canonical shape.
    """

    answer: str
    citations: list[Citation]
    used_context: list[dict[str, Any]]
    model: str
    ai_generated: bool
    disclaimer: str
    trust_score: int  # 0~100, 인용 신뢰 신호등(green/yellow) 집계 (인용 없으면 0)
    usage: dict[str, int]  # {"total_tokens": N} — 답변 생성 LLM 토큰(비용 미터)


# --------------------------------------------------------------------------- #
# Prompt assets (production constants — see contract (h))                      #
# --------------------------------------------------------------------------- #
# Senior Korean legal-research assistant for lawyers. Grounding + citation +
# hallucination-suppression + uncertainty-surfacing are all encoded here.
SYSTEM_PROMPT: str = (
    "당신은 대한민국 법령·판례 전반에 근거해 법률 질문에 답하는 'lawbot' "
    "법률 리서치 어시스턴트입니다. 이용자는 법률 전문가이므로, 일반 소비자용 "
    "면책(\"변호사와 상담하세요\" 식 회피)을 넣지 말고 정확한 법률 문어체로 충실하게 "
    "쟁점을 분석·정리하십시오.\n"
    "\n"
    "[도메인 가드 — 가장 먼저 판단] 이 서비스는 대한민국 법령·판례에 근거한 "
    "법률 질문 전반에 답합니다. 질문이 법률과 무관한 단순 인사·잡담·일상질문(예: "
    "'안녕', '점심 뭐 먹지')이거나 코딩·일반상식 등 법률 외 영역이면, [검색결과] "
    "내용과 무관하게 다음 안내문구를 answer에 그대로 넣고 citations는 반드시 빈 "
    "배열([])로 두십시오. 억지로 법조문·판례 인용을 만들지 마십시오. 안내문구: "
    "'저는 한국 법령·판례에 근거해 법률 질문에 답하는 lawbot입니다. 법률 관련 "
    "질문을 해주시면 근거와 함께 답변드리겠습니다.' "
    "단, 법률 질문이면(설령 표현이 간단하거나 검색결과가 빈약해도) "
    "이 안내로 회피하지 말고 아래 절차대로 근거 기반으로 답하십시오.\n"
    "\n"
    "[근거 강제] 반드시 아래 [검색결과] 블록에 포함된 내용만을 근거로 답하십시오. "
    "검색결과에 없는 사실·조문번호·사건번호·날짜·URL은 절대 지어내지 말고, "
    "확인되지 않으면 해당 부분을 '확인 필요' 또는 '근거 불충분'으로 명시하십시오. "
    "모델의 사전지식만으로 단정하지 마십시오.\n"
    "\n"
    "[인용 형식] 답변 본문의 각 주장 뒤에는 근거가 된 검색결과 블록 번호를 "
    "[1], [2]처럼 대괄호로 표기하십시오. 그리고 structured output의 citations "
    "배열에는 실제로 사용한 블록만 넣되, 각 citation의 source_id는 반드시 "
    "[검색결과]에 제시된 블록의 id 값과 '문자 그대로' 동일해야 합니다. "
    "id를 변형하거나 새로 만들지 마십시오. 근거를 고를 때는 질문에 직접 적용되는 "
    "법령 조문(법·시행령·시행규칙)을 우선 인용하고, 판례는 그 해석·적용을 보충하는 "
    "근거로 함께 제시하십시오. 관련 법령 조문이 검색결과에 있으면 판례만으로 답하지 "
    "마십시오.\n"
    "\n"
    "[불확실성 표면화] 현행 조문과 개정 전 조문, 본문이 있는 자료(A등급)와 "
    "메타데이터만 있는 자료(B등급)를 구분하십시오. B등급(본문 없음) 자료를 "
    "인용할 때는 본문이 없어 메타데이터에 한정됨을 답변에 명시하십시오. "
    "관할(국가법령·자치법규·행정규칙)이나 법원이 다른 자료가 충돌하면 임의로 "
    "통합하지 말고 각 출처와 함께 병기하십시오.\n"
    "\n"
    "[형식 규율] 한눈에 읽히도록 간결하고 정리된 답을 마크다운으로 쓰십시오. 구성은 "
    "(1) 핵심 답을 별도 소제목 없이 곧바로 1~3문장으로 제시하고(맨 앞에 '핵심 답' 같은 "
    "라벨·제목을 붙이지 말 것), (2) 필요하면 '**근거**' 소제목 아래 관련 조문·판례를 "
    "'- ' 불릿으로 짧게 정리하되 조문 전문을 그대로 옮기지 말고 핵심만 요약하며, (3) "
    "불확실하거나 검색결과에 없는 점이 있을 때만 끝에 '**유의**: …' 한 줄을 덧붙이십시오. "
    "굵은 소제목(**…**)과 불릿(- )으로 보기 좋게 나누되, 같은 내용을 여러 번 반복하지 "
    "말고(맨 끝에 중복 '요약' 금지), 답 끝에 인용번호를 몰아서 나열하지 마십시오. 각 근거 "
    "문장 끝에는 해당 블록번호 [n]만 붙이고, 번호 없는 빈 대괄호([])는 절대 쓰지 "
    "마십시오. 한국어 법률 문어체로, json_schema 필드 외의 텍스트는 금지.\n"
    "\n"
    "[검색결과가 빈약할 때] 관련성 높은 검색결과가 없으면 추측하지 말고 "
    "answer에 '제공된 검색결과만으로는 근거가 불충분합니다.'라는 취지를 적고 "
    "citations는 빈 배열로 두십시오."
)

# Strict Structured-Outputs JSON schema. The model must return exactly this
# shape; ``additionalProperties: false`` + ``strict`` block extra keys and force
# all listed properties to be present.
RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "cited_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "한국어 법률 문어체 답변. 각 주장 뒤에 근거 블록 번호를 "
                    "[1] 형식으로 표기."
                ),
            },
            "citations": {
                "type": "array",
                "description": "실제로 사용한 [검색결과] 블록만. 사용 안 했으면 빈 배열.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "source_id": {
                            "type": "string",
                            "description": "검색결과 블록의 id와 문자 그대로 동일해야 함.",
                        },
                        "title": {"type": "string"},
                        "location": {
                            "type": "string",
                            "description": "조문/섹션 핀사이트, 예) '제4조' 또는 '판결요지'.",
                        },
                        "source_url": {"type": "string"},
                    },
                    "required": ["source_id", "title", "location", "source_url"],
                },
            },
        },
        "required": ["answer", "citations"],
    },
}


# --------------------------------------------------------------------------- #
# Context construction                                                         #
# --------------------------------------------------------------------------- #
def _payload_of(hit: Any) -> dict[str, Any]:
    """Return a hit's payload as a plain dict (never ``None``)."""
    payload = getattr(hit, "payload", None)
    return dict(payload) if payload else {}


def _location_of(payload: dict[str, Any]) -> str:
    """Best-effort human-readable pin-cite from a payload.

    Uses the article number for statute-like documents; falls back to an empty
    string when nothing usable is present.
    """
    return str(payload.get("article_no") or "").strip()


def build_context(hits: list[Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    """Render retrieved hits into a numbered ``[n]`` context block.

    Each block is tagged with its ``source_id`` (the hit id, stringified) so the
    model can cite it verbatim, plus the metadata needed for honest grounding
    (doc_type, title, article/section, trust grade, url). The body text is
    truncated to keep the prompt bounded.

    Args:
        hits: Retrieved hits (Qdrant ``ScoredPoint``-like objects exposing
            ``.id``, ``.score``, ``.payload``).

    Returns:
        A tuple ``(context_text, source_index)`` where ``source_index`` maps each
        ``source_id`` to a metadata dict used later to enrich verified citations.
    """
    lines: list[str] = []
    source_index: dict[str, dict[str, Any]] = {}
    for i, hit in enumerate(hits, start=1):
        payload = _payload_of(hit)
        source_id = str(getattr(hit, "id", ""))
        doc_type = str(payload.get("doc_type", ""))
        title = str(payload.get("title", ""))
        location = _location_of(payload)
        url = str(payload.get("source_url") or "")
        trust = str(payload.get("trust_grade", "A"))
        # Body text may live under "text" (chunk payload) — truncate for budget.
        body = str(payload.get("text", "")).strip()
        if len(body) > 1500:
            body = body[:1500] + " …(생략)"

        grade_note = " [본문없음·메타데이터만(B등급)]" if trust == "B" else ""
        header = (
            f"[{i}] id={source_id} | 종류={doc_type or '-'} | "
            f"제목={title or '-'} | 위치={location or '-'} | 등급={trust}{grade_note}"
        )
        if url:
            header += f" | 출처={url}"
        lines.append(f"{header}\n{body}")

        source_index[source_id] = {
            "source_id": source_id,
            "title": title,
            "location": location,
            "source_url": url,
            "doc_type": doc_type,
            "trust_grade": trust,
            "effective_from": str(payload.get("effective_from") or ""),
        }
    return "\n\n".join(lines), source_index


# --------------------------------------------------------------------------- #
# Citation post-verification (pure function — unit-tested)                     #
# --------------------------------------------------------------------------- #
def verify_citations(
    citations: Iterable[dict[str, Any]],
    source_index: dict[str, dict[str, Any]],
) -> list[Citation]:
    """Drop hallucinated citations and enrich the survivors from retrieved data.

    A citation survives **iff** its ``source_id`` matches the id of a block that
    was actually shown to the model (i.e. is a key of ``source_index``). This is
    the always-on anti-hallucination control: the model cannot cite a source it
    was not given. Surviving citations have their ``title``/``location``/
    ``source_url``/``doc_type``/``trust_grade`` taken from the *retrieved* data
    (authoritative), not from whatever the model echoed back — so the model can
    never alter a citation's metadata either.

    Args:
        citations: Raw citation dicts as emitted by the model. Items missing a
            ``source_id``, or whose ``source_id`` is unknown, are discarded.
        source_index: Map of ``source_id -> metadata`` for the blocks that were
            placed in the context (from :func:`build_context`).

    Returns:
        A list of verified :class:`Citation` dicts, de-duplicated by
        ``source_id`` while preserving the model's citation order. Returns an
        empty list when nothing verifies.
    """
    verified: list[Citation] = []
    seen: set[str] = set()
    for raw in citations or []:
        if not isinstance(raw, dict):
            logger.warning("Dropping non-dict citation: %r", raw)
            continue
        source_id = raw.get("source_id")
        if not source_id:
            logger.warning("Dropping citation without source_id: %r", raw)
            continue
        source_id = str(source_id)
        meta = source_index.get(source_id)
        if meta is None:
            # Hallucinated / out-of-context citation — the core control.
            logger.warning(
                "Dropping unverifiable citation source_id=%r (not in retrieved context)",
                source_id,
            )
            continue
        if source_id in seen:
            continue
        seen.add(source_id)
        # Prefer the model's location label when it is non-empty (it may pin a
        # finer sub-point), but fall back to the retrieved metadata. All other
        # fields come from the authoritative retrieved data.
        model_loc = str(raw.get("location") or "").strip()
        # Effective-date status (LexDiff식 앵커링 차용): label future-effective
        # provisions as 시행예정 so a not-yet-in-force amendment is never presented
        # as 현행. effective_from rides the citation for the UI/answer to surface.
        eff = str(meta.get("effective_from") or "").strip()
        if not eff:
            status = "미상"
        elif eff > _today_iso():
            status = "시행예정"
        else:
            status = "현행"
        # Instant trust signal (인용 신호등) from what we RELIABLY hold — no extra
        # network/DB call (the authoritative 0-100 score is /v1/verify). The source
        # is guaranteed to exist (post-verified against retrieved context), so the
        # only reliable downgrade is: full article text not held (B grade).
        #
        # NOTE: effective_from is LAW-LEVEL (the law's latest amendment 시행일, not
        # the cited article's in-force date), so it produced false "시행예정" alarms
        # on in-force recently-amended laws (e.g. 근로기준법 제43조의8). It is no
        # longer used to drive the trust colour; the date is surfaced as neutral
        # info only, and currency caveats live in the answer disclaimer.
        if meta["trust_grade"] == "B":
            trust_flag, trust_note = "yellow", "본문 미확보(메타데이터만)"
        else:
            trust_flag = "green"
            trust_note = "본문 확보"
            if status == "시행예정":
                trust_note = f"본문 확보 · 개정 시행일 {eff}(현행 여부는 원문 확인 권장)"
        verified.append(
            Citation(
                source_id=source_id,
                title=meta["title"],
                location=model_loc or meta["location"],
                source_url=meta["source_url"],
                doc_type=meta["doc_type"],
                trust_grade=meta["trust_grade"],
                effective_from=eff,
                status=status,
                trust_flag=trust_flag,
                trust_note=trust_note,
            )
        )
    return verified


# --------------------------------------------------------------------------- #
# Model call + parsing                                                         #
# --------------------------------------------------------------------------- #
def _aggregate_trust(citations: list[Citation]) -> int:
    """0~100 신뢰점수: 인용별 신호등(green=100, yellow=60)의 평균. 인용 없으면 0.

    인용은 검색결과에 실재하는 것만 남은 상태(사후검증)이므로 red는 없고, 본문확보·
    현행(green)일수록 높다. 권위 있는 0~100 단건 검증은 별도 ``/v1/verify``.
    """
    if not citations:
        return 0
    vals = [100 if c.get("trust_flag") == "green" else 60 for c in citations]
    return round(sum(vals) / len(vals))


def _empty_result(model: str, message: str, used_context: list[dict[str, Any]]) -> AskResult:
    """Build a grounded "insufficient evidence" result with no citations."""
    return AskResult(
        answer=message,
        citations=[],
        used_context=used_context,
        model=model,
        ai_generated=True,
        disclaimer=config.ANSWER_DISCLAIMER,
        trust_score=0,
        usage={"total_tokens": 0},
    )


def _today_iso() -> str:
    """Server 'today' as ISO ``YYYY-MM-DD`` (default point-in-time for queries)."""
    import datetime

    return datetime.date.today().isoformat()


def _call_search(
    search_fn: Callable[..., Any],
    query: str,
    k: int,
    flt: dict[str, str] | None,
    as_of_date: str | None,
) -> list[Any]:
    """Invoke the retriever, forwarding ``as_of_date`` only if it is supported.

    The contract (`_BUILD_CONTRACT.md` (d)) specifies that ``search`` accepts an
    ``as_of_date`` point-in-time filter and that ``ask`` forwards it. To stay
    robust whether or not the retriever module has yet adopted that keyword (the
    retriever is owned by another builder), we introspect its signature: the
    keyword is passed through when present, and silently skipped otherwise. A
    skipped ``as_of_date`` is logged so the gap is visible during integration.

    Args:
        search_fn: The retriever ``search`` callable.
        query: The user's question.
        k: Number of chunks to retrieve.
        flt: Optional payload pre-filter.
        as_of_date: Optional ISO ``YYYY-MM-DD`` point-in-time constraint.

    Returns:
        The list of retrieved hits.
    """
    # NOTE (bug #1 — revised): a hard as_of=today default was tested and REVERTED.
    # The corpus sets ``effective_from`` to a law's latest-amendment date (not a
    # per-article in-force date), so recently-amended core laws (e.g. 형법, whose
    # articles are all dated 2026-09-13) would be hidden entirely by a today
    # filter — breaking retrieval of current law. Future-effective risk is instead
    # surfaced transparently: ``effective_from`` rides every citation and the
    # system prompt instructs distinguishing 현행 vs 개정전/시행예정. ``as_of_date``
    # remains an explicit opt-in (point-in-time lookup) when the caller passes it.
    kwargs: dict[str, Any] = {"k": k, "flt": flt}
    if as_of_date is not None:
        try:
            params = inspect.signature(search_fn).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins/C funcs
            params = {}
        accepts_kw = "as_of_date" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_kw:
            kwargs["as_of_date"] = as_of_date
        else:
            logger.warning(
                "retriever.search does not accept 'as_of_date'; point-in-time "
                "filter not applied for this query."
            )
    return list(search_fn(query, **kwargs))


def _ensure_statutes(
    search_fn: Callable[..., Any],
    query: str,
    hits: list[Any],
    flt: dict[str, Any] | None,
    as_of_date: str | None,
    want: int = 3,
) -> list[Any]:
    """Guarantee a few governing-statute chunks are present for citation.

    Colloquial fact-pattern queries ("보증금 안 돌려줘") retrieve mostly precedents
    (case law matches the fact pattern far more strongly than the terse statute),
    so the answer can only cite case law. When the top hits hold < 2 statute/admrule
    chunks, fetch a few law-filtered chunks for the SAME query (its rewrite is
    LRU-cached, so no extra LLM call) and append them, letting the model cite the
    governing 조문. Skipped when the caller already constrains ``doc_type``.
    """
    if not hits or (flt and "doc_type" in flt):
        return hits
    n_law = sum(
        1 for h in hits if str(_payload_of(h).get("doc_type", "")) in ("law", "admrule")
    )
    if n_law >= 2:
        return hits
    try:
        extra = _call_search(
            search_fn, query, want + 2, {"doc_type": ["law", "admrule"]}, as_of_date
        )
    except Exception:  # pragma: no cover - supplementary fetch is best-effort
        return hits
    have = {str(getattr(h, "id", "")) for h in hits}
    add = [h for h in extra if str(getattr(h, "id", "")) not in have][:want]
    return hits + add


def _generate(query: str, context: str, model: str) -> dict[str, Any]:
    """Call the chat model with strict Structured Outputs and parse the JSON.

    Args:
        query: The user's question.
        context: The numbered ``[n]`` context block from :func:`build_context`.
        model: The chat model id to use.

    Returns:
        The parsed model output dict (``{"answer": str, "citations": [...]}``).

    Raises:
        RuntimeError: If the response is missing content, was truncated by the
            content filter / length, or is not valid JSON.
    """
    client = _get_client()
    user_content = (
        f"[검색결과]\n{context}\n\n"
        f"[질문]\n{query}\n\n"
        "위 [검색결과]만 근거로, 지정된 json_schema 형식으로 답하십시오."
    )
    # gpt-5-mini (config.GEN_MODEL) rejects any ``temperature`` other than the
    # default (1) with a 400; omit it. Strict Structured Outputs still pins shape.
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        # Reasoning models (gpt-5/o-series) think before emitting tokens; cap that
        # phase via config (default "low") so it does not dominate latency. No-op
        # for non-reasoning models. See config.reasoning_effort_kwargs.
        **config.reasoning_effort_kwargs(model),
    )
    choice = response.choices[0]
    if choice.finish_reason not in ("stop", None):
        # length / content_filter — refuse rather than return partial JSON.
        raise RuntimeError(
            f"Model response did not complete cleanly (finish_reason="
            f"{choice.finish_reason!r})."
        )
    content = choice.message.content
    if not content:
        refusal = getattr(choice.message, "refusal", None)
        raise RuntimeError(
            f"Model returned no content (refusal={refusal!r})."
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:  # pragma: no cover - schema makes this rare
        raise RuntimeError("Model output was not valid JSON despite strict schema.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Model output JSON was not an object.")
    # Stash token usage for the caller's cost meter (best-effort). The ``_`` prefix
    # keeps it out of the model-output namespace (answer/citations).
    usage = getattr(response, "usage", None)
    parsed["_total_tokens"] = int(getattr(usage, "total_tokens", 0) or 0)
    return parsed


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def ask(
    query: str,
    k: int = config.DEFAULT_TOP_K,
    flt: dict[str, str] | None = None,
    model: str | None = None,
    as_of_date: str | None = None,
) -> AskResult:
    """Answer a legal question with grounded, post-verified citations.

    Pipeline: retrieve top-K → (low-score gate) → build numbered context → GPT
    (strict Structured Outputs) → verify citations against the retrieved hits →
    attach the AI / provenance notice. Expert mode (lawyers): full analysis, no
    consumer guard, but grounding and citation verification are always on.

    Args:
        query: The lawyer's natural-language question. Must be non-empty.
        k: Number of chunks to retrieve. Defaults to ``config.DEFAULT_TOP_K``.
        flt: Optional payload filter AND-ed into the Qdrant query, e.g.
            ``{"doc_type": "law"}`` or ``{"jurisdiction": "전라남도"}``.
        model: Optional chat model override (e.g. ``config.GEN_MODEL_FALLBACK``
            for hard queries). Defaults to ``config.GEN_MODEL``.
        as_of_date: Optional ISO ``YYYY-MM-DD`` point-in-time constraint,
            forwarded to the retriever to restrict to rows whose
            ``effective_from <= as_of_date`` (current-law-as-of lookup, 09 §A/E).

    Returns:
        An :class:`AskResult` (see ``_BUILD_CONTRACT.md`` (d)). When retrieval
        is empty *or* the best hit scores below ``config.MIN_RETRIEVAL_SCORE``,
        returns a grounded "근거 불충분" answer with no citations rather than
        calling the model (anti-hallucination, never fabricate).

    Raises:
        ValueError: If ``query`` is empty/whitespace or ``k`` < 1.
        RuntimeError: If the model call fails to produce parseable output.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k < 1:
        raise ValueError("k must be >= 1")
    chosen_model = model or config.GEN_MODEL

    # Import the retriever lazily so this module imports without a live vector
    # store (keeps the citation-verifier unit-testable in isolation).
    from search.retriever import search  # noqa: PLC0415

    hits = _call_search(search, query, k, flt, as_of_date)
    hits = _ensure_statutes(search, query, hits, flt, as_of_date)
    context, source_index = build_context(hits)
    used_context = list(source_index.values())

    if not hits:
        logger.info("No retrieval hits for query; returning grounded empty answer.")
        return _empty_result(
            chosen_model,
            "제공된 검색결과가 없어 근거가 불충분합니다. 질의를 더 구체화하거나 "
            "관할/문서종류 필터를 조정해 주십시오. (확인 필요)",
            used_context,
        )

    # Low-score gate (contract (d)): if even the best hit is below the minimum
    # retrieval score, the corpus has nothing relevant — do NOT call the model
    # (it would be tempted to answer from parametric knowledge). Return an honest
    # "근거 불충분" result with no citations. ``hits`` are score-descending, so
    # the first hit is the maximum.
    top_score = float(getattr(hits[0], "score", 0.0) or 0.0)
    if top_score < config.MIN_RETRIEVAL_SCORE:
        logger.info(
            "Top retrieval score %.4f < MIN_RETRIEVAL_SCORE %.4f; returning "
            "grounded empty answer without a model call.",
            top_score,
            config.MIN_RETRIEVAL_SCORE,
        )
        return _empty_result(
            chosen_model,
            "검색된 자료의 관련도가 낮아 제공된 근거만으로는 답변하기 어렵습니다. "
            "(근거 불충분) 질의를 더 구체화하거나 관할/문서종류 필터를 조정해 "
            "주십시오.",
            used_context,
        )

    parsed = _generate(query, context, chosen_model)
    answer = str(parsed.get("answer", "")).strip()
    verified = verify_citations(parsed.get("citations", []), source_index)

    if not answer:
        answer = "제공된 검색결과만으로는 근거가 불충분합니다. (확인 필요)"

    return AskResult(
        answer=answer,
        citations=verified,
        used_context=used_context,
        model=chosen_model,
        ai_generated=True,
        disclaimer=config.ANSWER_DISCLAIMER,
        trust_score=_aggregate_trust(verified),
        usage={"total_tokens": int(parsed.get("_total_tokens", 0) or 0)},
    )


# --------------------------------------------------------------------------- #
# Streaming entry point (09 §E-4.4 — SSE token streaming for low first-token)   #
# --------------------------------------------------------------------------- #
def _stream_generate(query: str, context: str, model: str) -> Iterator[str]:
    """Yield answer text deltas from the chat model as they arrive.

    Uses the same grounded prompt as :func:`_generate` but in plain streaming
    mode (no Structured Outputs — JSON-schema streaming would withhold tokens
    until the object is parseable, defeating the <1s first-token goal). The
    answer text streams live; citations are post-verified once afterward by the
    caller (which already holds the ``source_index``), preserving the always-on
    anti-hallucination control.
    """
    client = _get_client()
    user_content = (
        f"[검색결과]\n{context}\n\n"
        f"[질문]\n{query}\n\n"
        "위 [검색결과]만 근거로, 각 주장 뒤에 근거 블록 번호를 [1] 형식으로 표기하여 "
        "한국어 법률 문어체로 답하십시오. 검색결과에 없는 사실은 지어내지 마십시오."
    )
    # See ``_generate``: gpt-5-mini rejects non-default ``temperature``; omit it.
    stream = client.chat.completions.create(
        model=model,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        # Lower reasoning effort matters MOST for streaming: a reasoning model
        # finishes its hidden reasoning before the first content token, so a high
        # effort defeats the low-first-token goal. See config.reasoning_effort_kwargs.
        **config.reasoning_effort_kwargs(model),
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            yield piece


def ask_stream(
    query: str,
    k: int = config.DEFAULT_TOP_K,
    flt: dict[str, str] | None = None,
    model: str | None = None,
    as_of_date: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream a grounded answer as a sequence of event dicts (for SSE).

    Yields, in order:

    * ``{"type": "meta", "model", "used_context", "ai_generated", "disclaimer"}``
      once up front (so the client can render sources immediately);
    * zero or more ``{"type": "token", "text": <delta>}`` events as the answer
      streams (skipped entirely when the low-score gate fires — then a single
      token event carries the honest "근거 불충분" message);
    * a terminal ``{"type": "done", "citations": [...verified...],
      "answer": <full text>}`` event with the post-verified citations.

    Same grounding/anti-hallucination contract as :func:`ask`: retrieval is
    mandatory, the low-score gate avoids a model call when nothing is relevant,
    and citations are verified against the retrieved hits before being emitted.

    Args:
        query: The lawyer's natural-language question. Must be non-empty.
        k: Number of chunks to retrieve.
        flt: Optional payload pre-filter.
        model: Optional chat-model override.
        as_of_date: Optional ISO ``YYYY-MM-DD`` point-in-time constraint.

    Yields:
        Event dicts as described above.

    Raises:
        ValueError: If ``query`` is empty/whitespace or ``k`` < 1.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k < 1:
        raise ValueError("k must be >= 1")
    chosen_model = model or config.GEN_MODEL

    from search.retriever import search  # noqa: PLC0415

    hits = _call_search(search, query, k, flt, as_of_date)
    hits = _ensure_statutes(search, query, hits, flt, as_of_date)
    context, source_index = build_context(hits)
    used_context = list(source_index.values())

    yield {
        "type": "meta",
        "model": chosen_model,
        "used_context": used_context,
        "ai_generated": True,
        "disclaimer": config.ANSWER_DISCLAIMER,
    }

    # Same grounding gates as ask(): no hits, or best hit below the floor → emit
    # an honest "근거 불충분" message and stop, without calling the model.
    gate_msg: str | None = None
    if not hits:
        gate_msg = (
            "제공된 검색결과가 없어 근거가 불충분합니다. 질의를 더 구체화하거나 "
            "관할/문서종류 필터를 조정해 주십시오. (확인 필요)"
        )
    else:
        top_score = float(getattr(hits[0], "score", 0.0) or 0.0)
        if top_score < config.MIN_RETRIEVAL_SCORE:
            gate_msg = (
                "검색된 자료의 관련도가 낮아 제공된 근거만으로는 답변하기 어렵습니다. "
                "(근거 불충분) 질의를 더 구체화하거나 관할/문서종류 필터를 조정해 주십시오."
            )

    if gate_msg is not None:
        yield {"type": "token", "text": gate_msg}
        yield {"type": "done", "answer": gate_msg, "citations": []}
        return

    parts: list[str] = []
    for piece in _stream_generate(query, context, chosen_model):
        parts.append(piece)
        yield {"type": "token", "text": piece}

    answer = "".join(parts).strip()
    if not answer:
        answer = "제공된 검색결과만으로는 근거가 불충분합니다. (확인 필요)"

    # Post-verify citations the model referenced in-text against the retrieved
    # blocks. Streaming mode has no structured citation array, so we recover the
    # cited block numbers ([n]) from the answer text and map them back to the
    # source ids shown in build_context (block i -> hits[i-1].id).
    ordered_ids = [str(getattr(h, "id", "")) for h in hits]
    cited_raw: list[dict[str, Any]] = []
    seen_idx: set[int] = set()
    for m in re.finditer(r"\[(\d{1,3})\]", answer):
        idx = int(m.group(1))
        if 1 <= idx <= len(ordered_ids) and idx not in seen_idx:
            seen_idx.add(idx)
            cited_raw.append({"source_id": ordered_ids[idx - 1]})
    verified = verify_citations(cited_raw, source_index)

    yield {
        "type": "done",
        "answer": answer,
        "citations": verified,
        "trust_score": _aggregate_trust(verified),
    }


__all__ = [
    "ask",
    "ask_stream",
    "verify_citations",
    "build_context",
    "AskResult",
    "Citation",
    "SYSTEM_PROMPT",
    "RESPONSE_SCHEMA",
]
