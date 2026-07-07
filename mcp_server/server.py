r"""lawbot MCP server — Korean-legal RAG as Model Context Protocol tools.

A thin **stdio MCP server** that wraps the lawbot HTTP API so any MCP client
(Claude Desktop, Claude Code, IDE extensions, …) can use lawbot's grounded
Korean-legal answers, statute/precedent search, and citation verification as
first-class tools — with the user's own API key, against either a local server
or the deployed cloud service.

Why an HTTP wrapper (not direct Python calls)?
    So the exact same MCP server works against the **deployed cloud** lawbot
    (``LAWBOT_API_BASE=https://...`` + a tenant ``LAWBOT_API_KEY``) with no local
    index, corpus, or OpenAI key on the client side. The server holds nothing but
    the base URL + key read from the environment.

Configuration (environment)::

    LAWBOT_API_BASE   base URL of the lawbot API   (default http://localhost:8000)
    LAWBOT_API_KEY    tenant API key (Bearer)      (required for /v1/ask, /v1/verify)

Run (stdio)::

    cd /home/user1/lawbot && .venv/bin/python -m mcp_server.server

Claude Desktop / Claude Code config (``claude_desktop_config.json`` or
``.mcp.json``)::

    {
      "mcpServers": {
        "lawbot": {
          "command": "/home/user1/lawbot/.venv/bin/python",
          "args": ["-m", "mcp_server.server"],
          "cwd": "/home/user1/lawbot",
          "env": {
            "LAWBOT_API_BASE": "https://your-lawbot.example.com",
            "LAWBOT_API_KEY": "lk_xxxxxxxxxxxx"
          }
        }
      }
    }

Tools exposed:

* ``lawbot_ask``     — grounded AI legal answer with verified citations (LLM cost)
* ``lawbot_search``  — statute/precedent full-text search (no LLM)
* ``lawbot_verify``  — Citation Firewall: is a statute citation real & current?
* ``lawbot_review``  — medical-advertising compliance review of a text

Owner: mcp_server builder. Imports nothing from sibling builders; talks to the
running API over HTTP only.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
# Configuration (environment only — never hard-code keys)                     #
# --------------------------------------------------------------------------- #
API_BASE: str = os.environ.get("LAWBOT_API_BASE", "http://localhost:8000").rstrip("/")
API_KEY: str = os.environ.get("LAWBOT_API_KEY", "").strip()
# Generous timeout: /v1/ask and /v1/ad-review do LLM generation.
_TIMEOUT: float = float(os.environ.get("LAWBOT_TIMEOUT", "120"))

mcp = FastMCP("lawbot")


# --------------------------------------------------------------------------- #
# HTTP helpers                                                                 #
# --------------------------------------------------------------------------- #
def _headers(json_body: bool = True) -> dict[str, str]:
    """Build request headers, attaching the Bearer key when present."""
    h: dict[str, str] = {}
    if json_body:
        h["Content-Type"] = "application/json"
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _post(path: str, *, json: dict | None = None, data: dict | None = None,
          files: dict | None = None) -> dict[str, Any] | str:
    """POST to the lawbot API and return parsed JSON, or a readable error string.

    Errors are returned as plain strings (never raised) so a transient API
    problem surfaces to the model as tool output instead of crashing the tool.
    """
    url = f"{API_BASE}{path}"
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                url,
                headers=_headers(json_body=json is not None),
                json=json,
                data=data,
                files=files,
            )
    except httpx.RequestError as exc:
        return f"[lawbot 연결 실패] {API_BASE} 에 접속할 수 없습니다 ({type(exc).__name__}). LAWBOT_API_BASE 를 확인하세요."
    if resp.status_code == 401:
        return "[인증 실패] 유효한 API 키가 필요합니다. LAWBOT_API_KEY 를 확인하세요."
    if resp.status_code == 429:
        return "[요청 한도 초과] 잠시 후 다시 시도하세요(rate limit)."
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error", {}).get("detail", resp.text)
        except Exception:
            detail = resp.text
        return f"[오류 {resp.status_code}] {detail}"
    try:
        return resp.json()
    except Exception:
        return f"[응답 파싱 실패] {resp.text[:500]}"


def _fmt_citations(cits: list[dict[str, Any]]) -> str:
    """Format a list of citation dicts into a readable bullet list."""
    if not cits:
        return ""
    lines = ["\n\n[인용 근거]"]
    for c in cits:
        name = c.get("law_name") or c.get("title") or c.get("사건명") or ""
        art = c.get("article_no") or c.get("case_no") or c.get("사건번호") or ""
        url = c.get("source_url") or ""
        label = " ".join(p for p in (name, art) if p).strip() or "(출처)"
        lines.append(f"- {label}" + (f" — {url}" if url else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tools                                                                        #
# --------------------------------------------------------------------------- #
@mcp.tool()
def lawbot_ask(query: str, k: int = 8) -> str:
    """한국 법령·판례 원문에 근거해 법률 질문에 답한다(근거기반 RAG).

    검색된 한국 법령·판례·행정규칙·자치법규 **원문에만 근거**해 답하고, 인용은
    사후 검증되어 환각 인용이 제거된다. 한국법(의료법·형법·민법·행정법 등) 관련
    질문, "무면허 의료행위 처벌", "의료광고 사전심의 대상" 같은 조문·판례 근거가
    필요한 질의에 사용한다. (LLM 비용 발생 — API 키 필요)

    Args:
        query: 자연어 법률 질문(한국어).
        k: 검색해 근거로 쓸 원문 개수(기본 8).

    Returns:
        근거기반 답변 + 검증된 인용 목록(법령명·조문·출처 링크). 면책고지 포함.
    """
    res = _post("/v1/ask", json={"query": query, "k": k})
    if isinstance(res, str):
        return res
    answer = res.get("answer", "").strip() or "(답변 없음)"
    out = answer + _fmt_citations(res.get("citations", []) or [])
    if res.get("disclaimer"):
        out += f"\n\n※ {res['disclaimer']}"
    return out


@mcp.tool()
def lawbot_search(query: str, k: int = 5) -> str:
    """LLM 없이 관련 한국 법령·판례 원문을 정밀 검색한다(결정적·빠름).

    AI가 생성한 답변이 아니라 **관련 조문·판례 원문 자체**가 필요할 때 사용한다.
    각 결과는 법령명·조문번호·관련도 점수·본문 일부·출처를 담는다.

    Args:
        query: 자연어 검색 질의(한국어).
        k: 반환할 결과 개수(기본 5).

    Returns:
        관련 원문 목록(제목·조문·점수·본문 발췌·출처).
    """
    res = _post("/v1/statutes/search", json={"query": query, "k": k})
    if isinstance(res, str):
        return res
    rows = res.get("results", []) or []
    if not rows:
        return "관련 원문을 찾지 못했습니다."
    blocks = []
    for i, r in enumerate(rows, 1):
        title = r.get("title") or r.get("doc_id") or ""
        art = r.get("article_no") or ""
        score = r.get("score")
        text = (r.get("text") or "").strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + "…"
        head = f"{i}. {title} {art}".strip()
        if isinstance(score, (int, float)):
            head += f"  (관련도 {score:.2f})"
        url = r.get("source_url")
        block = head + "\n" + text + (f"\n출처: {url}" if url else "")
        blocks.append(block)
    return "\n\n".join(blocks)


@mcp.tool()
def lawbot_verify(law_name: str, article_no: str) -> str:
    """인용(법령 조문)이 실재하고 현행인지 검증한다(Citation Firewall).

    어떤 답변·문서가 제시한 "○○법 제△조" 인용이 진짜 존재하는지, 폐지/오인용은
    아닌지 확인할 때 사용한다.

    Args:
        law_name: 법령명(예: "의료법").
        article_no: 조문번호(예: "제56조" 또는 "56").

    Returns:
        검증 결과(실재 여부·현행 여부·출처·비고).
    """
    res = _post("/v1/verify", json={"citation": {"law_name": law_name, "article_no": article_no}})
    if isinstance(res, str):
        return res
    rows = res.get("results", []) or []
    if not rows:
        return "검증 결과가 비어 있습니다."
    r = rows[0]
    verified = r.get("verified")
    current = r.get("current")
    mark = "✅ 실재 확인" if verified else "❌ 확인 불가(오인용 가능)"
    cur = "현행" if current else ("폐지/구법 가능" if current is False else "시점 불명")
    out = f"{law_name} {article_no} → {mark} / {cur}"
    # Citation Firewall 신뢰도(0~100) + 신호등(green/yellow/red)이 있으면 표시.
    score = r.get("trust_score")
    flag = r.get("flag")
    if isinstance(score, (int, float)) or flag:
        light = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(flag, "")
        bits = [b for b in (f"신뢰도 {score}/100" if isinstance(score, (int, float)) else "",
                            f"{light}{flag}" if flag else "") if b]
        if bits:
            out += "\n" + " · ".join(bits)
    if r.get("source_url"):
        out += f"\n출처: {r['source_url']}"
    if r.get("note"):
        out += f"\n비고: {r['note']}"
    return out


@mcp.tool()
def lawbot_review(text: str, question: str | None = None) -> str:
    """의료광고 문안을 의료법 등 관련 법령에 비추어 검토한다.

    광고 카피·전단·홈페이지 문안의 의료법(특히 §56 의료광고) 위반 소지를 점검할
    때 사용한다. 위반/주의 항목과 근거 인용을 제시한다. (LLM 비용 — API 키 필요)

    Args:
        text: 검토할 광고 문안 텍스트.
        question: (선택) 검토 초점을 좁히는 질문(예: "효과 보장 표현만 봐줘").

    Returns:
        위반/주의 항목 목록 + 근거 인용 + 면책고지.
    """
    data = {"text": text}
    if question:
        data["question"] = question
    res = _post("/v1/ad-review", data=data)
    if isinstance(res, str):
        return res
    issues = res.get("issues", []) or []
    if not issues:
        out = "발견된 위반/주의 항목이 없습니다."
    else:
        lines = ["[검토 결과]"]
        for it in issues:
            # ad-review issue 스키마: {claim, verdict, severity, rationale,
            # suggested_fix, ...} — verdict(위반/위반소지/주의/적정/확인필요)와 근거를 표시.
            verdict = it.get("verdict") or it.get("severity") or ""
            claim = (it.get("claim") or "").strip()
            rationale = (it.get("rationale") or it.get("note") or it.get("text") or "").strip()
            fix = (it.get("suggested_fix") or "").strip()
            line = f"- [{verdict}] {claim}".rstrip()
            if rationale:
                line += f"\n  근거: {rationale}"
            if fix:
                line += f"\n  수정안: {fix}"
            lines.append(line)
        out = "\n".join(lines)
    if res.get("corrected_copy"):
        out += f"\n\n[교정본]\n{res['corrected_copy']}"
    out += _fmt_citations(res.get("citations", []) or [])
    if res.get("disclaimer"):
        out += f"\n\n※ {res['disclaimer']}"
    return out


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
