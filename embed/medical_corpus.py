"""Medical sub-corpus extractor (의료관련 빌드, _FAISS_BUILD_CONTRACT.md §5).

Materializes a *filtered* medical sub-corpus from the 6 GB source corpora
**without copying any raw ``.md``**: manifest YAMLs (the single source of truth)
point at whitelist folders / ministry scopes / precedent match-rules, and the
existing parsers (``ingest.parse_statute`` / ``parse_admrule`` /
``parse_precedent``) stream-parse only the selected files.

Public API (contract §5)::

    def build_medical_corpus() -> None: ...   # manifest(YAML) -> docs JSONL

Outputs (under ``config.MED_DIR``)::

    docs/국가법령.jsonl   docs/행정규칙.jsonl   docs/판례.jsonl     (filtered Documents)
    _audit/match_log.csv   _audit/absent_null.csv   _audit/dedup_log.csv

Each ``docs/*.jsonl`` line is exactly ``Document.model_dump_json()`` (the same
shape ``embed/chunk.py`` consumes), so the downstream pipeline
(chunk -> cached_embed(512d) -> normalize -> CHUNKS_VEC_JSONL ->
faiss_index.build_index) is unchanged.

Design rules enforced here (docs/의료관련_코퍼스_범위_추출전략.md):
  * 국가법령: whitelist 폴더의 법률/시행령/시행규칙(.md)만 ``parse_statute.parse_file``.
  * 행정규칙: 부처 폴더 스코핑 → 규칙명 키워드 매칭 → denylist 탈락 →
    ``parse_admrule.parse_file`` → 재발령 dedup(``parse_admrule._revision_rank``)
    → ``trust_grade`` A/B 분기(B = 메타링크만, 임베딩은 후속 chunk 단계가 A만 채택).
  * 판례: Phase-1(형사+일반행정 **대법원만**), ``parse_precedent.parse_all`` 스트리밍
    + ``## 참조조문`` 화이트리스트 법령명 **정확매칭**(공백·개행 제거, 구법 'ㄱ 의료법'
    허용, 약칭 정규화) → ``match_rule``/``matched_term`` 메타 기록.
  * 화이트리스트 폴더 실재 사전 assert(누락 = 빌드 실패).
  * 부재 4종을 ``absent_null.csv`` 에 명시(커버리지 착시 차단).
  * 멱등: 동일 manifest/원천 → 동일 산출물(원자적 교체, 정렬된 글롭 순서).

Cost rule: this stage performs **no embedding / no OpenAI call** — pure offline
filter-parse. ``--selftest`` exercises the matching logic on a few folders with
no billing.

Owner: builder ``medical_corpus``. Imports Contracts-owned ``config`` and the
existing ``ingest`` parsers (unmodified). Owns only this file +
``artifacts/의료관련/manifest/*.yaml``.

Usage (WSL venv, Python 3.12)::

    cd /home/user1/lawbot && .venv/bin/python -m embed.medical_corpus
    cd /home/user1/lawbot && .venv/bin/python -m embed.medical_corpus --selftest
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

# Allow both ``python -m embed.medical_corpus`` and script form.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from ingest import parse_admrule, parse_precedent, parse_statute  # noqa: E402
from ingest.schema import Document  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths (manifest SoT + outputs). All under config.MED_DIR.                     #
# --------------------------------------------------------------------------- #
MANIFEST_DIR = config.MED_DIR / "manifest"
DOCS_DIR = config.MED_DIR / "docs"
AUDIT_DIR = config.MED_DIR / "_audit"

STATUTES_WHITELIST = MANIFEST_DIR / "statutes_whitelist.yaml"
ADMRULE_TARGETS = MANIFEST_DIR / "admrule_targets.yaml"
ADMRULE_DENYLIST = MANIFEST_DIR / "admrule_denylist.yaml"
PRECEDENT_RULES = MANIFEST_DIR / "precedent_match_rules.yaml"

OUT_LAW = DOCS_DIR / "국가법령.jsonl"
OUT_ADMRULE = DOCS_DIR / "행정규칙.jsonl"
OUT_PREC = DOCS_DIR / "판례.jsonl"

AUDIT_MATCH = AUDIT_DIR / "match_log.csv"
AUDIT_ABSENT = AUDIT_DIR / "absent_null.csv"
AUDIT_DEDUP = AUDIT_DIR / "dedup_log.csv"

# Documented corpus-absent items (NULL 명시 + 조문 fallback). §7 of the strategy.
_ABSENT_NULL_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "의료광고 자율심의기준",
        "의협·치협·한의협 내규(정부고시 아님) — 원천 부재",
        "의료법 §56~57의3 + 시행령 fallback",
    ),
    (
        "비대면진료 시범사업 지침",
        "복지부 행정규칙 트리 무매칭 — 원천 부재",
        "의료법 §34(원격의료) fallback",
    ),
    (
        "의료기기 광고심의 위탁 고시",
        "의약품 광고심의만 존재(의료기기 부재)",
        "의료기기법 §24~26 시행규칙 fallback",
    ),
    (
        "비의료 건강관리서비스 가이드라인",
        "정부고시 부재(가이드라인 형태)",
        "의료법 §27 무면허 + 유권해석 fallback",
    ),
)


# --------------------------------------------------------------------------- #
# Small utilities                                                              #
# --------------------------------------------------------------------------- #
def _nfc(s: str) -> str:
    """NFC-normalize a string (folder names / law names compared after NFC)."""
    return unicodedata.normalize("NFC", s)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a manifest YAML mapping; raise a clear error if missing/invalid."""
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Expected under {MANIFEST_DIR}."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest {path} is not a YAML mapping.")
    return data


def _atomic_write_jsonl(out_path: Path, docs: Iterable[Document]) -> int:
    """Stream ``Document.model_dump_json()`` lines to ``out_path`` atomically.

    Writes to a sibling ``*.tmp`` then ``os.replace`` so a partial run never
    leaves a corrupt artifact (idempotent re-runs replace cleanly).

    Returns:
        The number of documents written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(doc.model_dump_json())
            fh.write("\n")
            n += 1
    os.replace(tmp, out_path)
    return n


class _CsvAudit:
    """Tiny atomic CSV writer for the ``_audit/*.csv`` logs."""

    def __init__(self, path: Path, header: list[str]) -> None:
        self.path = path
        self.header = header
        self.rows: list[list[Any]] = []

    def add(self, *row: Any) -> None:
        self.rows.append(list(row))

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.header)
            w.writerows(self.rows)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------- #
# 1) 국가법령 — whitelist 폴더의 법률/시행령/시행규칙.md                          #
# --------------------------------------------------------------------------- #
def _statute_folders(manifest: dict[str, Any]) -> list[str]:
    """Return the ordered, de-duplicated whitelist folder names (NFC)."""
    folders: list[str] = []
    seen: set[str] = set()
    for tier in ("tier1", "tier2"):
        for entry in manifest.get(tier) or []:
            name = _nfc(str(entry["folder"]).strip())
            if name and name not in seen:
                seen.add(name)
                folders.append(name)
    if not folders:
        raise ValueError("statutes_whitelist.yaml has no tier1/tier2 folders.")
    return folders


def assert_statute_folders_present(
    folders: list[str], law_dir: Path
) -> dict[str, Path]:
    """Assert every whitelist folder exists under ``law_dir`` (NFC-matched).

    Returns:
        A mapping ``folder_name -> resolved Path`` for the matched folders.

    Raises:
        FileNotFoundError: If any whitelist folder is missing (build must fail).
    """
    # Build an NFC index of on-disk directories once.
    on_disk: dict[str, Path] = {}
    if law_dir.exists():
        for child in law_dir.iterdir():
            if child.is_dir():
                on_disk[_nfc(child.name)] = child
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for name in folders:
        path = on_disk.get(name)
        if path is None:
            missing.append(name)
        else:
            resolved[name] = path
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} whitelist statute folder(s) absent under {law_dir}: "
            + ", ".join(missing)
        )
    return resolved


def _iter_statutes(
    manifest: dict[str, Any], law_dir: Path, audit: _CsvAudit
) -> Iterator[Document]:
    """Yield Documents for every ``*.md`` in each whitelist statute folder."""
    folders = _statute_folders(manifest)
    resolved = assert_statute_folders_present(folders, law_dir)
    include_kinds = {
        _nfc(str(k).strip()) for k in (manifest.get("include_kinds") or [])
    }
    for name in folders:
        folder = resolved[name]
        for md in sorted(folder.glob("*.md")):
            if include_kinds and _nfc(md.stem) not in include_kinds:
                continue
            doc = parse_statute.parse_file(md)
            if doc is None:
                audit.add("law", name, md.name, "skip", "no-front-matter")
                continue
            audit.add(
                "law", name, md.name, doc.law_kind or "", doc.trust_grade
            )
            yield doc


# --------------------------------------------------------------------------- #
# 2) 행정규칙 — 부처 스코핑 + 규칙명 키워드 + denylist + dedup                    #
# --------------------------------------------------------------------------- #
def _admrule_scopes(targets: dict[str, Any], admrule_dir: Path) -> list[Path]:
    """Resolve + assert each ministry scope path under ``admrule_dir``."""
    scopes = targets.get("scopes") or []
    if not scopes:
        raise ValueError("admrule_targets.yaml has no scopes.")
    resolved: list[Path] = []
    missing: list[str] = []
    for rel in scopes:
        p = admrule_dir / str(rel)
        if p.is_dir():
            resolved.append(p)
        else:
            missing.append(str(rel))
    if missing:
        raise FileNotFoundError(
            f"admrule scope(s) absent under {admrule_dir}: " + ", ".join(missing)
        )
    return resolved


def _admrule_title_of(path: Path) -> str:
    """The rule name = the folder that directly contains ``본문.md`` (NFC)."""
    return _nfc(path.parent.name)


def _matches_keywords(title: str, keywords: list[str]) -> str | None:
    """Return the first keyword contained in ``title`` (NFC substring), else None."""
    for kw in keywords:
        if kw and kw in title:
            return kw
    return None


def _denied(title: str, deny_keywords: list[str]) -> str | None:
    """Return the first denylist keyword contained in ``title``, else None."""
    for kw in deny_keywords:
        if kw and kw in title:
            return kw
    return None


def _iter_admrules(
    targets: dict[str, Any],
    denylist: dict[str, Any],
    admrule_dir: Path,
    audit: _CsvAudit,
    dedup_audit: _CsvAudit,
) -> Iterator[Document]:
    """Yield the canonical Document per 행정규칙ID within the scoped, keyworded set.

    Scoping → keyword include → denylist exclude → parse → dedup-by-doc_id using
    ``parse_admrule._revision_rank`` (latest revision / richest text wins).
    """
    scopes = _admrule_scopes(targets, admrule_dir)
    keywords = [_nfc(str(k)) for k in (targets.get("title_keywords") or [])]
    deny = [_nfc(str(k)) for k in (denylist.get("title_deny_keywords") or [])]

    best: dict[str, Document] = {}
    order: list[str] = []
    for scope in scopes:
        for md in sorted(scope.glob("**/본문.md")):
            title = _admrule_title_of(md)
            hit = _matches_keywords(title, keywords)
            if hit is None:
                continue  # not a keyword target
            den = _denied(title, deny)
            if den is not None:
                audit.add("admrule", title, md.parent.name, "deny", den)
                continue
            try:
                doc = parse_admrule.parse_file(md)
            except Exception as exc:  # noqa: BLE001 - resilience, never crash run
                audit.add("admrule", title, md.parent.name, "skip", repr(exc))
                continue
            existing = best.get(doc.doc_id)
            if existing is None:
                best[doc.doc_id] = doc
                order.append(doc.doc_id)
                audit.add("admrule", title, md.parent.name, hit, doc.trust_grade)
            elif parse_admrule._revision_rank(doc) > parse_admrule._revision_rank(
                existing
            ):
                dedup_audit.add(
                    doc.doc_id,
                    "superseded",
                    (existing.meta or {}).get("행정규칙일련번호"),
                    (doc.meta or {}).get("행정규칙일련번호"),
                )
                best[doc.doc_id] = doc
            else:
                dedup_audit.add(
                    doc.doc_id, "dropped-duplicate", md.parent.name, ""
                )
    for doc_id in order:
        yield best[doc_id]


# --------------------------------------------------------------------------- #
# 3) 판례 — Phase-1 형사+일반행정 대법원, 참조조문 정확매칭                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _LawMatcher:
    """Boundary-aware exact matcher for whitelist law names in 참조조문 text.

    The reference text is normalized (NFC + all whitespace removed). A law name
    matches when it is **not preceded by a Hangul syllable** — so ``의료법`` does
    not spuriously match inside ``국민의료법`` — *unless* the preceding char is
    ``구`` (the old-law/舊法 marker, e.g. ``구 의료법(개정 전의 것)``). Aliases map
    to their canonical whitelist name (recorded as ``matched_term``).
    """

    patterns: tuple[tuple[str, re.Pattern[str]], ...]

    @classmethod
    def build(
        cls, names: Iterable[str], aliases: dict[str, str]
    ) -> "_LawMatcher":
        # term -> canonical. Include canonical names (identity) + aliases.
        term_to_canon: dict[str, str] = {}
        for n in names:
            nn = _nfc(str(n).strip())
            if nn:
                term_to_canon[nn] = nn
        for alias, canon in (aliases or {}).items():
            a = _nfc(str(alias).strip())
            c = _nfc(str(canon).strip())
            if a and c:
                term_to_canon[a] = c
        # Longest term first so the most specific name wins on overlap.
        pats: list[tuple[str, re.Pattern[str]]] = []
        for term in sorted(term_to_canon, key=len, reverse=True):
            canon = term_to_canon[term]
            # (?:(?<=구)|(?<![가-힣]))<term>
            pat = re.compile(r"(?:(?<=구)|(?<![가-힣]))" + re.escape(term))
            pats.append((canon, pat))
        return cls(patterns=tuple(pats))

    @staticmethod
    def normalize(text: str) -> str:
        """NFC + remove all whitespace/newlines (참조조문 filler collapse)."""
        return re.sub(r"\s+", "", _nfc(text))

    def find(self, ref_text: str) -> list[str]:
        """Return the sorted unique canonical law names matched in ``ref_text``."""
        norm = self.normalize(ref_text)
        found: set[str] = set()
        for canon, pat in self.patterns:
            if pat.search(norm):
                found.add(canon)
        return sorted(found)


def _precedent_ref_article(doc: Document, section: str) -> str | None:
    """Return the body of the named precedent section (e.g. ``참조조문``), or None."""
    target = _nfc(section)
    for art in doc.articles:
        if _nfc(art.article_no) == target:
            return art.text
    return None


def _iter_precedents(
    rules: dict[str, Any], prec_dir: Path, audit: _CsvAudit
) -> Iterator[Document]:
    """Yield Phase-1 precedent Documents whose 참조조문 cites a whitelist law.

    Streams 형사/대법원 + 일반행정/대법원 via ``parse_precedent.parse_all`` (one
    file at a time), exact-matches the ``참조조문`` section, and tags accepted
    docs with ``meta.match_rule`` / ``meta.matched_term`` for downstream RAG
    citation + false-positive audit.
    """
    phase1 = rules.get("phase1") or {}
    scopes = phase1.get("scopes") or []
    if not scopes:
        raise ValueError("precedent_match_rules.yaml phase1 has no scopes.")
    section = str(phase1.get("match_section") or "참조조문")
    matcher = _LawMatcher.build(
        phase1.get("reference_law_names") or [],
        phase1.get("reference_law_aliases") or {},
    )

    # Assert scope dirs exist before streaming (missing = build failure).
    scope_dirs: list[Path] = []
    missing: list[str] = []
    for rel in scopes:
        p = prec_dir / str(rel)
        if p.is_dir():
            scope_dirs.append(p)
        else:
            missing.append(str(rel))
    if missing:
        raise FileNotFoundError(
            f"precedent scope(s) absent under {prec_dir}: " + ", ".join(missing)
        )

    for scope in scope_dirs:
        for doc in parse_precedent.parse_all(root=scope):
            ref = _precedent_ref_article(doc, section)
            if not ref:
                continue  # no 참조조문 -> cannot exact-match (Phase-1 precision)
            matched = matcher.find(ref)
            if not matched:
                continue
            term = ", ".join(matched)
            # Tag provenance for citation + audit (survives model_dump_json).
            doc.meta["match_rule"] = f"phase1:{section}:exact"
            doc.meta["matched_term"] = matched
            audit.add(
                "precedent",
                str(scope.relative_to(prec_dir)),
                doc.doc_id,
                f"{section}:exact",
                term,
            )
            yield doc


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def build_medical_corpus() -> None:
    """Build the medical sub-corpus docs JSONL + audit CSVs from the manifests.

    Idempotent: re-running with the same manifests/source yields identical
    artifacts (atomic replace, sorted glob order). Performs **no embedding**.

    Outputs under ``config.MED_DIR``:
      * ``docs/{국가법령,행정규칙,판례}.jsonl`` — filtered ``Document`` lines.
      * ``_audit/{match_log,absent_null,dedup}.csv``.
    """
    statutes = _load_yaml(STATUTES_WHITELIST)
    targets = _load_yaml(ADMRULE_TARGETS)
    denylist = _load_yaml(ADMRULE_DENYLIST)
    rules = _load_yaml(PRECEDENT_RULES)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    match_audit = _CsvAudit(
        AUDIT_MATCH, ["corpus", "scope", "item", "rule_or_kind", "matched_term"]
    )
    dedup_audit = _CsvAudit(
        AUDIT_DEDUP, ["doc_id", "action", "prev", "new"]
    )

    n_law = _atomic_write_jsonl(
        OUT_LAW, _iter_statutes(statutes, config.LAW_DIR, match_audit)
    )
    n_adm = _atomic_write_jsonl(
        OUT_ADMRULE,
        _iter_admrules(
            targets, denylist, config.ADMRULE_DIR, match_audit, dedup_audit
        ),
    )
    n_prec = _atomic_write_jsonl(
        OUT_PREC, _iter_precedents(rules, config.PRECEDENT_DIR, match_audit)
    )

    match_audit.flush()
    dedup_audit.flush()

    # absent_null.csv: documented corpus-absent items + 조문 fallback (§7).
    absent = _CsvAudit(AUDIT_ABSENT, ["item", "reason", "fallback"])
    for row in _ABSENT_NULL_ROWS:
        absent.add(*row)
    absent.flush()

    print(
        f"[medical_corpus] 국가법령={n_law} 행정규칙={n_adm} 판례={n_prec} "
        f"-> {DOCS_DIR}",
        file=sys.stderr,
    )
    print(
        f"[medical_corpus] audit -> {AUDIT_MATCH.name}, {AUDIT_DEDUP.name}, "
        f"{AUDIT_ABSENT.name} ({len(_ABSENT_NULL_ROWS)} absent rows)",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# Offline selftest (no embedding, no billing, tiny scope)                       #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Exercise matching logic on a few folders without any OpenAI call.

    Verifies: (1) the law-name matcher boundary/old-law/alias behavior, (2) a
    couple of whitelist statute folders parse to Documents, (3) the admrule
    keyword/denylist gate, (4) a known precedent exact-matches 의료법. Returns a
    process exit code (0 = pass).
    """
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        status = "OK  " if cond else "FAIL"
        if not cond:
            failures.append(name)
        print(f"  [{status}] {name}", file=sys.stderr)

    print("== medical_corpus selftest (offline, no embedding) ==", file=sys.stderr)

    # 1) Matcher unit behavior.
    rules = _load_yaml(PRECEDENT_RULES)
    p1 = rules["phase1"]
    matcher = _LawMatcher.build(
        p1["reference_law_names"], p1.get("reference_law_aliases") or {}
    )
    check("국민의료법 does NOT match 의료법", matcher.find("국민의료법 제40조") == [])
    check("의료법 제25조 -> [의료법]", matcher.find("의료법 제25조") == ["의료법"])
    check(
        "구법 표기 '구 의료법(개정 전)' -> [의료법]",
        matcher.find("구 의료법(2009. 1. 30. 법률 제9386호) 제25조") == ["의료법"],
    )
    check(
        "다중매칭 '의료법, 약사법'",
        matcher.find("의료법 제25조, 약사법 제36조 제2항") == ["약사법", "의료법"],
    )
    check(
        "alias 정보통신망법 -> 정식명",
        matcher.find("정보통신망법 제50조")
        == ["정보통신망이용촉진및정보보호등에관한법률"],
    )

    # 2) A couple of whitelist statute folders parse.
    statutes = _load_yaml(STATUTES_WHITELIST)
    folders = _statute_folders(statutes)
    try:
        resolved = assert_statute_folders_present(folders, config.LAW_DIR)
        check("all whitelist statute folders present", True)
    except FileNotFoundError as exc:
        check(f"all whitelist statute folders present ({exc})", False)
        resolved = {}
    sample = [f for f in ("의료법", "약사법") if f in resolved]
    parsed_ok = 0
    for name in sample:
        for md in sorted(resolved[name].glob("*.md"))[:1]:
            doc = parse_statute.parse_file(md)
            if doc is not None and doc.doc_type == "law" and doc.articles:
                parsed_ok += 1
    check("sample statute folders parse to A-grade docs", parsed_ok == len(sample))

    # 3) admrule keyword/denylist gate (pure-string, no I/O).
    targets = _load_yaml(ADMRULE_TARGETS)
    denylist = _load_yaml(ADMRULE_DENYLIST)
    kws = [_nfc(str(k)) for k in targets["title_keywords"]]
    deny = [_nfc(str(k)) for k in denylist["title_deny_keywords"]]
    check(
        "키워드 '의료광고 심의' 포섭",
        _matches_keywords(_nfc("의료광고 사전심의 기준"), kws) is not None,
    )
    check(
        "denylist '직제' 탈락",
        _denied(_nfc("보건복지부와 그 소속기관 직제"), deny) is not None,
    )
    check(
        "denylist '계약' 탈락",
        _denied(_nfc("보건복지부 협상에 의한 계약 제안서 평가 규정"), deny)
        is not None,
    )

    # 4) A known precedent file exact-matches (if the source is reachable).
    known = (
        config.PRECEDENT_DIR
        / "형사"
        / "대법원"
        / "대법원_1970-08-31_70도1393.md"
    )
    if known.exists():
        doc = parse_precedent.parse_file(known)
        ref = _precedent_ref_article(doc, "참조조문") if doc else None
        check(
            "known 형사 대법원 판례 참조조문 -> 의료법 매칭",
            bool(ref) and "의료법" in matcher.find(ref),
        )
    else:
        print(
            "  [SKIP] known precedent file not reachable (source offline)",
            file=sys.stderr,
        )

    if failures:
        print(f"FAILED: {failures}", file=sys.stderr)
        return 1
    print("selftest PASS", file=sys.stderr)
    return 0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the medical sub-corpus.")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run offline matching selftest (no embedding/billing) and exit.",
    )
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    build_medical_corpus()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
