"""Unit tests for the embedding wrapper + Batch path (embed_openai builder).

Covers ``embed/embed_client.py`` and ``embed/embed_batch.py``:

* content-hash determinism + NFC normalization,
* content-hash cache load/append round-trip,
* cache-aware ``cached_embed`` (cache miss -> embed -> reuse, no re-billing),
* sync ``embed_texts`` batching + dim validation (OpenAI client mocked),
* Batch ``estimate_cost`` (pure local, no network),
* Batch ``build_batch_input`` sharding + cache-skip + manifest fan-out,
* the **cost gate**: ``submit`` refuses without ``confirm=True`` and rejects
  over-cap demo limits,
* ``collect`` output parsing + chunk fan-out.

No real OpenAI call is made here (the 1-2 sanctioned real calls are the
``--selfcheck`` runs invoked separately). All network is mocked.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import config
from embed import embed_batch, embed_client


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _vec(seed: float = 1.0) -> list[float]:
    """Return a deterministic EMBED_DIM-length vector."""
    return [seed] * config.EMBED_DIM


def _redirect_artifacts(monkeypatch, tmp_path: Path) -> None:
    """Point all artifact paths used by the embed modules at a temp dir."""
    monkeypatch.setattr(config, "EMBED_CACHE_JSONL", tmp_path / "embed_cache.jsonl")
    monkeypatch.setattr(config, "CHUNKS_JSONL", tmp_path / "chunks.jsonl")
    monkeypatch.setattr(config, "EMBEDDINGS_JSONL", tmp_path / "embeddings.jsonl")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(embed_batch, "_BATCH_MANIFEST", tmp_path / "batch_manifest.json")


# --------------------------------------------------------------------------- #
# content_hash                                                                #
# --------------------------------------------------------------------------- #
def test_content_hash_is_deterministic_and_nfc_normalized():
    a = embed_client.content_hash("도로교통법 제17조")
    b = embed_client.content_hash("도로교통법 제17조")
    assert a == b
    assert len(a) == 64
    # NFC normalization: a decomposed-Hangul string hashes the same as composed.
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "도로교통법")
    assert embed_client.content_hash(decomposed) == embed_client.content_hash("도로교통법")
    # Surrounding whitespace does not change the key.
    assert embed_client.content_hash("  민법  ") == embed_client.content_hash("민법")


# --------------------------------------------------------------------------- #
# cache load/append                                                           #
# --------------------------------------------------------------------------- #
def test_cache_append_then_load_roundtrip(monkeypatch, tmp_path):
    _redirect_artifacts(monkeypatch, tmp_path)
    n = embed_client.append_cache([("h1", _vec(0.5)), ("h2", _vec(0.25))])
    assert n == 2
    cache = embed_client.load_cache()
    assert set(cache) == {"h1", "h2"}
    assert cache["h1"] == _vec(0.5)


def test_cache_load_skips_dim_mismatch(monkeypatch, tmp_path):
    _redirect_artifacts(monkeypatch, tmp_path)
    config.EMBED_CACHE_JSONL.write_text(
        json.dumps({"content_hash": "good", "vector": _vec()}) + "\n"
        + json.dumps({"content_hash": "bad", "vector": [1.0, 2.0]}) + "\n",
        encoding="utf-8",
    )
    cache = embed_client.load_cache()
    assert "good" in cache and "bad" not in cache


def test_append_cache_rejects_wrong_dim(monkeypatch, tmp_path):
    _redirect_artifacts(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        embed_client.append_cache([("h", [1.0, 2.0])])


# --------------------------------------------------------------------------- #
# embed_texts (sync, mocked client)                                           #
# --------------------------------------------------------------------------- #
class _FakeEmb:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _FakeEmbResponse:
    def __init__(self, inputs):
        # Return one distinct vector per input, preserving order via index.
        self.data = [
            _FakeEmb(i, [float(i + 1)] * config.EMBED_DIM) for i in range(len(inputs))
        ]


def _install_fake_client(monkeypatch, calls: list):
    """Install a fake OpenAI client capturing each embeddings.create call."""

    class _Emb:
        def create(self, *, model, input, dimensions=None):
            calls.append({"model": model, "input": list(input), "dimensions": dimensions})
            assert model == config.EMBED_MODEL  # large model forbidden
            # 512-dim Matryoshka pin: embed_client must request the shortened dim.
            assert dimensions == config.EMBED_DIM
            return _FakeEmbResponse(input)

    class _Client:
        embeddings = _Emb()

    embed_client._client.cache_clear()
    monkeypatch.setattr(embed_client, "_client", lambda: _Client())


def test_embed_texts_batches_and_validates_dim(monkeypatch):
    calls: list = []
    _install_fake_client(monkeypatch, calls)
    monkeypatch.setattr(embed_client, "_SYNC_BATCH_SIZE", 2)
    vecs = embed_client.embed_texts(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == config.EMBED_DIM for v in vecs)
    # 3 inputs, batch size 2 => 2 requests.
    assert len(calls) == 2


def test_embed_texts_rejects_empty_input(monkeypatch):
    calls: list = []
    _install_fake_client(monkeypatch, calls)
    with pytest.raises(ValueError):
        embed_client.embed_texts(["ok", "  "])
    assert embed_client.embed_texts([]) == []


# --------------------------------------------------------------------------- #
# cached_embed: cache miss -> embed -> reuse                                  #
# --------------------------------------------------------------------------- #
def test_cached_embed_reuses_cache_and_dedupes(monkeypatch, tmp_path):
    _redirect_artifacts(monkeypatch, tmp_path)
    calls: list = []
    _install_fake_client(monkeypatch, calls)

    items = [
        {"chunk_id": "c1", "text": "민법 제4조"},
        {"chunk_id": "c2", "text": "민법 제4조"},  # duplicate text -> 1 embed
        {"chunk_id": "c3", "text": "형법 제1조"},
    ]
    out = embed_client.cached_embed(items)
    assert set(out) == {"c1", "c2", "c3"}
    assert out["c1"] == out["c2"]  # same text, same vector
    # Two unique texts embedded in a single request batch.
    assert sum(len(c["input"]) for c in calls) == 2

    # Second call: everything cached -> no new OpenAI calls.
    calls.clear()
    out2 = embed_client.cached_embed(items)
    assert calls == []
    assert out2["c1"] == out["c1"]


# --------------------------------------------------------------------------- #
# Batch estimate (pure local, no network)                                     #
# --------------------------------------------------------------------------- #
def _write_chunks(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, t in enumerate(texts):
            fh.write(
                json.dumps(
                    {"chunk_id": f"k{i}", "doc_id": "D", "text": t, "payload": {}},
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_estimate_cost_counts_tokens_and_cache(monkeypatch, tmp_path):
    _redirect_artifacts(monkeypatch, tmp_path)
    _write_chunks(config.CHUNKS_JSONL, ["가나다 라마바", "사아자 차카타", "가나다 라마바"])
    est = embed_batch.estimate_cost()
    assert est["model"] == config.EMBED_MODEL
    assert est["n_chunks"] == 3
    # Two unique texts (the 1st and 3rd are identical).
    assert est["n_unique_misses"] == 2
    assert est["total_tokens"] > 0
    assert est["est_usd"] >= 0
    assert est["batch_discount"] == 0.5

    # Pre-cache one unique text -> it should drop from misses.
    h = embed_client.content_hash("가나다 라마바")
    embed_client.append_cache([(h, _vec())])
    est2 = embed_batch.estimate_cost()
    assert est2["n_cached"] == 2  # both copies of the cached text
    assert est2["n_unique_misses"] == 1


# --------------------------------------------------------------------------- #
# Batch input sharding + manifest + cache skip                                #
# --------------------------------------------------------------------------- #
def test_build_batch_input_shards_and_skips_cache(monkeypatch, tmp_path):
    _redirect_artifacts(monkeypatch, tmp_path)
    _write_chunks(config.CHUNKS_JSONL, ["alpha", "beta", "gamma", "alpha"])
    # Force tiny shards: 2 lines per shard.
    monkeypatch.setattr(embed_batch, "_MAX_LINES_PER_SHARD", 2)

    shards = embed_batch.build_batch_input()
    # 3 unique texts -> 2 shards (2 + 1) at 2 lines/shard.
    assert len(shards) == 2
    lines = []
    for s in shards:
        lines.extend(s.read_text(encoding="utf-8").splitlines())
    assert len(lines) == 3
    for ln in lines:
        rec = json.loads(ln)
        assert rec["method"] == "POST"
        assert rec["url"] == "/v1/embeddings"
        assert rec["body"]["model"] == config.EMBED_MODEL
        assert rec["custom_id"] == embed_client.content_hash(rec["body"]["input"])

    manifest = json.loads(embed_batch._BATCH_MANIFEST.read_text(encoding="utf-8"))
    # 'alpha' maps to two chunk_ids (k0, k3).
    halpha = embed_client.content_hash("alpha")
    assert sorted(manifest[halpha]) == ["k0", "k3"]

    # Now cache one text; rebuild -> it is skipped.
    embed_client.append_cache([(embed_client.content_hash("beta"), _vec())])
    shards2 = embed_batch.build_batch_input()
    lines2 = []
    for s in shards2:
        lines2.extend(s.read_text(encoding="utf-8").splitlines())
    assert len(lines2) == 2  # alpha, gamma (beta cached)


# --------------------------------------------------------------------------- #
# Cost gate: submit refuses without confirm                                   #
# --------------------------------------------------------------------------- #
def test_submit_refuses_without_confirm(monkeypatch, tmp_path):
    _redirect_artifacts(monkeypatch, tmp_path)
    _write_chunks(config.CHUNKS_JSONL, ["x", "y"])
    # Should never touch the network.
    embed_client._client.cache_clear()
    monkeypatch.setattr(
        embed_client, "_client",
        lambda: (_ for _ in ()).throw(AssertionError("network must not be called")),
    )
    with pytest.raises(PermissionError):
        embed_batch.submit(confirm=False)


def test_submit_rejects_over_demo_cap(monkeypatch, tmp_path):
    _redirect_artifacts(monkeypatch, tmp_path)
    _write_chunks(config.CHUNKS_JSONL, ["x"])
    with pytest.raises(ValueError):
        embed_batch.submit(confirm=True, limit=config.DEMO_MAX_CHUNKS + 1)


def test_submit_noop_when_all_cached(monkeypatch, tmp_path):
    _redirect_artifacts(monkeypatch, tmp_path)
    _write_chunks(config.CHUNKS_JSONL, ["only"])
    embed_client.append_cache([(embed_client.content_hash("only"), _vec())])
    # confirm=True but nothing to embed -> empty list, no network.
    embed_client._client.cache_clear()
    monkeypatch.setattr(
        embed_client, "_client",
        lambda: (_ for _ in ()).throw(AssertionError("network must not be called")),
    )
    assert embed_batch.submit(confirm=True) == []


# --------------------------------------------------------------------------- #
# collect: parse output + fan-out to chunk_ids                                #
# --------------------------------------------------------------------------- #
class _FakeBatch:
    def __init__(self, status, output_file_id):
        self.status = status
        self.output_file_id = output_file_id
        self.request_counts = types.SimpleNamespace(completed=1, total=1)


class _FakeFileContent:
    def __init__(self, text):
        self.text = text


def test_collect_parses_and_fans_out(monkeypatch, tmp_path):
    _redirect_artifacts(monkeypatch, tmp_path)
    # Manifest: one hash -> two chunk_ids.
    halpha = embed_client.content_hash("alpha")
    embed_batch._BATCH_MANIFEST.write_text(
        json.dumps({halpha: ["k0", "k3"]}), encoding="utf-8"
    )
    out_line = json.dumps(
        {
            "custom_id": halpha,
            "response": {"body": {"data": [{"embedding": _vec(0.7)}]}},
        }
    )

    class _Batches:
        def retrieve(self, bid):
            return _FakeBatch("completed", "outfile")

    class _Files:
        def content(self, fid):
            return _FakeFileContent(out_line + "\n")

    class _Client:
        batches = _Batches()
        files = _Files()

    embed_client._client.cache_clear()
    monkeypatch.setattr(embed_client, "_client", lambda: _Client())
    monkeypatch.setattr(embed_batch, "_client", lambda: _Client())

    embed_batch.collect(["b1"], poll=False)
    rows = [
        json.loads(l)
        for l in config.EMBEDDINGS_JSONL.read_text(encoding="utf-8").splitlines()
    ]
    assert {r["chunk_id"] for r in rows} == {"k0", "k3"}
    assert all(r["vector"] == _vec(0.7) for r in rows)
    # Vector cached for next run.
    assert halpha in embed_client.load_cache()
