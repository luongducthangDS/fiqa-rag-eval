"""Retrieval ablation on FiQA test: BM25 -> dense -> hybrid (RRF) -> cross-encoder rerank.

Every stage is cached in .cache/, so re-running only recomputes what changed.
    python scripts/run_eval.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.data import load  # noqa: E402
from rag.metrics import mrr_at_k, ndcg_at_k, recall_at_k, rrf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
DENSE = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge v1.5 query instruction
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP, RERANK_DEPTH = 100, 50


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cached(name, fn):
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = fn()
    path.write_text(json.dumps(out))
    return out


def main():
    d = load()
    qids = list(d.queries)
    qtexts = [d.queries[q] for q in qids]
    log(f"corpus {len(d.docs)} docs, {len(qids)} test queries")

    def bm25_stage():
        import bm25s
        import Stemmer
        stem = Stemmer.Stemmer("english")
        t = time.perf_counter()
        idx = bm25s.BM25()
        idx.index(bm25s.tokenize(d.docs, stopwords="en", stemmer=stem, show_progress=False), show_progress=False)
        build = time.perf_counter() - t
        runs, lat = {}, []
        for q, text in zip(qids, qtexts):
            s = time.perf_counter()
            res, _ = idx.retrieve(bm25s.tokenize([text], stopwords="en", stemmer=stem, show_progress=False),
                                  k=TOP, show_progress=False)
            lat.append((time.perf_counter() - s) * 1000)
            runs[q] = [d.doc_ids[i] for i in res[0]]
        return {"runs": runs, "latency_ms": lat, "build_s": build}

    def dense_stage():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(DENSE, device="cpu")
        emb_path = CACHE / "corpus_bge_small.npy"
        t = time.perf_counter()
        if emb_path.exists():
            emb = np.load(emb_path)
        else:
            emb = model.encode(d.docs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            np.save(emb_path, emb)
        build = time.perf_counter() - t
        runs, lat = {}, []
        for q, text in zip(qids, qtexts):
            s = time.perf_counter()
            v = model.encode([QUERY_PREFIX + text], normalize_embeddings=True)[0]
            top = np.argpartition(-(emb @ v), TOP)[:TOP]
            top = top[np.argsort(-(emb[top] @ v))]
            lat.append((time.perf_counter() - s) * 1000)
            runs[q] = [d.doc_ids[i] for i in top]
        return {"runs": runs, "latency_ms": lat, "build_s": build, "index_mb": emb.nbytes / 1e6}

    log("bm25 ...");  bm25 = cached("bm25", bm25_stage)
    log("dense ..."); dense = cached("dense", dense_stage)
    hybrid = {q: rrf([bm25["runs"][q], dense["runs"][q]], top=TOP) for q in qids}

    def rerank_stage():
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(RERANKER, device="cpu")
        text_of = dict(zip(d.doc_ids, d.docs))
        runs, lat = {}, []
        for q, text in zip(qids, qtexts):
            cands = hybrid[q][:RERANK_DEPTH]
            s = time.perf_counter()
            scores = ce.predict([(text, text_of[c]) for c in cands], batch_size=64, show_progress_bar=False)
            lat.append((time.perf_counter() - s) * 1000)
            runs[q] = [cands[i] for i in np.argsort(-scores)] + hybrid[q][RERANK_DEPTH:]
        return {"runs": runs, "latency_ms": lat}

    log("rerank ..."); rerank = cached("rerank_hybrid", rerank_stage)

    systems = {"bm25": bm25["runs"], "dense_bge_small": dense["runs"], "hybrid_rrf": hybrid,
               "hybrid_rrf+rerank": rerank["runs"]}
    per_query = {name: {"ndcg@10": [ndcg_at_k(r[q], d.qrels[q]) for q in qids],
                        "recall@100": [recall_at_k(r[q], d.qrels[q]) for q in qids],
                        "mrr@10": [mrr_at_k(r[q], d.qrels[q]) for q in qids]} for name, r in systems.items()}

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(qids), len(qids)) for _ in range(2000)]

    def paired(a, b):
        diff = np.array(per_query[b]["ndcg@10"]) - np.array(per_query[a]["ndcg@10"])
        bs = [diff[ix].mean() for ix in boots]
        return {"delta": round(float(diff.mean()), 4),
                "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                "queries_better": int((diff > 0).sum()), "queries_worse": int((diff < 0).sum())}

    lat = {"bm25": bm25["latency_ms"], "dense_bge_small": dense["latency_ms"],
           "hybrid_rrf": list(np.array(bm25["latency_ms"]) + np.array(dense["latency_ms"])),
           "hybrid_rrf+rerank": list(np.array(bm25["latency_ms"]) + np.array(dense["latency_ms"])
                                     + np.array(rerank["latency_ms"]))}
    results = {"corpus_docs": len(d.docs), "test_queries": len(qids), "dense_model": DENSE, "reranker": RERANKER,
               "rerank_depth": RERANK_DEPTH, "dense_index_build_s": round(dense["build_s"], 1),
               "dense_index_mb": round(dense.get("index_mb", 0), 1), "bm25_build_s": round(bm25["build_s"], 1),
               "systems": {n: {"ndcg@10": round(float(np.mean(m["ndcg@10"])), 4),
                               "recall@100": round(float(np.mean(m["recall@100"])), 4),
                               "mrr@10": round(float(np.mean(m["mrr@10"])), 4),
                               "latency_ms_p50": round(float(np.percentile(lat[n], 50)), 1),
                               "latency_ms_p95": round(float(np.percentile(lat[n], 95)), 1)}
                           for n, m in per_query.items()},
               "paired_ndcg@10": {"dense_vs_bm25": paired("bm25", "dense_bge_small"),
                                  "hybrid_vs_dense": paired("dense_bge_small", "hybrid_rrf"),
                                  "rerank_vs_hybrid": paired("hybrid_rrf", "hybrid_rrf+rerank")}}
    (ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2))
    (ROOT / "reports" / "per_query.json").write_text(json.dumps({"qids": qids, **per_query}))
    log(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
