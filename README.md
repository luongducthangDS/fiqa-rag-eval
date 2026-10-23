# fiqa-rag-eval: measuring retrieval trade-offs in financial RAG

A systematic retrieval ablation on the **FiQA-2018 financial QA benchmark** (57,638 financial documents, 648 test queries with human relevance judgements):
how much does each stage of a production RAG retriever actually add, and what does it cost in latency and memory?

1. **Lexical baseline**: BM25 with English stemming and stopwords (`bm25s`).
2. **Dense retriever**: `BAAI/bge-small-en-v1.5` (33M params, normalized embeddings, dot-product search).
3. **Hybrid retrieval**: Reciprocal Rank Fusion (RRF, $k=60$) combining BM25 and dense rank lists.
4. **Cross-encoder reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking the top-50 candidates.

## Core Methodology

- **Standard IR metrics**: NDCG@10 (ranking quality), Recall@100 (candidate pool ceiling for generation), and MRR@10 (first relevant hit).
- **Paired bootstrap testing**: 2,000 bootstrap resamples on the per-query metric differences to compute 95% confidence intervals on $\Delta\text{NDCG@10}$ and count queries improved vs degraded.
- **Latency accounting**: Per-query p50 and p95 retrieval latency measured end-to-end on CPU.
- **Stage caching**: Each stage caches candidate rankings in `.cache/`, so reranking depth or metric changes do not re-encode the corpus.

## Architecture

```
User Query
    │
    ├──► BM25 (bm25s) ───────────────────────► Top-100 Candidates ──┐
    │                                                                ▼
    └──► Dense (bge-small-en-v1.5, CPU) ─────► Top-100 Candidates ──┴─► RRF Fusion (k=60)
                                                                            │
                                                                            ▼ Top-50 candidates
                                                                    Cross-Encoder Reranker
                                                                            │
                                                                            ▼
                                                                    Final Top-K Contexts
```

## Run It

```bash
# 1. Run unit tests (IR metrics correctness, NDCG@k, Recall@k, MRR@k, RRF agreement)
pytest -q

# 2. Run the end-to-end evaluation pipeline
python scripts/run_eval.py
```

## Design Decisions (and why)

- **RRF over Convex Score Combination:** BM25 scores (unbounded log-odds) and cosine similarities have wildly different scales and distributions across queries. Reciprocal Rank Fusion uses only ordinal rank ($1 / (k + \text{rank})$), eliminating score calibration bugs.
- **Two-stage retrieval (Retrieve 100, Rerank 50):** Cross-encoders evaluate query-document pairs jointly, making them $O(N)$ transformer forward passes. Reranking top-50 balances latency ($<150\text{ ms}$) with capture rate.
- **Paired statistical testing:** Reporting only mean NDCG across 648 queries hides whether a gain comes from big improvements on 5 queries or consistent gains across 300 queries. Paired bootstrap reveals true consistency.
- **FiQA over general trivia:** Financial queries have dense terminology (tickers, financial metrics, ratios) where pure semantic dense models often lose to exact lexical matching, making hybrid search essential.

## Interview questions this project prepares for

1. Why do dense embedding models often underperform BM25 on specialized financial acronyms, and how does RRF fix this?
2. What is the mathematical difference between MRR@10 and NDCG@10, and when is each metric the right north star?
3. Why can a Bi-Encoder embed documents offline while a Cross-Encoder cannot?
4. How would you choose between $k=60$ vs $k=20$ in Reciprocal Rank Fusion?
