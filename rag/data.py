"""BEIR FiQA-2018: financial opinion QA. 57,638 forum answers, 648 test questions with graded relevance labels."""
from dataclasses import dataclass

import pandas as pd
from huggingface_hub import hf_hub_download


@dataclass
class FiQA:
    doc_ids: list[str]
    docs: list[str]
    qrels: dict[str, dict[str, int]]   # query id -> {doc id: relevance}
    queries: dict[str, str]            # test queries only


def load() -> FiQA:
    corpus = pd.read_parquet(hf_hub_download("BeIR/fiqa", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset"))
    queries = pd.read_parquet(hf_hub_download("BeIR/fiqa", "queries/queries-00000-of-00001.parquet", repo_type="dataset"))
    qrels = pd.read_csv(hf_hub_download("BeIR/fiqa-qrels", "test.tsv", repo_type="dataset"), sep="\t",
                        dtype={"query-id": str, "corpus-id": str})
    rel: dict[str, dict[str, int]] = {}
    for q, c, s in qrels[["query-id", "corpus-id", "score"]].itertuples(index=False):
        rel.setdefault(q, {})[c] = int(s)
    qmap = dict(zip(queries["_id"].astype(str), queries["text"]))
    text = (corpus["title"].fillna("") + " " + corpus["text"].fillna("")).str.strip()
    return FiQA(corpus["_id"].astype(str).tolist(), text.tolist(), rel, {q: qmap[q] for q in rel})
