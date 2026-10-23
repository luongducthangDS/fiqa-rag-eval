"""reports/results.json -> reports/figures/quality_vs_latency.png"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "reports/results.json").read_text())["systems"]
fig, ax = plt.subplots(figsize=(6.4, 4))
for name, m in r.items():
    ax.scatter(m["latency_ms_p50"], m["ndcg@10"], s=60, color="#1F3A5F")
    ax.annotate(name, (m["latency_ms_p50"], m["ndcg@10"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("p50 latency per query on laptop CPU (ms, log scale)")
ax.set_ylabel("NDCG@10 on FiQA test (648 queries)")
ax.grid(alpha=.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
(ROOT / "reports/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "reports/figures/quality_vs_latency.png", dpi=150)
