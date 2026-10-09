"""Retrieval evaluation metrics: Precision@K, Recall@K, mAP and MRR."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tsr.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Query:
    query: str
    relevant: list[str]
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> Query:
        relevant = d.get("relevant") or d.get("gt") or d.get("ground_truth") or []
        if isinstance(relevant, str):
            relevant = [relevant]
        return cls(query=d["query"], relevant=list(relevant), metadata=d.get("metadata", {}))


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / min(len(relevant), len(retrieved)) if retrieved else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary gains."""
    if k <= 0 or not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, doc in enumerate(retrieved[:k]) if doc in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_r(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Recall@R where R is the number of relevant items (standard IR practice)."""
    if not relevant:
        return 0.0
    r = len(relevant)
    hits = sum(1 for doc in retrieved[:r] if doc in relevant)
    return hits / r


def first_relevant_rank(retrieved: Sequence[str], relevant: set[str]) -> int | None:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return i
    return None


# --------------------------------------------------------------------------- #
# Full evaluation
# --------------------------------------------------------------------------- #
def evaluate_queries(
    retriever: Callable[[str, int], list[str]],
    queries: Sequence[Query],
    top_k: Iterable[int] = (1, 5, 10, 20),
    max_rank: int | None = None,
) -> dict:
    top_k = sorted({int(k) for k in top_k})
    limit = max_rank or max(top_k)

    sums = {f"precision@{k}": 0.0 for k in top_k}
    sums.update({f"recall@{k}": 0.0 for k in top_k})
    sums.update({f"ndcg@{k}": 0.0 for k in top_k})
    ap_sum = 0.0
    rr_sum = 0.0
    recall_r_sum = 0.0
    ranks: list[int] = []
    misses = 0
    per_query: list[dict] = []
    per_split: dict[str, dict[str, float]] = {}

    for q in queries:
        relevant = set(q.relevant)
        retrieved = retriever(q.query, limit)
        for k in top_k:
            sums[f"precision@{k}"] += precision_at_k(retrieved, relevant, k)
            sums[f"recall@{k}"] += recall_at_k(retrieved, relevant, k)
            sums[f"ndcg@{k}"] += ndcg_at_k(retrieved, relevant, k)
        ap = average_precision(retrieved, relevant)
        rr = reciprocal_rank(retrieved, relevant)
        r_at_r = recall_at_r(retrieved, relevant)
        ap_sum += ap
        rr_sum += rr
        recall_r_sum += r_at_r
        rank = first_relevant_rank(retrieved, relevant)
        if rank is None:
            misses += 1
        else:
            ranks.append(rank)
        per_query.append({"query": q.query, "rank": rank, "ap": round(ap, 4)})

        split = str(q.metadata.get("split", "all"))
        bucket = per_split.setdefault(split, {"num_queries": 0.0, "mAP": 0.0, "MRR": 0.0, "recall@R": 0.0})
        bucket["num_queries"] += 1
        bucket["mAP"] += ap
        bucket["MRR"] += rr
        bucket["recall@R"] += r_at_r

    n = max(1, len(queries))
    metrics = {name: value / n for name, value in sums.items()}
    metrics["mAP"] = ap_sum / n
    metrics["MRR"] = rr_sum / n
    metrics["recall@R"] = recall_r_sum / n
    metrics["median_rank"] = statistics.median(ranks) if ranks else None
    metrics["misses"] = misses
    metrics["num_queries"] = len(queries)

    splits_out: dict[str, dict[str, float]] = {}
    for split, totals in per_split.items():
        c = max(1.0, totals["num_queries"])
        splits_out[split] = {
            "num_queries": int(totals["num_queries"]),
            "mAP": round(totals["mAP"] / c, 4),
            "MRR": round(totals["MRR"] / c, 4),
            "recall@R": round(totals["recall@R"] / c, 4),
        }
    return {"metrics": metrics, "per_query": per_query, "per_split": splits_out}


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals / significance
# --------------------------------------------------------------------------- #
def per_query_metrics(
    retriever: Callable[[str, int], list[str]],
    queries: Sequence[Query],
    top_k: Iterable[int] = (1, 5, 10),
) -> dict[str, list[float]]:
    top_k = sorted({int(k) for k in top_k})
    limit = max(top_k) if top_k else 10
    values: dict[str, list[float]] = {"mAP": [], "MRR": [], "recall@R": []}
    for k in top_k:
        values[f"precision@{k}"] = []
        values[f"ndcg@{k}"] = []
    for q in queries:
        relevant = set(q.relevant)
        retrieved = retriever(q.query, limit)
        values["mAP"].append(average_precision(retrieved, relevant))
        values["MRR"].append(reciprocal_rank(retrieved, relevant))
        values["recall@R"].append(recall_at_r(retrieved, relevant))
        for k in top_k:
            values[f"precision@{k}"].append(precision_at_k(retrieved, relevant, k))
            values[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))
    return values


def bootstrap_ci(
    values: dict[str, list[float]],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float, float]]:
    """Mean and percentile CI for each metric given per-query values."""
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[float, float, float]] = {}
    for name, arr in values.items():
        data = np.asarray(arr, dtype=np.float64)
        if data.size == 0:
            out[name] = (0.0, 0.0, 0.0)
            continue
        n = data.size
        boot = data[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[name] = (float(data.mean()), float(lo), float(hi))
    return out


def paired_bootstrap_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Paired bootstrap for B-A: returns (delta, lo, hi, two-sided p-value)."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size == 0 or a.size != b.size:
        return (0.0, 0.0, 0.0, 1.0)
    diff = b - a
    n = diff.size
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    return (float(diff.mean()), float(lo), float(hi), min(1.0, p))


# --------------------------------------------------------------------------- #
# Eval-set construction
# --------------------------------------------------------------------------- #
def build_eval_queries_from_manifest(
    manifest: Sequence[dict],
    num_queries: int = 200,
    seed: int = 42,
) -> list[Query]:
    """Create single-relevant queries from gallery captions.

    For each sampled image one caption is used as the query and the image itself
    is the ground-truth relevant item.
    """
    rng = random.Random(seed)
    pool = [rec for rec in manifest if rec.get("captions")]
    rng.shuffle(pool)
    queries: list[Query] = []
    for rec in pool[:num_queries]:
        captions = rec["captions"]
        query_text = rng.choice(captions)
        queries.append(
            Query(
                query=query_text,
                relevant=[rec["image_id"]],
                metadata={"split": rec.get("split"), "image_id": rec["image_id"]},
            )
        )
    return queries


def save_queries(queries: Sequence[Query], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"query": q.query, "relevant": q.relevant, "metadata": q.metadata} for q in queries]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_queries(path: str | Path) -> list[Query]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Query.from_dict(d) for d in data]
