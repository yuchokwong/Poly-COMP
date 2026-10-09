from tsr.evaluate import (
    Query,
    average_precision,
    bootstrap_ci,
    evaluate_queries,
    ndcg_at_k,
    paired_bootstrap_test,
    per_query_metrics,
    precision_at_k,
    recall_at_k,
    recall_at_r,
    reciprocal_rank,
)


def test_precision_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert precision_at_k(retrieved, relevant, 0) == 0.0


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}
    assert recall_at_k(retrieved, relevant, 4) == 2 / 3
    assert recall_at_k(retrieved, relevant, 1) == 1 / 3
    assert recall_at_k(retrieved, set(), 4) == 0.0


def test_average_precision():
    assert average_precision(["a", "b", "c"], {"a", "c"}) == (1.0 + 2 / 3) / 2
    assert average_precision(["b", "c", "a"], {"a"}) == 1 / 3
    assert average_precision(["b", "c"], {"a"}) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["b", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["b", "c"], {"a"}) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0
    # relevant at rank 2: dcg = 1/log2(3); ideal = 1 -> ~0.6309
    assert abs(ndcg_at_k(["x", "a"], {"a"}, 2) - (1 / 1.584962500721156)) < 1e-9
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert ndcg_at_k(["a"], set(), 1) == 0.0


def test_recall_at_r():
    # R = number of relevant items; only the top-R results are considered.
    assert recall_at_r(["a", "c", "b"], {"a", "c"}) == 1.0
    assert recall_at_r(["a", "b", "c"], {"a", "c"}) == 0.5
    assert recall_at_r(["a", "b", "c"], {"a", "z"}) == 0.5
    assert recall_at_r(["a"], set()) == 0.0


def test_evaluate_queries_perfect_retriever():
    queries = [Query(query="q1", relevant=["1"]), Query(query="q2", relevant=["2"])]
    gallery = {"q1": ["1", "9", "8"], "q2": ["2", "9", "8"]}

    def retriever(query, k):
        return gallery[query][:k]

    report = evaluate_queries(retriever, queries, top_k=[1, 3])
    metrics = report["metrics"]
    assert metrics["precision@1"] == 1.0
    assert metrics["recall@1"] == 1.0
    assert metrics["mAP"] == 1.0
    assert metrics["MRR"] == 1.0
    assert metrics["misses"] == 0


def test_evaluate_queries_misses():
    queries = [Query(query="q1", relevant=["1"])]

    def retriever(query, k):
        return ["x", "y"]

    report = evaluate_queries(retriever, queries, top_k=[1, 2])
    assert report["metrics"]["precision@1"] == 0.0
    assert report["metrics"]["mAP"] == 0.0
    assert report["metrics"]["misses"] == 1
    assert report["metrics"]["median_rank"] is None


def test_evaluate_queries_per_split_and_ndcg():
    queries = [
        Query(query="q1", relevant=["1"], metadata={"split": "val"}),
        Query(query="q2", relevant=["2"], metadata={"split": "train"}),
    ]
    gallery = {"q1": ["1", "9"], "q2": ["9", "2"]}

    def retriever(query, k):
        return gallery[query][:k]

    report = evaluate_queries(retriever, queries, top_k=[1, 2])
    assert report["per_split"]["val"]["mAP"] == 1.0
    assert report["per_split"]["train"]["mAP"] == 0.5
    assert report["per_split"]["val"]["num_queries"] == 1
    assert report["metrics"]["ndcg@1"] == 0.5
    assert report["metrics"]["recall@R"] == 0.5


def test_per_query_metrics_and_bootstrap_ci():
    queries = [Query(query="q1", relevant=["1"]), Query(query="q2", relevant=["2"])]
    gallery = {"q1": ["1", "x"], "q2": ["x", "2"]}

    def retriever(query, k):
        return gallery[query][:k]

    values = per_query_metrics(retriever, queries, top_k=[1, 2])
    assert values["precision@1"] == [1.0, 0.0]
    assert "mAP" in values and "recall@R" in values

    ci = bootstrap_ci(values, n_boot=200, seed=0)
    mean, lo, hi = ci["mAP"]
    assert abs(mean - 0.75) < 1e-9
    assert lo <= mean <= hi


def test_paired_bootstrap_test_detects_difference():
    baseline = [0.0] * 20
    better = [1.0] * 20
    delta, lo, hi, p = paired_bootstrap_test(baseline, better, n_boot=200, seed=0)
    assert delta == 1.0
    assert lo == 1.0 and hi == 1.0
    assert p <= 0.05

    same = [0.5] * 20
    assert paired_bootstrap_test(same, same, n_boot=100)[0] == 0.0
    assert paired_bootstrap_test([], [], n_boot=10)[3] == 1.0


