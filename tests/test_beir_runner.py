"""正式 BEIR Runner 辅助函数测试。"""

from types import SimpleNamespace

from evaluation.beir_runner import (
    evaluate_beir_metrics,
    get_relevant_ids,
    get_relevant_ranks,
    normalize_qrels,
    ranking_to_beir_scores,
    result_chunk_ids,
    score_project_ranking,
    select_query_items,
    summarize_candidate_pool,
    summarize_records,
)


def make_result(
    chunk_id: str,
):
    """构造假的 RetrievalResult。"""

    return SimpleNamespace(
        chunk=SimpleNamespace(
            chunk_id=chunk_id,
        )
    )


def test_result_chunk_ids():
    results = [
        make_result("doc_1"),
        make_result("doc_2"),
    ]

    assert result_chunk_ids(
        results
    ) == [
        "doc_1",
        "doc_2",
    ]


def test_get_relevant_ids():
    qrels = {
        "doc_1": 1,
        "doc_2": 0,
        "doc_3": 2,
    }

    assert get_relevant_ids(
        qrels
    ) == [
        "doc_1",
        "doc_3",
    ]


def test_get_relevant_ranks():
    retrieved = [
        "wrong_1",
        "gold_1",
        "wrong_2",
        "gold_2",
    ]

    relevant = [
        "gold_1",
        "gold_2",
    ]

    ranks = get_relevant_ranks(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
    )

    assert ranks == [
        2,
        4,
    ]


def test_get_relevant_ranks_not_found():
    retrieved = [
        "wrong_1",
        "wrong_2",
    ]

    relevant = [
        "gold_1",
    ]

    assert get_relevant_ranks(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
    ) == []


def test_normalize_qrels():
    qrels = {
        1: {
            100: 1,
            200: 0,
        }
    }

    normalized = normalize_qrels(
        qrels
    )

    assert normalized == {
        "1": {
            "100": 1,
            "200": 0,
        }
    }


def test_ranking_to_beir_scores():
    results = [
        make_result("doc_a"),
        make_result("doc_b"),
        make_result("doc_c"),
    ]

    scores = ranking_to_beir_scores(
        results
    )

    assert scores == {
        "doc_a": 3.0,
        "doc_b": 2.0,
        "doc_c": 1.0,
    }

    assert (
        scores["doc_a"]
        > scores["doc_b"]
        > scores["doc_c"]
    )


def test_score_project_ranking():
    retrieved = [
        "wrong",
        "gold_1",
        "gold_2",
    ]

    relevant = [
        "gold_1",
        "gold_2",
    ]

    scores = score_project_ranking(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
        k=3,
    )

    assert scores["hit@3"] == 1.0
    assert scores["recall@3"] == 1.0

    # 第一个 Gold 在 Rank 2。
    assert scores["rr@3"] == 0.5


def test_summarize_records():
    records = [
        {
            "methods": {
                "bm25": {
                    "hit@5": 1.0,
                    "recall@5": 0.5,
                    "rr@5": 1.0,
                    "latency_ms": 10.0,
                }
            }
        },
        {
            "methods": {
                "bm25": {
                    "hit@5": 0.0,
                    "recall@5": 0.0,
                    "rr@5": 0.0,
                    "latency_ms": 20.0,
                }
            }
        },
    ]

    summary = summarize_records(
        records=records,
        method_name="bm25",
        k=5,
    )

    assert summary["hit@5"] == 0.5
    assert summary["recall@5"] == 0.25
    assert summary["mrr@5"] == 0.5
    assert summary["avg_latency_ms"] == 15.0


def test_summarize_candidate_pool():
    records = [
        {
            "candidate_analysis": {
                "rrf_recall@5": 0.0,
                "rrf_recall@10": 1.0,
                "rrf_recall@20": 1.0,
                "rrf_recall@100": 1.0,
                "gold_in_rerank_pool": True,
            }
        },
        {
            "candidate_analysis": {
                "rrf_recall@5": 0.0,
                "rrf_recall@10": 0.0,
                "rrf_recall@20": 0.0,
                "rrf_recall@100": 1.0,
                "gold_in_rerank_pool": False,
            }
        },
    ]

    summary = summarize_candidate_pool(
        records
    )

    assert summary["rrf_recall@5"] == 0.0
    assert summary["rrf_recall@10"] == 0.5
    assert summary["rrf_recall@20"] == 0.5
    assert summary["rrf_recall@100"] == 1.0

    assert summary["rrf_hit@20"] == 0.5

    assert (
        summary[
            "queries_with_gold_in_top20"
        ]
        == 1
    )

    assert (
        summary[
            "queries_without_gold_in_top20"
        ]
        == 1
    )


def test_select_query_items_is_deterministic():
    queries = {
        "1": "query 1",
        "2": "query 2",
        "3": "query 3",
        "4": "query 4",
    }

    qrels = {
        "1": {
            "doc_1": 1,
        },
        "2": {
            "doc_2": 1,
        },
        "3": {
            "doc_3": 1,
        },
        "4": {
            "doc_4": 1,
        },
    }

    first = select_query_items(
        queries=queries,
        qrels=qrels,
        limit=2,
        seed=42,
    )

    second = select_query_items(
        queries=queries,
        qrels=qrels,
        limit=2,
        seed=42,
    )

    assert first == second
    assert len(first) == 2


def test_select_query_items_full_dataset():
    queries = {
        "1": "query 1",
        "2": "query 2",
    }

    qrels = {
        "1": {
            "doc_1": 1,
        },
        "2": {
            "doc_2": 1,
        },
    }

    selected = select_query_items(
        queries=queries,
        qrels=qrels,
        limit=0,
        seed=42,
    )

    assert selected == [
        (
            "1",
            "query 1",
        ),
        (
            "2",
            "query 2",
        ),
    ]


def test_beir_official_metrics():
    """验证 BEIR 官方 Evaluator。"""

    qrels = {
        "q1": {
            "gold_doc": 1,
        }
    }

    results = {
        "q1": {
            "gold_doc": 2.0,
            "wrong_doc": 1.0,
        }
    }

    metrics = evaluate_beir_metrics(
        qrels=qrels,
        results=results,
        k_values=[
            1,
            2,
        ],
    )

    assert (
        metrics["ndcg"]["NDCG@1"]
        == 1.0
    )

    assert (
        metrics["map"]["MAP@1"]
        == 1.0
    )

    assert (
        metrics["recall"]["Recall@1"]
        == 1.0
    )

    assert (
        metrics["precision"]["P@1"]
        == 1.0
    )

    assert (
        metrics["mrr"]["MRR@1"]
        == 1.0
    )


def test_beir_metrics_when_gold_is_second():
    qrels = {
        "q1": {
            "gold_doc": 1,
        }
    }

    results = {
        "q1": {
            "wrong_doc": 2.0,
            "gold_doc": 1.0,
        }
    }

    metrics = evaluate_beir_metrics(
        qrels=qrels,
        results=results,
        k_values=[
            1,
            2,
        ],
    )

    assert (
        metrics["recall"]["Recall@1"]
        == 0.0
    )

    assert (
        metrics["recall"]["Recall@2"]
        == 1.0
    )

    assert (
        metrics["mrr"]["MRR@2"]
        == 0.5
    )