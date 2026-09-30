"""Failure Analysis 测试。"""

from evaluation.failure_analysis import (
    build_failure_cases,
    classify_record,
    default_output_paths,
    summarize_categories,
)


def make_record(
    *,
    query_id: str,
    rrf_hit: float,
    rerank_hit: float,
    rrf_recall_100: float,
    gold_in_pool: bool,
    first_rank=None,
):
    """构造一条假的 Benchmark record。"""

    return {
        "query_id": query_id,

        "query": (
            f"query {query_id}"
        ),

        "relevant_chunk_ids": [
            f"gold_{query_id}"
        ],

        "candidate_analysis": {
            "rrf_recall@5": (
                float(rrf_hit)
            ),

            "rrf_recall@10": (
                float(rrf_hit)
            ),

            "rrf_recall@20": (
                1.0
                if gold_in_pool
                else 0.0
            ),

            "rrf_recall@100": (
                rrf_recall_100
            ),

            "rrf_relevant_ranks": (
                [first_rank]
                if first_rank
                is not None
                else []
            ),

            "rrf_first_relevant_rank": (
                first_rank
            ),

            "gold_in_rerank_pool": (
                gold_in_pool
            ),
        },

        "methods": {
            "bm25": {
                "hit@5": 1.0,
            },

            "dense": {
                "hit@5": 0.0,
            },

            "rrf": {
                "hit@5": rrf_hit,
            },

            "rerank": {
                "hit@5": rerank_hit,
            },
        },
    }


def test_classify_stable_success():
    record = make_record(
        query_id="1",
        rrf_hit=1.0,
        rerank_hit=1.0,
        rrf_recall_100=1.0,
        gold_in_pool=True,
        first_rank=1,
    )

    assert classify_record(
        record,
        top_k=5,
    ) == "stable_success"


def test_classify_reranker_rescue():
    record = make_record(
        query_id="2",
        rrf_hit=0.0,
        rerank_hit=1.0,
        rrf_recall_100=1.0,
        gold_in_pool=True,
        first_rank=12,
    )

    assert classify_record(
        record,
        top_k=5,
    ) == "reranker_rescue"


def test_classify_reranker_harm():
    record = make_record(
        query_id="3",
        rrf_hit=1.0,
        rerank_hit=0.0,
        rrf_recall_100=1.0,
        gold_in_pool=True,
        first_rank=3,
    )

    assert classify_record(
        record,
        top_k=5,
    ) == "reranker_harm"


def test_classify_reranker_failure():
    record = make_record(
        query_id="4",
        rrf_hit=0.0,
        rerank_hit=0.0,
        rrf_recall_100=1.0,
        gold_in_pool=True,
        first_rank=18,
    )

    assert classify_record(
        record,
        top_k=5,
    ) == "reranker_failure"


def test_classify_candidate_pool_miss():
    record = make_record(
        query_id="5",
        rrf_hit=0.0,
        rerank_hit=0.0,
        rrf_recall_100=1.0,
        gold_in_pool=False,
        first_rank=37,
    )

    assert classify_record(
        record,
        top_k=5,
    ) == "candidate_pool_miss"


def test_classify_retrieval_miss():
    record = make_record(
        query_id="6",
        rrf_hit=0.0,
        rerank_hit=0.0,
        rrf_recall_100=0.0,
        gold_in_pool=False,
        first_rank=None,
    )

    assert classify_record(
        record,
        top_k=5,
    ) == "retrieval_miss"


def test_summarize_categories():
    records = [
        make_record(
            query_id="1",
            rrf_hit=1.0,
            rerank_hit=1.0,
            rrf_recall_100=1.0,
            gold_in_pool=True,
            first_rank=1,
        ),

        make_record(
            query_id="2",
            rrf_hit=0.0,
            rerank_hit=1.0,
            rrf_recall_100=1.0,
            gold_in_pool=True,
            first_rank=8,
        ),

        make_record(
            query_id="3",
            rrf_hit=1.0,
            rerank_hit=0.0,
            rrf_recall_100=1.0,
            gold_in_pool=True,
            first_rank=2,
        ),

        make_record(
            query_id="4",
            rrf_hit=0.0,
            rerank_hit=0.0,
            rrf_recall_100=1.0,
            gold_in_pool=True,
            first_rank=18,
        ),

        make_record(
            query_id="5",
            rrf_hit=0.0,
            rerank_hit=0.0,
            rrf_recall_100=1.0,
            gold_in_pool=False,
            first_rank=30,
        ),

        make_record(
            query_id="6",
            rrf_hit=0.0,
            rerank_hit=0.0,
            rrf_recall_100=0.0,
            gold_in_pool=False,
            first_rank=None,
        ),
    ]

    summary = summarize_categories(
        records,
        top_k=5,
    )

    assert (
        summary["total_queries"]
        == 6
    )

    assert (
        summary["counts"][
            "stable_success"
        ]
        == 1
    )

    assert (
        summary["counts"][
            "reranker_rescue"
        ]
        == 1
    )

    assert (
        summary["counts"][
            "reranker_harm"
        ]
        == 1
    )

    assert (
        summary["counts"][
            "reranker_failure"
        ]
        == 1
    )

    assert (
        summary["counts"][
            "candidate_pool_miss"
        ]
        == 1
    )

    assert (
        summary["counts"][
            "retrieval_miss"
        ]
        == 1
    )

    # stable_success + reranker_harm
    assert (
        summary["derived"][
            "rrf_hit@5_queries"
        ]
        == 2
    )

    # stable_success + reranker_rescue
    assert (
        summary["derived"][
            "rerank_hit@5_queries"
        ]
        == 2
    )


def test_failure_cases_exclude_stable_success():
    records = [
        make_record(
            query_id="1",
            rrf_hit=1.0,
            rerank_hit=1.0,
            rrf_recall_100=1.0,
            gold_in_pool=True,
            first_rank=1,
        ),

        make_record(
            query_id="2",
            rrf_hit=0.0,
            rerank_hit=1.0,
            rrf_recall_100=1.0,
            gold_in_pool=True,
            first_rank=10,
        ),
    ]

    rows = build_failure_cases(
        records,
        top_k=5,
    )

    assert len(rows) == 1

    assert (
        rows[0]["query_id"]
        == "2"
    )

    assert (
        rows[0]["category"]
        == "reranker_rescue"
    )


def test_default_output_paths():
    summary_path, csv_path = (
        default_output_paths(
            "evaluation/results/test.json"
        )
    )

    assert (
        summary_path.name
        == "test_failure_analysis.json"
    )

    assert (
        csv_path.name
        == "test_failure_cases.csv"
    )