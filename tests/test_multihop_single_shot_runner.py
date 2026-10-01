"""MultiHop Single-shot Runner 测试。"""

import pytest

from evaluation.multihop_single_shot_runner import (
    compute_context_diagnostics,
    extract_answer,
    load_checkpoint,
    save_checkpoint,
    summarize_records,
)


def make_record(
    *,
    question_type: str,
    strict_correct: float,
    gold_doc_recall: float,
    all_gold_docs_present: bool,
):
    return {
        "question_type": (
            question_type
        ),

        "qa_metrics": {
            "official_match": (
                strict_correct
            ),

            "exact_match": (
                strict_correct
            ),

            "gold_containment": (
                strict_correct
            ),

            "token_f1": (
                strict_correct
            ),

            "strict_answer_match": (
                strict_correct
            ),

            "semantic_review_candidate": (
                False
            ),
        },

        "context_diagnostics": {
            "gold_doc_recall": (
                gold_doc_recall
            ),

            "all_gold_docs_present": (
                all_gold_docs_present
            ),

            "unique_context_documents": 4,
        },

        "retrieval_latency_ms": 100.0,
        "llm_latency_ms": 900.0,

        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
        },
    }


def test_extract_answer_from_json():
    assert (
        extract_answer(
            '{"answer": "Google"}'
        )
        == "Google"
    )


def test_extract_answer_plain_text():
    assert (
        extract_answer(
            "Google"
        )
        == "Google"
    )


def test_compute_context_diagnostics():
    diagnostics = (
        compute_context_diagnostics(
            retrieved_document_ids=[
                "a",
                "x",
                "b",
            ],

            gold_document_ids=[
                "a",
                "b",
            ],

            top_k=3,
        )
    )

    assert (
        diagnostics[
            "gold_doc_recall"
        ]
        == 1.0
    )

    assert (
        diagnostics[
            "all_gold_docs_present"
        ]
        is True
    )


def test_compute_context_diagnostics_partial():
    diagnostics = (
        compute_context_diagnostics(
            retrieved_document_ids=[
                "a",
                "x",
            ],

            gold_document_ids=[
                "a",
                "b",
            ],

            top_k=2,
        )
    )

    assert (
        diagnostics[
            "gold_doc_recall"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        diagnostics[
            "all_gold_docs_present"
        ]
        is False
    )


def test_summarize_records():
    records = [
        make_record(
            question_type=(
                "inference_query"
            ),

            strict_correct=1.0,

            gold_doc_recall=1.0,

            all_gold_docs_present=True,
        ),

        make_record(
            question_type=(
                "comparison_query"
            ),

            strict_correct=0.0,

            gold_doc_recall=0.5,

            all_gold_docs_present=False,
        ),
    ]

    summary = (
        summarize_records(
            records
        )
    )

    assert (
        summary[
            "strict_answer_accuracy"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        summary[
            "context"
        ][
            "avg_gold_doc_recall"
        ]
        == pytest.approx(
            0.75
        )
    )

    assert (
        summary[
            "latency"
        ][
            "avg_online_pipeline_latency_ms"
        ]
        == pytest.approx(
            1000.0
        )
    )


def test_checkpoint_round_trip(
    tmp_path,
):
    path = (
        tmp_path
        / "checkpoint.json"
    )

    metadata = {
        "model": "test-model",
        "top_k": 5,
    }

    records = [
        {
            "case_id": "case_1",
            "error": None,
        }
    ]

    save_checkpoint(
        path,
        metadata,
        records,
    )

    loaded = load_checkpoint(
        path,
        metadata,
    )

    assert loaded == records


def test_checkpoint_rejects_wrong_metadata(
    tmp_path,
):
    path = (
        tmp_path
        / "checkpoint.json"
    )

    save_checkpoint(
        path,
        {
            "model": "a",
        },
        [],
    )

    with pytest.raises(
        ValueError
    ):
        load_checkpoint(
            path,
            {
                "model": "b",
            },
        )