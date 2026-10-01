"""MultiHop-RAG Retrieval Runner 测试。"""

import pytest

from evaluation.multihop_adapter import (
    MultiHopCase,
)
from evaluation.multihop_retrieval_runner import (
    score_evidence_documents,
    score_official_facts,
    select_benchmark_cases,
)
from retrieval.document import (
    DocumentChunk,
)
from retrieval.result import (
    RetrievalResult,
)


def make_result(
    chunk_id: str,
    document_id: str,
    content: str,
    rank: int,
) -> RetrievalResult:
    """创建测试用 RetrievalResult。"""

    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        filename="test.txt",
    )

    return RetrievalResult(
        chunk=chunk,
        score=float(
            100 - rank
        ),
        rank=rank,
    )


def make_case(
    case_id: str,
    question_type: str,
) -> MultiHopCase:
    """创建测试用 MultiHopCase。"""

    return MultiHopCase(
        case_id=case_id,
        query="test query",
        answer="test answer",
        question_type=(
            question_type
        ),
        evidence_facts=(
            "Fact A",
            "Fact B",
        ),
        evidence_document_ids=(
            "doc_a",
            "doc_b",
        ),
    )


def test_score_official_facts():
    """
    Gold A 在 Rank 2。
    Gold B 在 Rank 4。

    因此：
        Hits@4  = 1
        Hits@10 = 1
        MRR@10  = 1/2

        MAP@10
        =
        (1/2 + 1/4) / 2
        =
        0.375
    """

    retrieved_texts = [
        "irrelevant",
        "This contains Fact A.",
        "still irrelevant",
        "This contains Fact B.",
    ]

    metrics = (
        score_official_facts(
            retrieved_texts=(
                retrieved_texts
            ),
            gold_facts=(
                "Fact A",
                "Fact B",
            ),
        )
    )

    assert (
        metrics["hits@4"]
        == 1.0
    )

    assert (
        metrics["hits@10"]
        == 1.0
    )

    assert (
        metrics["mrr@10"]
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics["map@10"]
        == pytest.approx(
            0.375
        )
    )


def test_score_official_facts_ignores_duplicate_gold():
    """
    同一个 Gold Fact
    被两个重叠 Chunk 找到时，
    MAP 不能重复计分。
    """

    metrics = (
        score_official_facts(
            retrieved_texts=[
                "Fact A",
                "Fact A again",
            ],
            gold_facts=[
                "Fact A",
            ],
        )
    )

    assert (
        metrics["map@10"]
        == pytest.approx(
            1.0
        )
    )


def test_score_official_facts_empty_gold():
    metrics = (
        score_official_facts(
            retrieved_texts=[
                "anything"
            ],
            gold_facts=[],
        )
    )

    assert (
        metrics["hits@4"]
        == 0.0
    )

    assert (
        metrics["hits@10"]
        == 0.0
    )

    assert (
        metrics["map@10"]
        == 0.0
    )

    assert (
        metrics["mrr@10"]
        == 0.0
    )


def test_score_evidence_documents_partial_coverage():
    """
    Gold Document:
        A
        B

    Top-2 Chunk:
        A
        X

    因此：
        Hit = 1
        Recall = 1/2
        All Evidence = 0
    """

    results = [
        make_result(
            chunk_id="a1",
            document_id="doc_a",
            content="A",
            rank=1,
        ),

        make_result(
            chunk_id="x1",
            document_id="doc_x",
            content="X",
            rank=2,
        ),

        make_result(
            chunk_id="b1",
            document_id="doc_b",
            content="B",
            rank=3,
        ),
    ]

    metrics = (
        score_evidence_documents(
            results=results,
            gold_document_ids=(
                "doc_a",
                "doc_b",
            ),
            k=2,
        )
    )

    assert (
        metrics["doc_hit@2"]
        == 1.0
    )

    assert (
        metrics["doc_recall@2"]
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics[
            "all_evidence_hit@2"
        ]
        == 0.0
    )


def test_score_evidence_documents_full_coverage():
    """
    Top-3 已同时包含 A 和 B，
    因此 All-Evidence 成功。
    """

    results = [
        make_result(
            chunk_id="a1",
            document_id="doc_a",
            content="A",
            rank=1,
        ),

        make_result(
            chunk_id="x1",
            document_id="doc_x",
            content="X",
            rank=2,
        ),

        make_result(
            chunk_id="b1",
            document_id="doc_b",
            content="B",
            rank=3,
        ),
    ]

    metrics = (
        score_evidence_documents(
            results=results,
            gold_document_ids=(
                "doc_a",
                "doc_b",
            ),
            k=3,
        )
    )

    assert (
        metrics["doc_recall@3"]
        == 1.0
    )

    assert (
        metrics[
            "all_evidence_hit@3"
        ]
        == 1.0
    )


def test_score_evidence_documents_duplicate_chunks_do_not_fake_coverage():
    """
    同一个父文档出现多个 Chunk，
    不能假装找到了多个 Gold Document。
    """

    results = [
        make_result(
            chunk_id="a1",
            document_id="doc_a",
            content="A part 1",
            rank=1,
        ),

        make_result(
            chunk_id="a2",
            document_id="doc_a",
            content="A part 2",
            rank=2,
        ),
    ]

    metrics = (
        score_evidence_documents(
            results=results,
            gold_document_ids=(
                "doc_a",
                "doc_b",
            ),
            k=2,
        )
    )

    assert (
        metrics["doc_recall@2"]
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics[
            "all_evidence_hit@2"
        ]
        == 0.0
    )


def test_score_evidence_documents_invalid_k():
    with pytest.raises(
        ValueError
    ):
        score_evidence_documents(
            results=[],
            gold_document_ids=[
                "doc_a"
            ],
            k=0,
        )


def test_select_benchmark_cases_is_stratified():
    """
    limit=20 时：

        inference  = 7
        comparison = 7
        temporal   = 6

    null_query 不参与 Retrieval Eval。
    """

    cases = []

    for index in range(30):
        cases.append(
            make_case(
                case_id=(
                    f"i_{index:02d}"
                ),
                question_type=(
                    "inference_query"
                ),
            )
        )

        cases.append(
            make_case(
                case_id=(
                    f"c_{index:02d}"
                ),
                question_type=(
                    "comparison_query"
                ),
            )
        )

        cases.append(
            make_case(
                case_id=(
                    f"t_{index:02d}"
                ),
                question_type=(
                    "temporal_query"
                ),
            )
        )

    for index in range(10):
        cases.append(
            make_case(
                case_id=(
                    f"n_{index:02d}"
                ),
                question_type=(
                    "null_query"
                ),
            )
        )

    selected = (
        select_benchmark_cases(
            cases=cases,
            limit=20,
            seed=42,
        )
    )

    assert len(selected) == 20

    counts = {
        question_type: sum(
            1
            for case in selected
            if (
                case.question_type
                == question_type
            )
        )
        for question_type in (
            "inference_query",
            "comparison_query",
            "temporal_query",
            "null_query",
        )
    }

    assert (
        counts[
            "inference_query"
        ]
        == 7
    )

    assert (
        counts[
            "comparison_query"
        ]
        == 7
    )

    assert (
        counts[
            "temporal_query"
        ]
        == 6
    )

    assert (
        counts[
            "null_query"
        ]
        == 0
    )


def test_select_benchmark_cases_is_deterministic():
    """
    同一个 seed 必须得到相同 Case。
    """

    cases = []

    for index in range(20):
        for question_type in (
            "inference_query",
            "comparison_query",
            "temporal_query",
        ):
            cases.append(
                make_case(
                    case_id=(
                        f"{question_type}_"
                        f"{index}"
                    ),
                    question_type=(
                        question_type
                    ),
                )
            )

    first = (
        select_benchmark_cases(
            cases=cases,
            limit=12,
            seed=42,
        )
    )

    second = (
        select_benchmark_cases(
            cases=cases,
            limit=12,
            seed=42,
        )
    )

    assert [
        case.case_id
        for case in first
    ] == [
        case.case_id
        for case in second
    ]