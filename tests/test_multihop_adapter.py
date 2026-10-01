"""MultiHop-RAG Adapter 测试。"""

import json

import pytest

from evaluation.multihop_adapter import (
    CORPUS_FILENAME,
    QUERY_FILENAME,
    corpus_to_documents,
    documents_to_chunks,
    load_multihop_rag,
    queries_to_cases,
)


def create_corpus_data():
    """
    构造两个假的 MultiHop-RAG Corpus 文档。
    """

    return [
        {
            "title": "Company A acquires Company B",
            "body": (
                "Company A acquired "
                "Company B in 2023."
            ),
            "author": "Alice",
            "source": "News One",
            "published_at": (
                "2023-10-01T00:00:00+00:00"
            ),
            "category": "business",
            "url": "https://example.com/a",
        },

        {
            "title": "Company B appoints a CEO",
            "body": (
                "Bob became the CEO "
                "of Company B."
            ),
            "author": "Carol",
            "source": "News Two",
            "published_at": (
                "2023-10-02T00:00:00+00:00"
            ),
            "category": "business",
            "url": "https://example.com/b",
        },
    ]


def test_corpus_to_documents_includes_metadata():
    """
    MultiHop-RAG 问题可能依赖 metadata，
    所以 metadata 必须进入可检索文本。
    """

    documents = corpus_to_documents(
        create_corpus_data()
    )

    assert len(documents) == 2

    first = documents[0]

    assert (
        "Company A acquires Company B"
        in first.content
    )

    assert (
        "News One"
        in first.content
    )

    assert (
        "2023-10-01"
        in first.content
    )

    assert (
        "Company A acquired Company B"
        in first.content
    )

    assert (
        first.metadata["source"]
        == "News One"
    )


def test_document_ids_are_stable():
    """
    相同 Corpus 多次转换，
    document_id 必须一致。
    """

    first_run = (
        corpus_to_documents(
            create_corpus_data()
        )
    )

    second_run = (
        corpus_to_documents(
            create_corpus_data()
        )
    )

    assert (
        first_run[0].document_id
        == second_run[0].document_id
    )


def test_queries_to_cases_maps_evidence_by_url():
    """
    Gold Evidence 优先通过 URL
    映射回父文档。
    """

    documents = (
        corpus_to_documents(
            create_corpus_data()
        )
    )

    query_data = [
        {
            "query": (
                "Who became CEO of "
                "the acquired company?"
            ),
            "answer": "Bob",
            "question_type": (
                "inference_query"
            ),
            "evidence_list": [
                {
                    "title": (
                        "Company A acquires "
                        "Company B"
                    ),
                    "url": (
                        "https://example.com/a"
                    ),
                    "fact": (
                        "Company A acquired "
                        "Company B in 2023."
                    ),
                },
                {
                    "title": (
                        "Company B appoints "
                        "a CEO"
                    ),
                    "url": (
                        "https://example.com/b"
                    ),
                    "fact": (
                        "Bob became the CEO "
                        "of Company B."
                    ),
                },
            ],
        }
    ]

    cases = queries_to_cases(
        query_data=query_data,
        documents=documents,
    )

    assert len(cases) == 1

    case = cases[0]

    assert (
        case.answer
        == "Bob"
    )

    assert (
        len(
            case.evidence_document_ids
        )
        == 2
    )

    assert (
        len(
            case.evidence_facts
        )
        == 2
    )


def test_queries_to_cases_falls_back_to_title():
    """
    Evidence URL 缺失时，
    唯一 Title 仍然可以映射。
    """

    documents = (
        corpus_to_documents(
            create_corpus_data()
        )
    )

    query_data = [
        {
            "query": "Who is the CEO?",
            "answer": "Bob",
            "question_type": (
                "inference_query"
            ),
            "evidence_list": [
                {
                    "title": (
                        "Company B appoints "
                        "a CEO"
                    ),
                    "fact": (
                        "Bob became the CEO "
                        "of Company B."
                    ),
                }
            ],
        }
    ]

    cases = queries_to_cases(
        query_data=query_data,
        documents=documents,
    )

    assert (
        len(
            cases[0]
            .evidence_document_ids
        )
        == 1
    )


def test_queries_to_cases_rejects_unmapped_gold():
    """
    非 null Query 的 Gold Evidence
    映射失败时必须报错。

    Benchmark 不能悄悄丢 Gold。
    """

    documents = (
        corpus_to_documents(
            create_corpus_data()
        )
    )

    query_data = [
        {
            "query": "Unknown question",
            "answer": "Unknown",
            "question_type": (
                "inference_query"
            ),
            "evidence_list": [
                {
                    "title": (
                        "Document does not exist"
                    ),
                    "url": (
                        "https://example.com/missing"
                    ),
                    "fact": "Missing fact",
                }
            ],
        }
    ]

    with pytest.raises(
        ValueError
    ):
        queries_to_cases(
            query_data=query_data,
            documents=documents,
            strict=True,
        )


def test_null_query_can_have_no_evidence():
    """
    null_query 没有 supporting evidence
    是合法情况。
    """

    documents = (
        corpus_to_documents(
            create_corpus_data()
        )
    )

    query_data = [
        {
            "query": (
                "What is not present "
                "in the knowledge base?"
            ),
            "answer": "Insufficient information",
            "question_type": (
                "null_query"
            ),
            "evidence_list": [],
        }
    ]

    cases = queries_to_cases(
        query_data=query_data,
        documents=documents,
    )

    case = cases[0]

    assert (
        case.question_type
        == "null_query"
    )

    assert (
        case.evidence_document_ids
        == ()
    )

    assert (
        case.evidence_facts
        == ()
    )


def test_documents_to_chunks_preserves_metadata():
    """
    Chunk 必须保留父文档 metadata，
    后续 Failure Analysis 才能定位来源。
    """

    documents = (
        corpus_to_documents(
            create_corpus_data()
        )
    )

    chunks = documents_to_chunks(
        documents=documents,
        chunk_size=100,
        overlap=20,
    )

    assert chunks

    first_chunk = chunks[0]

    assert (
        first_chunk.document_id
        == documents[0].document_id
    )

    assert (
        first_chunk.metadata[
            "dataset"
        ]
        == "MultiHop-RAG"
    )

    assert (
        first_chunk.metadata[
            "source"
        ]
        == "News One"
    )


def test_load_multihop_rag_from_local_files(
    tmp_path,
):
    """
    测试 Loader 本身，
    不访问真实网络。
    """

    corpus_data = (
        create_corpus_data()
    )

    query_data = [
        {
            "query": "Test question",
            "answer": "Test answer",
            "question_type": (
                "null_query"
            ),
            "evidence_list": [],
        }
    ]

    corpus_path = (
        tmp_path
        / CORPUS_FILENAME
    )

    query_path = (
        tmp_path
        / QUERY_FILENAME
    )

    corpus_path.write_text(
        json.dumps(
            corpus_data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    query_path.write_text(
        json.dumps(
            query_data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (
        loaded_corpus,
        loaded_queries,
    ) = load_multihop_rag(
        data_dir=tmp_path,
        download=False,
    )

    assert (
        loaded_corpus
        == corpus_data
    )

    assert (
        loaded_queries
        == query_data
    )


def test_load_multihop_rag_missing_files(
    tmp_path,
):
    """
    download=False 时，
    缺文件应该明确失败。
    """

    with pytest.raises(
        FileNotFoundError
    ):
        load_multihop_rag(
            data_dir=tmp_path,
            download=False,
        )