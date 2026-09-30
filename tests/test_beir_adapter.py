"""BEIR Adapter 测试。"""

from evaluation.beir_adapter import (
    beir_corpus_to_chunks,
)


def test_beir_corpus_to_chunks():
    corpus = {
        "doc_1": {
            "title": "RAG",
            "text": "RAG 是检索增强生成。",
        },
        "doc_2": {
            "title": "",
            "text": "FAISS 用于向量检索。",
        },
    }

    chunks = beir_corpus_to_chunks(
        corpus=corpus,
        dataset_name="fake_dataset",
    )

    assert len(chunks) == 2

    first = chunks[0]

    assert first.chunk_id == "doc_1"
    assert first.document_id == "doc_1"

    assert first.content == (
        "RAG\n"
        "RAG 是检索增强生成。"
    )

    assert first.filename == (
        "beir:fake_dataset:doc_1"
    )

    assert first.metadata["dataset"] == (
        "fake_dataset"
    )

    assert first.metadata["title"] == "RAG"


def test_beir_chunk_id_matches_document_id():
    corpus = {
        "123": {
            "title": "Example",
            "text": "Example text",
        }
    }

    chunks = beir_corpus_to_chunks(
        corpus=corpus,
        dataset_name="scifact",
    )

    assert chunks[0].chunk_id == "123"
    assert chunks[0].document_id == "123"


def test_beir_document_without_title():
    corpus = {
        "doc_1": {
            "text": "只有正文。",
        }
    }

    chunks = beir_corpus_to_chunks(
        corpus=corpus,
        dataset_name="scifact",
    )

    assert chunks[0].content == "只有正文。"


def test_beir_document_without_text():
    corpus = {
        "doc_1": {
            "title": "只有标题",
            "text": "",
        }
    }

    chunks = beir_corpus_to_chunks(
        corpus=corpus,
        dataset_name="scifact",
    )

    assert chunks[0].content == "只有标题"