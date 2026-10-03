"""
BEIR SciFact 真实数据加载检查：python -m evaluation.smoke_beir。

首次运行可能下载公开数据；只打印加载摘要，不生成正式实验报告。
"""

from evaluation.beir_adapter import (
    beir_corpus_to_chunks,
    load_beir_dataset,
)


def main():
    print("正在加载 BEIR SciFact...")

    corpus, queries, qrels = (
        load_beir_dataset(
            dataset_name="scifact",
            split="test",
        )
    )

    chunks = beir_corpus_to_chunks(
        corpus=corpus,
        dataset_name="scifact",
    )

    print(
        f"Corpus documents: {len(corpus)}"
    )

    print(
        f"Queries: {len(queries)}"
    )

    print(
        f"Qrels queries: {len(qrels)}"
    )

    print(
        f"Converted chunks: {len(chunks)}"
    )

    # 展示一条真实查询及其相关性标注，便于确认数据加载结果。
    query_id = next(
        iter(queries)
    )

    print()
    print("Example query:")
    print(
        f"id = {query_id}"
    )
    print(
        f"text = {queries[query_id]}"
    )

    print()
    print("Relevant documents:")

    for document_id, score in (
        qrels.get(query_id, {}).items()
    ):
        print(
            f"doc_id={document_id}, "
            f"relevance={score}"
        )


if __name__ == "__main__":
    main()
