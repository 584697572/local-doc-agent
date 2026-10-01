"""
MultiHop-RAG 数据集适配器。

职责：

1. 下载官方 MultiHop-RAG 数据集；
2. 加载 corpus.json 和 MultiHopRAG.json；
3. 转换成 LocalDoc-Agent 的 Document / DocumentChunk；
4. 把每道题转换成统一的 MultiHopCase；
5. 同时保存：
   - Gold supporting facts
   - Gold supporting document IDs

注意：
这个模块只负责“数据适配”，
不负责 Retrieval / Agent / QA Benchmark。
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from retrieval.chunker import chunk_document
from retrieval.document import (
    Document,
    DocumentChunk,
)


# ======================================================
# Official Dataset
# ======================================================

MULTIHOP_RAG_BASE_URL = (
    "https://huggingface.co/datasets/"
    "yixuantt/MultiHopRAG/resolve/main"
)

CORPUS_FILENAME = "corpus.json"
QUERY_FILENAME = "MultiHopRAG.json"

DEFAULT_DATA_DIR = Path(
    "evaluation/data/multihop_rag"
)


@dataclass(frozen=True)
class MultiHopCase:
    """
    一道标准化后的 MultiHop-RAG Case。

    case_id:
        项目内部稳定使用的 Case ID。

    query:
        用户问题。

    answer:
        官方 Gold Answer。

    question_type:
        官方问题类型。

    evidence_facts:
        官方 evidence_list 中的 supporting facts。

        后面可用于兼容官方 Retrieval Eval。

    evidence_document_ids:
        supporting evidence 所在的父文档 ID。

        后面可用于计算：
            Evidence Recall@K
            All-Evidence Hit@K
            Multi-hop Coverage
    """

    case_id: str
    query: str
    answer: str
    question_type: str

    evidence_facts: tuple[str, ...]
    evidence_document_ids: tuple[str, ...]


def _normalize_lookup_text(
    value: str | None,
) -> str:
    """
    对 title / URL 做稳定匹配。

    casefold 比 lower 更适合一般字符串标准化。
    """

    if not value:
        return ""

    return " ".join(
        str(value)
        .strip()
        .casefold()
        .split()
    )


def _make_document_id(
    item: dict,
) -> str:
    """
    为 Corpus 文档生成稳定 document_id。

    优先使用 URL；
    URL 缺失时使用 title/source/published_at。

    最终取 SHA1 前 16 位，
    避免把很长的 URL 直接塞进 ID。
    """

    url = str(
        item.get("url", "")
        or ""
    ).strip()

    if url:
        identity = url

    else:
        identity = "|".join(
            [
                str(
                    item.get("title", "")
                    or ""
                ),
                str(
                    item.get("source", "")
                    or ""
                ),
                str(
                    item.get(
                        "published_at",
                        "",
                    )
                    or ""
                ),
            ]
        )

    digest = hashlib.sha1(
        identity.encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    return (
        f"multihop_doc_{digest}"
    )


def _download_file(
    url: str,
    destination: Path,
) -> None:
    """
    下载一个数据文件。

    使用临时文件后再 rename，
    避免下载中断时留下半个 JSON。
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        destination.with_suffix(
            destination.suffix
            + ".tmp"
        )
    )

    request = Request(
        url,
        headers={
            "User-Agent": (
                "LocalDoc-Agent/1.0"
            )
        },
    )

    try:
        with urlopen(
            request,
            timeout=120,
        ) as response:
            content = response.read()

        temporary_path.write_bytes(
            content
        )

        temporary_path.replace(
            destination
        )

    finally:
        # 如果下载中途失败，
        # 清理残留临时文件。
        if temporary_path.exists():
            temporary_path.unlink()


def ensure_multihop_rag_files(
    data_dir: str | Path = (
        DEFAULT_DATA_DIR
    ),
) -> tuple[Path, Path]:
    """
    保证两个官方数据文件已经存在。

    已存在时不会重复下载。
    """

    data_dir = Path(
        data_dir
    )

    corpus_path = (
        data_dir
        / CORPUS_FILENAME
    )

    query_path = (
        data_dir
        / QUERY_FILENAME
    )

    files = [
        (
            CORPUS_FILENAME,
            corpus_path,
        ),
        (
            QUERY_FILENAME,
            query_path,
        ),
    ]

    for (
        filename,
        path,
    ) in files:
        if path.exists():
            continue

        url = (
            f"{MULTIHOP_RAG_BASE_URL}/"
            f"{filename}"
        )

        print(
            f"[MultiHop-RAG] "
            f"Downloading {filename}..."
        )

        _download_file(
            url=url,
            destination=path,
        )

    return (
        corpus_path,
        query_path,
    )


def load_multihop_rag(
    data_dir: str | Path = (
        DEFAULT_DATA_DIR
    ),
    download: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    加载官方 Corpus 和 Query 数据。

    返回：
        corpus_data
        query_data
    """

    data_dir = Path(
        data_dir
    )

    if download:
        (
            corpus_path,
            query_path,
        ) = ensure_multihop_rag_files(
            data_dir
        )

    else:
        corpus_path = (
            data_dir
            / CORPUS_FILENAME
        )

        query_path = (
            data_dir
            / QUERY_FILENAME
        )

        if not corpus_path.exists():
            raise FileNotFoundError(
                f"找不到 MultiHop-RAG Corpus："
                f"{corpus_path}"
            )

        if not query_path.exists():
            raise FileNotFoundError(
                f"找不到 MultiHop-RAG Query："
                f"{query_path}"
            )

    with corpus_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        corpus_data = json.load(
            file
        )

    with query_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        query_data = json.load(
            file
        )

    if not isinstance(
        corpus_data,
        list,
    ):
        raise ValueError(
            "corpus.json 顶层必须是 list"
        )

    if not isinstance(
        query_data,
        list,
    ):
        raise ValueError(
            "MultiHopRAG.json 顶层必须是 list"
        )

    return (
        corpus_data,
        query_data,
    )


def corpus_to_documents(
    corpus_data: list[dict],
) -> list[Document]:
    """
    把 MultiHop-RAG Corpus
    转成 LocalDoc-Agent Document。

    很重要：

    MultiHop-RAG 的问题可能依赖文档 metadata，
    因此检索文本不能只有 body。

    我们把：

        title
        source
        published_at
        author
        category

    一并写入 Document.content。

    这样 BM25 / Dense 都能检索这些字段。
    """

    documents = []

    for item in corpus_data:
        title = str(
            item.get(
                "title",
                "",
            )
            or ""
        ).strip()

        body = str(
            item.get(
                "body",
                "",
            )
            or ""
        ).strip()

        source = str(
            item.get(
                "source",
                "",
            )
            or ""
        ).strip()

        published_at = str(
            item.get(
                "published_at",
                "",
            )
            or ""
        ).strip()

        author = str(
            item.get(
                "author",
                "",
            )
            or ""
        ).strip()

        category = str(
            item.get(
                "category",
                "",
            )
            or ""
        ).strip()

        url = str(
            item.get(
                "url",
                "",
            )
            or ""
        ).strip()

        document_id = (
            _make_document_id(
                item
            )
        )

        # Metadata 进入可检索文本。
        content_parts = [
            f"Title: {title}",
            f"Source: {source}",
            (
                "Published At: "
                f"{published_at}"
            ),
            f"Author: {author}",
            f"Category: {category}",
            "",
            body,
        ]

        content = "\n".join(
            content_parts
        ).strip()

        document = Document(
            document_id=(
                document_id
            ),

            content=content,

            filename=(
                f"multihop:"
                f"{document_id}"
            ),

            file_type="json",

            metadata={
                "dataset": (
                    "MultiHop-RAG"
                ),
                "title": title,
                "source": source,
                "published_at": (
                    published_at
                ),
                "author": author,
                "category": category,
                "url": url,
            },
        )

        documents.append(
            document
        )

    return documents


def documents_to_chunks(
    documents: list[Document],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """
    使用项目自己的 Chunker
    切分 MultiHop-RAG Corpus。

    这样 Benchmark 与真实 Runtime
    使用同一种切块逻辑。

    Document metadata 会补回 Chunk.metadata，
    方便后续分析 Gold 文档来源。
    """

    chunks = []

    for document in documents:
        document_chunks = (
            chunk_document(
                document=document,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

        for chunk in (
            document_chunks
        ):
            # chunk_document 当前只保存 file_type。
            # Benchmark 需要更多 metadata，
            # 因此这里补充父文档 metadata。
            chunk.metadata.update(
                document.metadata
            )

        chunks.extend(
            document_chunks
        )

    return chunks


def _build_document_lookup(
    documents: list[Document],
) -> tuple[
    dict[str, str],
    dict[str, list[str]],
]:
    """
    建立 Gold Evidence → Corpus Document 映射。

    返回：
        URL -> document_id
        Title -> [document_id, ...]

    Title 使用 list，
    因为理论上可能出现重名标题。
    """

    url_lookup = {}
    title_lookup = {}

    for document in documents:
        url = _normalize_lookup_text(
            document.metadata.get(
                "url"
            )
        )

        title = _normalize_lookup_text(
            document.metadata.get(
                "title"
            )
        )

        if url:
            url_lookup[url] = (
                document.document_id
            )

        if title:
            title_lookup.setdefault(
                title,
                [],
            ).append(
                document.document_id
            )

    return (
        url_lookup,
        title_lookup,
    )


def _find_evidence_document_id(
    evidence: dict,
    url_lookup: dict[str, str],
    title_lookup: dict[
        str,
        list[str],
    ],
) -> str | None:
    """
    将一条 Gold Evidence
    映射回 Corpus 父文档。

    匹配优先级：

        1. URL 精确匹配
        2. 唯一 Title 匹配
    """

    url = _normalize_lookup_text(
        evidence.get(
            "url"
        )
    )

    if (
        url
        and url in url_lookup
    ):
        return url_lookup[
            url
        ]

    title = _normalize_lookup_text(
        evidence.get(
            "title"
        )
    )

    title_matches = (
        title_lookup.get(
            title,
            [],
        )
    )

    if len(
        title_matches
    ) == 1:
        return title_matches[0]

    return None


def queries_to_cases(
    query_data: list[dict],
    documents: list[Document],
    strict: bool = True,
) -> list[MultiHopCase]:
    """
    把官方 Query 转成 MultiHopCase。

    strict=True：
        非 null_query 的 Evidence
        如果无法映射到 Corpus 文档，
        立即报错。

    Benchmark 阶段宁可直接发现数据映射问题，
    不要悄悄吞掉 Gold。
    """

    (
        url_lookup,
        title_lookup,
    ) = _build_document_lookup(
        documents
    )

    cases = []

    for index, item in enumerate(
        query_data
    ):
        query = str(
            item.get(
                "query",
                "",
            )
            or ""
        ).strip()

        answer = str(
            item.get(
                "answer",
                "",
            )
            or ""
        ).strip()

        question_type = str(
            item.get(
                "question_type",
                "",
            )
            or ""
        ).strip()

        evidence_list = (
            item.get(
                "evidence_list",
                [],
            )
            or []
        )

        evidence_facts = []
        evidence_document_ids = []

        for evidence in (
            evidence_list
        ):
            fact = str(
                evidence.get(
                    "fact",
                    "",
                )
                or ""
            ).strip()

            if fact:
                evidence_facts.append(
                    fact
                )

            document_id = (
                _find_evidence_document_id(
                    evidence=(
                        evidence
                    ),
                    url_lookup=(
                        url_lookup
                    ),
                    title_lookup=(
                        title_lookup
                    ),
                )
            )

            if document_id is None:
                if (
                    strict
                    and question_type
                    != "null_query"
                ):
                    raise ValueError(
                        "无法把 Gold Evidence "
                        "映射到 Corpus："
                        f"query_index={index}, "
                        f"title="
                        f"{evidence.get('title')!r}, "
                        f"url="
                        f"{evidence.get('url')!r}"
                    )

                continue

            # 去重但保持原始顺序。
            if (
                document_id
                not in
                evidence_document_ids
            ):
                evidence_document_ids.append(
                    document_id
                )

        case = MultiHopCase(
            case_id=(
                f"multihop_"
                f"{index:04d}"
            ),

            query=query,
            answer=answer,

            question_type=(
                question_type
            ),

            evidence_facts=tuple(
                evidence_facts
            ),

            evidence_document_ids=tuple(
                evidence_document_ids
            ),
        )

        cases.append(
            case
        )

    return cases


def load_multihop_benchmark(
    data_dir: str | Path = (
        DEFAULT_DATA_DIR
    ),
    chunk_size: int = 500,
    overlap: int = 100,
    download: bool = True,
) -> tuple[
    list[Document],
    list[DocumentChunk],
    list[MultiHopCase],
]:
    """
    Benchmark 统一入口。

    返回：
        documents
        chunks
        cases
    """

    (
        corpus_data,
        query_data,
    ) = load_multihop_rag(
        data_dir=data_dir,
        download=download,
    )

    documents = (
        corpus_to_documents(
            corpus_data
        )
    )

    chunks = (
        documents_to_chunks(
            documents=documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    )

    cases = (
        queries_to_cases(
            query_data=query_data,
            documents=documents,
            strict=True,
        )
    )

    return (
        documents,
        chunks,
        cases,
    )


def main() -> None:
    """
    Adapter Smoke Test CLI。

    这里只验证数据，
    不加载 Embedding / Reranker。
    """

    parser = argparse.ArgumentParser(
        description=(
            "Load and inspect "
            "MultiHop-RAG dataset."
        )
    )

    parser.add_argument(
        "--data-dir",
        default=str(
            DEFAULT_DATA_DIR
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    (
        documents,
        chunks,
        cases,
    ) = load_multihop_benchmark(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        download=True,
    )

    type_counts = Counter(
        case.question_type
        for case in cases
    )

    non_null_cases = [
        case
        for case in cases
        if (
            case.question_type
            != "null_query"
        )
    ]

    unmapped_non_null = [
        case.case_id
        for case in non_null_cases
        if not (
            case.evidence_document_ids
        )
    ]

    print()
    print(
        "[MultiHop-RAG] "
        f"Documents: {len(documents)}"
    )

    print(
        "[MultiHop-RAG] "
        f"Chunks: {len(chunks)}"
    )

    print(
        "[MultiHop-RAG] "
        f"Cases: {len(cases)}"
    )

    print(
        "[MultiHop-RAG] "
        f"Question types: "
        f"{dict(type_counts)}"
    )

    print(
        "[MultiHop-RAG] "
        "Unmapped non-null cases: "
        f"{len(unmapped_non_null)}"
    )


if __name__ == "__main__":
    main()