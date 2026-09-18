from dataclasses import dataclass

from retrieval.document import DocumentChunk


@dataclass
class RetrievalResult:
    """
    一条统一的检索结果。

    chunk:
        命中的 DocumentChunk。

    score:
        当前 Retriever 给出的相关性分数。

    rank:
        当前结果的排名，从 1 开始。
    """

    chunk: DocumentChunk
    score: float
    rank: int