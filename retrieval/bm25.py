import re
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

from retrieval.document import DocumentChunk

@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    rank: int

def tokenize(text: str) -> list[str]:
    """
    将中英文混合文本转换成 BM25 使用的 token 列表。
    """

    text = text.lower()

    pieces = re.findall(
        r"[\u4e00-\u9fff]+|[a-z0-9_]+",
        text,
    )

    tokens = []

    for piece in pieces:
        if re.fullmatch(r"[\u4e00-\u9fff]+", piece):
            tokens.extend(
                token.strip()
                for token in jieba.lcut(piece)
                if token.strip()
            )
        else:
            tokens.append(piece)

    return tokens
class BM25Retriever:
    """
    基于 DocumentChunk 的 BM25 检索器。
    """

    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = list(chunks)

        self.tokenized_corpus = [
            tokenize(chunk.content)
            for chunk in self.chunks
        ]

        if self.tokenized_corpus:
            self.index = BM25Okapi(self.tokenized_corpus)
        else:
            self.index = None

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if not query.strip():
            return []

        if self.index is None:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        # 如果所有 Chunk 都完全没有匹配，
        # 不返回一堆没有意义的结果。
        if not any(score != 0 for score in scores):
            return []

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for rank, index in enumerate(
            ranked_indices[:top_k],
            start=1,
        ):
            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    score=float(scores[index]),
                    rank=rank,
                )
            )

        return results