"""把文档和问题分词，再用 BM25 计算关键词匹配分数。"""

import re

import jieba
from rank_bm25 import BM25Okapi

from retrieval.document import DocumentChunk
from retrieval.result import RetrievalResult


def tokenize(text: str) -> list[str]:
    """
    将中英文混合文本转换成 BM25 使用的 token 列表。
    """

    text = text.lower()

    pieces = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", text)

    tokens = []

    for piece in pieces:
        if re.fullmatch(r"[\u4e00-\u9fff]+", piece):
            # 正则先找出连续中文，再交给 jieba 切成词。
            chinese_words = jieba.lcut(piece)
            for word in chinese_words:
                cleaned_word = word.strip()
                if cleaned_word:
                    tokens.append(cleaned_word)
        else:
            tokens.append(piece)

    return tokens


class BM25Retriever:
    """
    基于 DocumentChunk 的 BM25 检索器。
    """

    def __init__(self, chunks: list[DocumentChunk]):
        # 复制外层列表；列表里的每个元素仍是原来的 DocumentChunk 对象。
        self.chunks = list(chunks)

        # 一个文档片段对应一个词列表，外层列表保存所有片段的分词结果。
        self.tokenized_corpus = []
        for chunk in self.chunks:
            chunk_tokens = tokenize(chunk.content)
            self.tokenized_corpus.append(chunk_tokens)

        if self.tokenized_corpus:
            self.index = BM25Okapi(self.tokenized_corpus)
        else:
            self.index = None

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:

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
        has_match = False
        for score in scores:
            if score != 0:
                has_match = True
                break

        if not has_match:
            return []

        # 排的是片段下标，排序依据是该下标对应的分数。
        def get_score(index):
            return scores[index]

        ranked_indices = sorted(range(len(scores)), key=get_score, reverse=True)

        results = []

        top_indices = ranked_indices[:top_k]
        for rank, index in enumerate(top_indices, start=1):
            result = RetrievalResult(
                chunk=self.chunks[index], score=float(scores[index]), rank=rank
            )
            results.append(result)

        return results
