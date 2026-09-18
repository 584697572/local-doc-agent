from pathlib import Path

from retrieval.chunker import chunk_document
from retrieval.hybrid import HybridRetriever
from retrieval.loader import load_document
from retrieval.result import RetrievalResult


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class RetrievalEngine:
    """
    本地知识库检索总入口。

    负责：
    1. 扫描知识库目录；
    2. 加载 txt / md / pdf；
    3. 切成 DocumentChunk；
    4. 建立 HybridRetriever；
    5. 对外提供统一 search()。
    """

    def __init__(
        self,
        data_dir: str | Path,
        chunk_size: int = 500,
        overlap: int = 100,
        dense_encoder=None,
        reranker_model=None,
    ):
        self.data_dir = Path(data_dir)

        self.chunk_size = chunk_size
        self.overlap = overlap

        # 测试时可以注入 FakeEncoder / FakeReranker。
        self.dense_encoder = dense_encoder
        self.reranker_model = reranker_model

        # build() 之后保存所有 chunk。
        self.chunks = []

        # build() 之前还没有 Retriever。
        self.retriever = None

    def build(self) -> None:
        """
        扫描 data_dir 中所有支持的文档，
        加载、切块，并建立 HybridRetriever。
        """

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"知识库目录不存在: {self.data_dir}"
            )

        if not self.data_dir.is_dir():
            raise ValueError(
                f"知识库路径不是目录: {self.data_dir}"
            )

        chunks = []

        # 按路径排序，保证每次建立索引的顺序稳定。
        file_paths = sorted(
            path
            for path in self.data_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        )

        for file_path in file_paths:
            document = load_document(file_path)

            document_chunks = chunk_document(
                document,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )

            chunks.extend(document_chunks)

        self.chunks = chunks

        self.retriever = HybridRetriever(
            self.chunks,
            dense_encoder=self.dense_encoder,
            reranker_model=self.reranker_model,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[RetrievalResult]:
        """
        使用已经建立好的知识库进行检索。
        """

        if self.retriever is None:
            raise RuntimeError(
                "RetrievalEngine 尚未 build()，"
                "请先建立知识库索引。"
            )

        return self.retriever.search(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
        )