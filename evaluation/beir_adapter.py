"""把 BEIR 数据集转换成 LocalDoc-Agent 可使用的结构。"""

from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from retrieval.document import DocumentChunk


BEIR_DATASET_BASE_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets"
)


def load_beir_dataset(
    dataset_name: str = "scifact",
    split: str = "test",
    data_dir: str | Path = "evaluation/data",
):
    """
    下载并加载一个 BEIR 数据集。

    返回：
        corpus
        queries
        qrels
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    url = (
        f"{BEIR_DATASET_BASE_URL}/"
        f"{dataset_name}.zip"
    )

    # BEIR 会自动下载并解压。
    dataset_path = util.download_and_unzip(
        url,
        str(data_dir),
    )

    corpus, queries, qrels = (
        GenericDataLoader(
            data_folder=dataset_path
        ).load(
            split=split
        )
    )

    return corpus, queries, qrels


def beir_corpus_to_chunks(
    corpus: dict,
    dataset_name: str,
) -> list[DocumentChunk]:
    """
    把 BEIR corpus 转成 DocumentChunk。

    注意：
    初始 Benchmark 中一个 BEIR document
    对应一个 DocumentChunk。

    不进行额外 chunking，
    因为 BEIR 的 qrels 是按 document_id 标注的。
    """

    chunks = []

    for document_id, document in corpus.items():
        title = (
            document.get("title", "")
            or ""
        ).strip()

        text = (
            document.get("text", "")
            or ""
        ).strip()

        # 标题和正文合在一起参与检索。
        if title and text:
            content = f"{title}\n{text}"
        else:
            content = title or text

        chunk = DocumentChunk(
            # chunk_id 必须等于 BEIR document_id，
            # 后面才能和官方 qrels 直接比较。
            chunk_id=str(document_id),

            document_id=str(document_id),

            content=content,

            filename=(
                f"beir:{dataset_name}:"
                f"{document_id}"
            ),

            metadata={
                "dataset": dataset_name,
                "title": title,
            },
        )

        chunks.append(chunk)

    return chunks