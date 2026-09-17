from retrieval.document import Document, DocumentChunk


def _find_page_number(document: Document, char_position: int) -> int | None:
    """
    根据字符在 Document.content 中的位置，
    查找它来自 PDF 的第几页。

    非 PDF 或没有 page_boundaries 时返回 None。
    """
    page_boundaries = document.metadata.get("page_boundaries", [])

    for boundary in page_boundaries:
        if boundary["start_char"] <= char_position < boundary["end_char"]:
            return boundary["page"]

    return None


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """
    将完整 Document 按固定字符窗口切成多个 DocumentChunk。

    chunk_size:
        每个 chunk 最大字符数。

    overlap:
        相邻 chunk 之间重复的字符数。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if overlap < 0:
        raise ValueError("overlap 不能小于 0")

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    if not document.content:
        return []

    chunks = []

    start = 0
    chunk_index = 0
    content_length = len(document.content)

    while start < content_length:
        end = min(start + chunk_size, content_length)

        chunk_content = document.content[start:end]

        # 如果这一块只有空白字符，就不生成 chunk。
        if chunk_content.strip():
            page = _find_page_number(document, start)

            chunk = DocumentChunk(
                chunk_id=f"{document.document_id}_chunk_{chunk_index:04d}",
                document_id=document.document_id,
                content=chunk_content,
                filename=document.filename,
                page=page,
                start_char=start,
                end_char=end,
                metadata={
                    "file_type": document.file_type,
                },
            )

            chunks.append(chunk)
            chunk_index += 1

        # 已经到达文档结尾，就结束循环。
        if end >= content_length:
            break

        # 下一块从“当前结束位置 - overlap”开始。
        start = end - overlap

    return chunks