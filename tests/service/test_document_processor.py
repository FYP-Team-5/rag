from pathlib import Path

import pytest

from app.service import (
    DocumentProcessor,
    EmptyDocumentError,
    UnsupportedDocumentError,
)


def test_processes_and_chunks_text_rubric(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.md"
    rubric.write_text(
        "# Short answer rubric\n\n## Accuracy\n" + ("Evidence and reasoning. " * 80),
        encoding="utf-8",
    )
    processor = DocumentProcessor(chunk_size=250, chunk_overlap=30)

    chunks = processor.process(rubric)

    assert len(chunks) > 1
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


def test_rejects_unsupported_files(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.csv"
    rubric.write_text("criterion,points", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentError):
        DocumentProcessor(250, 30).process(rubric)


@pytest.mark.parametrize("extension", [".txt", ".md"])
def test_processes_supported_text_formats(tmp_path: Path, extension: str) -> None:
    rubric = tmp_path / f"rubric{extension}"
    rubric.write_text("Accuracy\nUse relevant evidence.", encoding="utf-8")

    chunks = DocumentProcessor(250, 30).process(rubric)

    assert [chunk.page_content for chunk in chunks] == [
        "Accuracy\nUse relevant evidence."
    ]


def test_rejects_documents_without_extractable_text(tmp_path: Path) -> None:
    rubric = tmp_path / "empty.txt"
    rubric.write_text("   \n\n", encoding="utf-8")

    with pytest.raises(EmptyDocumentError, match="no extractable text"):
        DocumentProcessor(250, 30).process(rubric)
