from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedDocumentError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


class DocumentProcessor:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )

    def process(self, path: Path) -> list[Document]:
        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedDocumentError(
                f"Unsupported file type '{extension}'. Supported types: "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            )

        if extension == ".pdf":
            loader = PyPDFLoader(str(path))
        elif extension == ".docx":
            loader = Docx2txtLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)

        documents = loader.load()
        documents = [document for document in documents if document.page_content.strip()]
        if not documents:
            raise EmptyDocumentError("The uploaded document contains no extractable text.")

        chunks = self._splitter.split_documents(documents)
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        if not chunks:
            raise EmptyDocumentError("The uploaded document contains no extractable text.")
        return chunks
