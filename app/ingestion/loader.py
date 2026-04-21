from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_documents(source: str) -> list[Document]:
    """
    Load documents from a file path or raw text string.
    Supports: .pdf, .txt, raw text
    """
    path = Path(source)

    if path.exists() and path.is_file():
        ext = path.suffix.lower()

        if ext == ".pdf":
            logger.info(f"Loading PDF: {source}")
            loader = PyPDFLoader(str(path))
            docs = loader.load()

        elif ext == ".txt":
            logger.info(f"Loading TXT: {source}")
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()

        else:
            raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")

    else:
        # Treat as raw text string
        logger.info("Loading raw text input")
        docs = [Document(page_content=source, metadata={"source": "raw_text"})]

    logger.info(f"Loaded {len(docs)} document(s)")
    return docs