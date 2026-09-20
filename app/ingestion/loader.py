from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import fitz  # pymupdf
import os
import logging

logger = logging.getLogger(__name__)

def load_documents(source: str) -> list[Document]:
    if source.startswith("http://") or source.startswith("https://"):
        return _load_url(source)

    path = Path(source)

    if path.exists() and path.is_file():
        ext = path.suffix.lower()

        if ext == ".pdf":
            logger.info(f"Loading PDF: {source}")
            docs = _load_pdf(str(path))

        elif ext == ".txt":
            logger.info(f"Loading TXT: {source}")
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()

        elif ext == ".docx":
            logger.info(f"Loading DOCX: {source}")
            docs = _load_docx(str(path))

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    else:
        logger.info("Loading raw text input")
        docs = [Document(page_content=source, metadata={"source": "raw_text"})]

    # Filter empty docs
    docs = [d for d in docs if d.page_content.strip()]
    logger.info(f"Loaded {len(docs)} non-empty document(s)")
    return docs


def _load_pdf(path: str) -> list[Document]:
    """
    Extract text from PDF using PyMuPDF.
    Works for both text-based and many scanned PDFs.
    Merges all pages into one document for better semantic chunking.
    """
    doc = fitz.open(path)
    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages_text.append(text)

    doc.close()

    if not pages_text:
        raise ValueError(
            "This PDF contains no extractable text — it appears to be a scanned/image PDF. "
            "Please convert it using Adobe Acrobat, SmallPDF, or ILovePDF, "
            "then try again. Alternatively, copy the text into a .txt file."
        )
    full_text = "\n\n".join(pages_text)
    logger.info(f"Extracted {len(full_text):,} characters from {len(pages_text)} pages")

    return [Document(
        page_content=full_text,
        metadata={"source": path, "type": "pdf", "pages": len(pages_text)}
    )]


def _load_docx(path: str) -> list[Document]:
    from docx import Document as DocxDocument
    doc = DocxDocument(path)
    full_text = "\n\n".join(
        p.text.strip() for p in doc.paragraphs if p.text.strip()
    )
    if not full_text:
        raise ValueError("DOCX file appears to be empty or has no extractable text")
    return [Document(
        page_content=full_text,
        metadata={"source": path, "type": "docx"}
    )]


def _load_url(url: str) -> list[Document]:
    logger.info(f"Fetching URL: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; HybridRAG/1.0)"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "iframe", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n\n".join(lines)

    if len(clean_text) < 100:
        raise ValueError("Could not extract meaningful text from URL")

    title = soup.find("title")
    title_text = title.get_text(strip=True) if title else url
    logger.info(f"Extracted {len(clean_text):,} chars from: {title_text}")

    return [Document(
        page_content=clean_text,
        metadata={"source": url, "title": title_text, "type": "url"}
    )]