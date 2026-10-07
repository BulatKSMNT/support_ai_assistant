import io
import re
from dataclasses import dataclass

from pypdf import PdfReader

from app.errors import ServiceError


@dataclass(frozen=True)
class Passage:
    text: str
    page: int | None = None


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_file(filename: str, content: bytes, max_chars: int) -> list[Passage]:
    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix in {"txt", "md"}:
        try:
            passages = [Passage(normalize_text(content.decode("utf-8-sig")))]
        except UnicodeDecodeError as exc:
            raise ServiceError("invalid_encoding", "Используйте текст в UTF-8.", 422) from exc
    elif suffix == "pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted or len(reader.pages) > 50:
                raise ValueError("Encrypted or oversized PDF")
            passages = []
            total = 0
            for index, page in enumerate(reader.pages, 1):
                text = normalize_text(page.extract_text() or "")
                total += len(text)
                if total > max_chars:
                    raise ServiceError("document_too_large", "Слишком много текста в PDF.", 413)
                if text:
                    passages.append(Passage(text, index))
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(
                "invalid_pdf", "Нужен читаемый PDF без пароля, до 50 страниц.", 422
            ) from exc
    else:
        raise ServiceError("unsupported_file", "Поддерживаются TXT, Markdown и текстовые PDF.", 415)
    size = sum(len(p.text) for p in passages)
    if size < 20:
        raise ServiceError("empty_document", "Недостаточно текста. Для сканов нужен OCR.", 422)
    if size > max_chars:
        raise ServiceError("document_too_large", "Документ превышает лимит текста.", 413)
    return passages


def chunk_passages(passages: list[Passage], size: int, overlap: int) -> list[Passage]:
    chunks = []
    for passage in passages:
        text = normalize_text(passage.text)
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                # Prefer a paragraph/sentence boundary, retaining long unbroken tokens.
                boundary = max(
                    text.rfind("\n", start + size // 2, end),
                    text.rfind(". ", start + size // 2, end),
                )
                if boundary > start:
                    end = boundary + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append(Passage(piece, passage.page))
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
