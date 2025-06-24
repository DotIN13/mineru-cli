import fitz  # PyMuPDF
from typing import Optional

from loguru import logger


def extract_pdf_bytes_by_pymupdf(
    pdf_bytes: bytes,
    start_page: int = 0,
    end_page: Optional[int] = None
) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = doc.page_count

    # clamp end_page
    if end_page is None or end_page < 0:
        end = total - 1
    elif end_page >= total:
        logger.warning("end_page (%d) out of range, using last page", end_page)
        end = total - 1
    else:
        end = end_page

    # keep only the desired pages
    doc.select(list(range(start_page, end + 1)))

    # write out (returns bytes)
    result = doc.write(garbage=3, deflate=True)
    doc.close()
    return result
