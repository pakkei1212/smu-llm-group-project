import re
from typing import List, Tuple, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from configs.parsing import ITEM_PATTERN
from src.utils.sec_items import normalize_item_code

'''def normalize_item_code(match) -> str:
    """
    Normalize a regex match into canonical SEC_ITEM_MAP key.
    """
    raw = match.group(0)

    # 1. Uppercase for consistency
    raw = raw.upper()

    # 2. Remove trailing dot
    raw = re.sub(r"\.\s*$", "", raw)

    # 3. Normalize whitespace
    raw = re.sub(r"\s+", " ", raw).strip()

    # 4. Ensure 'ITEM ' prefix with space
    if not raw.startswith("ITEM "):
        raw = raw.replace("ITEM", "ITEM ", 1)

    # 5. Convert to title case (Item, not ITEM)
    return raw.title()'''

# -------------------------------------------------
# Section splitting
# -------------------------------------------------
def split_sections_with_items(text: str) -> List[Tuple[str, str]]:
    """
    Splits SEC filing text into (item_code, section_text).

    - Preserves pre-ITEM content as ('PREAMBLE', text)
    - Uses centralized ITEM_PATTERN
    """
    matches = list(ITEM_PATTERN.finditer(text))
    sections: List[Tuple[str, str]] = []

    # No ITEMs found → whole doc as one section
    if not matches:
        return [("FULL_DOCUMENT", text.strip())]

    # Handle preamble
    if matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.append(("PREAMBLE", preamble))

    # Handle ITEM sections
    for i, match in enumerate(matches):
        item_code = normalize_item_code(match)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()
        if section_text:
            sections.append((item_code, section_text))

    return sections

# -------------------------------------------------
# Chunking
# -------------------------------------------------
def chunk_text(
    text: str,
    header: Optional[str] = None,
    max_size: int = 800,
    overlap: int = 160,
    min_chunk_size: int = 100,
) -> List[str]:
    """
    SEC-optimized semantic chunker.

    - Preserves paragraph & sentence boundaries
    - Adds section header context
    - Produces embedding-friendly chunks
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",     # paragraphs
            "\n",       # lines
            ". ",       # sentences
            "; ",       # legal clauses
            ", ",       # fallback
            " ",        # words
            ""          # characters (last resort)
        ],
    )

    chunks = splitter.split_text(text)

    # Filter tiny / low-signal chunks
    chunks = [c.strip() for c in chunks if len(c.strip()) >= min_chunk_size]

    # Prepend section header (critical for RAG grounding)
    if header:
        chunks = [
            f"[{header}]\n{chunk}"
            for chunk in chunks
        ]

    return chunks
