import time
from time import perf_counter
import re
import uuid
from statistics import mean
from typing import List, Dict
from collections import Counter
from src.utils.logger import setup_logger

from tqdm.auto import tqdm

from configs.paths import LOG_DIR
from configs.sec import BASE_ARCHIVE
from configs.vector_db import VECTOR_DB_PATH
from configs.models import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_ALIAS,
    BATCH_SIZE,
    OLLAMA_HOST,
)
from configs.parsing import SEC_ITEM_MAP

from src.ingestion.sec_client import get_submissions, download_filing, load_all_submissions
from src.storage.chroma_manager import ChromaManager
from src.embeddings.embedding_manager import TransformerEmbeddingManager

from src.ingestion.html_parser import parse_html
from src.ingestion.chunker import split_sections_with_items, chunk_text


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def get_fiscal_year(filing_date: str) -> int:
    return int(filing_date[:4])


def make_doc_id(cik, year, acc, item, chunk_idx) -> str:
    uid = uuid.uuid4().hex  # 32-char lowercase hex, no hyphens
    return (
        f"{cik}_10K_{year}_"
        f"{acc.replace('-', '')}_"
        f"{item.replace(' ', '')}_"
        f"{chunk_idx:06d}_"
        f"{uid}"
    )

# -------------------------------------------------
# SEC rate limiter
# -------------------------------------------------
REQUESTS_PER_SECOND = 2.0
_INTERVAL = 1.0 / REQUESTS_PER_SECOND
_last_request = 0.0


def sec_wait():
    global _last_request
    now = time.time()
    elapsed = now - _last_request
    if elapsed < _INTERVAL:
        time.sleep(_INTERVAL - elapsed)
    _last_request = time.time()


# -------------------------------------------------
# Optional debug utility
# -------------------------------------------------
def normalize_header(line: str) -> str:
    # Remove trailing page numbers like "| 1", "| 23"
    return re.sub(r"\|\s*\d+\s*$", "| <PAGE>", line)

def list_repeated_lines(
    text: str,
    min_repeats: int = 3,
    min_length: int = 10,
    max_length: int = 120,
    top_k: int = 30,
):
    raw_lines = [
        line.strip()
        for line in text.splitlines()
        if min_length <= len(line.strip()) <= max_length
    ]

    normalized_lines = [normalize_header(l) for l in raw_lines]

    counts = Counter(normalized_lines)
    repeated = [(l, c) for l, c in counts.items() if c >= min_repeats]
    repeated.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Repeated lines (normalized) ===")
    for line, cnt in repeated[:top_k]:
        print(f"[{cnt:>3}x] {line}")


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------
def main(debug: bool = False):

    # ---- Setup logger ----
    logger = setup_logger(
        log_dir=LOG_DIR,
        debug=debug,
    )

    # ---- Initialize vector store ----
    chroma = ChromaManager(
        persist_directory=VECTOR_DB_PATH,
        embedding_model=EMBEDDING_MODEL_ALIAS,
        collection_name="sec_filings",
        base_url=OLLAMA_HOST,
        verbose=0,
    )

    embedding_manager = TransformerEmbeddingManager(
        embedding_model=EMBEDDING_MODEL_NAME,
        batch_size=BATCH_SIZE,
        logger=logger
    )

    chroma.reset_collection()
    chroma.get_collection_stats()

    # ---- Load SEC submissions ----
    submissions_by_cik = load_all_submissions()
    for cik, submission_path in submissions_by_cik.items():
        data = get_submissions(submission_path)
        recent = data["filings"]["recent"]

        ten_ks = [
            (form, acc, date, doc)
            for form, acc, date, doc in zip(
                recent["form"],
                recent["accessionNumber"],
                recent["filingDate"],
                recent["primaryDocument"],
            )
            if form == "10-K"
        ]

        all_tables: List[Dict] = []

        # ---- Process filings ----
        for form, acc, date, primary_doc in tqdm(ten_ks, desc="Indexing 10-K filings"):
            #if acc not in ["0001628280-16-020309", "0001193125-15-356351"] :
            #    continue  # known problematic filing

            logger.info(
                "processing_filing | %s",
                {
                    "event": "processing_filing",
                    "cik": cik,
                    "accession": acc,
                    "filing_date": date,
                }
            )

            base_url = f"{BASE_ARCHIVE}/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"

            sec_wait()
            html = download_filing(cik, acc, primary_doc)

            text, tables, images = parse_html(html, base_url)

            fiscal_year = get_fiscal_year(date)

            # ---- Index images ----
            _index_images(
                chroma=chroma,
                embedder=embedding_manager,
                images=images,
                cik=cik,
                company=data["name"],
                filing_date=date,
                fiscal_year=fiscal_year,
                accession=acc,
                logger=logger
            )

            #if debug:
            #    list_repeated_lines(text)

            sections = split_sections_with_items(text)

            # ---- Store tables (future structured index) ----
            indexed_tables = 0
            
            for idx, table in enumerate(tables):
                all_tables.append({
                    "cik": cik,
                    "company": data["name"],
                    "filing_type": "10-K",
                    "filing_date": date,
                    "fiscal_year": fiscal_year,
                    "accession": acc,
                    "table_index": idx,
                    "table": table,
                    "source": "SEC EDGAR",
                })
    
                indexed_tables += 1

                logger.debug(
                    "table_stored| %s",
                    {
                        "event": "table_stored",
                        "table_index": idx,
                        "num_rows": len(table.get("rows", [])),
                        "num_columns": len(table.get("headers", [])),
                        "headers": table.get("headers", [])[:5],  # preview only
                        "table": table
                    }
                )
            
            # ---- Chunk & embed narrative text ----
            batch_texts, batch_meta, batch_ids = [], [], []
            indexed = 0
            chunk_lengths = []

            t0 = perf_counter()

            for item_code, section_text in sections:         

                section_title = SEC_ITEM_MAP.get(item_code)
                if not section_title:
                    continue
                
                chunks = chunk_text(
                    section_text,
                    header=f"{data['name']} {fiscal_year} {item_code} - {section_title}",
                )
                
                chunk_lengths.extend(len(c) for c in chunks)

                for idx, chunk in enumerate(chunks):
                    logger.debug(
                        "chunk_indexed | %s",
                        {
                            "event": "chunk_indexed",
                            "cik": cik,
                            "accession": acc,
                            "section": item_code,
                            "chunk_index": idx,
                            "chunk_length": len(chunk),
                            "chunk_preview": chunk[:800]
                        }
                    )

                    batch_texts.append(chunk)
                    batch_ids.append(make_doc_id(cik, fiscal_year, acc, item_code, idx))
                    batch_meta.append({
                        "cik": cik,
                        "company": data["name"],
                        "filing_type": "10-K",
                        "filing_date": date,
                        "fiscal_year": fiscal_year,
                        "accession": acc,
                        "section": item_code,
                        "section_title": section_title,
                        "chunk_index": idx,
                        "content_type": "narrative",
                        "source": "SEC EDGAR",
                    })

                    if len(batch_texts) >= BATCH_SIZE:
                        _flush_batch(chroma, embedding_manager, batch_texts, batch_meta, batch_ids)
                        indexed += len(batch_texts)
                        batch_texts.clear()
                        batch_meta.clear()
                        batch_ids.clear()

            if batch_texts:
                _flush_batch(chroma, embedding_manager, batch_texts, batch_meta, batch_ids)
                indexed += len(batch_texts)

            elapsed = perf_counter() - t0

            if chunk_lengths:
                avg_len = round(mean(chunk_lengths), 1)
                print(
                    f"Indexed {indexed} chunks | "
                    f"{indexed_tables} tables | "
                    f"{elapsed:.2f}s | "
                    f"Avg len {avg_len} | "
                    f"Min/Max {min(chunk_lengths)}/{max(chunk_lengths)}"
                )

def _flush_batch(chroma, embedder, texts, metas, ids):
    embeddings = embedder.generate_text_embeddings(texts)
    chroma.add_with_embeddings(
        texts=texts,
        embeddings=embeddings,
        metadatas=metas,
        ids=ids,
    )

def _index_images(
    chroma,
    embedder,
    images: List[Dict],
    cik: str,
    company: str,
    filing_date: str,
    fiscal_year: int,
    accession: str,
    logger
):
    """
    Index image descriptions into vector DB.
    """

    logger.debug(
        "image_index_start | %s",
        {
            "event": "image_index_start",
            "cik": cik,
            "accession": accession,
            "num_images": len(images),
        },
    )

    texts, metas, ids = [], [], []
    skipped = 0

    for idx, img in enumerate(images):
        desc = img.get("image_description")
        if not desc:
            skipped += 1
            logger.debug(
                "image_skipped | %s",
                {
                    "event": "image_skipped",
                    "reason": "missing_description",
                    "image_id": img.get("image_id"),
                    "image_url": img.get("image_url"),
                },
            )
            continue

        doc_id = (
            f"{cik}_10K_{fiscal_year}_"
            f"{accession.replace('-', '')}_"
            f"IMAGE_{img['image_id']}_"
            f"{uuid.uuid4().hex}"
        )

        texts.append(desc)
        ids.append(doc_id)

        metas.append({
            "cik": cik,
            "company": company,
            "filing_type": "10-K",
            "filing_date": filing_date,
            "fiscal_year": fiscal_year,
            "accession": accession,
            "content_type": "image",
            "image_id": img["image_id"],
            "image_url": img["image_url"],
            "image_path": img["image_path"],
            "alt_text": img.get("alt_text", ""),
            "item_code": img.get("item_code"),
            "source": "SEC EDGAR",
        })

        logger.debug(
            "image_queued | %s",
            {
                "event": "image_queued",
                "image_index": idx,
                "image_id": img["image_id"],
                "item_code": img.get("item_code"),
                "doc_id": doc_id,
                "desc_preview": desc[:300],
            },
        )

    if not texts:
        logger.debug(
            "image_index_empty | %s",
            {
                "event": "image_index_empty",
                "cik": cik,
                "accession": accession,
                "skipped": skipped,
            },
        )
        return

    logger.debug(
        "image_embedding_start | %s",
        {
            "event": "image_embedding_start",
            "num_images": len(texts),
        },
    )

    embeddings = embedder.generate_text_embeddings(texts)

    logger.debug(
        "image_embedding_done | %s",
        {
            "event": "image_embedding_done",
            "num_images": len(embeddings),
        },
    )

    chroma.add_with_embeddings(
        texts=texts,
        embeddings=embeddings,
        metadatas=metas,
        ids=ids,
    )

    logger.info(
        "image_indexed | %s",
        {
            "event": "image_indexed",
            "cik": cik,
            "accession": accession,
            "indexed": len(texts),
            "skipped": skipped,
        },
    )

# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    main(debug=True)
