
# Retrieval Preparation (Chunking, Dense Embeddings, BM25)

This document explains how to **query the prepared indexes** produced by the chunking and embedding pipelines.

The system supports:

1. **Dense Retrieval (Chroma + PubMedBERT)**
2. **BM25 Lexical Retrieval**

Two chunking strategies are available:

- **section_400token** → section‑aware chunks (~400 tokens)
- **fixed_500char** → fixed length chunks (500 characters)

Prepared indexes must be downloaded before querying.

---

# 1. Download Prepared Indexes

Download the files from:

https://drive.google.com/drive/u/1/folders/1GsSlv4QWTTcBjnaZ8q2Dor8wsdfDwSDh

Download:

```
bm25_index.zip
vector_store.zip
```

Place them inside:

```
data/
```

Extract them:

```bash
unzip bm25_index.zip -d data/
unzip vector_store.zip -d data/
```

After extraction:

```
data/
├── bm25_index/
│   ├── section_400token.pkl
│   └── fixed_500char.pkl
│
└── vector_store/
    ├── section_400token/pubmedbert/
    └── fixed_500char/pubmedbert/
```

---

# 2. Dense Retrieval (Chroma + PubMedBERT)

## Section‑Aware Chunking

```python
db_section = ChromaManager(
    base_directory=VECTOR_DIR,
    chunk_strategy="section_400token",
    embedding_model="pubmedbert",
    collection_name="medical_rag"
)

query = "What causes Alzheimer's disease?"

results = db_section.query(query, n_results=3)

for i in range(len(results["documents"][0])):
    print("\nResult", i+1)
    print(results["documents"][0][i][:300])
    print("PMID:", results["metadatas"][0][i]["pmid"])
```

---

## Fixed‑Length Chunking (500 char)

```python
db_fixed = ChromaManager(
    base_directory=VECTOR_DIR,
    chunk_strategy="fixed_500char",
    embedding_model="pubmedbert",
    collection_name="medical_rag"
)

query = "What causes Alzheimer's disease?"

results = db_fixed.query(query, n_results=3)

for i in range(len(results["documents"][0])):
    print("\nResult", i+1)
    print(results["documents"][0][i][:300])
    print("PMID:", results["metadatas"][0][i]["pmid"])
```

---

# 3. BM25 Retrieval

BM25 indexes are stored in:

```
data/bm25_index/
```

Example biomedical query:

```
amyloid beta plaque formation in Alzheimer's disease
```

Tokenize the query using the same tokenizer used for indexing.

---

## Section‑Aware BM25

```python
query = "amyloid beta plaque formation in Alzheimer's disease"

tokenized_query = spacy_tokenize_texts([query])[0]

bm25_section = pickle.load(open(BM25_DIR / "section_400token.pkl", "rb"))

scores = bm25_section.get_scores(tokenized_query)

import numpy as np

top_k = 10
top_idx = np.argsort(scores)[-top_k:][::-1]

results = section_chunks_df.iloc[top_idx]

results[["pmid", "section", "chunk_text"]]
```

---

## Fixed‑Length BM25

```python
bm25_fixed = pickle.load(open(BM25_DIR / "fixed_500char.pkl", "rb"))

scores = bm25_fixed.get_scores(tokenized_query)

import numpy as np

top_k = 10
top_idx = np.argsort(scores)[-top_k:][::-1]

results = fixed_chunks_df.iloc[top_idx]

results[["pmid", "section", "chunk_text"]]
```

---

# 4. Example Biomedical Queries

```
amyloid beta plaque formation in Alzheimer's disease
EGFR mutation lung cancer
BRCA1 breast cancer DNA repair
interleukin 6 inflammatory cytokine
ACE2 receptor SARS‑CoV‑2 spike protein
```

---

# 5. Notes

This document only covers **retrieval preparation artifacts**:

- chunking pipelines
- embedding pipelines
- BM25 index creation

The **full RAG QA system** is documented separately in the main project README.
