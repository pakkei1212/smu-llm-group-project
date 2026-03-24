# 📘 PubMed QA with Retrieval-Augmented Generation (RAG)

## 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system for **biomedical question answering** using PubMed abstracts.

The objective is to reduce **LLM hallucination** by grounding answers in retrieved medical evidence, and to systematically evaluate how different RAG design choices affect performance.

---

## 🏗️ Project Structure

```
.
├── src/                      # Core RAG modules
├── notebooks/                # Main pipelines (run these)
├── data/
├── vector_store/
├── bm25_index/
├── Dockerfile
├── docker-compose.jupyter.yml
└── requirements.txt
```

---

## ⚙️ How to Run

⚠️ This project is **not script-based**. It is designed to run via **Jupyter notebooks**.

### 1. Start Environment (Recommended)
```bash
docker-compose -f docker-compose.jupyter.yml up
```

### 2. Run Pipeline (in order)
1. `pubmed_extraction.ipynb`
2. `pubmed_chunking_pipelines.ipynb`
3. `pubmed_embeddings_pipelines.ipynb`
4. `RAG_Strategies_All6.ipynb`
5. `medical-rag-eval.ipynb`

---

## 📊 Data Sources

### BioASQ13 Dataset
- Link: https://participants-area.bioasq.org/datasets/
- Task: **BioASQ Task B**

**Training Dataset**
- Training 13b

**Test Dataset**
- 13b golden enriched:
  - 13B1_golden
  - 13B2_golden
  - 13B3_golden
  - 13B4_golden

**Question Types**
- factoid
- list
- yesno
- summary

---

### Post-Processed Dataset (CSV)
- https://drive.google.com/drive/folders/1Con2hF37SrS7FofwvEhynU5gPIGVsIkr?usp=sharing

---

### PubMed Corpus
- Abstracts retrieved via PubMed API
- Parsed using Biopython XML

---

## 📦 Prebuilt Indexes (Download)

To skip heavy preprocessing, download:

https://drive.google.com/drive/u/1/folders/1GsSlv4QWTTcBjnaZ8q2Dor8wsdfDwSDh

Files:
- `bm25_index.zip`
- `vector_store.zip`

### After Download
1. Extract into project root:
```
bm25_index/
vector_store/
```

2. Ensure structure matches:
```
vector_store/<chunk_strategy>/pubmedbert/
bm25_index/<chunk_strategy>.pkl
```

---

## ✂️ Chunking Strategies

| Strategy | Description |
|----------|------------|
| Fixed-Length (500 chars) | Baseline |
| Section-Aware (400 tokens) | Structure-preserving |
| Contextual Section-Aware (400 tokens) | Adds context |

---

## 🔍 Retrieval Components

- Dense: PubMedBERT + Chroma
- Sparse: BM25 + spaCy
- Hybrid: Dense + BM25 fusion

---

## 🧠 RAG Variants

- RAG1: Dense  
- RAG2: Hybrid  
- RAG3: + Reranking  
- RAG4: + Gradient selection  
- RAG5: + Diverse prompting  
- RAG6: + K-means voting  

---

## ⚙️ Generation

- Qwen (local)
- OpenAI GPT (optional)

Strategies:
- Greedy
- Diverse personas
- K-means selection

---

## 📈 Evaluation

- F1 Score
- Recall
- Factuality
- Correctness

Method: DeepEval (LLM-as-a-judge)

---

## 🧾 Key Findings

- Hybrid > Dense
- Reranking gives biggest gain
- Gradient pruning may remove useful info
- K-means underperforms
- High factuality ≠ correctness

Best model: **RAG3**

---

## 🔮 Future Improvements

- Adaptive chunking
- Metadata-based retrieval
- Better semantic matching
- Agentic RAG

---

## 👥 Team

- Yip Pak Kei  
- Calvin Li  
- Low Min Yee  
- Chen Qihang  
- Koh We Kiat  
