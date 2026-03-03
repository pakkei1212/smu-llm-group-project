# 🏥 Medical QA RAG Project

Retrieval-Augmented Generation (RAG) system for Medical Question
Answering using:

-   🤗 HuggingFace Transformers
-   🧠 ChromaDB (Persistent Vector Store)
-   🖥 NVIDIA GPU (CUDA)
-   📓 JupyterLab
-   🐳 Docker

------------------------------------------------------------------------

# 📁 Project Structure
```
SMU-LLM-GROUP-PROJECT 
├── data/ # Raw & processed datasets 
├── notebooks/ # Jupyter notebooks 
├── src/ # Core RAG pipeline code 
├── vector_store/ # Persistent Chroma DB storage 
├── docker-compose.jupyter.yml 
├── Dockerfile 
├── requirements.txt 
└── README.md
```
------------------------------------------------------------------------

# 🚀 Quick Start (For Group Members)

## 1️⃣ Prerequisites

Before starting, ensure you have:

### ✅ Required Software

-   Docker (latest version)
-   Docker Compose (v2+)
-   NVIDIA GPU (if using GPU acceleration)
-   NVIDIA Container Toolkit (for GPU support)

Check installation:
```
docker --version
docker compose version
nvidia-smi
```
------------------------------------------------------------------------

# 🐳 Setup Instructions

## 2️⃣ Clone the Repository
```
git clone `<your-repo-url>`{=html}
cd SMU-LLM-GROUP-PROJECT
```
------------------------------------------------------------------------

## 3️⃣ Build and Start the Environment
```
docker compose -f docker-compose.jupyter.yml up --build
```
First build may take several minutes.

------------------------------------------------------------------------

## 4️⃣ Access JupyterLab

Open browser:
```
http://localhost:8888
```
------------------------------------------------------------------------

# 🖥 GPU Verification (Important)

Inside Jupyter notebook, run:
```
import torch
torch.cuda.is_available()
```
If `True` → GPU is working ✅
If `False` → check NVIDIA Container Toolkit setup.

------------------------------------------------------------------------

# 💾 Persistent Storage Explained

These folders are mounted from your local machine into the container:
```
  Local Folder    Container Path            Purpose
  --------------- ------------------------- ----------------------
  data/           /workspace/data           Datasets
  notebooks/      /workspace/notebooks      Experiments
  src/            /workspace/src            Core code
  vector_store/   /workspace/vector_store   Chroma persistent DB
```
⚠️ Important: Vector embeddings stored in `vector_store/` will persist
even if container is restarted.

------------------------------------------------------------------------

# 🧠 ChromaDB Usage (Persistent Mode)

Example:
```
import chromadb

client = chromadb.PersistentClient(path="./vector_store")
collection = client.get_or_create_collection("medical_qa")
```
Do NOT change the path unless necessary.

------------------------------------------------------------------------

# 🛑 Stop the Environment

To stop container:
```
Ctrl + C
```
To stop and remove container:
```
docker compose -f docker-compose.jupyter.yml down
```
------------------------------------------------------------------------

# 🔄 Rebuild After Dependency Changes

If `requirements.txt` is updated:
```
docker compose -f docker-compose.jupyter.yml build --no-cache
docker compose -f docker-compose.jupyter.yml up
```
------------------------------------------------------------------------

# 🧪 Development Workflow

### ✅ Where to write code?

-   Core logic → src/
-   Experiments → notebooks/
-   Data → data/
-   Vector DB auto-saves → vector_store/

------------------------------------------------------------------------

### ✅ Recommended Workflow

1.  Develop RAG logic in src/
2.  Test via Jupyter notebook
3.  Refactor stable logic into modules
4.  Commit frequently

------------------------------------------------------------------------

# 📦 Installing Additional Python Packages

If you need extra packages:

1.  Add to requirements.txt
2.  Rebuild container

Do NOT install manually inside container --- changes will be lost after
restart.

------------------------------------------------------------------------

# 🛠 Common Issues

### ❌ GPU Not Detected

Check:
```
nvidia-smi
```
Ensure NVIDIA Container Toolkit is installed.

------------------------------------------------------------------------

### ❌ Port 8888 Already In Use

Change port in `docker-compose.jupyter.yml`:
```
ports: - "8890:8888"
```
Then access:
```
http://localhost:8890
```
------------------------------------------------------------------------

# 📌 Best Practices For Team

-   Do NOT commit vector_store/
-   Do NOT commit large datasets
-   Use .gitignore
-   Always rebuild after dependency change
-   Keep RAG logic modular

------------------------------------------------------------------------

# 🎯 Summary

This Docker setup provides:

-   Reproducible environment
-   GPU acceleration
-   Persistent vector database
-   Clean separation of code & data
-   Easy onboarding for new members
