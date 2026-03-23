"""
Generate answer from question + retrieved context using a configurable LLM.
"""
import ast
import torch
import gc
import numpy as np
import os
from typing import Optional
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# ==============================
# LOAD ENV 🔥
# ==============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
_client = OpenAI(api_key=OPENAI_API_KEY)

# ==============================
# GLOBAL GENERATION CONFIG 🔥
# ==============================
GRAD_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GEN_MODEL  = "gpt-4.1-mini"   # 🔥 CHANGED
MAX_CTX = 3072
MAX_GRAD_TOKENS = 1024 
MAX_NEW_TOKENS = 160

# ==============================
# GLOBAL MODEL CACHE ✅ (UNIFIED)
# ==============================
_MODEL_CACHE: dict[str, tuple] = {}

# 🔥 sentence encoder (used in RAG6)
_SENT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_sent_model = SentenceTransformer(_SENT_MODEL_NAME)

# 🔥 Optimized prompt (no duplicate system role)
DEFAULT_PROMPT_TEMPLATE = (
    "You are a biomedical QA system.\n\n"
    "RULES:\n"
    "1. Use the provided context as the primary evidence.\n"
    "2. Prefer explicit evidence; allow cautious synthesis across multiple context snippets.\n"
    "3. If evidence is partial, give the best supported answer and briefly state uncertainty.\n"
    "4. Reply \"Not enough information\" only when the context has no relevant evidence.\n\n"
    "OUTPUT FORMAT:\n"
    "- Maximum 120 words.\n"
    "- Answer first, then one short evidence note.\n\n"
    "Question:\n{question}\n\n"
    "Context:\n{context}\n\n"
    "Final Answer:"
)

# ==============================
# SHARED MODEL LOADER 🔥
# ==============================
def _get_model(model_name: str, model_cache: Optional[dict] = None):
    cache = model_cache if model_cache is not None else _MODEL_CACHE

    model_name = model_name.strip()

    if model_name not in cache:
        print(f"🔥 Loading model: {model_name}")

        cache.clear()
        torch.cuda.empty_cache()
        gc.collect()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_buffers=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        model.eval()
        cache[model_name] = (tokenizer, model)

    return cache[model_name]


def format_context(chunks: list[tuple], sep: str = "\n\n---\n\n") -> str:
    return sep.join(t[1] for t in chunks)


# ==============================
# PMID extraction (UNCHANGED)
# ==============================
def extract_pmids(chunks: list[tuple]) -> list[str]:
    pmids = set()

    for c in chunks:
        pmid = None

        if isinstance(c, tuple) and len(c) >= 3:
            meta = c[2]

            if isinstance(meta, dict):
                pmid = meta.get("pmid")

            elif isinstance(meta, str) and meta.startswith("{"):
                try:
                    meta_dict = ast.literal_eval(meta)
                    pmid = meta_dict.get("pmid")
                except:
                    pass

        elif isinstance(c, str) and c.startswith("{"):
            try:
                meta_dict = ast.literal_eval(c)
                pmid = meta_dict.get("pmid")
            except:
                pass

        if pmid:
            pmids.add(str(pmid))

    return sorted(pmids)


# ==============================
# GENERATION ENTRY
# ==============================
def generate_answer(
    question: str,
    context: str,
    model_name: str = GEN_MODEL,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: Optional[str] = None,
    model_cache: Optional[dict] = None,
) -> str:

    prompt = prompt_template.format(question=question, context=context)

    return _generate_with_openai(   # 🔥 switched
        prompt,
        model_name=model_name,
        max_new_tokens=max_new_tokens
    )


# ==============================
# CORE GENERATION (OPENAI VERSION 🔥)
# ==============================
def _generate_with_openai(
    prompt: str,
    model_name: str,
    max_new_tokens: int,
) -> str:

    response = _client.responses.create(
        model=model_name,
        input=[
            {"role": "user", "content": prompt}
        ],
        max_output_tokens=max_new_tokens,
    )

    return response.output_text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Gradient-Based Context Selection — used in RAG4, RAG5, RAG6
# ═══════════════════════════════════════════════════════════════════════════════

def gradient_select_chunks(
    question: str,
    chunks: list,
    retain_ratio: float = 0.6,
    sep: str = "\n\n---\n\n",
    model_name: str = GRAD_MODEL,
    model_cache: Optional[dict] = None,
) -> list:

    if len(chunks) <= 1:
        return chunks

    tokenizer, model = _get_model(model_name, model_cache)

    chunk_texts = [c[1] for c in chunks]
    prompt = f"Question: {question}\nContext: {sep.join(chunk_texts)}\nAnswer:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_GRAD_TOKENS
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(input_ids).detach().requires_grad_(True)

    model.zero_grad()
    
    outputs = model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        use_cache=False
    )
    
    loss = outputs.logits.mean()   # 🔥 lightweight proxy
    loss.backward()

    saliency = embeds.grad.norm(dim=-1).squeeze(0).detach().cpu().numpy()

    chunk_saliency = []
    offset = 0

    for ct in chunk_texts:
        token_ids = tokenizer(ct, add_special_tokens=False)["input_ids"]
        n_tok = min(len(token_ids), len(saliency) - offset)

        if n_tok <= 0:
            chunk_saliency.append(0.0)
            continue

        chunk_saliency.append(float(saliency[offset:offset + n_tok].mean()))
        offset += n_tok

    n_keep = max(1, int(len(chunks) * retain_ratio))
    ranked = sorted(range(len(chunks)), key=lambda i: chunk_saliency[i], reverse=True)

    return [chunks[i] for i in sorted(ranked[:n_keep])]


# ═══════════════════════════════════════════════════════════════════════════════
# Diverse Prompted Generation — RAG5
# ═══════════════════════════════════════════════════════════════════════════════

GLOBAL_RULE = """
You MUST follow these rules:
- Use ONLY the provided context
- Do NOT use prior knowledge
- Do NOT infer beyond the context
- If the answer is not present or cannot be reasonably inferred, reply EXACTLY: "Not enough information"

"""

PERSONAS = [
    {
        "system": GLOBAL_RULE + "You are a concise medical assistant. Answer strictly using the context. No extra words or explanation.",
        "template": "Answer concisely based on the context.\nQuestion: {q}\nContext: {ctx}\nAnswer:"
    },
    {
        "system": GLOBAL_RULE + "You are a molecular geneticist focusing on mechanisms. Use only explicitly stated mechanisms from the context.",
        "template": "Focus on molecular genetic mechanisms.\nQuestion: {q}\nContext: {ctx}\nAnswer:"
    },
    {
        "system": GLOBAL_RULE + "You are a clinical researcher prioritizing strong evidence. Only return clinically supported facts from the context.",
        "template": "Prioritize clinically supported evidence.\nQuestion: {q}\nContext: {ctx}\nAnswer:"
    },
    {
        "system": GLOBAL_RULE + "You are a specialist in rare diseases and atypical presentations. Only use information explicitly mentioned in the context.",
        "template": "Highlight rare conditions or atypical presentations if relevant.\nQuestion: {q}\nContext: {ctx}\nAnswer:"
    },
    {
        "system": GLOBAL_RULE + "You apply strict evidence-based medicine principles. Do not infer beyond the provided context.",
        "template": "Apply evidence-based reasoning.\nQuestion: {q}\nContext: {ctx}\nAnswer:"
    },
    {
        "system": GLOBAL_RULE + "You are a paediatric specialist. Focus only on paediatric-relevant information if explicitly stated.",
        "template": "Focus on paediatric relevance if applicable.\nQuestion: {q}\nContext: {ctx}\nAnswer:"
    },
    {
        "system": GLOBAL_RULE + "You are an epidemiologist focusing on population-level insights. Use only population-level evidence from the context.",
        "template": "Consider population-level or epidemiological insights.\nQuestion: {q}\nContext: {ctx}\nAnswer:"
    },
]

def _token_overlap(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if (sa and sb) else 0.0


def diverse_prompted_generate(
    question: str,
    context: str,
    n_generations: int = 7,
    temperature: float = 0.3,
    model_name: str = GEN_MODEL,
    model_cache: Optional[dict] = None,
) -> str:
    
    """
    Generate N candidate answers using distinct expert persona prompts at high
    temperature, then return the candidate with the highest average token overlap
    with all other candidates (most semantically central / consensus answer).

    Personas (in order):
        1. Medical assistant (generic baseline)
        2. Molecular geneticist
        3. Clinical researcher
        4. Rare disease / comorbidity specialist
        5. Evidence-based medicine reviewer
        6. Paediatric specialist
        7. Epidemiologist

    Args:
        question      : User query.
        context       : Formatted retrieved context string.
        n_generations : Number of persona prompts to use (up to 7).
        temperature   : Sampling temperature (higher = more diverse).
        model_name    : HuggingFace generator model identifier.

    Returns:
        Consensus candidate answer string.
    """
    
    candidates = []

    for i, persona in enumerate(PERSONAS[:n_generations]):
        system_role = persona["system"]
        prompt = persona["template"].format(q=question, ctx=context)

        response = _client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_role},   # ✅ HERE
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
        )

        decoded = response.output_text.strip()

        candidates.append(decoded)

    scores = [
        np.mean([
            _token_overlap(c, candidates[j])
            for j in range(len(candidates)) if j != i
        ])
        for i, c in enumerate(candidates)
    ]

    return candidates[int(np.argmax(scores))]


print(f"Diverse prompted generation ready ({len(PERSONAS)} personas defined).")


# ═══════════════════════════════════════════════════════════════════════════════
# K-Means Majority Vote Generation — RAG6
# ═══════════════════════════════════════════════════════════════════════════════

def kmeans_majority_generate(
    question: str,
    context: str,
    n_generations: int = 7,
    temperature: float = 0.8,
    n_clusters: int = 3,
    model_name: str = GEN_MODEL,
    model_cache: Optional[dict] = None,
) -> str:
    
    """
    Generate N candidate answers, embed them, cluster with K-Means, and return
    the response closest to the centroid of the largest (majority) cluster.

    Algorithm:
        1. Generate N answers by sampling the generator at `temperature`.
        2. Encode each answer with a sentence transformer (all-MiniLM-L6-v2).
        3. Run K-Means (k = n_clusters) on the embedding matrix.
        4. Identify the majority cluster (most members = plurality vote).
        5. Return the member closest to that cluster's centroid.

    Advantage over RAG5: operates in embedding space, so it is invariant to
    surface-level paraphrasing and more robust than token-overlap voting.

    Args:
        question      : User query.
        context       : Formatted retrieved context string.
        n_generations : Number of candidate answers to sample.
        temperature   : Sampling temperature.
        n_clusters    : K for K-Means (should be < n_generations).
        model_name    : HuggingFace generator model identifier.

    Returns:
        Answer string closest to the majority cluster centroid.
    """
    
    prompt = f"""
    You MUST follow these rules:
    - Use ONLY the provided context
    - Do NOT use prior knowledge
    - Do NOT infer beyond the context
    - If the answer is not present or cannot be reasonably inferred, reply EXACTLY: "Not enough information"
    
    Question: {question}
    Context: {context}
    Answer:
    """

    candidates = []

    for _ in range(n_generations):
        response = _client.responses.create(
            model=model_name,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
        )

        decoded = response.output_text.strip()

        candidates.append(decoded)

    embeddings = _sent_model.encode(candidates, convert_to_numpy=True)

    k = min(n_clusters, len(candidates))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(embeddings)

    majority = int(np.argmax(np.bincount(labels)))
    members = np.where(labels == majority)[0]

    dists = np.linalg.norm(
        embeddings[members] - km.cluster_centers_[majority],
        axis=1
    )

    return candidates[int(members[np.argmin(dists)])]


print(f"K-Means majority vote generator ready (sentence encoder: {_SENT_MODEL_NAME}).")
print("\n✓ Section 0 complete — all helpers loaded. Proceed to Section 1.")