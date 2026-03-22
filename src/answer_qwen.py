"""
Generate answer from question + retrieved context using a configurable LLM.
"""
import ast
import torch
import gc
import numpy as np
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# ==============================
# GLOBAL GENERATION CONFIG 🔥
# ==============================
GRAD_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GEN_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_CTX = 3072
MAX_NEW_TOKENS = 64

# ==============================
# GLOBAL MODEL CACHE ✅ (UNIFIED)
# ==============================
_MODEL_CACHE: dict[str, tuple] = {}

# 🔥 sentence encoder (used in RAG6)
_SENT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_sent_model = SentenceTransformer(_SENT_MODEL_NAME)

# 🔥 Optimized prompt (no duplicate system role)
DEFAULT_PROMPT_TEMPLATE = (
    "You are a biomedical expert assistant.\n\n"
    
    "Answer the question using the provided context.\n"
    "You may combine and infer information across the context to form the answer.\n"
    
    "If the context provides partial information, give the best possible answer.\n"
    "Only say \"Not enough information\" if nothing relevant is found.\n\n"
    
    "Keep the answer concise and factual.\n\n"
    
    "Question:\n{question}\n\n"
    "Context:\n{context}\n\n"
    
    "Answer:"
)

# ==============================
# SHARED MODEL LOADER 🔥
# ==============================
def _get_model(model_name: str, model_cache: Optional[dict] = None):
    cache = model_cache if model_cache is not None else _MODEL_CACHE

    model_name = model_name.strip()

    if model_name not in cache:
        print(f"🔥 Loading model: {model_name}")

        # 🚨 prevent OOM (important for your pipeline)
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

    return _generate_with_hf(
        prompt,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        device=device,
        model_cache=model_cache
    )


# ==============================
# CORE GENERATION (FAST VERSION)
# ==============================
def _generate_with_hf(
    prompt: str,
    model_name: str,
    max_new_tokens: int,
    device: Optional[str],
    model_cache: Optional[dict] = None,
) -> str:

    tokenizer, model = _get_model(model_name, model_cache)

    # ==========================
    # BUILD CHAT INPUT (QWEN)
    # ==========================
    messages = [
        {"role": "system", "content": "You are a biomedical expert assistant."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # ==========================
    # TOKENIZE
    # ==========================
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CTX
    ).to(model.device)

    # ==========================
    # GENERATE (FAST + SAFE)
    # ==========================
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            max_time=10,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # ==========================
    # CLEAN OUTPUT
    # ==========================
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


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
        max_length=MAX_CTX
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # ==========================
    # Get embeddings (Qwen-compatible)
    # ==========================
    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(input_ids).detach().requires_grad_(True)

    # ==========================
    # Forward + backward
    # ==========================
    model.zero_grad()
    
    outputs = model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        labels=input_ids
    )

    outputs.loss.backward()

    saliency = embeds.grad.norm(dim=-1).squeeze(0).detach().cpu().numpy()

    # ==========================
    # Map token saliency → chunks
    # ==========================
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

PERSONA_SYSTEM_ROLES = [
    "You are a concise medical assistant.",
    "You are a molecular geneticist focusing on mechanisms.",
    "You are a clinical researcher prioritizing strong evidence.",
    "You are a specialist in rare diseases and atypical presentations.",
    "You apply strict evidence-based medicine principles.",
    "You are a paediatric specialist.",
    "You are an epidemiologist focusing on population-level insights.",
]

PERSONA_TEMPLATES = [
    "Answer concisely based on the context.\nQuestion: {q}\nContext: {ctx}\nAnswer:",
    "Focus on molecular genetic mechanisms.\nQuestion: {q}\nContext: {ctx}\nAnswer:",
    "Prioritize clinically supported evidence.\nQuestion: {q}\nContext: {ctx}\nAnswer:",
    "Highlight rare conditions or atypical presentations if relevant.\nQuestion: {q}\nContext: {ctx}\nAnswer:",
    "Apply evidence-based reasoning.\nQuestion: {q}\nContext: {ctx}\nAnswer:",
    "Focus on paediatric relevance if applicable.\nQuestion: {q}\nContext: {ctx}\nAnswer:",
    "Consider population-level or epidemiological insights.\nQuestion: {q}\nContext: {ctx}\nAnswer:",
]

def _token_overlap(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if (sa and sb) else 0.0


def diverse_prompted_generate(
    question: str,
    context: str,
    n_generations: int = 7,
    temperature: float = 0.8,
    model_name: str = GEN_MODEL,
    model_cache: Optional[dict] = None,
) -> str:

    tok, model = _get_model(model_name, model_cache)

    candidates = []

    for i, tmpl in enumerate(PERSONA_TEMPLATES[:n_generations]):
        system_role = PERSONA_SYSTEM_ROLES[i]
        prompt = tmpl.format(q=question, ctx=context)

        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ]

        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_CTX).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temperature
            )

        gen_tokens = out[0][enc["input_ids"].shape[1]:]
        decoded = tok.decode(gen_tokens, skip_special_tokens=True).strip()

        candidates.append(decoded)

    scores = [
        np.mean([
            _token_overlap(c, candidates[j])
            for j in range(len(candidates)) if j != i
        ])
        for i, c in enumerate(candidates)
    ]

    return candidates[int(np.argmax(scores))]


print(f"Diverse prompted generation ready ({len(PERSONA_TEMPLATES)} personas defined).")


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

    tok, model = _get_model(model_name, model_cache)

    prompt = (
        f"Answer the question based only on the context.\n"
        f"Question: {question}\nContext: {context}\nAnswer:"
    )

    messages = [
        {"role": "system", "content": "You are a biomedical expert assistant."},
        {"role": "user", "content": prompt},
    ]

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_CTX).to(model.device)

    input_len = enc["input_ids"].shape[1]

    candidates = []

    with torch.no_grad():
        for _ in range(n_generations):
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temperature
            )

            gen_tokens = out[0][input_len:]
            decoded = tok.decode(gen_tokens, skip_special_tokens=True).strip()
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