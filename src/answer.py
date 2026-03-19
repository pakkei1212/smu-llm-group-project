"""
Generate answer from question + retrieved context using a configurable LLM.
"""
from typing import Optional

DEFAULT_PROMPT_TEMPLATE = (
    "Answer the following question based only on the given context.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)


def format_context(chunks: list[tuple], sep: str = "\n\n---\n\n") -> str:
    """Format list of (chunk_id, chunk_text, meta) into a single context string."""
    return sep.join(t[1] for t in chunks)
    

# ==============================
# PMID extraction (NEW)
# ==============================
def extract_pmids(chunks: list[tuple]) -> list[str]:
    """
    Extract unique PMIDs from chunks.

    Supports:
    - (id, text, metadata_dict)
    - stringified metadata dict
    """
    pmids = set()

    for c in chunks:
        pmid = None

        # Case 1: tuple with metadata
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

        # Case 2: chunk itself is metadata string
        elif isinstance(c, str) and c.startswith("{"):
            try:
                meta_dict = ast.literal_eval(c)
                pmid = meta_dict.get("pmid")
            except:
                pass

        if pmid:
            pmids.add(str(pmid))

    return sorted(pmids)

    
def generate_answer(
    question: str,
    context: str,
    model_name: str = "google/flan-t5-small",
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_new_tokens: int = 256,
    device: Optional[str] = None,
) -> str:
    """
    Generate a single answer using HuggingFace transformers.
    Model is loaded on first call and reused.
    """
    prompt = prompt_template.format(question=question, context=context)
    return _generate_with_hf(prompt, model_name=model_name, max_new_tokens=max_new_tokens, device=device)


_model_cache: dict = {}


def _generate_with_hf(
    prompt: str,
    model_name: str = "google/flan-t5-small",
    max_new_tokens: int = 256,
    device: Optional[str] = None,
) -> str:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    if model_name not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _model_cache[model_name] = (tokenizer, model, device)

    tokenizer, model, dev = _model_cache[model_name]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(dev)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)