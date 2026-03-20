"""
Generate answer from question + retrieved context using a configurable LLM.
"""
import ast
from typing import Optional

# 🔥 Optimized prompt (no duplicate system role)
DEFAULT_PROMPT_TEMPLATE = (
    "Answer the question using ONLY the provided context.\n"
    "Do not use external knowledge.\n\n"
    
    "If the answer can be reasonably inferred from the context, provide the answer.\n"
    "If the context does not contain enough information, reply:\n"
    "\"Not enough information\"\n\n"
    
    "Be concise and extract precise biomedical facts.\n\n"
    
    "Question:\n{question}\n\n"
    "Context:\n{context}\n\n"
    
    "Final Answer:"
)


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
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",  # 🔥 smaller model
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_new_tokens: int = 128,
    device: Optional[str] = None,
) -> str:

    prompt = prompt_template.format(question=question, context=context)

    return _generate_with_hf(
        prompt,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        device=device
    )


_model_cache: dict = {}
   
# ==============================
# CORE GENERATION (FAST VERSION)
# ==============================
def _generate_with_hf(
    prompt: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 32,   # 🔥 reduced for speed
    device: Optional[str] = None,
) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_name = model_name.strip()

    # ==========================
    # LOAD MODEL (ONCE ONLY)
    # ==========================
    if model_name not in _model_cache:
        print(f"🔥 Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        _model_cache[model_name] = (tokenizer, model)

    tokenizer, model = _model_cache[model_name]

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
    MAX_CTX = 1536   # 🔥 safe + fast

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
            max_time=10,   # 🔥 prevents hanging
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # ==========================
    # CLEAN OUTPUT
    # ==========================
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()