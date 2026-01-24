import re
import unicodedata
import ast
from typing import List, Optional, Union, Dict, Any, Tuple

import pandas as pd
import numpy as np
import torch
import faiss
import tiktoken
from chonkie import TokenChunker, RecursiveChunker, RecursiveRules
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from langdetect import detect, detect_langs


def strip_beginning(text: str, expression: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(
        expression,
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

def strip_until_title(text: str, title: str) -> str:
    if not text or not title:
        return text

    title = title.strip()
    words = re.findall(r"\w+", title)
    if not words:
        return text

    title_pattern = r"\s+".join(map(re.escape, words))
    match = re.search(title_pattern, text, re.IGNORECASE)
    if match:
        return text[match.start():]

    return text

def clean_book(string: Optional[Union[str, bytes]], title: str) -> Optional[str]:
    if string is None:
        return None

    text: str = ""
    if isinstance(string, str):
        if string.startswith("b'") or string.startswith('b"'):
            # Convert string representation of bytes back to actual string
            text = ast.literal_eval(string).decode("utf-8-sig") if isinstance(ast.literal_eval(string), bytes) else ast.literal_eval(string)
        else:
            text = string
    elif isinstance(string, bytes):
        text = string.decode("utf-8-sig")

    start = re.search(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
    end = re.search(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)

    if start and end and start.end() < end.start():
        text = text[start.end():end.start()]

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    text = strip_until_title(text, title)
    text = strip_beginning(text, r'^Produced by.*?\n\s*\n')
    text = strip_beginning(text, r'^\[Illustration\]\s*\n\s*\n')
    
    return text.strip()


def chunk_text(text: str, size: int = 512, overlap: int = 128) -> List[Any]:
    encoding = tiktoken.get_encoding("gpt2")
    chunker = TokenChunker(
        tokenizer=encoding,   
        chunk_size=size,         
        chunk_overlap=overlap      
    )
    return chunker.chunk(text)

def recursive_chunk_text(text: str, size: int = 512, overlap: int = 128) -> List[Any]:
    encoding = tiktoken.get_encoding("gpt2")
    chunker = RecursiveChunker(
        tokenizer=encoding,    
        chunk_size=size,         
        chunk_overlap=overlap,      
        rules=RecursiveRules(),
        min_characters_per_chunk=24
    )
    return chunker.chunk(text)

def create_chunk_df(book_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, row in book_df.iterrows():
        chunks = chunk_text(row['cleaned_text'])
        for j, chunk in enumerate(chunks):
            rows.append({
                "book_id": i,
                "title": row["title"],
                "chunk_id": j,
                "chunk": chunk.text,
                "token_count": chunk.token_count
            })
    return pd.DataFrame(rows)



def embed(model_name: str, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(model_name, device=device)
    model.eval()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if not batch:
            continue
        batch_embeddings = model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
        embeddings.extend(batch_embeddings.tolist())

    print(f"Generated {len(embeddings)} embeddings.")
    return embeddings

def create_faiss_index(embeddings: Union[np.ndarray, List[List[float]]]) -> faiss.IndexFlatIP:
    data = np.array(embeddings).astype("float32")
    faiss.normalize_L2(data)
    
    index = faiss.IndexFlatIP(data.shape[1])
    index.add(data)
    return index



def load_embedding_model(model_name: str, device: Optional[str] = None) -> SentenceTransformer:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)

def load_generation_model(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct", 
    device: Optional[str] = None
) -> Tuple[PreTrainedTokenizer, PreTrainedModel, str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device != "cuda":
        model.to(device)
        
    model.eval()
    return tokenizer, model, device

def retrieve(
    query: str,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    chunks_df: pd.DataFrame,
    k: int = 8
) -> pd.DataFrame:
    q_emb = embed_model.encode(query, normalize_embeddings=True)
    q_emb = np.asarray([q_emb], dtype="float32")

    _, I = index.search(q_emb, k)
    return chunks_df.iloc[I[0]].reset_index(drop=True)

def build_context(retrieved_rows: pd.DataFrame) -> str:
    blocks = []
    for _, row in retrieved_rows.iterrows():
        blocks.append(
            f"[Source: {row['title']} | Chunk {row['chunk_id']}]\n{row['chunk']}"
        )
    return "\n\n---\n\n".join(blocks)

def answer_query(
    query: str,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    chunks_df: pd.DataFrame,
    gen_model: PreTrainedModel,
    gen_tokenizer: PreTrainedTokenizer,
    device: str,
    k: int = 8,
    max_new_tokens: int = 200
) -> Dict[str, Any]:
    
    retrieved_rows = retrieve(
        query=query,
        embed_model=embed_model,
        index=index,
        chunks_df=chunks_df,
        k=k
    )

    context = build_context(retrieved_rows)

    prompt = f"""You are a librarian and literary assistant managing a library.
You are ONLY allowed to answer questions using the context below.
Do NOT make up any information.
If the context does not contain the answer, respond exactly with:
"I don't know."

CONTEXT:
{context}

QUESTION:
{query}

Answer strictly based on the context:
"""

    inputs = gen_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.6,
            do_sample=True,
            eos_token_id=gen_tokenizer.eos_token_id,
        )

    output_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = output_text[len(prompt):].strip()

    return {
        "answer": answer,
        "sources": retrieved_rows[["title", "chunk_id"]]
    }

