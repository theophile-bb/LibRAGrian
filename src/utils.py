import pandas as pd
import numpy as np

import re
import unicodedata
import ast
from langdetect import detect, detect_langs

import tiktoken
from chonkie import TokenChunker
from chonkie import RecursiveChunker, RecursiveRules
from chonkie import Visualizer

import transformers
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import faiss


#--------------------- Book cleaning ------------------------

def strip_beginning(text, expression):
    if not isinstance(text, str):
        return text
    return re.sub(
        expression,
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

def strip_until_title(text, title):
    if not text or not title:
        return text

    title = title.strip()

    #whitespace-tolerant regex
    words = re.findall(r"\w+", title)
    if not words:
        return text

    title_pattern = r"\s+".join(map(re.escape, words))

    match = re.search(title_pattern, text, re.IGNORECASE)
    if match:
        return text[match.start():]

    return text

def Clean_book(string, title):

  # Decode UTF-8 BOM
  if string is None:
        return None

  if isinstance(string, str):
      if string.startswith("b'") or string.startswith('b"'):
        string = ast.literal_eval(string)
      else:
        text = string
  if isinstance(string, bytes):
      text = string.decode("utf-8-sig")

  # Keep the content between the tags
  start = re.search(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
  end   = re.search(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)

  if start and end and start.end() < end.start():
        text = text[start.end():end.start()]

  # Remove unncessary line jumps
  text = unicodedata.normalize("NFKC", text)
  text = text.replace("\r\n", "\n")
  text = re.sub(r"\n{3,}", "\n\n", text)
  text = text.strip()

  # Strip redundant noise
  text = strip_until_title(text, title)
  text = strip_beginning(text, r'^Produced by.*?\n\s*\n')
  text = strip_beginning(text, r'^\[Illustration\]\s*\n\s*\n')
  text = text.strip()

  return text

#--------------------- Book chunking ------------------------

def Chunk_text(text, size = 512, overlap = 128):
  encoding = tiktoken.get_encoding("gpt2")

  chunker = TokenChunker(
      tokenizer=encoding,    # Pass the encoding object
      chunk_size=size,        # Maximum tokens per chunk
      chunk_overlap=overlap      # Overlap between chunks
  )
  chunks = chunker.chunk(text)
  return chunks

def Recursive_chunk_text(text, size = 512, overlap = 128):
  encoding = tiktoken.get_encoding("gpt2")

  chunker = RecursiveChunker(
      tokenizer=encoding,    # Pass the encoding object
      chunk_size=size,        # Maximum tokens per chunk
      chunk_overlap=overlap,      # Overlap between chunks
      rules = RecursiveRules(),
      min_characters_per_chunk = 24
  )
  chunks = chunker.chunk(text)
  return chunks


def Create_chunk_df(book_df):
  rows = []

  for i, row in book_df.iterrows():
    chunks = Chunk_text(row['cleaned_text'])

    for j, chunk in enumerate(chunks):
          rows.append({
              "book_id": i,
              "title": row["title"],
              "chunk_id": j,
              "chunk": chunk.text,
              "token_count": chunk.token_count
          })
  chunks_df = pd.DataFrame(rows)
  return chunks_df

#--------------------- Embedding ------------------------

def Embedding(model_name, texts, batch_size = 32):
  device = "cuda" if torch.cuda.is_available() else "cpu"

  print(f"Using device: {device}")

  model = SentenceTransformer(model_name, device=device)

  model.eval()

  embeddings = []
  for i in range(0, len(texts), batch_size):
      batch = texts[i:i+batch_size]

      if not batch:
          continue

      batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
      embeddings.extend(batch_embeddings.tolist())

  print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}.")

  return embeddings

#--------------------- Create Faiss index ------------------------

def Create_Faiss_index(embeddings):
  embeddings = np.array(embeddings).astype("float32")
  faiss.normalize_L2(embeddings)

  index = faiss.IndexFlatIP(embeddings.shape[1])
  index.add(embeddings)
  return index


#--------------------- Models loading ------------------------

def load_embedding_model(model_name, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        model_name,
        device=device
    )
    return model

def load_generation_model(model_name="Qwen/Qwen2.5-3B-Instruct", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    model.eval()
    return tokenizer, model, device

#--------------------- Retrieval from Faiss index ------------------------

def retrieve(
    query,
    embed_model,
    index,
    chunks_df,
    k=8
):

    q_emb = embed_model.encode(query, normalize_embeddings=True)
    q_emb = np.asarray([q_emb], dtype="float32")

    _, I = index.search(q_emb, k)

    return chunks_df.iloc[I[0]].reset_index(drop=True)


def build_context(retrieved_rows):
    blocks = []

    for _, row in retrieved_rows.iterrows():
        blocks.append(
            f"[Source: {row['title']} | Chunk {row['chunk_id']}]\n{row['chunk']}"
        )

    return "\n\n---\n\n".join(blocks)

#--------------------- RAG pipeline ------------------------

def answer_query(
    query,
    embed_model,
    index,
    chunks_df,
    gen_model,
    gen_tokenizer,
    device,
    k=8,
    max_new_tokens=200
):
    # retrieve → build context → generate answer.
    
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