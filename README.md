# LibRAGrian

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/stas/gutenberg-100)


<img width="399" height="417" alt="LibRAGrian illustration" src="https://github.com/user-attachments/assets/eb196d7f-b2d3-4b22-ab3b-adb176b66399" />




---

## 📋 Prerequisites

This project requires:

- Python 3.10+
- A working Python environment (venv, conda, etc.)

---

## ⚙️ Installation

Clone the repository and install dependencies:

```
$ git clone https://github.com/theophile-bb/LibRAGrian.git
$ cd LibRAGrian
$ pip install -r requirements.txt
```
---

## Getting the data

The data used for this project are available on Hugging Face at this address : https://huggingface.co/datasets/stas/gutenberg-100.

---

## Notebook

The main notebook LibRAGrian.ipynb walks through:

- Text processing and cleaning.

- Chunking of the books using **Chonkie Chunker**.

- Embedding of the chunks using the **bge-small-en-v1.5** model.

- Indexing of the embeddings with **Faiss**.

- Retrieval with the **Qwen 2.5-3b** model.

- Querying with examples.

---
