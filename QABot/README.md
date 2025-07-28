# QABot: PDF Question-Answering with LLaMA 3 & Chroma

**QABot** is a lightweight retrieval-augmented question-answering system that reads content from PDF files and returns highly accurate answers using Meta's [`Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-3.2B-Instruct) model, combined with vector search via [Chroma](https://www.trychroma.com/).

---

## 🚀 Features

- 📄 **PDF Ingestion**: Load and split PDF documents automatically.
- 🔍 **Retrieval-Augmented Generation (RAG)**: Combines semantic retrieval with LLMs for contextual answers.
- 🧠 **Powered by Meta’s LLaMA 3.2 3B**: Leverages a strong instruct-tuned model from Hugging Face.
- ⚡ **Fast Local Vector Search**: Uses Chroma for low-latency document retrieval.
- 🛠️ **Modular & Extensible**: Easy to replace components (e.g., vector store or LLM backend).
