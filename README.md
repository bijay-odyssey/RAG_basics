# RAG Basics – Minimal Retrieval-Augmented Generation (RAG) Chatbot

This repository demonstrates a **minimal and easy-to-understand Retrieval-Augmented Generation (RAG) pipeline** using:

* **Qdrant** for vector-based retrieval
* **Sentence Transformers** for embeddings
* **Groq LLM API** for fast text generation

The goal of this project is to explain **RAG fundamentals** with clean, readable code—ideal for learning, interviews, and quick prototyping.

---

##  What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:

1. **Retrieval** – fetch relevant documents from a vector database
2. **Generation** – use an LLM to generate answers grounded in retrieved context

This approach improves factual accuracy and allows models to answer questions based on **external knowledge**.

---

##  Features

* Lightweight **end-to-end RAG implementation**
* Uses **Qdrant** for semantic retrieval
* Uses **Groq-hosted LLMs** for fast inference
* Automatic model selection from available Groq chat models
* CLI-based interactive chatbot
* Safe prompting: answers only from retrieved context

---

##  Tech Stack

* **Python 3.8+**
* **Sentence Transformers** (`all-MiniLM-L6-v2`)
* **Qdrant Vector Database**
* **Groq API**
* **dotenv** for environment variables

---

##  Project Structure

```text
RAG_basics/
│
├── rag_basic.py   # Minimal RAG chatbot implementation
└── README.md
```

---

##  Prerequisites

Before running the project, ensure you have:

* Python 3.8+
* Docker (for Qdrant)
* A **Groq API key**

---

##  Setup Instructions

### 1️ Run Qdrant Locally

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Qdrant will be available at:

```
http://localhost:6333
```

> This script assumes a **Qdrant collection named `documents` already exists** and contains vectors with a `text` payload.

---

### 2️ Install Dependencies

```bash
pip install sentence-transformers qdrant-client python-dotenv groq
```

---

### 3️ Set Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

##  Running the RAG Chatbot

```bash
python rag_basic.py
```

You’ll see:

```text
=== BASIC RAG CHATBOT ===
Ask a question (or 'exit'):
```

---

##  Example Interaction

```text
Ask a question (or 'exit'): What is a vector database?
```

**Output:**

```text
Context:
- Vector databases store embeddings of text for semantic search.
- Qdrant is a vector database optimized for embeddings.

Answer:
A vector database stores embeddings and is optimized for semantic search.
```

If the answer is **not found in retrieved context**, the model responds:

```text
I don't know.
```

---

##  How the RAG Pipeline Works

### 1. **Retrieve Context**

* Converts the user query into an embedding
* Retrieves top-K similar documents from Qdrant

### 2. **Build Prompt**

* Injects retrieved context into a strict prompt
* Prevents hallucination by limiting answers to context

### 3. **Generate Answer**

* Uses Groq’s fastest available chat model
* Produces a grounded, context-aware response

---

##  Key Functions Explained

| Function                     | Purpose                              |
| ---------------------------- | ------------------------------------ |
| `retrieve_context()`         | Fetch relevant documents from Qdrant |
| `build_prompt()`             | Create a safe, context-only prompt   |
| `get_available_chat_model()` | Auto-select Groq chat model          |
| `generate_answer()`          | Generate LLM response                |
| `rag_query()`                | Full RAG pipeline execution          |

---

##  Use Cases

* Learning RAG fundamentals
* Interview preparation
* Knowledge-grounded chatbots
* Prototyping AI search systems
* Base template for production RAG apps

---

##  Possible Enhancements

* Add document ingestion pipeline
* Support PDFs / text files
* Add FastAPI REST interface
* Streaming responses
* Hybrid search (BM25 + vectors)
* Multi-collection support

 
---

##  Author

**Bijay**
GitHub: [bijay-odyssey](https://github.com/bijay-odyssey)

---
 
