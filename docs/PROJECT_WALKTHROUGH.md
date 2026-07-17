# Reasona — Project Walkthrough

> A plain-English guide to the entire codebase.  
> No jargon where it can be avoided; diagrams where they help.

---

## 1. What Is This Project?

Reasona is a **question-answering system** that reads your documents (PDFs, Word files, plain text) and answers questions about them.

What makes it different from a basic chatbot:

- It **self-corrects**. After generating an answer, it checks its own work. If the answer is wrong, it fixes it and remembers the correction for next time.
- It uses two research techniques — **HyDE** and **SEAL** — to be smarter about finding and fixing information.

In simple terms:

> You upload files → ask questions → get answers grounded in your documents.  
> If the system gets something wrong, it learns from the mistake automatically.

---

## 2. The Big Picture

Here is how a question flows through the system from start to finish:

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                         YOU (the user)                          │
 │  1. Open the web UI (Streamlit, port 8501)                      │
 │  2. Upload a PDF / DOCX / TXT                                   │
 │  3. Type a question like "What is X?"                           │
 └────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   FRONTEND  (ui.py)                             │
 │  • Shows the chat interface                                     │
 │  • Sends your file / question to the backend over HTTP          │
 └────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   BACKEND  (main.py — FastAPI)                  │
 │  • /upload   → receives files, extracts text, stores in DB     │
 │  • /query    → runs the full HyDE + SEAL pipeline              │
 │  • /health   → quick "am I alive?" check                        │
 └────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   CORE ENGINE  (core/)                          │
 │                                                                  │
 │   config.py ──→ loads settings from .env                        │
 │   models.py ──→ defines request/response shapes                 │
 │   ingest.py ──→ extracts text from uploaded files               │
 │   llm_factory.py ──→ picks the right AI model                   │
 │   vectorstore.py ──→ manages the vector database (ChromaDB)     │
 │   rag_engine.py ──→ runs the HyDE + SEAL reasoning loop        │
 └──────────────────────────────────────────────────────────────────┘
```

---

## 3. How the Answer Pipeline Works (Step by Step)

When you ask a question, the system does the following **in order**:

```
  Question: "What is X?"
       │
       ▼
  ┌─────────────────────────────────┐
  │  STEP 1 — HyDE (Hypothesize)   │
  │  Ask the AI: "What do you think │
  │  the answer might be?"          │
  │  It generates 3 guesses.        │
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │  STEP 2 — Retrieve             │
  │  Search the vector DB using     │
  │  BOTH the original question     │
  │  AND the 3 guesses.            │
  │  This finds more relevant docs  │
  │  than searching with just the   │
  │  question alone.               │
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │  STEP 3 — Generate Answer      │
  │  Give the AI the retrieved docs │
  │  + the original question.       │
  │  It writes a 3-5 sentence      │
  │  answer based only on those     │
  │  documents.                    │
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │  STEP 4 — Critic (Fact-Check)  │
  │  A separate AI call checks:     │
  │  "Is this answer actually       │
  │   supported by the context?"    │
  │  Returns CORRECT or INCORRECT.  │
  └──────────────┬──────────────────┘
                 │
          ┌──────┴──────┐
          │             │
     CORRECT       INCORRECT
          │             │
          ▼             ▼
  ┌───────────┐  ┌──────────────────────┐
  │ Show      │  │ STEP 5 — SEAL        │
  │ answer    │  │ (Self-Edit & Learn)  │
  │ to user   │  │                      │
  │ ✅        │  │ • Show answer + ⚠️    │
  └───────────┘  │   warning badge      │
                 │ • Generate a better  │
                 │   text chunk         │
                 │ • Create Q&A pairs   │
                 │ • Save corrections   │
                 │   into the DB so     │
                 │   future queries     │
                 │   get it right       │
                 │ • Show 🧠 "Learned"  │
                 │   badge to user      │
                 └──────────────────────┘
```

**Why this matters:** A normal RAG system would stop at Step 3. Reasona adds Steps 4 and 5 — the user still gets an answer, but the system also **learns from the mistake** in the background so future queries are more accurate.

---

## 4. File-by-File Breakdown

### `main.py` — The Backend Server

| What it does | How it does it |
|---|---|
| Runs the web server | FastAPI on port 8000 |
| Accepts file uploads | `POST /upload` — reads the file, extracts text, stores embeddings |
| Handles questions | `POST /query` — runs the full HyDE+SEAL pipeline |
| Health check | `GET /health` — returns provider info |

Key design choices:
- Uses a **thread pool** (4 workers) so heavy AI calls don't block the server.
- Wraps blocking calls in `run_in_executor` so the async event loop stays responsive.
- CORS is wide open (`*`) for development convenience.

---

### `ui.py` — The Frontend

A **Streamlit** app that gives users a chat-style interface.

- **Sidebar**: pick your AI provider (Ollama local, OpenAI cloud, Google cloud) and model.
- **Upload area**: drag-and-drop PDF / DOCX / TXT files.
- **Chat area**: type questions, see answers with timing info.
- Shows badges when the system self-corrected or learned something new.

---

### `core/config.py` — Settings Loader

Uses **Pydantic Settings** to read from a `.env` file. Defines:

| Setting | Default | Purpose |
|---|---|---|
| `llm_provider` | `ollama` | Which AI to use |
| `model_name` | `qwen3:1.7b` | Which specific model |
| `ollama_host` | `http://localhost:11434` | Local Ollama URL |
| `chroma_path` | `./data/chroma` | Where to store the vector DB |
| `upload_path` | `./data/uploads` | Where uploaded files go |
| `openai_api_key` | None | Optional cloud API key |
| `google_api_key` | None | Optional cloud API key |

---

### `core/models.py` — Data Shapes

Pydantic models that define the shape of data moving between frontend and backend:

- **QueryRequest**: `question` + `provider` + `model`
- **QueryResponse**: `answer` + `retrieved_docs` + `was_corrected` + `self_edit_performed`
- **SelfEdit**: `original_chunk` + `improved_chunk` + `qa_pairs`

---

### `core/ingest.py` — File Reader

Takes raw file bytes and a filename, detects the type, and extracts plain text:

```
  .pdf  → PyMuPDFLoader
  .docx → UnstructuredWordDocumentLoader
  .txt  → TextLoader
```

Writes to a temp file, reads it, cleans up. Returns one big string of text.

---

### `core/llm_factory.py` — AI Model Picker

A factory function: give it a provider name and model name, get back a ready-to-use LLM object.

```
  "ollama"  → ChatOllama  (runs locally, no API key needed)
  "openai"  → ChatOpenAI  (needs OPENAI_API_KEY)
  "google"  → ChatGoogleGenerativeAI  (needs GOOGLE_API_KEY)
```

Falls back to `qwen3:1.7b` if the model name is missing or contains "llama".

---

### `core/vectorstore.py` — The Memory

Manages a **ChromaDB** vector database — this is where all document knowledge lives.

- **Embedding model**: `all-MiniLM-L6-v2` from HuggingFace (small, fast, good quality).
- **Chunking**: Splits long texts into ~1000 character pieces with 150 character overlap so no information falls between the cracks.
- **add_documents()**: Chunks text → embeds → stores.
- **get_retriever()**: Returns a search function that finds the 3 most similar chunks for a given query.

---

### `core/rag_engine.py` — The Brain

This is where HyDE and SEAL come together. The `HyDE_SEAL_Engine` class:

1. **HyDE step** — generates 3 hypothetical answers (higher temperature = more creative guesses).
2. **Retrieve** — searches the DB with the question + all hypotheses, deduplicates results, keeps top 4.
3. **RAG answer** — generates a final answer from the retrieved context.
4. **Critic** — a strict fact-checker judges if the answer is supported by the context (returns JSON with `verdict` and `rationale`).
5. **SEAL self-edit** — if the critic says INCORRECT:
   - Generates an `improved_chunk` (better version of the context).
   - Generates `qa_pairs` (question-answer pairs for future retrieval).
   - Generates `edit_directives` (meta-notes about what was wrong).
   - **Stores all of this back into ChromaDB** so the system is smarter next time.

---

## 5. Data Flow Diagram

```
                    ┌─────────────────┐
                    │   User's Files  │
                    │  (PDF/DOCX/TXT) │
                    └────────┬────────┘
                             │ upload
                             ▼
                    ┌─────────────────┐
                    │    ingest.py    │
                    │  extract text   │
                    └────────┬────────┘
                             │ raw text
                             ▼
                    ┌─────────────────┐
                    │ vectorstore.py  │
                    │  chunk → embed  │──────────────────┐
                    │  → store        │                  │
                    └────────┬────────┘                  │
                             │                           │
                             ▼                           ▼
                    ┌─────────────────┐        ┌─────────────────┐
                    │    ChromaDB     │        │   HuggingFace   │
                    │  (persistent    │◄───────│   Embeddings    │
                    │   vector DB)    │        │  (MiniLM-L6-v2) │
                    └────────┬────────┘        └─────────────────┘
                             │ retrieve
                             ▼
                    ┌─────────────────┐
                    │  rag_engine.py  │
                    │  HyDE → RAG →  │
                    │  Critic → SEAL  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  LLM (Ollama /  │
                    │  OpenAI/Google)  │
                    └────────┬────────┘
                             │ answer
                             ▼
                    ┌─────────────────┐
                    │   User sees     │
                    │   the answer    │
                    └─────────────────┘
```

---

## 6. How to Run It

```
Terminal 1 (backend):    python main.py        → starts FastAPI on port 8000
Terminal 2 (frontend):   streamlit run ui.py   → starts Streamlit on port 8501
```

Open `http://localhost:8501`, upload documents, ask questions.

---

## 7. Key Concepts in Plain English

| Term | What It Actually Means |
|---|---|
| **RAG** | "Retrieval-Augmented Generation" — look up relevant documents first, then let the AI answer based on what it found. |
| **HyDE** | Instead of searching with your raw question, first guess what the answer might look like, then search with that guess. Finds better results. |
| **SEAL** | After answering, check the answer. If it's wrong, create a corrected version and save it so the system doesn't make the same mistake again. |
| **Vector Database** | A database that stores text as numbers (embeddings) so you can find "similar" text quickly, even if the exact words are different. |
| **Embeddings** | Converting text into a list of numbers that capture its meaning. Similar meanings → similar numbers. |
| **Chunking** | Breaking a long document into smaller pieces so each piece can be searched and retrieved independently. |
| **LLM** | "Large Language Model" — the AI that reads text and generates answers (like GPT, Gemini, or Llama). |

---

## 8. Tech Stack Summary

```
  Frontend:      Streamlit (Python web framework for data apps)
  Backend:       FastAPI (high-performance Python API framework)
  AI Framework:  LangChain (tools for building LLM applications)
  Vector DB:     ChromaDB (stores and searches document embeddings)
  Embeddings:    HuggingFace all-MiniLM-L6-v2 (converts text to vectors)
  AI Models:     Ollama (local) / OpenAI (cloud) / Google (cloud)
  Config:        Pydantic Settings (type-safe .env loading)
```
