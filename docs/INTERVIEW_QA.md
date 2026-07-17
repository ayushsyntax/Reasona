# Reasona — Interview Q&A

> Common questions an interviewer might ask about this project, with clear answers.  
> Organized from general to technical.

---

## General / "Tell Me About Your Project"

---

### Q1: What is Reasona? Explain it in one sentence.

**A:** Reasona is a self-correcting question-answering system — you upload documents, ask questions about them, and the system not only answers using those documents but also checks its own answers and learns from mistakes.

---

### Q2: What problem does it solve?

**A:** Standard RAG (Retrieval-Augmented Generation) systems often retrieve the wrong chunks of text or generate answers that aren't faithful to the source material. They have no way to catch or fix their own mistakes. Reasona adds a **fact-checking step** and a **self-correction loop** so the system improves its knowledge base over time, reducing repeated errors.

---

### Q3: What makes this different from a basic RAG system?

**A:** Two things:

1. **HyDE (Hypothetical Document Embeddings)** — Instead of searching the database with the raw question, Reasona first generates "guesses" of what the answer might look like, then uses those guesses to search. This finds more relevant documents because the search query now looks like an answer, not a question.

2. **SEAL (Self-Edit And Learn)** — After generating an answer, a separate AI call fact-checks it against the source documents. If it's wrong, the system generates a corrected version and stores it back in the database. Next time someone asks a similar question, the correct information is already there.

```
  Normal RAG:     Question → Search → Answer → Done
  
  Reasona:        Question → Hypothesize → Search → Answer 
                           → Fact-Check → If Wrong → Fix → Store → Done
```

---

### Q4: Walk me through what happens when a user asks a question.

**A:** Here's the step-by-step:

1. User types a question in the Streamlit UI.
2. The UI sends a POST request to the FastAPI backend (`/query`).
3. The backend calls `rag_engine.py`, which:
   - **HyDE**: Generates 3 hypothetical answers using a higher-temperature LLM call.
   - **Retrieve**: Searches ChromaDB using the question + all 3 hypotheses. Deduplicates and keeps the top 4 results.
   - **Generate**: Feeds the retrieved documents + original question to the LLM. Gets a 3-5 sentence answer.
   - **Critic**: A separate LLM call acts as a strict fact-checker. Returns `CORRECT` or `INCORRECT` with a rationale.
   - **SEAL** (only if INCORRECT): Generates an improved text chunk, Q&A pairs, and edit directives, and stores all of these back into ChromaDB for future improvement.
4. The answer is **always returned** to the UI and displayed.
5. If correct → shown as-is. If incorrect → shown with a ⚠️ warning badge, plus a 🧠 "Learned and updated knowledge base" notification so the user knows the system is improving for next time.

---

### Q5: How many LLM calls happen per question?

**A:** At minimum **5 calls**, at maximum **6 calls**:

| Call # | Purpose | Always runs? |
|---|---|---|
| 1-3 | HyDE — generate 3 hypothetical answers | Yes |
| 4 | RAG — generate the final answer | Yes |
| 5 | Critic — fact-check the answer | Yes |
| 6 | SEAL — generate correction (if critic says INCORRECT) | Only if wrong |

This is something to be aware of for cost and latency — each question takes 5-6 round trips to the LLM.

---

## Architecture & Design

---

### Q6: Why did you separate the frontend and backend?

**A:** Three reasons:

1. **Scalability** — The backend can run independently. You could swap the Streamlit UI for a mobile app or a different web frontend without touching any backend code.
2. **Async processing** — FastAPI handles requests asynchronously. Heavy LLM calls run in a thread pool so the server stays responsive to other requests.
3. **Deployment flexibility** — The backend can be deployed on a GPU server while the frontend runs anywhere.

---

### Q7: Why FastAPI instead of Flask or Django?

**A:** FastAPI gives us:
- **Async support** out of the box — critical because LLM calls are slow and we don't want to block the server.
- **Automatic API docs** — Swagger UI at `/docs` for free.
- **Pydantic integration** — Request/response validation is built in, matching our `models.py` setup.
- **Performance** — One of the fastest Python web frameworks thanks to Starlette and uvicorn.

---

### Q8: Why ChromaDB for the vector database?

**A:** ChromaDB was chosen because:
- **Persistent storage** — Data survives server restarts (important for the SEAL learning loop).
- **No external service** — Runs in-process, no need to set up a separate database server.
- **LangChain integration** — Works seamlessly with LangChain's retriever interface.
- **Good enough for this scale** — For a single-user or small-team tool, ChromaDB handles the load well. For production at scale, you'd consider Pinecone, Weaviate, or Qdrant.

---

### Q9: Explain the chunking strategy.

**A:** Documents are split using LangChain's `RecursiveCharacterTextSplitter`:
- **Chunk size**: 1000 characters
- **Overlap**: 150 characters

The overlap ensures that if an important sentence spans two chunks, it still appears in both. The splitter tries to break at natural boundaries (paragraphs → sentences → words) rather than cutting mid-word.

For short texts (under 1500 characters), no splitting is done — they're stored as-is.

---

### Q10: Why did you use HuggingFace embeddings instead of OpenAI embeddings?

**A:** The `all-MiniLM-L6-v2` model:
- **Runs locally** — No API calls, no cost, no latency.
- **Small** — Only ~80MB, loads fast.
- **Good quality** — Consistently ranks well on sentence similarity benchmarks.
- **Offline capable** — The system can work entirely offline with Ollama + local embeddings.

Using OpenAI embeddings would add API cost and dependency on an external service.

---

## HyDE & SEAL (The Core Ideas)

---

### Q11: Explain HyDE like I'm five.

**A:** Imagine you're looking for a book in a library. 

- **Normal way**: You tell the librarian "I want to know about dinosaurs." The librarian looks for books with the word "dinosaurs" in the title.
- **HyDE way**: You first think "A book about dinosaurs would probably talk about T-Rex, fossils, and the Jurassic period." Then you tell the librarian THAT. Now the librarian finds much better books because your description sounds like the actual content of the books you want.

That's what HyDE does — it generates a guess of what the answer looks like, then uses that guess to search, finding more relevant results.

---

### Q12: Explain SEAL like I'm five.

**A:** Imagine you're a student who answers a question on a test. 

- **Normal student**: Writes the answer, hands it in.
- **SEAL student**: Writes the answer, then asks a teacher "Is this right?" If the teacher says no, the student writes a corrected answer on a flashcard and keeps it. Next time a similar question comes up, the student checks the flashcard first.

That's SEAL — check your own work, and if it's wrong, create a correction and save it for future reference.

---

### Q13: What happens to the SEAL corrections over time?

**A:** Three types of data get stored back into ChromaDB:

1. **Improved chunks** — Better versions of the original context, tagged with `source: self_edit, type: improved_chunk`.
2. **Q&A pairs** — Explicit question-answer pairs, tagged with `type: qa_pair`.
3. **Edit directives** — Meta-notes about what was wrong, tagged with `type: edit_directives`.

Over time, the vector database accumulates these corrections alongside the original documents. Future retrieval queries will naturally surface the corrections because they're more semantically aligned with the questions that triggered them.

**Potential issue**: If the system keeps adding corrections without cleanup, the database grows and could introduce noise. A production system would need a strategy for merging or pruning old corrections.

---

### Q14: What if the critic makes a wrong judgment?

**A:** This is a real limitation:

- **False CORRECT** (critic says right, but it's wrong): The bad answer goes through. No correction happens.
- **False INCORRECT** (critic says wrong, but it's right): The system unnecessarily generates a "correction" and stores it in the DB, potentially introducing noise.

Mitigation strategies:
- Use a stronger model for the critic than for the answer generation.
- Add confidence thresholds instead of a binary CORRECT/INCORRECT.
- Include human feedback as an additional validation layer.

---

## Code-Level Questions

---

### Q15: Why do you use `run_in_executor` in `main.py`?

**A:** LangChain's LLM calls are **synchronous** (blocking). FastAPI is **asynchronous**. If we call a blocking function directly inside an `async def` endpoint, it blocks the entire event loop — no other requests can be served until the LLM responds (which can take 10-30 seconds).

`run_in_executor` moves the blocking call to a separate thread so the event loop stays free. The thread pool has 4 workers, so up to 4 queries can be processed concurrently.

---

### Q16: How does the LLM factory work?

**A:** It's a simple factory pattern:

```
  get_llm("ollama", "qwen3:1.7b")  → returns ChatOllama(...)
  get_llm("openai", "gpt-4o")      → returns ChatOpenAI(...)
  get_llm("google", "gemini-2.5-flash") → returns ChatGoogleGenerativeAI(...)
```

The function checks for API keys and raises clear errors if they're missing. This lets the rest of the code work with any LLM without knowing which provider it's using — they all conform to LangChain's `BaseLanguageModel` interface.

---

### Q17: How does the vector store handle metadata for SEAL corrections?

**A:** When SEAL stores corrections, it tags them with metadata:

```python
# Improved chunk
metadatas=[{"source": "self_edit", "original_question": question, "type": "improved_chunk"}]

# Q&A pairs
metadatas=[{"source": "self_edit", "type": "qa_pair"}]

# Edit directives
metadatas=[{"source": "self_edit", "type": "edit_directives"}]
```

This metadata lets you:
- Distinguish original documents from learned corrections.
- Trace which question triggered a correction.
- Filter by type if you need to audit or prune the corrections later.

---

### Q18: What is the `_prepare_texts` method doing in `vectorstore.py`?

**A:** It's a smart chunking function:

- If you pass a **single string**, it chunks it using the text splitter.
- If you pass a **list of strings**, it checks each one:
  - **Longer than 1500 characters** → split it into chunks.
  - **1500 characters or shorter** → keep it as-is (it's already small enough).

This prevents unnecessarily splitting short SEAL corrections that are already concise.

---

### Q19: Why does the HyDE step use a higher temperature?

**A:** Temperature controls how "creative" the LLM is. The HyDE step uses `temperature=0.7` (more creative) while the final answer uses `temperature=0.1` (more precise).

The reasoning: HyDE **wants** diverse guesses. If all 3 hypotheses are nearly identical, they'll retrieve the same documents. By using higher temperature, we get varied guesses that cast a wider net across the vector database, improving retrieval coverage.

---

## Deployment & Production

---

### Q20: How would you deploy this to production?

**A:** Several changes would be needed:

1. **Containerize** — Dockerfile for both backend and frontend, Docker Compose to orchestrate.
2. **Swap ChromaDB** — Use a managed vector DB (Pinecone, Weaviate) for reliability and scale.
3. **Add authentication** — The current CORS allows everything. Add API keys or OAuth.
4. **Rate limiting** — Each question makes 5-6 LLM calls. Without rate limiting, costs could explode.
5. **Monitoring** — Track query latency, LLM costs, correction frequency, and retrieval quality.
6. **Caching** — Cache repeated queries to avoid redundant LLM calls.
7. **Queue system** — For high traffic, use a task queue (Celery/Redis) instead of thread pool.

---

### Q21: What are the current limitations?

**A:**

| Limitation | Why it matters |
|---|---|
| No multi-user support | Everyone shares one ChromaDB instance — corrections from one user affect everyone |
| No authentication | Anyone can upload files and query |
| SEAL corrections can accumulate noise | No pruning or merging strategy for old corrections |
| 5-6 LLM calls per question | High latency (30-60s with local models) and cost (with cloud APIs) |
| No hybrid search | Only vector similarity search, no keyword/BM25 fallback |
| No structure-aware chunking | Tables, code blocks, and headers are split like plain text |
| Single collection | All documents go into one "docs" collection, no per-user or per-project isolation |

---

### Q22: How would you evaluate this system's performance?

**A:** Three dimensions:

1. **Retrieval quality** — Are the right chunks being retrieved? Metrics: Recall@k, MRR (Mean Reciprocal Rank). Build a test set of questions with known relevant passages.

2. **Answer quality** — Is the generated answer factually correct and faithful to the source? Metrics: Faithfulness (does the answer stick to the context?), Answer relevance (does it address the question?). Use frameworks like RAGAS or DeepEval.

3. **SEAL effectiveness** — Does the self-correction actually improve future answers? Test by asking the same question before and after a correction is stored. Measure whether the corrected version leads to better retrieval and answers.

---

### Q23: What's the difference between this and fine-tuning?

**A:** 

| Aspect | SEAL (Reasona's approach) | Fine-tuning |
|---|---|---|
| What changes | The retrieval database (new chunks added) | The model's weights |
| Speed | Instant (just add to ChromaDB) | Hours/days of training |
| Cost | Low (just a few extra LLM calls) | High (GPU compute) |
| Reversibility | Easy (delete the correction from DB) | Hard (retrain without the data) |
| Scope | Per-question corrections | Broad behavioral changes |
| Risk | Can add noise to the DB | Can cause catastrophic forgetting |

SEAL is more like "adding notes to a textbook" while fine-tuning is like "rewriting the textbook."

---

## Behavioral / Soft Questions

---

### Q24: What was the hardest part of building this?

**A:** Getting the **critic prompt** right. If the critic is too strict, it flags correct answers as wrong and the system generates unnecessary corrections (adding noise). If it's too lenient, it misses real errors and the self-correction loop never triggers. Finding the balance required iterating on the prompt and testing with diverse question types.

---

### Q25: If you had more time, what would you add first?

**A:** **Hybrid retrieval** — combining vector similarity search with keyword search (BM25). Vector search is great for semantic matching but can miss exact terms (like specific names, numbers, or acronyms). Adding keyword search as a fallback would significantly improve retrieval accuracy with minimal additional complexity.

---

### Q26: How did you decide on the tech stack?

**A:** Each choice was driven by a specific need:

- **FastAPI** over Flask → needed async support for non-blocking LLM calls.
- **Streamlit** over React → fastest path to a working UI for a Python-centric project.
- **ChromaDB** over Pinecone → wanted everything to run locally, no cloud dependency.
- **LangChain** → provides ready-made abstractions for prompts, chains, and vector store integration.
- **Ollama** → enables fully offline operation for privacy-sensitive use cases.

---

### Q27: How does error handling work?

**A:** Multiple layers:

1. **API level** (`main.py`): Try/catch around every endpoint. Upload failures return 400, query failures return 500.
2. **Engine level** (`rag_engine.py`): If the critic's JSON response can't be parsed, it falls back to a heuristic (if the answer contains "I don't have enough information" → treat as INCORRECT). If SEAL fails, it prints the error and returns the original answer without self-editing.
3. **File level** (`ingest.py`): Unsupported file types raise a clear `ValueError`. Temp files are always cleaned up in a `finally` block.

---

### Q28: Can this work completely offline?

**A:** Yes. Set `LLM_PROVIDER=ollama` in `.env`, have Ollama running locally with a downloaded model (e.g., `qwen3:1.7b`), and the entire system runs without any internet connection. The embeddings model (`all-MiniLM-L6-v2`) also runs locally after initial download.

---

### Q29: What would break if two users uploaded files at the same time?

**A:** ChromaDB's `PersistentClient` is thread-safe for basic operations, so concurrent uploads wouldn't crash. However:
- Both users' documents go into the same `"docs"` collection — there's no isolation.
- User A could ask a question and get results from User B's documents.
- SEAL corrections from one user's queries would affect retrieval for the other.

Fix: Use per-user or per-session collections, or add metadata filtering on retrieval.

---

### Q30: Walk me through a scenario where SEAL actually helps.

**A:** 

1. You upload a company handbook that says "Annual leave is 25 days per year."
2. You ask: "How many vacation days do I get?"
3. The retriever pulls a different chunk that says "Employees are entitled to statutory leave" (not the specific number).
4. The LLM generates: "You are entitled to statutory leave as per company policy." (vague, not grounded in the specific number).
5. The **critic** checks this against the context and says: `INCORRECT` — the answer doesn't mention the specific 25 days even though the document has this info.
6. The user **still sees** the answer, but with a ⚠️ warning badge and a 🧠 "Learned and updated knowledge base" notification.
7. Meanwhile, **SEAL** generates in the background:
   - `improved_chunk`: "Employees receive 25 days of annual leave per year as per company policy."
   - `qa_pair`: `{"question": "How many vacation days?", "answer": "25 days per year"}`
8. Both get stored in ChromaDB.
9. **Next time** someone asks about vacation days, the retriever finds the SEAL-generated Q&A pair and gives a better, specific answer: "You get 25 days of annual leave per year."
