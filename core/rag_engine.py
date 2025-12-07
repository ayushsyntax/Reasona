# Flow:
# 1. Generates multiple hypothetical answers (HyDE*) to drive robust retrieval.
# 2. Retrieves context using both the user query and hypotheses.
# 3. Produces a concise, fact-rich answer from aggregated context.
# 4. Uses a stricter critic to judge faithfulness and coherence.
# 5. If incorrect, performs SEAL-style self-edit: improved chunk, QA pairs, and edit directives; persists for future learning

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .vectorstore import VectorStoreManager
from .llm_factory import get_llm
import json

class HyDE_SEAL_Engine:
    def __init__(self, vector_manager: VectorStoreManager, hyde_k: int = 3, top_k: int = 4):
        self.vector_manager = vector_manager
        self.hyde_k = hyde_k
        self.top_k = top_k

    def _hyde_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", "Produce a helpful, self-contained answer to guide retrieval. No disclaimers."),
            ("human", "Question: {question}")
        ])

    def _rag_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a factual, concise assistant. Use only the context to answer naturally.\n"
             "Include short facts (figures/names) when helpful. If unsure, say \"I don't have enough information.\""),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nRespond in 3–5 sentences.")
        ])

    def _critic_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a strict fact-checker. Judge if the Answer is fully supported by the Context.\n"
             "Return JSON with fields: verdict ('CORRECT' or 'INCORRECT'), rationale (one sentence)."),
            ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}")
        ])

    def _seal_edit_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system",
             "Generate a self-edit to improve future answers.\n"
             "Return JSON with keys:\n"
             "  improved_chunk: revised, compact factual passage\n"
             "  qa_pairs: list of {\"question\":\"...\",\"answer\":\"...\"}\n"
             "  edit_directives: {\"augmentation\":[],\"notes\":\"...\"}\n"),
            ("human",
             "Original Context:\n{chunk}\n\nQuestion:\n{question}\n\nWrong Answer:\n{wrong_answer}\n\n"
             "Create improved_chunk that directly supports the correct answer, 3–6 sentences.")
        ])

    def _run_chain(self, prompt, llm, **kwargs):
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(kwargs)

    def _generate_hypotheses(self, llm, question: str):
        hyde_llm = get_llm("ollama" if hasattr(llm, "model") else "openai", getattr(llm, "model", ""), temperature=0.7)
        prompt = self._hyde_prompt()
        hypos = []
        for _ in range(self.hyde_k):
            hypos.append(self._run_chain(prompt, hyde_llm, question=question))
        return hypos

    def _retrieve(self, question: str, hypotheses: list):
        retriever = self.vector_manager.get_retriever()
        seeds = [question] + hypotheses
        docs = []
        for q in seeds:
            docs.extend(retriever.invoke(q))
        uniq = []
        seen = set()
        for d in docs:
            key = hash(d.page_content)
            if key not in seen:
                seen.add(key)
                uniq.append(d)
        return uniq[: self.top_k]

    def process_query(self, question: str, provider: str, model: str):
        llm = get_llm(provider, model)

        hypos = self._generate_hypotheses(llm, question)
        retrieved = self._retrieve(question, hypos)
        context = "\n\n".join(d.page_content for d in retrieved) if retrieved else ""

        answer = self._run_chain(self._rag_prompt(), llm, context=context, question=question)

        critic_raw = self._run_chain(self._critic_prompt(), llm, question=question, context=context, answer=answer)
        try:
            critic = json.loads(critic_raw) if isinstance(critic_raw, str) else critic_raw
            verdict = str(critic.get("verdict", "")).upper()
        except Exception:
            verdict = "CORRECT" if "I don't have enough information" not in answer else "INCORRECT"

        result = {
            "answer": answer,
            "retrieved_docs": [d.page_content for d in retrieved],
            "was_corrected": verdict == "INCORRECT",
            "self_edit_performed": False
        }

        if verdict == "INCORRECT" and context.strip():
            edit_raw = self._run_chain(
                self._seal_edit_prompt(),
                llm,
                chunk=context,
                question=question,
                wrong_answer=answer
            )
            try:
                edit = json.loads(edit_raw) if isinstance(edit_raw, str) else edit_raw
                improved_chunk = edit.get("improved_chunk", "").strip()
                qa_pairs = edit.get("qa_pairs", [])
                directives = edit.get("edit_directives", {})

                if improved_chunk:
                    self.vector_manager.add_documents(
                        texts=[improved_chunk],
                        metadatas=[{"source": "self_edit", "original_question": question, "type": "improved_chunk"}]
                    )

                if isinstance(qa_pairs, list) and qa_pairs:
                    qa_texts = [f"Q: {qa.get('question','')}\nA: {qa.get('answer','')}" for qa in qa_pairs if qa]
                    if qa_texts:
                        self.vector_manager.add_documents(
                            texts=qa_texts,
                            metadatas=[{"source": "self_edit", "type": "qa_pair"}] * len(qa_texts)
                        )

                if directives:
                    self.vector_manager.add_documents(
                        texts=[json.dumps(directives, ensure_ascii=False)],
                        metadatas=[{"source": "self_edit", "type": "edit_directives"}]
                    )

                result["self_edit_performed"] = True
            except Exception as e:
                print(f"Self-edit failed: {e}")

        return result
