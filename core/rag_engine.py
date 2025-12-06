# Flow:
# 1. Generates a hypothetical answer (HyDE) to improve retrieval.
# 2. Retrieves context from the vector store.
# 3. Produces an answer using RAG.
# 4. Validates answer accuracy with a critic.
# 5. Performs self-editing (SEAL) if the answer is incorrect and updates storage.

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .vectorstore import VectorStoreManager
from .llm_factory import get_llm
import json

class HyDE_SEAL_Engine:
    """Combines HyDE (Hypothetical Document Embeddings) with SEAL (Self-Edit And Learn)."""
    
    def __init__(self, vector_manager: VectorStoreManager):
        self.vector_manager = vector_manager
    
    def create_hyde_chain(self, llm):
        """Generate hypothetical answers to guide document retrieval"""
        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a helpful answer to this question. "
                      "This will be used to find relevant documents."),
            ("human", "Question: {question}")
        ])
        return hyde_prompt | llm | StrOutputParser()
    
    def create_rag_chain(self, llm):
        """Answer user questions using retrieved context"""
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using only the provided context. "
                      "If uncertain, say 'I don't have enough information'."),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
        return rag_prompt | llm | StrOutputParser()
    
    def create_critic_chain(self, llm):
        """Evaluate if the generated answer is fully supported by the context"""
        critic_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a fact-checker. Analyze if the answer is fully "
                      "supported by the context.\n\n"
                      "Question: {question}\n"
                      "Context: {context}\n"
                      "Answer: {answer}\n\n"
                      "Respond with only 'CORRECT' or 'INCORRECT'."),
            ("human", "Verify the answer.")
        ])
        return critic_prompt | llm | StrOutputParser()
    
    def create_self_edit_chain(self, llm):
        """Improve content and generate Q&A pairs if the answer is incorrect"""
        edit_prompt = ChatPromptTemplate.from_messages([
            ("system", "The provided answer was incorrect. Improve the context "
                      "and create Q&A pairs for better learning.\n\n"
                      "Original Context: {chunk}\n"
                      "Question: {question}\n"
                      "Wrong Answer: {wrong_answer}\n\n"
                      "Respond in JSON format:\n"
                      "{{\"improved_chunk\": \"...\", "
                      "\"qa_pairs\": [{{\"question\": \"...\", \"answer\": \"...\"}}]}}"),
            ("human", "Generate improved content.")
        ])
        return edit_prompt | llm | StrOutputParser()
    
    def process_query(self, question: str, provider: str, model: str):
        """Run the full RAG pipeline with HyDE and SEAL"""
        llm = get_llm(provider, model)
        retriever = self.vector_manager.get_retriever()
        
        # Step 1: HyDE - Generate hypothetical answer for better retrieval
        hyde_chain = self.create_hyde_chain(llm)
        hypothetical_answer = hyde_chain.invoke({"question": question})
        
        # Step 2: Retrieve documents based on hypothesis
        retrieved_docs = retriever.invoke(hypothetical_answer)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Step 3: Generate final answer using RAG
        rag_chain = self.create_rag_chain(llm)
        answer = rag_chain.invoke({"context": context, "question": question})
        
        # Step 4: Critic evaluation
        critic_chain = self.create_critic_chain(llm)
        critique = critic_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        }).strip().upper()
        
        result = {
            "answer": answer,
            "retrieved_docs": [doc.page_content for doc in retrieved_docs],
            "was_corrected": "INCORRECT" in critique,
            "self_edit_performed": False
        }
        
        # Step 5: Self-edit (SEAL) if needed
        if "INCORRECT" in critique:
            self_edit_chain = self.create_self_edit_chain(llm)
            edit_result = self_edit_chain.invoke({
                "chunk": context,
                "question": question,
                "wrong_answer": answer
            })
            
            try:
                edit_data = json.loads(edit_result) if isinstance(edit_result, str) else edit_result
                
                # Add improved context
                self.vector_manager.add_documents(
                    texts=[edit_data["improved_chunk"]],
                    metadatas=[{"source": "self_edit", "original_question": question}]
                )
                
                # Add Q&A pairs for future retrieval
                qa_texts = [f"Q: {qa['question']}\nA: {qa['answer']}" 
                            for qa in edit_data.get("qa_pairs", [])]
                if qa_texts:
                    self.vector_manager.add_documents(
                        texts=qa_texts,
                        metadatas=[{"type": "qa_pair"}] * len(qa_texts)
                    )
                
                result["self_edit_performed"] = True
            
            except Exception as e:
                print(f"Self-edit failed: {e}")
        
        return result
