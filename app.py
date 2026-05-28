"""
Crochet Companion - Flask Backend
RAG-powered crochet pattern recommendation system using LangChain, FAISS, and Gemini Flash.
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from patterns_data import CROCHET_PATTERNS

# ─── LangChain / Gemini / FAISS imports ──────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend

# ─── Configuration ────────────────────────────────────────────────────────────
# Set your Gemini API key here or via environment variable GOOGLE_API_KEY
GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")

FAISS_INDEX_PATH = "faiss_index"

# ─── Build Documents for FAISS ────────────────────────────────────────────────
def build_documents():
    """Convert pattern dicts into LangChain Document objects."""
    docs = []
    for p in CROCHET_PATTERNS:
        content = (
            f"Pattern Name: {p['name']}\n"
            f"Skill Level: {p['skill_level']}\n"
            f"Item Type: {p['item_type']}\n"
            f"Yarn Type: {p['yarn_type']}\n"
            f"Yarn Weight: {p['yarn_weight']}\n"
            f"Season: {p['season']}\n"
            f"Difficulty: {p['difficulty']}\n"
            f"Estimated Time: {p['estimated_time']}\n"
            f"Description: {p['description']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "id": p["id"],
                "name": p["name"],
                "skill_level": p["skill_level"],
                "item_type": p["item_type"],
                "difficulty": p["difficulty"],
                "estimated_time": p["estimated_time"],
                "season": p["season"],
            }
        ))
    return docs


# ─── Initialize Embeddings + Vector Store ─────────────────────────────────────
def init_vector_store():
    """Create or load the FAISS vector store."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        print("Building FAISS index from scratch...")
        docs = build_documents()
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"FAISS index saved to '{FAISS_INDEX_PATH}'")
    
    return vectorstore


# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Granny, a warm, wise, and lovable crochet companion. You speak with gentle affection, 
calling users "my child" or "dear" occasionally. Your job is to recommend crochet patterns from the provided context 
based on the user's needs, skill level, and preferences.

When recommending patterns:
1. Always explain WHY a pattern suits the user's request
2. Mention the estimated time and difficulty
3. Give 2-3 recommendations when possible
4. Be encouraging and supportive — crochet can be intimidating for beginners!
5. If the user asks about something not in the patterns, gently redirect to what you have

Format your response naturally and warmly. When listing patterns, present them clearly.
Use this format for each recommended pattern:

✨ **[Pattern Name]**
⏱ Time: [estimated time]
🧶 Level: [skill level] | Difficulty: [difficulty]
💬 Why this suits you: [your reasoning]

Context from pattern database:
{context}

Chat History:
{chat_history}

User Question: {question}

Granny's Response:"""

CONDENSE_QUESTION_PROMPT = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question about crochet patterns.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""


# ─── Global RAG Chain ─────────────────────────────────────────────────────────
# Stores one memory object per session_id
session_memories = {}

vectorstore = None
llm = None

def get_rag_chain(session_id: str):
    """Get or create a RAG chain with conversation memory for a session."""
    global vectorstore, llm
    
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    memory = session_memories[session_id]
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    qa_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT
    )
    
    condense_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=CONDENSE_QUESTION_PROMPT
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_prompt,
        return_source_documents=True,
        verbose=False
    )
    
    return chain


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle a chat message and return Granny's recommendation."""
    data = request.get_json()
    
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_message = data["message"].strip()
    session_id = data.get("session_id", "default")
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    try:
        chain = get_rag_chain(session_id)
        result = chain({"question": user_message})
        
        answer = result.get("answer", "I'm sorry, dear, I couldn't find a good answer for that. 🧶")
        
        # Extract source pattern names for frontend display
        source_docs = result.get("source_documents", [])
        sources = []
        seen = set()
        for doc in source_docs:
            name = doc.metadata.get("name", "")
            if name and name not in seen:
                seen.add(name)
                sources.append({
                    "name": name,
                    "skill_level": doc.metadata.get("skill_level", ""),
                    "estimated_time": doc.metadata.get("estimated_time", ""),
                    "difficulty": doc.metadata.get("difficulty", ""),
                    "season": doc.metadata.get("season", ""),
                })
        
        return jsonify({
            "response": answer,
            "sources": sources,
            "session_id": session_id
        })
    
    except Exception as e:
        print(f"Error in /api/chat: {e}")
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500


@app.route("/api/reset", methods=["POST"])
def reset_session():
    """Clear conversation memory for a session."""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    
    if session_id in session_memories:
        del session_memories[session_id]
    
    return jsonify({"message": "Session reset successfully", "session_id": session_id})


@app.route("/api/patterns", methods=["GET"])
def get_patterns():
    """Return all available patterns (for debugging/preview)."""
    return jsonify({"patterns": CROCHET_PATTERNS, "count": len(CROCHET_PATTERNS)})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Granny is ready to help! 🧶"})


# ─── App Init ─────────────────────────────────────────────────────────────────
def initialize_app():
    global vectorstore, llm
    print("Initializing Crochet Companion backend...")
    
    vectorstore = init_vector_store()
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    print("✅ Backend ready! Granny is here to help.")


if __name__ == "__main__":
    initialize_app()
    app.run(debug=True, host="0.0.0.0", port=5000)