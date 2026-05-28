# 🧶 Crochet Companion — AI Pattern Assistant

A RAG-powered crochet assistant using LangChain, FAISS, Gemini Flash, and Flask.

---

## 📁 Project Structure

```
crochet_app/
├── app.py              ← Flask backend (RAG pipeline)
├── patterns_data.py    ← Crochet dataset (40 patterns)
├── requirements.txt    ← Python dependencies
├── index.html          ← Frontend UI
├── .env.example        ← API key template
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install Python dependencies

```bash
cd crochet_app
pip install -r requirements.txt
```

### 2. Set your Gemini API key

**Option A — .env file (recommended):**
```bash
cp .env.example .env
# Edit .env and paste your API key:
# GOOGLE_API_KEY=your_key_here
```

**Option B — Edit app.py directly:**
Find this line and replace the placeholder:
```python
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 3. Run the Flask backend

```bash
python app.py
```

On first run, this will:
- Build FAISS embeddings for all 40 patterns (~30 seconds)
- Save the index to `faiss_index/` for fast future loads
- Start the server at `http://localhost:5000`

### 4. Open the frontend

Open `index.html` in your browser — or serve it:

```bash
# Simple HTTP server (Python)
python -m http.server 8080
# Then visit: http://localhost:8080
```

> ⚠️ The frontend connects to `http://localhost:5000` by default.
> Change `API_BASE` in `index.html` if your Flask runs elsewhere.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[LangChain] — Query Embedding (Gemini embedding-001)
    │
    ▼
[FAISS] — Semantic similarity search → Top 4 patterns
    │
    ▼
[Gemini Flash] — Generate response with context + chat history
    │
    ▼
[Flask API] — JSON response → Frontend display
```

### Key Components

| Component | Role |
|---|---|
| `GoogleGenerativeAIEmbeddings` | Converts patterns + queries to vectors |
| `FAISS` | Vector database for semantic search |
| `ChatGoogleGenerativeAI` | Gemini Flash LLM for response generation |
| `ConversationalRetrievalChain` | Orchestrates multi-turn RAG |
| `ConversationBufferMemory` | Maintains chat history per session |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat` | Send a message, get a recommendation |
| POST | `/api/reset` | Clear conversation memory |
| GET | `/api/patterns` | List all 40 patterns |
| GET | `/api/health` | Health check |

### Example request:
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want a quick beginner plushie for gifting", "session_id": "user1"}'
```

---

## ✨ Features

- **RAG Architecture**: Semantic search over 40 real crochet patterns
- **Multi-turn Conversations**: Remembers context across messages
- **Granny Persona**: Warm, encouraging responses from "Granny"
- **First-time Modal**: Welcoming modal for new users
- **Quick Chips**: Suggested prompts for easy onboarding
- **Pattern Cards**: Visual display of retrieved patterns with metadata
- **Cute UI**: Pastel, handmade aesthetic with micro-animations

---

## 🔧 Customisation

- **Add patterns**: Edit `patterns_data.py` and delete `faiss_index/` to rebuild
- **Change persona**: Edit `SYSTEM_PROMPT` in `app.py`
- **Backend URL**: Change `API_BASE` in `index.html`
- **Model**: Change `model="gemini-1.5-flash"` in `app.py` for other Gemini models
