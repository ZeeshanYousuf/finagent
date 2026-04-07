# 💰 FinAgent — AI-Powered Personal Finance Assistant

A production-grade RAG (Retrieval-Augmented Generation) application that lets users have natural conversations about their personal finances using Claude AI.

Built as a portfolio project to demonstrate agentic AI development, RAG pipelines, and full-stack Python engineering.

## 🌐 Live Demo
👉 [https://finagent.zeeshanyousuf.io](https://finagent.zeeshanyousuf.io)

---

## 🎯 What It Does

- **Natural language financial queries** — Ask "How much did I spend on food?" and get intelligent, accurate answers
- **Multi-turn conversation memory** — Remembers context across questions in a session
- **Anomaly detection** — Automatically flags unusual transactions, duplicate charges, and spending patterns
- **Spending categorization** — Groups transactions into categories with a full monthly report
- **Multi-user sessions** — Each user gets isolated data with privacy controls
- **File upload** — Upload your own bank/credit card CSV statements
- **Data privacy** — Session-based storage with one-click data deletion

---

## 🏗️ Architecture

```
User Question
     ↓
FastAPI REST API
     ↓
ChromaDB Vector Search (semantic retrieval)
     ↓
Claude API (reasoning + answer generation)
     ↓
Structured Response with memory
```

**This is RAG (Retrieval-Augmented Generation):**
1. **Retrieval** — ChromaDB searches transactions by semantic meaning, not just keywords
2. **Augmented** — Retrieved transactions are attached to the prompt as context
3. **Generation** — Claude reasons over the data and generates accurate answers

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| API Framework | FastAPI |
| AI/LLM | Anthropic Claude (claude-sonnet-4-5) |
| Vector Database | ChromaDB |
| Data Processing | Pandas |
| Web Server | Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |

---

## 📁 Project Structure

```
finagent/
├── app/
│   ├── main.py          # FastAPI application + all endpoints
│   ├── agent.py         # RAG pipeline + conversation memory
│   ├── ingest.py        # CSV ingestion + ChromaDB storage
│   ├── anomaly.py       # Anomaly detection engine
│   ├── categories.py    # Spending categorization
│   ├── logger.py        # Centralized logging
│   └── static/
│       └── index.html   # Chat web UI
├── app/data/
│   ├── sample_bank.csv      # Sample bank statement
│   └── sample_credit.csv    # Sample credit card statement
├── logs/                # Application logs (auto-created)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Anthropic API key ([get one here](https://console.anthropic.com))

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/finagent.git
cd finagent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Add your Anthropic API key to .env
```

### Configuration

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_api_key_here
```

### Run the application

```bash
# Start the server
uvicorn app.main:app --reload

# Open in browser
open http://127.0.0.1:8000/chat
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/chat` | Chat web UI |
| POST | `/ask` | Ask a financial question |
| GET | `/insights` | Generate anomaly insights |
| GET | `/report` | Monthly spending report |
| GET | `/transactions/count` | Transaction count |
| POST | `/session/init` | Initialize session with sample data |
| DELETE | `/session/{id}` | Clear session data |
| POST | `/ingest/csv` | Upload CSV file |
| POST | `/ingest/reload` | Reload sample data |

---

## 💡 Key Technical Decisions

**Why ChromaDB?**
Semantic vector search finds transactions by meaning, not keywords. Searching "dining out" finds "Restaurant Dinner" and "Burger King" — exact keyword matching would miss these.

**Why session-based collections?**
Each user gets an isolated ChromaDB collection named transactions_{session_id}. User A cannot access User B's financial data.

**Why conversation history trimming?**
Sending full conversation history to Claude grows token usage linearly. We trim to the last 10 messages (5 exchanges) to control costs while maintaining context.

**Why FastAPI over Flask?**
Automatic request validation via Pydantic models, built-in OpenAPI docs at /docs, and async support out of the box.

---

## 🔒 Privacy

- All data is session-scoped and isolated per user
- No data persists after session deletion
- Sample data uses anonymized fictional transactions
- API keys are stored in environment variables, never in code

---

## 📊 Sample Questions to Try

- How much did I spend on food this month?
- What were my biggest expenses?
- Show me all my subscriptions
- What is my total income?
- Are there any unusual charges?
- How does my housing cost compare to my income?

---

## 🏦 Built For

This project was built to demonstrate:
- Production-grade RAG pipeline implementation
- Agentic AI development patterns
- Python full-stack development
- Financial data processing and analysis
- Multi-user session management

---

## .env.example

Create a `.env` file with the following:

```
ANTHROPIC_API_KEY=your_api_key_here
```

---
