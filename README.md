# 🗞️ NewsGenie - AI-Powered News & Information Assistant

NewsGenie is a multi-agent AI chatbot that fetches real-time news, filters misinformation, and answers general questions- all in one unified Streamlit interface.

Built with **LangGraph**, **NewsAPI**, **SerpAPI**, and **OpenAI**.

---

## 📸 Demo

> Ask NewsGenie anything:
> - *"Latest technology news"* → fetches headlines, fact-checks each one, summarises
> - *"What is quantitative easing?"* → web search + LLM answer
> - *"Top sports headlines today"* → categorised news with source verification

After every response, an **agent state flow diagram** shows exactly which agents ran and what decisions were made.

---

## 🏗️ Architecture

NewsGenie uses a **5-agent LangGraph pipeline**. Each agent has one job and passes a shared state object to the next.

```
User Query
    │
    ▼
classify_agent   - Is this news or a general question? What category?
    │
    ├── [news]    ──► news_agent      - Fetch from NewsAPI (falls back to SerpAPI)
    └── [general] ──► general_agent   - SerpAPI web search
                              │
                              ▼
                      factcheck_agent  - LLM checks every result individually
                              │
                              ▼
                       format_agent   - Composes the final answer
                              │
                              ▼
                        Streamlit UI
```

### Agent Responsibilities

| Agent | Job |
|---|---|
| `classify_agent` | Uses LLM to detect query type (`news` / `general`) and category. Falls back to keyword matching if LLM fails. |
| `news_agent` | Fetches top headlines from NewsAPI. Falls back to SerpAPI if NewsAPI returns nothing. |
| `general_agent` | Runs a SerpAPI web search for non-news queries. |
| `factcheck_agent` | Checks **every result individually** using the LLM — flags unreliable sources, clickbait headlines, or unsupported claims. |
| `format_agent` | Composes the final response using only verified content. Flagged results appear as warnings at the bottom. |

---

## 📁 File Structure

```
newsgenie/
├── app.py            # Streamlit UI - chat interface and state flow diagram
├── workflow.py       # LangGraph wiring - connects agents into a directed graph
├── agents.py         # All 5 agent functions + API helper utilities
├── state.py          # Shared state definition (TypedDict)
├── requirements.txt  # Python dependencies
└── .env              # API keys 
```

Each file has **one responsibility only** - the wiring, the logic, the UI, and the state are all kept separate.

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/newsgenie.git
cd newsgenie
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

Create a file called `.env` in the project root:

```
OPENAI_API_KEY=sk-...
NEWSAPI_KEY=your-newsapi-key
SERPAPI_KEY=your-serpapi-key
```

**Where to get the keys:**

| Key | Link | Free Tier |
|---|---|---|
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | Pay per use |
| NewsAPI | [newsapi.org](https://newsapi.org/register) | 100 requests/day |
| SerpAPI | [serpapi.com](https://serpapi.com) | 100 searches/month |

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔑 API Keys — Sidebar Option

You can also enter API keys directly in the app sidebar at runtime without a `.env` file. Click **"Save & Apply Keys"** after entering them.

---

## 🛡️ Fallback Mechanisms

The system degrades gracefully at every step — it never fully breaks.

| Failure | What happens |
|---|---|
| NewsAPI returns nothing | Automatically falls back to SerpAPI |
| NewsAPI key missing | Skips directly to SerpAPI |
| SerpAPI key missing | LLM answers from its own knowledge |
| LLM classify fails | Keyword-based fallback classifies the query |
| factcheck LLM fails for an item | Item is passed through as verified by default |
| format LLM fails | Raw context is shown directly with a warning |

---

## 📊 State Flow Diagram

After every response, click **"🔍 View agent state flow"** to see a colour-coded diagram of what happened:

- 🟢 **Green** — agent completed successfully
- 🟡 **Yellow** — agent used a fallback strategy
- 🔴 **Red** — agent encountered an error

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Multi-agent workflow orchestration |
| [LangChain](https://github.com/langchain-ai/langchain) | LLM interface and message types |
| [OpenAI GPT-4o-mini](https://openai.com) | Classification, fact-checking, response generation |
| [NewsAPI](https://newsapi.org) | Real-time news headlines |
| [SerpAPI](https://serpapi.com) | Web search results |
| [Streamlit](https://streamlit.io) | Frontend chat interface |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Environment variable management |

---

## 📝 Supported News Categories

`technology` · `finance` · `sports` · `health` · `science` · `entertainment`

---

## ⚠️ Known Limitations

- **NewsAPI free tier** only covers articles from the last 30 days and has limited category support. SerpAPI fallback handles this automatically.
- **Fact-checking** is LLM-based and not authoritative — it uses heuristics like source name, headline tone, and content quality. It is not a replacement for professional fact-checking.
- **No persistent memory** — conversation history is stored in the Streamlit session only and resets when the app restarts.

---

## 📄 License

MIT License — free to use, modify, and distribute.
