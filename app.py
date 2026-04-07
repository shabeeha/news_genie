"""
app.py — The Streamlit frontend for NewsGenie.

This file only handles the UI. All the AI logic lives in workflow.py and agents.py.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from workflow import build_graph

load_dotenv()


# ── State flow renderer ───────────────────────────────────────────────────────

def _render_state_flow(state_log: list[dict]):
    """Displays the agent flow as a visual step-by-step diagram."""
    if not state_log:
        return

    STATUS_COLOR = {
        "success":  ("#d4edda", "#155724", "✅"),
        "fallback": ("#fff3cd", "#856404", "⚠️"),
        "error":    ("#f8d7da", "#721c24", "❌"),
    }

    with st.expander("🔍 View agent state flow", expanded=False):
        cols = st.columns(len(state_log))
        for i, entry in enumerate(state_log):
            status                  = entry.get("status", "success")
            bg, fg, icon            = STATUS_COLOR.get(status, ("#e2e3e5", "#383d41", "ℹ️"))
            agent                   = entry.get("agent", "unknown")
            detail                  = entry.get("detail", "")

            with cols[i]:
                st.markdown(
                    f"""<div style="background:{bg}; color:{fg}; padding:12px;
                        border-radius:10px; text-align:center; font-size:0.85rem;">
                        <div style="font-size:1.4rem">{icon}</div>
                        <div style="font-weight:700; margin:6px 0">{agent}</div>
                        <div style="font-size:0.75rem">{detail}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                # Draw arrow between steps (except after last)
                if i < len(state_log) - 1:
                    st.markdown(
                        "<div style='text-align:center; font-size:1.2rem; margin-top:8px'>→</div>",
                        unsafe_allow_html=True
                    )



st.set_page_config(page_title="NewsGenie", page_icon="🗞️", layout="wide")

st.title("🗞️ NewsGenie")
st.caption("Your AI-powered news and information assistant")
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("🔑 API Keys")

    openai_key  = st.text_input("OpenAI API Key",  type="password", placeholder="sk-...")
    newsapi_key = st.text_input("NewsAPI Key",      type="password", placeholder="Your NewsAPI key")
    serpapi_key = st.text_input("SerpAPI Key",      type="password", placeholder="Your SerpAPI key")

    if st.button("💾 Save & Apply Keys"):
        st.session_state.openai_key  = openai_key
        st.session_state.newsapi_key = newsapi_key
        st.session_state.serpapi_key = serpapi_key
        st.session_state.graph = build_graph(openai_key, newsapi_key, serpapi_key)
        st.success("Keys saved!")

    st.divider()

    st.subheader("📰 Quick News")
    category = st.selectbox("Pick a category", [
        "technology", "finance", "sports", "health", "science", "entertainment"
    ])
    if st.button("Fetch News"):
        st.session_state.quick_query = f"Latest {category} news"

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ── Session state setup ───────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "graph" not in st.session_state:
    st.session_state.graph = build_graph(
        openai_key  = os.getenv("OPENAI_API_KEY", ""),
        newsapi_key = os.getenv("NEWSAPI_KEY", ""),
        serpapi_key = os.getenv("SERPAPI_KEY", ""),
    )


# ── Display chat history ──────────────────────────────────────────────────────

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # Show the state flow diagram below each assistant message
    if message["role"] == "assistant" and "state_log" in message:
        _render_state_flow(message["state_log"])


# ── Handle quick-news button ──────────────────────────────────────────────────

if "quick_query" in st.session_state:
    user_input = st.session_state.pop("quick_query")
else:
    user_input = st.chat_input("Ask me anything or request news…")



# ── Process user input ────────────────────────────────────────────────────────

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = st.session_state.graph.invoke({
                    "user_query":     user_input,
                    "query_type":     "",
                    "news_category":  "",
                    "raw_news":       [],
                    "search_results": "",
                    "final_answer":   "",
                    "verified_news":  [],
                    "flagged_news":   [],
                    "state_log":      [],
                })
                response  = result["final_answer"]
                state_log = result.get("state_log", [])
            except Exception as e:
                response  = f"⚠️ Something went wrong: {e}\n\nPlease check your API keys in the sidebar."
                state_log = []

        st.markdown(response)
        _render_state_flow(state_log)

    st.session_state.chat_history.append({
        "role":      "assistant",
        "content":   response,
        "state_log": state_log,
    })