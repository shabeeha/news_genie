"""
agents.py — Every agent (node) in NewsGenie lives here.

Each agent is a plain Python function that:
  1. Receives the current state (the shared sticky note)
  2. Does its job
  3. Returns a dict with the fields it wants to update
"""

import os
import json
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from state import NewsGenieState


# ── Shared LLM instance ───────────────────────────────────────────────────────

def get_llm(api_key: str = "") -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=api_key or os.getenv("OPENAI_API_KEY", "placeholder"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — Classify Agent
# Job: Read the user's message and decide if it's a news request or a general
#      question. Also figure out which news category it belongs to.
# ─────────────────────────────────────────────────────────────────────────────

def classify_agent(state: NewsGenieState, llm: ChatOpenAI) -> dict:
    log = list(state.get("state_log", []))

    system = SystemMessage(content="""You are a query classifier.
Given a user message, reply ONLY with a JSON object like this:
{"type": "news", "category": "technology"}

Rules:
- "type" must be "news" if the user wants headlines/articles/updates, else "general"
- "category" must be one of: technology, finance, sports, health, science, entertainment, general
- Reply with ONLY the JSON. No extra text.""")

    user_msg = HumanMessage(content=state["user_query"])

    try:
        response      = llm.invoke([system, user_msg])
        parsed        = json.loads(response.content.strip())
        query_type    = parsed.get("type", "general")
        news_category = parsed.get("category", "general")
        log.append({
            "agent":  "classify_agent",
            "status": "success",
            "detail": f"Detected type: {query_type} | category: {news_category}"
        })

    except Exception:
        # ── Keyword fallback ──────────────────────────────────────────────
        query      = state["user_query"].lower()
        news_words = ["news", "headline", "latest", "update", "article", "today"]
        query_type = "news" if any(w in query for w in news_words) else "general"

        category_map = {
            "tech":      "technology",
            "finance":   "finance",
            "business":  "finance",
            "sport":     "sports",
            "health":    "health",
            "science":   "science",
            "entertain": "entertainment",
        }
        news_category = next(
            (v for k, v in category_map.items() if k in query), "general"
        )
        log.append({
            "agent":  "classify_agent",
            "status": "fallback",
            "detail": f"LLM parse failed — used keywords | type: {query_type} | category: {news_category}"
        })

    return {
        "query_type":    query_type,
        "news_category": news_category,
        "state_log":     log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — News Agent
# Job: Fetch real news articles for the detected category.
#      Falls back to SerpAPI web search if NewsAPI returns nothing.
# ─────────────────────────────────────────────────────────────────────────────

def news_agent(state: NewsGenieState, newsapi_key: str, serpapi_key: str) -> dict:
    log            = list(state.get("state_log", []))
    articles       = _call_newsapi(state["user_query"], state["news_category"], newsapi_key)
    search_results = ""

    if not articles or "error" in articles[0]:
        print("⚠️  NewsAPI returned nothing — falling back to SerpAPI")
        search_results = _call_serpapi(state["user_query"], serpapi_key)
        articles = []
        log.append({
            "agent":  "news_agent",
            "status": "fallback",
            "detail": "NewsAPI returned nothing — used SerpAPI"
        })
    else:
        print(f"✅ NewsAPI returned {len(articles)} articles")
        log.append({
            "agent":  "news_agent",
            "status": "success",
            "detail": f"NewsAPI returned {len(articles)} articles"
        })

    return {
        "raw_news":       articles,
        "search_results": search_results,
        "state_log":      log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — General Agent
# Job: For non-news questions, run a web search and store the results.
# ─────────────────────────────────────────────────────────────────────────────

def general_agent(state: NewsGenieState, serpapi_key: str) -> dict:
    log            = list(state.get("state_log", []))
    search_results = _call_serpapi(state["user_query"], serpapi_key)

    log.append({
        "agent":  "general_agent",
        "status": "success",
        "detail": "Ran SerpAPI web search"
    })

    return {
        "raw_news":       [],
        "search_results": search_results,
        "state_log":      log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — Fact Check Agent
# Job: Review each article and decide if it's reliable or should be flagged.
#      Uses the LLM to assess credibility based on source, headline, and content.
#      Only runs for news queries — skipped for general questions.
# ─────────────────────────────────────────────────────────────────────────────

def factcheck_agent(state: NewsGenieState, llm: ChatOpenAI) -> dict:
    log           = list(state.get("state_log", []))
    articles      = state.get("raw_news", [])
    search_result = state.get("search_results", "")

    # If there are no articles (e.g. SerpAPI was used), fact check the search results as a block
    if not articles and not search_result:
        log.append({
            "agent":  "factcheck_agent",
            "status": "fallback",
            "detail": "No content to fact check"
        })
        return {
            "verified_news": [],
            "flagged_news":  [],
            "state_log":     log,
        }

    # ── Case 1: We have structured articles from NewsAPI ─────────────────
    if articles:
        verified = []
        flagged  = []

        system = SystemMessage(content="""You are a news credibility checker.
Given a news article's title, source, and description, assess its reliability.
Reply ONLY with a JSON object:
{"verdict": "reliable", "reason": "one sentence explanation"}
or
{"verdict": "flagged", "reason": "one sentence explanation"}

Flag an article if:
- The source is unknown or not a recognised news outlet
- The headline is exaggerated, sensationalist, or uses clickbait language
- The description contains unsupported claims or contradicts itself
- The content seems like opinion presented as fact

Reply with ONLY the JSON. No extra text.""")

        for article in articles:
            title  = article.get("title", "")
            source = article.get("source", {}).get("name", "Unknown")
            desc   = article.get("description", "")

            user_msg = HumanMessage(content=(
                f"Title: {title}\n"
                f"Source: {source}\n"
                f"Description: {desc}"
            ))

            try:
                response = llm.invoke([system, user_msg])
                parsed   = json.loads(response.content.strip())
                verdict  = parsed.get("verdict", "reliable")
                reason   = parsed.get("reason", "")

                if verdict == "flagged":
                    flagged.append({**article, "flag_reason": reason})
                else:
                    verified.append({**article, "verified_reason": reason})

            except Exception:
                # If LLM fails to assess, pass the article through as reliable
                verified.append(article)

        passed  = len(verified)
        removed = len(flagged)
        log.append({
            "agent":  "factcheck_agent",
            "status": "success" if removed == 0 else "fallback",
            "detail": f"{passed} verified ✔ | {removed} flagged ✘"
        })

        return {
            "verified_news": verified,
            "flagged_news":  flagged,
            "state_log":     log,
        }

    # ── Case 2: We have SerpAPI search results (a single string) ─────────
    # Ask the LLM to assess the overall reliability of the search results
    system = SystemMessage(content="""You are a news credibility checker.
Given a set of web search results, assess their overall reliability.
Reply ONLY with a JSON object:
{"verdict": "reliable", "reason": "one sentence explanation"}
or
{"verdict": "flagged", "reason": "one sentence explanation"}

Flag if results appear to come from unreliable, biased, or sensationalist sources.
Reply with ONLY the JSON. No extra text.""")

    user_msg = HumanMessage(content=f"Search results:\n{search_result}")

    try:
        response = llm.invoke([system, user_msg])
        parsed   = json.loads(response.content.strip())
        verdict  = parsed.get("verdict", "reliable")
        reason   = parsed.get("reason", "")

        log.append({
            "agent":  "factcheck_agent",
            "status": "success" if verdict == "reliable" else "fallback",
            "detail": f"Search results: {verdict} — {reason}"
        })
    except Exception:
        log.append({
            "agent":  "factcheck_agent",
            "status": "fallback",
            "detail": "Could not assess search results — passed through"
        })

    return {
        "verified_news": [],
        "flagged_news":  [],
        "state_log":     log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5 — Format Agent
# Job: Take verified content and write a clean reply.
#      Also mentions any flagged articles at the bottom as a warning.
# ─────────────────────────────────────────────────────────────────────────────

def format_agent(state: NewsGenieState, llm: ChatOpenAI) -> dict:
    log = list(state.get("state_log", []))

    system = SystemMessage(content="""You are NewsGenie, a helpful news and information assistant.
Use the provided context to write a clear, well-formatted markdown response.
- For news: summarise the top stories, mention sources, highlight key facts.
- For general questions: answer concisely using the search results.
- Be factual and objective. Use bullet points and bold text for clarity.
- Only use the verified articles provided. Do not include flagged content.""")

    context_parts = []

    # Use verified_news if available, otherwise fall back to raw_news
    verified = state.get("verified_news", [])
    raw      = state.get("raw_news", [])
    articles = verified if verified else raw

    if articles:
        formatted = _format_articles(articles)
        context_parts.append(f"Verified News Articles:\n{formatted}")

    search = state.get("search_results", "")
    if search:
        context_parts.append(f"Web Search Results:\n{search}")

    context = "\n\n".join(context_parts) if context_parts else "No external data available."

    user_msg = HumanMessage(content=(
        f"User asked: {state['user_query']}\n\n"
        f"Context:\n{context}\n\n"
        f"Write a helpful response."
    ))

    try:
        response     = llm.invoke([system, user_msg])
        final_answer = response.content

        # Append flagged articles as a warning at the bottom
        flagged = state.get("flagged_news", [])
        if flagged:
            final_answer += "\n\n---\n⚠️ **The following articles were flagged as potentially unreliable and excluded:**\n"
            for a in flagged:
                title  = a.get("title", "Unknown")
                reason = a.get("flag_reason", "")
                final_answer += f"\n- ~~{title}~~ — *{reason}*"

        log.append({
            "agent":  "format_agent",
            "status": "success",
            "detail": "LLM composed the final answer"
        })
    except Exception as e:
        final_answer = f"⚠️ Could not generate a response: {e}\n\n**Raw results:**\n{context}"
        log.append({
            "agent":  "format_agent",
            "status": "error",
            "detail": f"LLM failed: {e}"
        })

    return {
        "final_answer": final_answer,
        "state_log":    log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _call_newsapi(query: str, category: str, api_key: str) -> list[dict]:
    if not api_key:
        return [{"error": "No NewsAPI key provided"}]
    try:
        if category and category != "general":
            url    = "https://newsapi.org/v2/top-headlines"
            params = {"apiKey": api_key, "category": category, "country": "us", "pageSize": 5}
        else:
            url    = "https://newsapi.org/v2/everything"
            params = {"apiKey": api_key, "q": query, "pageSize": 5, "language": "en"}

        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except Exception as e:
        return [{"error": str(e)}]


def _call_serpapi(query: str, api_key: str) -> str:
    if not api_key:
        return ""
    try:
        params  = {"q": query, "api_key": api_key, "num": 5, "hl": "en"}
        resp    = requests.get("https://serpapi.com/search", params=params, timeout=8)
        resp.raise_for_status()
        results = resp.json().get("organic_results", [])
        lines   = [
            f"- **{r.get('title', 'No title')}**: {r.get('snippet', '')[:200]}... ([source]({r.get('link', '#')}))"
            for r in results
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def _format_articles(articles: list[dict]) -> str:
    lines = []
    for a in articles[:5]:
        title  = a.get("title", "No title")
        source = a.get("source", {}).get("name", "Unknown")
        desc   = a.get("description", "")
        url    = a.get("url", "#")
        lines.append(f"- **{title}** ({source}): {desc} [Read more]({url})")
    return "\n".join(lines)