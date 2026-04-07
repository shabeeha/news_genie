"""
workflow.py — Connects all agents into a LangGraph flowchart.

This file only handles WIRING — which agent runs first, which runs next,
and how to decide which branch to take. The actual work is in agents.py.
"""

from functools import partial
from typing import Literal

from langgraph.graph import StateGraph, START, END

from state import NewsGenieState
from agents import classify_agent, news_agent, general_agent, factcheck_agent, format_agent, get_llm


# ── Routing function ──────────────────────────────────────────────────────────
# After classify_agent runs, this decides which path to take next.

def route_query(state: NewsGenieState) -> Literal["news_agent", "general_agent"]:
    if state.get("query_type") == "news":
        return "news_agent"
    return "general_agent"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(openai_key: str = "", newsapi_key: str = "", serpapi_key: str = ""):
    """
    Builds and returns the compiled LangGraph.

    Step 1 — Create the graph with our state definition
    Step 2 — Add each agent as a node
    Step 3 — Connect the nodes with edges
    Step 4 — Compile and return
    """

    llm = get_llm(openai_key)

    # ── Step 1: Create graph ──────────────────────────────────────────────
    graph = StateGraph(NewsGenieState)

    # ── Step 2: Add nodes ─────────────────────────────────────────────────
    graph.add_node("classify_agent",  partial(classify_agent,  llm=llm))
    graph.add_node("news_agent",      partial(news_agent,      newsapi_key=newsapi_key, serpapi_key=serpapi_key))
    graph.add_node("general_agent",   partial(general_agent,   serpapi_key=serpapi_key))
    graph.add_node("factcheck_agent", partial(factcheck_agent, llm=llm))
    graph.add_node("format_agent",    partial(format_agent,    llm=llm))

    # ── Step 3: Connect nodes with edges ──────────────────────────────────

    # Always start at classify_agent
    graph.add_edge(START, "classify_agent")

    # After classify_agent, branch based on query type
    graph.add_conditional_edges(
        "classify_agent",
        route_query,
        {
            "news_agent":    "news_agent",
            "general_agent": "general_agent",
        }
    )

    # Both branches go through factcheck_agent before formatting
    graph.add_edge("news_agent",      "factcheck_agent")
    graph.add_edge("general_agent",   "factcheck_agent")

    # factcheck_agent always leads to format_agent
    graph.add_edge("factcheck_agent", "format_agent")

    # format_agent is the last step
    graph.add_edge("format_agent", END)

    # ── Step 4: Compile ───────────────────────────────────────────────────
    return graph.compile()