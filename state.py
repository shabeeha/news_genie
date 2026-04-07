from typing import TypedDict


class NewsGenieState(TypedDict):
    """
    This is the shared 'sticky note' passed between every agent.
    Each agent reads from it and writes its result back into it.
    """

    # The user's original message
    user_query: str

    # classify_agent fills these two
    query_type: str        # "news" or "general"
    news_category: str     # e.g. "technology", "sports", "finance"

    # news_agent or general_agent fills these
    raw_news: list[dict]   # list of articles from NewsAPI
    search_results: str    # web search results from SerpAPI

    # factcheck_agent fills these
    verified_news: list[dict]   # articles that passed the fact check
    flagged_news: list[dict]    # articles that were flagged as unreliable

    # format_agent fills this — the final reply shown to the user
    final_answer: str

    # each agent appends a log entry here so we can visualize the flow
    state_log: list[dict]