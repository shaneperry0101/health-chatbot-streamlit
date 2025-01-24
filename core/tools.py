from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from duckduckgo_search import DDGS


TavilySearch = TavilySearchResults(max_results=5)


@tool
def VideoSearch(
    query: str,
    max_results=5
) -> str:
    """Use this to get videos."""

    try:
        results = DDGS().videos(
            keywords=query,
            max_results=max_results
        )
        videos = [i["embed_url"] for i in results]

        return str(videos)
    except Exception:
        return ""


@tool
def ImageSearch(
    query: str,
    max_results=5
) -> str:
    """Use this to get images."""

    try:
        results = DDGS().images(
            keywords=query,
            max_results=max_results
        )
        images = [{
            "title": i["title"],
            "thumbnail": i["thumbnail"]
        } for i in results]

        return str(images)
    except Exception:
        return ""
