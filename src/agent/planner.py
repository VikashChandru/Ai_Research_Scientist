"""Stage 1: turn the user's research question into a concrete plan."""
from src.grok_client import GrokClient

SYSTEM = """You are the planning module of an autonomous AI research scientist.
Given a research question (and optionally a summary of user-supplied documents),
break it into a small, executable research plan. Be concrete and scoped down
to what could actually be investigated with literature search + small
computational experiments (simulations, statistical tests, small models on
synthetic or provided data) -- not real-world lab work.

Return JSON with this exact shape:
{
  "restated_question": "...",
  "subquestions": ["...", "..."],
  "search_queries": ["...", "..."],
  "planned_experiment_types": ["...", "..."]
}
Keep subquestions to at most 4, search_queries to at most 5, all short and specific.
"""


def build_plan(grok: GrokClient, question: str, uploaded_summary: str = "") -> dict:
    user = f"Research question: {question}\n"
    if uploaded_summary:
        user += f"\nUser-provided document summary:\n{uploaded_summary}\n"
    return grok.chat_json(SYSTEM, user)