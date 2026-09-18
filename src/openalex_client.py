"""
Minimal OpenAlex client: keyword search + abstract reconstruction.
OpenAlex is free/open; supplying an email puts requests in the "polite pool"
(faster, more reliable). An optional premium API key is supported too.
"""
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    positions = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions.keys()))


class OpenAlexClient:
    def __init__(self, email: str = "", api_key: str = ""):
        self.email = email
        self.api_key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search(self, query: str, per_page: int = 8) -> list:
        params = {
            "search": query,
            "per-page": per_page,
            "sort": "relevance_score:desc",
        }
        if self.email:
            params["mailto"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        resp = requests.get(BASE_URL, params=params, timeout=20)
        resp.raise_for_status()
        results = resp.json().get("results", [])

        papers = []
        for r in results:
            abstract = _reconstruct_abstract(r.get("abstract_inverted_index"))
            primary_location = r.get("primary_location") or {}
            oa_pdf_url = None
            if primary_location.get("is_oa") and primary_location.get("pdf_url"):
                oa_pdf_url = primary_location.get("pdf_url")
            best_oa = r.get("best_oa_location") or {}
            if not oa_pdf_url and best_oa.get("pdf_url"):
                oa_pdf_url = best_oa.get("pdf_url")

            authors = [
                a.get("author", {}).get("display_name", "")
                for a in r.get("authorships", [])
            ]

            papers.append({
                "id": r.get("id"),
                "title": r.get("title") or "Untitled",
                "abstract": abstract,
                "year": r.get("publication_year"),
                "doi": r.get("doi"),
                "authors": [a for a in authors if a],
                "venue": (r.get("primary_location") or {}).get("source", {}).get("display_name")
                if (r.get("primary_location") or {}).get("source") else None,
                "cited_by_count": r.get("cited_by_count", 0),
                "oa_pdf_url": oa_pdf_url,
                "landing_page_url": (r.get("primary_location") or {}).get("landing_page_url"),
            })
        return papers