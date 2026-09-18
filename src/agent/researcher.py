"""Stage 2: search OpenAlex, read uploaded docs, populate vector store + KG."""
from src.grok_client import GrokClient
from src.openalex_client import OpenAlexClient
from src.neo4j_client import Neo4jClient
from src.embeddings import Embedder
from src.vector_store import VectorStore
from src.pdf_utils import chunk_text
from src.utils import ProvenanceLog

CONCEPT_SYSTEM = """You extract key scientific concepts/keywords from paper
abstracts for a knowledge graph. For each paper given, return 3-6 short
concept phrases (lowercase, 1-3 words each, no punctuation).

Return JSON: {"concepts_by_paper": {"<paper_id>": ["concept1", "concept2", ...], ...}}
"""


def search_and_ingest(
    grok: GrokClient,
    openalex: OpenAlexClient,
    neo4j: Neo4jClient,
    embedder: Embedder,
    store: VectorStore,
    search_queries: list,
    max_papers_per_query: int,
    provenance: ProvenanceLog,
    progress_cb=None,
) -> list:
    """Runs OpenAlex searches, embeds abstracts, writes Paper/Concept nodes.
    Returns the deduplicated list of paper dicts collected this run."""
    all_papers = {}

    for q in search_queries:
        if progress_cb:
            progress_cb(f"Searching OpenAlex for: {q}")
        try:
            results = openalex.search(q, per_page=max_papers_per_query)
        except Exception as e:
            provenance.log("research", "openalex_search_failed", {"query": q, "error": str(e)})
            continue
        provenance.log("research", "openalex_search", {"query": q, "num_results": len(results)})
        for p in results:
            if p["id"] and p["id"] not in all_papers:
                all_papers[p["id"]] = p

    papers = list(all_papers.values())

    # Embed + store abstracts for retrieval
    texts, metas = [], []
    for p in papers:
        if p.get("abstract"):
            texts.append(p["abstract"])
            metas.append({"type": "paper_abstract", "paper_id": p["id"], "title": p["title"]})
    if texts:
        vecs = embedder.embed(texts)
        store.add(vecs, texts, metas)

    # Write Paper nodes to the KG
    for p in papers:
        try:
            neo4j.upsert_paper({
                "id": p["id"], "title": p["title"], "year": p["year"],
                "doi": p["doi"], "venue": p["venue"], "authors": p["authors"],
                "source": "openalex",
            })
        except Exception as e:
            provenance.log("research", "kg_paper_write_failed", {"paper_id": p["id"], "error": str(e)})

    # Extract concepts in small batches via Grok, then link Paper-HAS_CONCEPT->Concept
    batch_size = 6
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        payload = "\n".join(
            f"- id: {p['id']} | title: {p['title']} | abstract: {(p['abstract'] or '')[:600]}"
            for p in batch
        )
        try:
            result = grok.chat_json(CONCEPT_SYSTEM, payload)
            concepts_by_paper = result.get("concepts_by_paper", {})
            for pid, concepts in concepts_by_paper.items():
                for c in concepts:
                    neo4j.link_paper_concept(pid, c)
            provenance.log("research", "concepts_extracted", {"num_papers": len(batch)})
        except Exception as e:
            provenance.log("research", "concept_extraction_failed", {"error": str(e)})

    return papers


def ingest_uploaded_documents(embedder: Embedder, store: VectorStore, documents: list, provenance: ProvenanceLog):
    """documents: list of {"filename": str, "text": str}"""
    texts, metas = [], []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            texts.append(chunk)
            metas.append({"type": "user_document", "filename": doc["filename"]})
    if texts:
        vecs = embedder.embed(texts)
        store.add(vecs, texts, metas)
    provenance.log("research", "uploaded_docs_ingested", {"num_docs": len(documents), "num_chunks": len(texts)})