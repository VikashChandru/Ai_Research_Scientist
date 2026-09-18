"""Stage 3: retrieve relevant evidence and generate testable hypotheses."""
from src.grok_client import GrokClient
from src.embeddings import Embedder
from src.vector_store import VectorStore
from src.neo4j_client import Neo4jClient
from src.utils import new_id, ProvenanceLog

SYSTEM = """You are the hypothesis-generation module of an autonomous AI
research scientist. You are given the research question, a knowledge-graph
summary, and a set of retrieved evidence passages (each tagged with a source
id). Propose hypotheses that are:
- Specific and falsifiable
- Testable via a SMALL, self-contained Python computational experiment
  (e.g. a simulation, a statistical test, fitting a small model to synthetic
  or user-provided data) -- not something requiring physical lab work.
- Grounded in the evidence given; cite the evidence source ids you used.

Return JSON:
{
  "hypotheses": [
    {
      "text": "...",
      "rationale": "...",
      "evidence_source_ids": ["...", "..."],
      "suggested_experiment": "one paragraph describing a concrete computational experiment to test this"
    }
  ]
}
Propose at most max_hypotheses hypotheses.
"""


def generate_hypotheses(
    grok: GrokClient,
    embedder: Embedder,
    store: VectorStore,
    neo4j: Neo4jClient,
    question: str,
    max_hypotheses: int,
    provenance: ProvenanceLog,
) -> list:
    query_vec = embedder.embed([question])[0]
    evidence = store.search(query_vec, k=10)

    evidence_block = "\n".join(
        f"[source_id={i}] ({e['metadata'].get('type')}, "
        f"{e['metadata'].get('title') or e['metadata'].get('filename', '')}): {e['text'][:500]}"
        for i, e in enumerate(evidence)
    )

    kg_summary = neo4j.graph_summary()

    user = (
        f"Research question: {question}\n\n"
        f"Knowledge graph summary: {kg_summary}\n\n"
        f"Retrieved evidence:\n{evidence_block}\n\n"
        f"max_hypotheses = {max_hypotheses}"
    )

    result = grok.chat_json(SYSTEM.replace("max_hypotheses", str(max_hypotheses)), user, max_tokens=3000)
    hyps = result.get("hypotheses", [])[:max_hypotheses]

    enriched = []
    for h in hyps:
        hyp_id = new_id("hyp")
        source_ids = h.get("evidence_source_ids", [])
        based_on_paper_ids = []
        for sid in source_ids:
            try:
                idx = int(sid)
                meta = evidence[idx]["metadata"]
                if meta.get("type") == "paper_abstract":
                    based_on_paper_ids.append(meta["paper_id"])
            except (ValueError, IndexError, KeyError):
                continue
        try:
            neo4j.create_hypothesis(hyp_id, h.get("text", ""), h.get("rationale", ""), based_on_paper_ids)
        except Exception as e:
            provenance.log("hypothesis", "kg_write_failed", {"error": str(e)})

        enriched.append({
            "id": hyp_id,
            "text": h.get("text", ""),
            "rationale": h.get("rationale", ""),
            "suggested_experiment": h.get("suggested_experiment", ""),
            "based_on_paper_ids": based_on_paper_ids,
        })

    provenance.log("hypothesis", "hypotheses_generated", {"count": len(enriched)})
    return enriched