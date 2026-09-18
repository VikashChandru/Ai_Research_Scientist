"""Main autonomous loop: plan -> research -> hypothesize -> experiment -> critique -> report."""
from src.config import Settings
from src.grok_client import GrokClient
from src.openalex_client import OpenAlexClient
from src.neo4j_client import Neo4jClient
from src.embeddings import Embedder
from src.vector_store import VectorStore
from src.utils import ProvenanceLog
from src.agent import planner, researcher, hypothesis as hyp_module, experimenter, reporter


def run_pipeline(
    settings: Settings,
    question: str,
    documents: list,
    embedder: Embedder,
    progress_cb=None,
) -> dict:
    """documents: list of {"filename": str, "text": str} already extracted from uploads."""

    def report_progress(msg):
        if progress_cb:
            progress_cb(msg)

    provenance = ProvenanceLog()
    grok = GrokClient(settings.groq_api_key, settings.grok_model)
    openalex = OpenAlexClient(settings.openalex_email, settings.openalex_api_key)
    neo4j = Neo4jClient(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password, settings.neo4j_database)
    store = VectorStore()

    try:
        neo4j.ensure_constraints()

        # 1. Plan
        report_progress("Planning research approach with Grok...")
        uploaded_summary = ""
        if documents:
            uploaded_summary = f"{len(documents)} document(s) uploaded: " + ", ".join(d["filename"] for d in documents)
        plan = planner.build_plan(grok, question, uploaded_summary)
        provenance.log("plan", "plan_created", plan)

        # 2. Ingest uploaded documents (if any) into the vector store
        if documents:
            report_progress("Reading and embedding uploaded documents...")
            researcher.ingest_uploaded_documents(embedder, store, documents, provenance)

        # 3. Search + ingest literature
        search_queries = plan.get("search_queries", [question])[: settings.max_search_queries]
        papers = researcher.search_and_ingest(
            grok, openalex, neo4j, embedder, store,
            search_queries, settings.max_papers_per_query,
            provenance, progress_cb=report_progress,
        )

        # 4. Generate hypotheses
        report_progress("Generating hypotheses from evidence + knowledge graph...")
        hypotheses = hyp_module.generate_hypotheses(
            grok, embedder, store, neo4j, question, settings.max_hypotheses, provenance
        )

        # 5. Experiment + critique loop, per hypothesis
        bundles = []
        for i, h in enumerate(hypotheses):
            report_progress(f"Experimenting on hypothesis {i + 1}/{len(hypotheses)}...")
            bundle = experimenter.run_and_critique(
                grok, neo4j, h,
                settings.sandbox_docker_image, settings.sandbox_timeout_seconds,
                settings.max_experiment_retries, provenance, progress_cb=report_progress,
            )
            bundles.append(bundle)

        # 6. Build reference list (papers actually cited/used) + final report
        report_progress("Compiling final cited report...")
        kg_summary = neo4j.graph_summary()
        report_md = reporter.build_report(grok, question, papers, kg_summary, bundles)

        return {
            "plan": plan,
            "papers": papers,
            "hypotheses": hypotheses,
            "bundles": bundles,
            "kg_summary": kg_summary,
            "report_md": report_md,
            "provenance": provenance.as_list(),
        }
    finally:
        neo4j.close()