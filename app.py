"""
AI Research Scientist -- a single-file Streamlit app that autonomously plans
research, searches & reads papers (OpenAlex), builds/queries a knowledge
graph (Neo4j AuraDB), generates hypotheses, runs sandboxed Python experiments
(Docker), critiques results, iterates, and produces a fully cited report.

Run with:  streamlit run app.py
"""
import json
import traceback

import streamlit as st

from src.config import load_settings, missing_required
from src.embeddings import Embedder
from src.neo4j_client import Neo4jClient
from src.pdf_utils import extract_text
from src.sandbox import docker_available
from src.agent.orchestrator import run_pipeline

st.set_page_config(page_title="Automated ai research for quant finance", page_icon="🔬", layout="wide")


@st.cache_resource(show_spinner="Loading local embedding model (first run only)...")
def get_embedder():
    return Embedder()


def check_neo4j(settings) -> tuple:
    try:
        client = Neo4jClient(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password, settings.neo4j_database)
        client.verify_connectivity()
        client.close()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def main():
    settings = load_settings()

    st.title("🔬 Automated AI Research for Quantitative Finance")
    st.caption("Autonomous research: plan → search papers → build knowledge graph → hypothesize → experiment → critique → report")

    with st.sidebar:
        st.header("System status")

        missing = missing_required(settings)
        if missing:
            st.error(f"Missing required .env values: {', '.join(missing)}")
            st.markdown("Copy `.env.example` to `.env` and fill these in, then restart the app.")
        else:
            st.success("Required credentials found in .env")

        if st.button("Test Neo4j AuraDB connection"):
            with st.spinner("Connecting..."):
                ok, msg = check_neo4j(settings)
            if ok:
                st.success("Neo4j: " + msg)
            else:
                st.error("Neo4j connection failed: " + msg)
                st.info("AuraDB instances can take ~60s to wake up after creation/pause. Check status at https://console.neo4j.io")

        if docker_available():
            st.success("Docker: available (experiments run sandboxed)")
        else:
            st.warning("Docker: not detected. Experiments will fall back to running as a local subprocess (NOT isolated). Install Docker Desktop for real sandboxing.")

        st.divider()
        st.caption(f"Groq model: `{settings.grok_model}`")
        st.caption(f"Max search queries: {settings.max_search_queries} | Max papers/query: {settings.max_papers_per_query}")
        st.caption(f"Max hypotheses: {settings.max_hypotheses} | Max experiment retries: {settings.max_experiment_retries}")

    st.subheader("1. Ask a research question")
    question = st.text_area(
        "Research question",
        placeholder="e.g. Does inventory-skewed quoting improve market-maker PnL under adverse selection in binary options?",
        height=80,
    )

    uploaded_files = st.file_uploader(
        "Optional: upload supporting PDFs / CSV / TXT files",
        type=["pdf", "csv", "txt", "md"],
        accept_multiple_files=True,
    )

    run_clicked = st.button("🚀 Run autonomous research", type="primary", disabled=bool(missing))

    if run_clicked:
        if not question.strip():
            st.warning("Please enter a research question first.")
            st.stop()

        documents = []
        for f in uploaded_files or []:
            text = extract_text(f.read(), f.name)
            if text.strip():
                documents.append({"filename": f.name, "text": text})

        embedder = get_embedder()

        status_box = st.status("Starting autonomous research pipeline...", expanded=True)

        def progress_cb(msg):
            status_box.write(msg)

        try:
            result = run_pipeline(settings, question, documents, embedder, progress_cb=progress_cb)
            status_box.update(label="Research pipeline complete", state="complete", expanded=False)
            st.session_state["result"] = result
        except Exception as e:
            status_box.update(label="Pipeline failed", state="error")
            st.error(f"Pipeline error: {e}")
            st.code(traceback.format_exc())
            st.stop()

    result = st.session_state.get("result")
    if result:
        st.divider()
        st.subheader("2. Research plan")
        st.json(result["plan"])

        st.subheader("3. Papers found")
        st.write(f"{len(result['papers'])} unique papers retrieved from OpenAlex")
        for p in result["papers"]:
            with st.expander(f"{p['title']} ({p.get('year', 'n.d.')})"):
                st.write(f"Authors: {', '.join(p.get('authors', [])) or 'Unknown'}")
                st.write(f"Venue: {p.get('venue') or 'n/a'}")
                st.write(f"DOI: {p.get('doi') or 'n/a'}")
                st.write(p.get("abstract") or "_No abstract available._")

        st.subheader("4. Knowledge graph summary")
        st.json(result["kg_summary"])

        st.subheader("5. Hypotheses, experiments & critiques")
        for i, b in enumerate(result["bundles"]):
            h = b["hypothesis"]
            crit = b["critique"] or {}
            res = b["result"] or {}
            verdict = crit.get("verdict", "unknown")
            badge = {"supported": "🟢", "refuted": "🔴", "inconclusive": "🟡", "experiment_broken": "⚠️"}.get(verdict, "⚪")
            with st.expander(f"{badge} Hypothesis {i + 1}: {h['text']}"):
                st.markdown(f"**Rationale:** {h['rationale']}")
                st.markdown(f"**Experiment sandbox mode:** `{res.get('mode', 'n/a')}`")
                st.markdown("**Experiment code:**")
                st.code(b["code"], language="python")
                st.markdown("**stdout:**")
                st.code(res.get("stdout", ""))
                if res.get("stderr"):
                    st.markdown("**stderr:**")
                    st.code(res.get("stderr", ""))
                st.markdown("**Parsed results (output.json):**")
                st.json(res.get("artifacts", {}))
                st.markdown(f"**Critique verdict:** {verdict} (confidence: {crit.get('confidence', 'n/a')})")
                st.markdown(f"**Critique notes:** {crit.get('critique', '')}")

        st.subheader("6. Final cited report")
        st.markdown(result["report_md"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "📄 Download report (Markdown)",
                data=result["report_md"],
                file_name="research_report.md",
                mime="text/markdown",
            )
        with col2:
            st.download_button(
                "🧾 Download provenance log (JSON)",
                data=json.dumps(result["provenance"], indent=2, default=str),
                file_name="provenance_log.json",
                mime="application/json",
            )
        with col3:
            st.download_button(
                "🕸️ Download knowledge graph summary (JSON)",
                data=json.dumps(result["kg_summary"], indent=2, default=str),
                file_name="kg_summary.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()