"""Stage 5: compile the final cited report with full provenance."""
from src.grok_client import GrokClient

SYSTEM = """You are the report-writing module of an autonomous AI research
scientist. You will be given: the original research question, a numbered
reference list, a knowledge-graph summary, and a list of hypothesis/
experiment/critique bundles. Write a rigorous but readable research report in
Markdown with these sections:

## Executive Summary
## Background & Related Work
(synthesize the references; cite with [n] matching the given reference numbers -- ONLY use numbers that exist in the reference list, never invent one)
## Hypotheses & Experiments
(for each hypothesis: the hypothesis, why it was proposed [cite], what the
experiment did, what was found, and the critique's verdict)
## Findings
## Limitations
## Suggested Next Steps

Do not invent citations. Only cite reference numbers that are provided to you.
If a claim isn't backed by a reference, don't add a citation marker for it.
"""


def build_report(
    grok: GrokClient,
    question: str,
    reference_list: list,
    kg_summary: dict,
    hypothesis_bundles: list,
) -> str:
    refs_block = "\n".join(
        f"[{i+1}] {r['title']} ({r.get('year', 'n.d.')}) - "
        f"{', '.join(r.get('authors', [])[:5]) or 'Unknown authors'}"
        f"{' - DOI: ' + r['doi'] if r.get('doi') else ''}"
        for i, r in enumerate(reference_list)
    )

    bundles_block = ""
    for b in hypothesis_bundles:
        h = b["hypothesis"]
        res = b["result"] or {}
        crit = b["critique"] or {}
        bundles_block += (
            f"\n---\nHypothesis: {h['text']}\n"
            f"Rationale: {h['rationale']}\n"
            f"Based on paper ids: {h['based_on_paper_ids']}\n"
            f"Experiment stdout: {(res.get('stdout') or '')[:800]}\n"
            f"Experiment artifacts: {res.get('artifacts')}\n"
            f"Critique verdict: {crit.get('verdict')} (confidence {crit.get('confidence')})\n"
            f"Critique notes: {crit.get('critique')}\n"
        )

    user = (
        f"Research question: {question}\n\n"
        f"Reference list (cite these by number in brackets, e.g. [1]):\n{refs_block}\n\n"
        f"Knowledge graph summary: {kg_summary}\n\n"
        f"Hypothesis/experiment/critique bundles:\n{bundles_block}\n"
    )

    body = grok.chat(SYSTEM, user, temperature=0.4, max_tokens=4000)

    references_section = "\n\n## References\n" + "\n".join(
        f"{i+1}. {r['title']} ({r.get('year', 'n.d.')}). "
        f"{', '.join(r.get('authors', [])[:8]) or 'Unknown authors'}. "
        f"{r.get('venue') or ''} "
        f"{('DOI: ' + r['doi']) if r.get('doi') else (r.get('landing_page_url') or '')}"
        for i, r in enumerate(reference_list)
    )

    return body + references_section