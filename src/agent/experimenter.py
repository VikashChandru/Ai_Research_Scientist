"""Stage 4: turn a hypothesis into runnable Python code and execute it in the sandbox."""
from src.grok_client import GrokClient
from src import sandbox
from src.neo4j_client import Neo4jClient
from src.utils import new_id, ProvenanceLog

CODE_SYSTEM = """You write small, self-contained Python 3.11 experiment scripts
for an autonomous research agent. Rules:
- Only use: numpy, pandas, scipy, scikit-learn, matplotlib, and the Python
  standard library. No internet access is available at runtime.
- The script must run standalone with `python script.py`, no CLI args.
- If you need input data and none is realistically available, generate a
  reasonable synthetic dataset that lets the hypothesis be meaningfully tested,
  and clearly print any assumptions made.
- The script MUST write its final results to a file named output.json in the
  current directory, as a flat JSON object with numeric/string/boolean values
  (e.g. {"metric_name": 0.83, "p_value": 0.02, "conclusion": "..."}).
- Also print a short human-readable summary of the result to stdout.
- Keep total runtime under a few seconds; keep the script under ~80 lines.
- Do not use matplotlib GUI backends; if plotting, save to a PNG file with
  matplotlib and use the "Agg" backend, but plots are optional -- output.json
  is what matters.

Return JSON: {"code": "<the full python script as a string>"}
"""

CRITIQUE_SYSTEM = """You are the critic module of an autonomous AI research
scientist. You are given a hypothesis, the experiment code that was run, and
its stdout/stderr/output. Assess rigor and correctness, then decide:
- verdict: one of "supported", "refuted", "inconclusive", "experiment_broken"
- confidence: float 0-1
- critique: a few sentences on strengths/weaknesses of the experiment design
- next_action: one of "accept", "revise_experiment", "abandon_hypothesis"
- revision_notes: if next_action is "revise_experiment", concrete instructions
  for how to fix or improve the script (empty string otherwise)

Return JSON:
{"verdict": "...", "confidence": 0.0, "critique": "...", "next_action": "...", "revision_notes": "..."}
"""


def generate_experiment_code(grok: GrokClient, hypothesis_text: str, suggested_experiment: str, revision_notes: str = "") -> str:
    user = f"Hypothesis: {hypothesis_text}\nSuggested experiment: {suggested_experiment}\n"
    if revision_notes:
        user += f"\nThe previous attempt needs revision. Notes: {revision_notes}\n"
    result = grok.chat_json(CODE_SYSTEM, user, temperature=0.3, max_tokens=2500)
    return result.get("code", "")


def run_and_critique(
    grok: GrokClient,
    neo4j: Neo4jClient,
    hypothesis: dict,
    sandbox_image: str,
    sandbox_timeout: int,
    max_retries: int,
    provenance: ProvenanceLog,
    progress_cb=None,
) -> dict:
    revision_notes = ""
    last_result = None
    last_code = ""
    critique_result = {}

    for attempt in range(max_retries + 1):
        if progress_cb:
            progress_cb(f"Generating experiment code (attempt {attempt + 1})")
        code = generate_experiment_code(grok, hypothesis["text"], hypothesis["suggested_experiment"], revision_notes)
        last_code = code
        provenance.log("experiment", "code_generated", {"hypothesis_id": hypothesis["id"], "attempt": attempt + 1})

        if progress_cb:
            progress_cb(f"Running experiment in sandbox (attempt {attempt + 1})")
        result = sandbox.run_experiment(code, sandbox_image, sandbox_timeout)
        last_result = result
        provenance.log("experiment", "sandbox_run", {
            "hypothesis_id": hypothesis["id"], "attempt": attempt + 1,
            "success": result["success"], "mode": result["mode"],
        })

        exp_id = new_id("exp")
        try:
            neo4j.create_experiment(exp_id, hypothesis["id"], code, result["stdout"], result["stderr"], result["success"])
        except Exception as e:
            provenance.log("experiment", "kg_write_failed", {"error": str(e)})

        if progress_cb:
            progress_cb("Critiquing experiment result")
        critique_user = (
            f"Hypothesis: {hypothesis['text']}\n\n"
            f"Experiment code:\n{code}\n\n"
            f"stdout:\n{result['stdout'][:2000]}\n\n"
            f"stderr:\n{result['stderr'][:1000]}\n\n"
            f"artifacts (parsed output.json): {result['artifacts']}\n"
        )
        try:
            critique_result = grok.chat_json(CRITIQUE_SYSTEM, critique_user, max_tokens=1200)
        except Exception as e:
            critique_result = {
                "verdict": "inconclusive", "confidence": 0.0,
                "critique": f"Critique generation failed: {e}",
                "next_action": "accept", "revision_notes": "",
            }

        critique_id = new_id("crit")
        try:
            neo4j.create_critique(
                critique_id, exp_id, critique_result.get("critique", ""),
                critique_result.get("verdict", "inconclusive"),
                float(critique_result.get("confidence", 0) or 0),
            )
            neo4j.set_hypothesis_status(hypothesis["id"], critique_result.get("verdict", "inconclusive"))
        except Exception as e:
            provenance.log("experiment", "kg_critique_write_failed", {"error": str(e)})

        provenance.log("experiment", "critique_done", {
            "hypothesis_id": hypothesis["id"], "verdict": critique_result.get("verdict"),
            "next_action": critique_result.get("next_action"),
        })

        if critique_result.get("next_action") != "revise_experiment" or attempt == max_retries:
            break
        revision_notes = critique_result.get("revision_notes", "")

    return {
        "hypothesis": hypothesis,
        "code": last_code,
        "result": last_result,
        "critique": critique_result,
    }