"""
Knowledge-graph wrapper around the Neo4j Python driver, targeting AuraDB.

Schema used by this project:
  (:Paper {id, title, year, doi, venue, authors, source})
  (:Concept {name})
  (:Hypothesis {id, text, rationale, status, created_at})
  (:Experiment {id, code, stdout, stderr, success, created_at})
  (:Critique {id, text, verdict, confidence, created_at})

  (Paper)-[:HAS_CONCEPT]->(Concept)
  (Hypothesis)-[:BASED_ON]->(Paper)
  (Experiment)-[:TESTS]->(Hypothesis)
  (Critique)-[:EVALUATES]->(Experiment)
"""
from neo4j import GraphDatabase
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError


class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    def close(self):
        self._driver.close()

    def verify_connectivity(self):
        self._driver.verify_connectivity()

    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((ServiceUnavailable, SessionExpired, TransientError)),
    )
    def _run(self, query: str, **params):
        def _work(tx):
            return list(tx.run(query, **params))
        with self._driver.session(database=self._database) as session:
            return session.execute_write(_work)

    def ensure_constraints(self):
        self._run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
        self._run("CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
        self._run("CREATE CONSTRAINT hyp_id IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE")
        self._run("CREATE CONSTRAINT exp_id IF NOT EXISTS FOR (e:Experiment) REQUIRE e.id IS UNIQUE")

    def upsert_paper(self, paper: dict):
        self._run(
            """
            MERGE (p:Paper {id: $id})
            SET p.title = $title, p.year = $year, p.doi = $doi,
                p.venue = $venue, p.authors = $authors, p.source = $source
            """,
            id=paper["id"], title=paper.get("title", ""), year=paper.get("year"),
            doi=paper.get("doi"), venue=paper.get("venue"),
            authors=paper.get("authors", []), source=paper.get("source", "openalex"),
        )

    def link_paper_concept(self, paper_id: str, concept: str):
        self._run(
            """
            MERGE (c:Concept {name: $concept})
            WITH c
            MATCH (p:Paper {id: $paper_id})
            MERGE (p)-[:HAS_CONCEPT]->(c)
            """,
            concept=concept.strip().lower(), paper_id=paper_id,
        )

    def create_hypothesis(self, hyp_id: str, text: str, rationale: str, based_on_paper_ids: list):
        self._run(
            """
            MERGE (h:Hypothesis {id: $id})
            SET h.text = $text, h.rationale = $rationale, h.status = 'proposed',
                h.created_at = datetime()
            """,
            id=hyp_id, text=text, rationale=rationale,
        )
        for pid in based_on_paper_ids:
            self._run(
                """
                MATCH (h:Hypothesis {id: $hid}), (p:Paper {id: $pid})
                MERGE (h)-[:BASED_ON]->(p)
                """,
                hid=hyp_id, pid=pid,
            )

    def set_hypothesis_status(self, hyp_id: str, status: str):
        self._run("MATCH (h:Hypothesis {id: $id}) SET h.status = $status", id=hyp_id, status=status)

    def create_experiment(self, exp_id: str, hyp_id: str, code: str, stdout: str, stderr: str, success: bool):
        self._run(
            """
            MERGE (e:Experiment {id: $id})
            SET e.code = $code, e.stdout = $stdout, e.stderr = $stderr,
                e.success = $success, e.created_at = datetime()
            WITH e
            MATCH (h:Hypothesis {id: $hid})
            MERGE (e)-[:TESTS]->(h)
            """,
            id=exp_id, code=code, stdout=stdout[:5000], stderr=stderr[:5000],
            success=success, hid=hyp_id,
        )

    def create_critique(self, critique_id: str, exp_id: str, text: str, verdict: str, confidence: float):
        self._run(
            """
            MERGE (c:Critique {id: $id})
            SET c.text = $text, c.verdict = $verdict, c.confidence = $confidence,
                c.created_at = datetime()
            WITH c
            MATCH (e:Experiment {id: $eid})
            MERGE (c)-[:EVALUATES]->(e)
            """,
            id=critique_id, text=text, verdict=verdict, confidence=confidence, eid=exp_id,
        )

    def get_related_papers_for_concepts(self, concepts: list, limit: int = 10) -> list:
        if not concepts:
            return []
        records = self._run(
            """
            MATCH (p:Paper)-[:HAS_CONCEPT]->(c:Concept)
            WHERE c.name IN $concepts
            RETURN DISTINCT p.id AS id, p.title AS title, p.year AS year
            LIMIT $limit
            """,
            concepts=[c.lower() for c in concepts], limit=limit,
        )
        return [dict(r) for r in records]

    def graph_summary(self) -> dict:
        def _count(label: str) -> int:
            rows = self._run(f"MATCH (n:{label}) RETURN count(n) AS c")
            return rows[0]["c"] if rows else 0

        return {
            "papers": _count("Paper"),
            "concepts": _count("Concept"),
            "hypotheses": _count("Hypothesis"),
            "experiments": _count("Experiment"),
        }