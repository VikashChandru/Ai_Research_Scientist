"""
Central configuration loader.
Reads every setting from environment variables (populated from a local .env
file via python-dotenv). No secret is ever hard-coded here.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)  # loads .env from the project root if present,
                             # and always wins over any stray shell/system env var  # loads .env from the project root if present


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    groq_api_key: str
    grok_model: str
    openalex_email: str
    openalex_api_key: str
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    sandbox_docker_image: str
    sandbox_timeout_seconds: int
    max_search_queries: int
    max_papers_per_query: int
    max_hypotheses: int
    max_experiment_retries: int


def load_settings() -> Settings:
    return Settings(
        groq_api_key=_get("GROQ_API_KEY"),
        grok_model=_get("GROK_MODEL", "llama-3.3-70b-versatile"),
        openalex_email=_get("OPENALEX_EMAIL"),
        openalex_api_key=_get("OPENALEX_API_KEY"),
        neo4j_uri=_get("NEO4J_URI"),
        neo4j_username=_get("NEO4J_USERNAME"),
        neo4j_password=_get("NEO4J_PASSWORD"),
        neo4j_database=_get("NEO4J_DATABASE", "neo4j"),
        sandbox_docker_image=_get("SANDBOX_DOCKER_IMAGE", "ai-research-sandbox:latest"),
        sandbox_timeout_seconds=_get_int("SANDBOX_TIMEOUT_SECONDS", 60),
        max_search_queries=_get_int("MAX_SEARCH_QUERIES", 4),
        max_papers_per_query=_get_int("MAX_PAPERS_PER_QUERY", 8),
        max_hypotheses=_get_int("MAX_HYPOTHESES", 3),
        max_experiment_retries=_get_int("MAX_EXPERIMENT_RETRIES", 2),
    )


def missing_required(settings: Settings) -> list:
    """Return a list of human-readable names of required settings that are empty."""
    missing = []
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    if not settings.neo4j_uri:
        missing.append("NEO4J_URI")
    if not settings.neo4j_username:
        missing.append("NEO4J_USERNAME")
    if not settings.neo4j_password:
        missing.append("NEO4J_PASSWORD")
    return missing