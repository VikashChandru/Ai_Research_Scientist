"""
Run this BEFORE `streamlit run app.py` to sanity-check every external
dependency using your own .env credentials.

    python check_setup.py
"""
import sys

from src.config import load_settings, missing_required


def main():
    settings = load_settings()
    print("=== AI Research Scientist -- setup check ===\n")

    missing = missing_required(settings)
    if missing:
        print(f"[FAIL] Missing required .env values: {', '.join(missing)}")
        print("        Copy .env.example to .env and fill these in.\n")
    else:
        print("[OK] All required .env values are present.\n")

    # --- Groq ---
    print("Checking Groq...")
    try:
        from src.grok_client import GrokClient
        client = GrokClient(settings.groq_api_key, settings.grok_model)
        reply = client.chat("You are a test assistant.", "Reply with the single word: OK", max_tokens=10)
        print(f"[OK] Groq responded: {reply.strip()[:50]}\n")
    except Exception as e:
        print(f"[FAIL] Groq check failed: {e}\n")

    # --- OpenAlex ---
    print("Checking OpenAlex...")
    try:
        from src.openalex_client import OpenAlexClient
        client = OpenAlexClient(settings.openalex_email, settings.openalex_api_key)
        results = client.search("machine learning", per_page=1)
        print(f"[OK] OpenAlex returned {len(results)} result(s)\n")
    except Exception as e:
        print(f"[FAIL] OpenAlex check failed: {e}\n")

    # --- Neo4j AuraDB ---
    print("Checking Neo4j AuraDB (this can take up to ~60s if the instance just resumed)...")
    try:
        from src.neo4j_client import Neo4jClient
        client = Neo4jClient(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password, settings.neo4j_database)
        client.verify_connectivity()
        client.close()
        print("[OK] Neo4j AuraDB connection succeeded\n")
    except Exception as e:
        print(f"[FAIL] Neo4j check failed: {e}")
        print("        If your instance was just created/paused, wait ~60s and try again,")
        print("        or check status at https://console.neo4j.io\n")

    # --- Docker ---
    print("Checking Docker (used to sandbox experiments)...")
    from src.sandbox import docker_available
    if docker_available():
        print("[OK] Docker is available. Remember to build the sandbox image once:")
        print("     docker build -t ai-research-sandbox:latest -f Dockerfile.sandbox .\n")
    else:
        print("[WARN] Docker not detected. The app will still run, but experiments will")
        print("       execute as an unsandboxed local subprocess. Install Docker Desktop")
        print("       for real isolation.\n")

    # --- Local embedding model ---
    print("Checking local embedding model (downloads ~90MB on first run)...")
    try:
        from src.embeddings import Embedder
        embedder = Embedder()
        vec = embedder.embed(["hello world"])
        print(f"[OK] Embedding model loaded, vector shape {vec.shape}\n")
    except Exception as e:
        print(f"[FAIL] Embedding model check failed: {e}\n")

    print("Done.")


if __name__ == "__main__":
    sys.exit(main())