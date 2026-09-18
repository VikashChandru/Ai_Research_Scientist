"""
Thin wrapper around the Groq chat completion API.
Groq exposes an OpenAI-compatible endpoint, so we reuse the `openai` SDK
with a custom base_url. The API key is read only from environment
variables (never hard-coded, never logged).
"""
import json
import re
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GrokClient:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def chat(self, system: str, user: str, temperature: float = 0.4, max_tokens: int = 2000) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_json(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 2000) -> dict:
        """Ask Grok for a reply and force-parse it as JSON, retrying on malformed output."""
        sys_json = system + "\n\nRespond with ONLY valid JSON. No markdown fences, no preamble, no trailing commas."
        last_error = None
        current_user = user
        for attempt in range(3):
            raw = self.chat(sys_json, current_user, temperature=temperature, max_tokens=max_tokens)
            try:
                return self._extract_json(raw)
            except (json.JSONDecodeError, AttributeError) as e:
                last_error = e
                current_user = user + "\n\nYour previous reply was not valid JSON. Return ONLY a single valid JSON object, nothing else, no explanation."
        raise ValueError(f"Grok did not return valid JSON after 3 attempts: {last_error}")

    @staticmethod
    def _extract_json(raw: str) -> dict:
        text = raw.strip()
        text = re.sub(r"^```(json)?", "", text.strip())
        text = re.sub(r"```$", "", text.strip())
        text = text.strip()
        text = re.sub(r",(\s*[}\]])", r"\1", text)  # strip trailing commas before } or ]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                candidate = re.sub(r",(\s*[}\]])", r"\1", match.group(0))
                return json.loads(candidate)
            raise