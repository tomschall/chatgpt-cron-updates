#!/usr/bin/env python3
"""
Weekly LINK/RWA Update → Slack
- Ruft OpenAI Responses API auf, erzeugt ein kurzes Update
- Postet den Text an einen Slack Incoming Webhook

ENV Variablen (über GitHub Secrets setzen):
- OPENAI_API_KEY
- SLACK_WEBHOOK_URL
Optional:
- OPENAI_MODEL (Default: gpt-4o-mini)
- MAX_TOKENS (Default: 1200)
- TEMPERATURE (Default: 0.3)
"""
import os
import json
import sys
from datetime import datetime, timezone
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY fehlt.")
    sys.exit(1)
if not SLACK_WEBHOOK_URL:
    print("[ERROR] SLACK_WEBHOOK_URL fehlt.")
    sys.exit(1)

PROMPT = f"""
Erstelle ein kurzes Wochenupdate zu Chainlink (LINK) im Kontext von RWAs.
- Neue Partnerschaften/Integrationen mit Institutionen
- Nutzung von Data Streams und CCIP
- Entwicklung des RWA-Gesamtmarkts (z. B. BUIDL-Fonds, RWA.xyz)
- On-Chain-/Revenue-Signale
- Kritische Punkte oder Gegenargumente

Form:
- 1 kurze Einleitung mit Datum
- 4–6 Bulletpoints
- Jede Bullet mit Quelle/Referenz, wenn möglich
- Schluss: Takeaway + 1–2 Signalschwellen

Datum: {datetime.now(timezone.utc).strftime('%Y-%m-%d')} (UTC)
"""


def _extract_text_from_responses(json_obj: dict) -> str:
    """Extrahiert Text aus OpenAI Responses API Antworten."""
    if isinstance(json_obj, dict):
        if "output_text" in json_obj:
            return str(json_obj["output_text"]).strip()
        out = json_obj.get("output")
        if isinstance(out, list) and out:
            content = out[0].get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") in ("output_text", "text"):
                            return str(part.get("text", "")).strip()
                        if part.get("type") == "message":
                            for sub in part.get("content", []):
                                if sub.get("type") in ("output_text", "text"):
                                    return str(sub.get("text", "")).strip()
        choices = json_obj.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {}).get("content")
            if msg:
                return str(msg).strip()
    return ""


def fetch_update_from_openai() -> str:
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "input": PROMPT,
        "max_output_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API Error: {resp.status_code} {resp.text}")
    data = resp.json()
    text = _extract_text_from_responses(data)
    if not text:
        text = json.dumps(data, ensure_ascii=False, indent=2)
    return text


def send_to_slack(text: str) -> None:
    payload = {"text": f"🚀 Weekly LINK/RWA Update\n\n{text}"}
    r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(
            f"Slack Webhook Error: {r.status_code} {r.text}"
        )


def main():
    try:
        report = fetch_update_from_openai()
        send_to_slack(report)
        print("[OK] Update gesendet.")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()