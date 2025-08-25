#!/usr/bin/env python3
# followup.py — interactive terminal version
import os, json, re, sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

def _extract_json_array(text: str):
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array found in LLM output:\n{text}")
    arr = json.loads(m.group(0))
    if not isinstance(arr, list):
        raise ValueError("Expected a JSON array")
    return arr

def main():
    if not GROQ_API_KEY:
        print("❌ Missing GROQ_API_KEY in .env")
        sys.exit(1)

    # Ask user for inputs
    competitors_path = input("Enter path to competitors.json (from findcomp.py): ").strip()
    summary_path = input("Enter path to summary.json (from questions.py): ").strip()
    out_path = input("Enter path for output (default: followups.json): ").strip() or "followups.json"

    with open(competitors_path, "r", encoding="utf-8") as f:
        comp_data = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    competitors = [c.get("name") or c.get("brand") for c in comp_data.get("competitors", []) if (c.get("name") or c.get("brand"))]
    summary = json.dumps(summary_data, ensure_ascii=False)[:4000]

    client = Groq(api_key=GROQ_API_KEY)
    out = {}

    for brand in competitors:
        system = (
            "You are a business consultant. "
            "Given a business summary and a competitor, generate EXACTLY 3 follow-up questions "
            "in JSON array format. Keep them factual and researchable."
        )
        user = f"""Business Summary:
{summary}

Target competitor: {brand}

Return ONLY a JSON array of 3 strings, nothing else."""
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content
        questions = _extract_json_array(raw)[:3]
        out[brand] = questions

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ Follow-up questions saved to {out_path}")

if __name__ == "__main__":
    main()
