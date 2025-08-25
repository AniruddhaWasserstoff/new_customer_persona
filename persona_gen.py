#!/usr/bin/env python3
"""
persona_gen.py — STRICT schema + flat array output

- Generates exactly ONE persona JSON per cluster.
- Each persona object contains ONLY these keys:
  "persona_name", "demographics", "goals", "pain_points",
  "channels", "content_preferences", "marketing_strategy"

- The final output file is a flat JSON array of persona objects (no wrapper).
- Supports OpenAI or Groq; falls back to heuristic if no API key or call fails.

Usage:
  python persona_gen.py --clusters clusters.json --index comments_index.json --output personas.json
  # Optional provider controls:
  --provider auto|openai|groq|none
  --model gpt-4o-mini
  --temperature 0.2
  --max-quotes 3
"""

import os, json, argparse, re, sys
from pathlib import Path
from typing import List, Dict, Any, Optional

REQUIRED_KEYS = [
    "persona_name",
    "demographics",
    "goals",
    "pain_points",
    "channels",
    "content_preferences",
    "marketing_strategy",
]

TEMPLATE_INSTRUCTION = (
    'You are an expert market researcher. Create exactly one JSON persona for cluster {label}. '
    'Previously generated persona names: {prev_list}. '
    'Ensure this persona is distinct in persona_name, demographics, goals, and pain_points from all previous ones. '
    'Include ONLY these keys: persona_name (string), demographics (object), goals (object), '
    'pain_points (array of strings), channels (object), content_preferences (object), marketing_strategy (object). '
    'All keys and values must be double-quoted. Numeric ranges (e.g. "28-35") must be strings. '
    'Do NOT include comments, markdown, or trailing commas. Use JSON arrays for lists.'
)

# --- Optional provider imports (lazy) ---
_openai_available = False
_groq_available = False
try:
    from openai import OpenAI as _OpenAI
    _openai_available = True
except Exception:
    pass

try:
    from groq import Groq as _Groq
    _groq_available = True
except Exception:
    pass

def _shorten(s: str, n: int = 600) -> str:
    s = s.strip().replace("\\n", " ").replace("\\r", " ")
    return s if len(s) <= n else s[:n] + "…"

def build_cluster_context(cluster: Dict[str, Any], by_idx: Dict[int, Dict[str, Any]], max_quotes: int = 3) -> str:
    parts = []
    parts.append(f'Cluster Label: {cluster.get("cluster_id", "N/A")}')
    parts.append(f'Cluster Name: {cluster.get("name", "")}')
    kws = cluster.get("top_keywords", [])
    if kws:
        parts.append(f'Top Keywords: {", ".join(kws)}')
    tb = cluster.get("top_brands", [])
    if tb:
        parts.append("Top Brands: " + ", ".join([f"{b}:{c}" for b,c in tb]))
    tq = cluster.get("top_questions", [])
    if tq:
        parts.append("Top Questions: " + ", ".join([_shorten(q, 120) for q,_ in tq]))
    indices = cluster.get("indices", [])
    items = [by_idx[i] for i in indices if i in by_idx]
    items_sorted = sorted(items, key=lambda it: it.get("likeCount", 0), reverse=True)
    if items_sorted:
        parts.append("Representative Comments:")
        for it in items_sorted[:max_quotes]:
            parts.append(f'- "{_shorten(it.get("comment_text", ""), 280)}" (brand={it.get("brand","")}, likes={it.get("likeCount",0)})')
    return "\\n".join(parts)

def extract_json_block(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    count = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            count += 1
        elif ch == "}":
            count -= 1
            if count == 0:
                return text[start:i+1]
    return None

def llm_call(provider: str, model: str, temperature: float, system_prompt: str, user_prompt: str) -> Optional[str]:
    try:
        if provider == "openai":
            if not _openai_available: return None
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key: return None
            client = _OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role":"system","content": system_prompt},
                    {"role":"user","content": user_prompt},
                ],
                response_format={"type":"json_object"},
            )
            return resp.choices[0].message.content
        elif provider == "groq":
            if not _groq_available: return None
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key: return None
            client = _Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role":"system","content": system_prompt},
                    {"role":"user","content": user_prompt},
                ],
            )
            return resp.choices[0].message.content
        else:
            return None
    except Exception as e:
        print(f"[warn] LLM call failed: {e}", file=sys.stderr)
        return None

def ensure_only_required_keys(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Keep only the required keys; coerce missing to empty appropriate types
    clean = {}
    for k in REQUIRED_KEYS:
        v = obj.get(k, None)
        if k == "pain_points":
            clean[k] = v if isinstance(v, list) else ([] if v is None else [str(v)])
        elif k in ("demographics","goals","channels","content_preferences","marketing_strategy"):
            clean[k] = v if isinstance(v, dict) else {}
        else:
            clean[k] = "" if v is None else str(v)
    return clean

def heuristic_persona(cluster: Dict[str, Any]) -> Dict[str, Any]:
    kws = cluster.get("top_keywords", [])
    base = (", ".join(kws[:2]) or "General Interest").title()
    persona = {
        "persona_name": f"{base} Buyer",
        "demographics": {
            "age_range": "24-34",
            "gender": "Any",
            "location": "Tier-1 and Tier-2 cities",
            "income_range": "₹6L-₹14L per year",
            "occupation": "Early-career professionals",
            "family_status": "Single or newly married, no kids"
        },
        "goals": {
            "primary": "Find reliable products with strong value for money",
            "secondary": "Reduce post-purchase regret via credible reviews",
            "discovery": "Use social search and video reviews to validate options",
            "long_term": "Build a durable, minimal set of essentials"
        },
        "pain_points": [
            "Hard-to-verify product claims",
            "Inconsistent sizing and fit info",
            "Shipping delays and hidden fees"
        ],
        "channels": {
            "social": ["Instagram", "YouTube"],
            "search": ["Google"],
            "email": ["Brand newsletters"]
        },
        "content_preferences": {
            "formats": ["Short comparison videos", "Infographics", "Review roundups"],
            "topics": ["Price vs. quality", "Transparency", "Care and repair tips"],
            "tone": ["Practical", "Evidence-based", "Transparent"]
        },
        "marketing_strategy": {
            "messaging": "Lead with total cost of ownership and proof of durability.",
            "offers": "Bundle savings, free first exchange, transparent shipping policies.",
            "targeting": "Retarget visitors who checked sizing and warranty pages."
        }
    }
    return persona

def main():
    ap = argparse.ArgumentParser(description="Generate strict-schema personas from clusters (flat array output)")
    ap.add_argument("--clusters", default="clusters.json", help="Path to clusters.json")
    ap.add_argument("--index", default="comments_index.json", help="Path to comments_index.json")
    ap.add_argument("--output", default="personas.json", help="Output file (flat JSON array)")
    ap.add_argument("--provider", default="auto", choices=["auto","openai","groq","none"], help="LLM provider")
    ap.add_argument("--model", default=None, help="Model name")
    ap.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    ap.add_argument("--max-quotes", type=int, default=3, help="Representative comments to pass into prompt")
    args = ap.parse_args()

    # Resolve provider
    provider = args.provider
    if provider == "auto":
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("GROQ_API_KEY"):
            provider = "groq"
        else:
            provider = "none"

    # Default model
    model = args.model
    if not model:
        if provider == "openai":
            model = "gpt-4o-mini"
        elif provider == "groq":
            model = "llama-3.1-70b-versatile"
        else:
            model = ""

    # Load inputs
    with Path(args.clusters).open("r", encoding="utf-8") as f:
        clusters = json.load(f)
    with Path(args.index).open("r", encoding="utf-8") as f:
        index = json.load(f)
    by_idx = {c["idx"]: c for c in index}

    personas: List[Dict[str, Any]] = []
    existing_names = set()

    for c in clusters.get("clusters", []):
        label = c.get("cluster_id", "N/A")
        prev_list = ", ".join(sorted(existing_names)) if existing_names else "[]"
        context = build_cluster_context(c, by_idx, max_quotes=args.max_quotes)

        system_prompt = "Return only valid JSON with exactly the requested keys. No markdown."
        user_prompt = TEMPLATE_INSTRUCTION.format(label=label, prev_list=prev_list) + "\\n\\n" + context

        persona_obj: Optional[Dict[str, Any]] = None

        if provider in ("openai","groq") and model:
            raw = llm_call(provider, model, args.temperature, system_prompt, user_prompt)
            if raw:
                try:
                    persona_obj = json.loads(raw)
                except Exception:
                    block = extract_json_block(raw)
                    if block:
                        try:
                            persona_obj = json.loads(block)
                        except Exception:
                            persona_obj = None

        if persona_obj is None:
            persona_obj = heuristic_persona(c)

        persona_obj = ensure_only_required_keys(persona_obj)

        # Ensure distinct persona_name
        base_name = persona_obj.get("persona_name","Persona").strip() or "Persona"
        name = base_name
        i = 2
        while name in existing_names:
            name = f"{base_name} #{i}"
            i += 1
        persona_obj["persona_name"] = name
        existing_names.add(name)

        personas.append(persona_obj)

    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)

    print(f"✓ Wrote {len(personas)} strict personas → {args.output} (provider={provider}{' model='+model if model else ''})")

if __name__ == "__main__":
    main()
