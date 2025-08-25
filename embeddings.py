#!/usr/bin/env python3
"""
embeddings.py — build embeddings for YouTube Q&A comments
"""

import os, json, argparse, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect, lang_detect_exception

DEFAULT_MODEL = os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2")

def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except lang_detect_exception.LangDetectException:
        return False

def _flatten_comments(ydata):
    out = []
    for comp in ydata.get("competitors_data", []):
        brand = comp.get("brand", "")
        website = comp.get("website", "")
        for q in comp.get("results", []):
            question = q.get("question", "")
            for v in q.get("videos", []):
                video = v.get("video", {})
                vid = video.get("video_id", "")
                url = video.get("url", "")
                for c in v.get("top_comments", []):
                    text = (c.get("text") or "").strip()
                    if not text:
                        continue
                    out.append({
                        "brand": brand,
                        "website": website,
                        "question": question,
                        "video_id": vid,
                        "video_url": url,
                        "comment_text": text,
                        "likeCount": c.get("likeCount", 0),
                        "publishedAt": c.get("publishedAt", ""),
                        "author": c.get("author", "Anonymous"),
                    })
    return out

def main():
    ap = argparse.ArgumentParser(description="Embed YouTube comments for clustering")
    ap.add_argument("--youtube", required=True, help="Path to youtube_analysis.json")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    ap.add_argument("--min-chars", type=int, default=25, help="Minimum characters for a comment")
    ap.add_argument("--out-prefix", default="embeddings", help="Output prefix")
    args = ap.parse_args()

    yt_path = Path(args.youtube)
    with yt_path.open("r", encoding="utf-8") as f:
        ydata = json.load(f)

    comments = _flatten_comments(ydata)

    # Dedup
    seen = set()
    deduped = []
    for c in comments:
        if len(c["comment_text"]) < args.min_chars:
            continue
        key = (c["brand"].lower().strip(), c["question"].strip(), c["comment_text"].strip())
        if key in seen:
            continue
        seen.add(key)
        if _is_english(c["comment_text"]):
            deduped.append(c)

    if not deduped:
        raise SystemExit("No comments to embed after filtering.")

    print(f"Embedding {len(deduped)} comments using model: {args.model}")
    model = SentenceTransformer(args.model)
    texts = [c["comment_text"] for c in deduped]
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    vec_path = Path(args.out_prefix).with_suffix(".npy")
    idx_path = Path("comments_index.json")
    meta_path = Path("embeddings.json")

    np.save(vec_path, embs)

    for i, c in enumerate(deduped):
        c["idx"] = i
        c["text_hash"] = _hash_text(c["comment_text"])

    with idx_path.open("w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    meta = {
        "vectors_file": str(vec_path),
        "index_file": str(idx_path),
        "count": len(deduped),
        "model": args.model,
        "source_youtube_file": str(yt_path),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved vectors to {vec_path}")
    print(f"✓ Saved index to {idx_path}")
    print(f"✓ Wrote metadata {meta_path}")

if __name__ == "__main__":
    main()
