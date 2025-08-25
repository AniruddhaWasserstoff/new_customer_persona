#!/usr/bin/env python3
"""
pipeline.py — one-command runner with dynamic clustering params

Flow:
  youtube_analysis.json
    -> embeddings.py  (writes embeddings.json, comments_index.json, embeddings.npy)
    -> clustering.py  (auto-picks min_cluster_size/min_keep based on dataset size unless overridden)
    -> persona_gen.py (writes personas.json)

Usage:
  python pipeline.py --youtube youtube_analysis.json
Optional:
  --model all-MiniLM-L6-v2
  --min-chars 25
  --min-cluster-size N      # override auto
  --min-keep M              # override auto
  --personas-out personas.json
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    if p.returncode != 0:
        sys.exit(p.returncode)

def pick_params(n: int) -> tuple[int, int]:
    """
    Heuristic for small-to-large datasets:
      n < 10   -> mcs=2, keep=1
      10-29    -> mcs=max(2, round(0.2*n)), keep=max(1, round(0.15*n))
      30-199   -> mcs=max(3, round(0.1*n)), keep=max(2, round(0.07*n))
      >=200    -> mcs=round(0.07*n), keep=round(0.05*n)
    Caps/Bounds included.
    """
    if n < 10:
        mcs, keep = 2, 1
    elif n < 30:
        mcs = max(2, round(0.2 * n))
        keep = max(1, round(0.15 * n))
    elif n < 200:
        mcs = max(3, round(0.1 * n))
        keep = max(2, round(0.07 * n))
    else:
        mcs = max(5, round(0.07 * n))
        keep = max(3, round(0.05 * n))

    # Safety caps
    mcs = min(mcs, max(2, n // 2))  # never require > half the data
    keep = min(keep, mcs)           # keep threshold shouldn't exceed cluster size
    return mcs, keep

def main():
    ap = argparse.ArgumentParser(description="Run embeddings -> clustering -> persona generation (dynamic params)")
    ap.add_argument("--youtube", required=True, help="Path to youtube_analysis.json (from youtube.py)")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    ap.add_argument("--min-chars", type=int, default=25, help="Min chars per comment for embeddings.py")
    ap.add_argument("--min-cluster-size", type=int, default=None, help="Override HDBSCAN min cluster size")
    ap.add_argument("--min-keep", type=int, default=None, help="Override minimum items to keep a cluster")
    ap.add_argument("--personas-out", default="personas.json", help="Output file for personas")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    emb = here / "embeddings.py"
    clu = here / "clustering.py"
    per = here / "persona_gen.py"

    if not emb.exists() or not clu.exists() or not per.exists():
        print("Error: embeddings.py, clustering.py, persona_gen.py must be in the same folder as pipeline.py", file=sys.stderr)
        sys.exit(2)

    # 1) Embeddings
    run([sys.executable, str(emb),
         "--youtube", args.youtube,
         "--model", args.model,
         "--min-chars", str(args.min_chars)])

    # Read number of comments to pick clustering params
    idx_path = Path("comments_index.json")
    if not idx_path.exists():
        print("Error: comments_index.json not found after embeddings step.", file=sys.stderr)
        sys.exit(3)

    with idx_path.open("r", encoding="utf-8") as f:
        comments = json.load(f)
    n = len(comments)
    auto_mcs, auto_keep = pick_params(n)

    mcs = args.min_cluster_size if args.min_cluster_size is not None else auto_mcs
    keep = args.min_keep if args.min_keep is not None else auto_keep

    print(f"[auto] comments={n} → min_cluster_size={mcs}, min_keep={keep}", flush=True)

    # 2) Clustering
    run([sys.executable, str(clu),
         "--meta", "embeddings.json",
         "--min-cluster-size", str(mcs),
         "--min-keep", str(keep),
         "--output", "clusters.json"])

    # 3) Personas
    run([sys.executable, str(per),
         "--clusters", "clusters.json",
         "--index", "comments_index.json",
         "--output", args.personas_out])

    print(f"✓ Done. Personas written to {args.personas_out}")

if __name__ == "__main__":
    main()
