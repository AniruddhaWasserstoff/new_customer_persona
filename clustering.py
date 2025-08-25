#!/usr/bin/env python3
"""
clustering.py — cluster comment embeddings into themes (robust write)

Inputs:
  embeddings.json (writes by embeddings.py)
  comments_index.json
  embeddings.npy

Outputs:
  clusters.json (always written, even if zero clusters)
"""

import json, argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import re
from collections import Counter, defaultdict

USE_HDBSCAN = True
try:
    import hdbscan  # type: ignore
except Exception:
    USE_HDBSCAN = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

def load_inputs(meta_path: Path):
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    vecs = np.load(meta["vectors_file"])
    with Path(meta["index_file"]).open("r", encoding="utf-8") as f:
        idx = json.load(f)
    return vecs, idx, meta

def _keywords(texts: List[str], top_k=8) -> List[str]:
    if not texts:
        return []
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=5000, stop_words="english")
    X = vec.fit_transform(texts)
    means = np.asarray(X.mean(axis=0)).ravel()
    feats = np.array(vec.get_feature_names_out())
    top_idx = means.argsort()[::-1][:top_k]
    return feats[top_idx].tolist()

def cluster_with_hdbscan(embs: np.ndarray, min_cluster_size=8, min_samples=None) -> np.ndarray:
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples or max(2, min_cluster_size//2),
                         metric="euclidean")
    labels = cl.fit_predict(embs)
    return labels

def cluster_with_agglo(embs: np.ndarray, distance_threshold=0.6) -> np.ndarray:
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage="average")
    labels = agg.fit_predict(embs)
    return labels

def summarize_clusters(labels: np.ndarray, comments: List[Dict[str, Any]], top_k_keywords=8, min_keep=4) -> Dict[str, Any]:
    clusters = defaultdict(list)
    for lbl, c in zip(labels, comments):
        if lbl == -1:
            continue
        clusters[int(lbl)].append(c)

    out = {"total_clusters": 0, "clusters": []}
    for lbl, items in clusters.items():
        if len(items) < min_keep:
            continue
        texts = [it["comment_text"] for it in items]
        kws = _keywords(texts, top_k=top_k_keywords)
        brands = Counter([it["brand"] for it in items]).most_common()
        questions = Counter([it["question"] for it in items]).most_common()
        exemplar = max(items, key=lambda it: it.get("likeCount", 0))
        out["clusters"].append({
            "cluster_id": int(lbl),
            "name": ", ".join(kws[:3]) if kws else f"Theme {lbl}",
            "slug": f"cluster-{lbl}",
            "size": len(items),
            "top_keywords": kws,
            "top_brands": brands[:5],
            "top_questions": questions[:5],
            "representative_comment": {
                "text": exemplar["comment_text"],
                "likeCount": exemplar.get("likeCount", 0),
                "video_url": exemplar.get("video_url", ""),
                "author": exemplar.get("author", "Anonymous"),
            },
            "indices": [it["idx"] for it in items],
        })
    out["total_clusters"] = len(out["clusters"])
    return out

def main():
    ap = argparse.ArgumentParser(description="Cluster comment embeddings into themes")
    ap.add_argument("--meta", default="embeddings.json", help="Path to embeddings.json")
    ap.add_argument("--min-cluster-size", type=int, default=8, help="Min cluster size (HDBSCAN)")
    ap.add_argument("--distance-threshold", type=float, default=0.6, help="Fallback agglomerative distance threshold")
    ap.add_argument("--min-keep", type=int, default=4, help="Minimum items to keep a cluster")
    ap.add_argument("--output", default="clusters.json", help="Output file")
    args = ap.parse_args()

    vecs, idx, meta = load_inputs(Path(args.meta))

    if USE_HDBSCAN:
        print(f"Clustering with HDBSCAN (min_cluster_size={args.min_cluster_size})")
        labels = cluster_with_hdbscan(vecs, min_cluster_size=args.min_cluster_size)
    else:
        print(f"HDBSCAN not available, using AgglomerativeClustering (distance_threshold={args.distance_threshold})")
        labels = cluster_with_agglo(vecs, distance_threshold=args.distance_threshold)

    summary = summarize_clusters(labels, idx, min_keep=args.min_keep)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    kept = sum(c["size"] for c in summary["clusters"])
    print(f"✓ Clusters written to {args.output} — kept {summary['total_clusters']} clusters covering {kept}/{len(idx)} comments")

if __name__ == "__main__":
    main()
