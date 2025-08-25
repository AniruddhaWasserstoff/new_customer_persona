
import os
import json
import logging
import argparse
import time
import socket
import re
from urllib.parse import urlparse
from datetime import datetime

from dotenv import load_dotenv
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sentence_transformers import SentenceTransformer, util
import httplib2

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("youtube_competitor_scraper")

# -------------------------
# Config
# -------------------------
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY not found in .env file.")

HTTP_TIMEOUT_SECONDS = int(os.getenv("YT_HTTP_TIMEOUT", "30"))
RETRIES = int(os.getenv("YT_RETRIES", "3"))
MIN_SIMILARITY = float(os.getenv("YT_MIN_SIMILARITY", "0.25"))  # stricter
MAX_COMMENTS_PAGE = int(os.getenv("YT_MAX_COMMENTS_PAGE", "100"))
MIN_WORDS = int(os.getenv("YT_MIN_WORDS", "6"))
MIN_CHARS = int(os.getenv("YT_MIN_CHARS", "25"))

_http = httplib2.Http(timeout=HTTP_TIMEOUT_SECONDS)
yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, http=_http)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

STOPWORDS = set(
    """the a an and or for to of in on with by from into at over under about as is are was were be been being this that those these it its
    how what why when where who which does do did has have had their them they you we i your our ours his her hers him he she
    brand brands product products pricing price compare comparison unique selling proposition usp reviews review testimonial testimonials pain points"""
    .split()
)

BRAND_ALIASES = {
    "brother vellies": ["brother vellies", "aurora james", "bvellies"],
    "able": ["able carry", "able bags", "able brand", "able leather"],
    "soko": ["soko jewelry", "shop soko", "soko kenya"],
    "accompany": ["accompany", "accompanyus", "accompany shop", "accompany fair trade"],
    "mz fair trade": ["mz fair trade", "mz fairtrade", "mz made by"],
}

NEGATIVE_KWS = {
    "soko": ["grand seiko", "seiko", "butcher", "restaurant", "rooftop"],
    "accompany": ["bob seger", "song", "reaction", "kneecap", "film", "movie"],
    "mz fair trade": ["zainuddin", "mz kh", "k.h zainuddin", "sermon"],
}

CATEGORY_KWS = [
    "bag", "leather", "handbag", "wallet", "tote", "backpack",
    "jewelry", "earrings", "necklace", "bracelet", "ring",
    "fair trade", "artisan", "craft", "handcrafted", "ethical", "sustainable",
    "review", "unboxing", "brand", "story", "pricing", "price", "testimonial",
]

GENERIC_TRASH_PATTERNS = [
    r"subscribe", r"follow me", r"check (out )?my channel", r"giveaway",
    r"http[s]?://", r"www\.", r"#\w+", r"@\w+", r"\b(?:like|share|comment)\b",
    r"^nice( video)?!?$", r"^cool!?$", r"^awesome!?$", r"^first!?$", r"^lol!?$"
]

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def tolc(s: str) -> str:
    return normalize(s).lower()

def domain_from_url(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        return ".".join(host.split(".")[-2:]) if host else ""
    except Exception:
        return ""

# -------------------------
# Retry helper
# -------------------------
def _is_retryable_http_error(err: HttpError) -> bool:
    status = getattr(getattr(err, "resp", None), "status", None)
    return status in (500, 502, 503, 504)

def execute_with_retry(request, tries: int, backoff: float = 2.0, what: str = "YouTube API"):
    last_err = None
    for attempt in range(tries):
        try:
            return request.execute()
        except HttpError as e:
            if not _is_retryable_http_error(e):
                raise
            last_err = e
            logger.warning("%s transient HTTP error (attempt %d/%d): %s", what, attempt + 1, tries, e)
        except (socket.timeout, TimeoutError, OSError) as e:
            last_err = e
            logger.warning("%s network/timeout (attempt %d/%d): %s", what, attempt + 1, tries, e)
        except Exception as e:
            last_err = e
            logger.warning("%s error (attempt %d/%d): %s", what, attempt + 1, tries, e)
        if attempt < tries - 1:
            time.sleep(backoff ** attempt)
    raise last_err

# -------------------------
# Query building & filtering
# -------------------------
def extract_keywords(text: str):
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", tolc(text))
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def build_search_queries(brand: str, website: str, question: str, max_q=6):
    brand_lc = tolc(brand)
    aliases = BRAND_ALIASES.get(brand_lc, [brand_lc])
    dom = domain_from_url(website)

    qk = extract_keywords(question)
    key_terms = (qk[:4] + CATEGORY_KWS)[:8]

    queries = []
    for alias in aliases:
        for kt in key_terms:
            queries.append(f'"{alias}" {kt} review')
        queries.extend([
            f'"{alias}" testimonial',
            f'"{alias}" customer review',
            f'"{alias}" pricing',
            f'"{alias}" artisan',
            f'"{alias}" sustainable',
        ])
    if dom:
        queries.append(f'"{brand}" "{dom}"')
    # dedupe & cap
    seen, deduped = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
        if len(deduped) >= max_q:
            break
    return deduped

def is_video_on_topic(video_snippet: dict, brand: str) -> bool:
    brand_lc = tolc(brand)
    aliases = BRAND_ALIASES.get(brand_lc, [brand_lc])
    negatives = NEGATIVE_KWS.get(brand_lc, [])
    title = tolc(video_snippet.get("title", ""))
    desc = tolc(video_snippet.get("description", ""))
    channel = tolc(video_snippet.get("channelTitle", ""))
    text = f"{title} {desc} {channel}"
    if not any(a in text for a in aliases):
        return False
    if any(neg in text for neg in negatives):
        return False
    return True

def search_video_ids(queries, brand, max_results=5, tries=RETRIES):
    found = []
    for q in queries:
        try:
            req = yt.search().list(part="snippet", q=q, type="video", maxResults=min(10, max_results), order="relevance", regionCode="US")
            resp = execute_with_retry(req, tries=tries, what=f"search('{q}')")
            for item in resp.get("items", []):
                if is_video_on_topic(item.get("snippet", {}), brand):
                    found.append(item["id"]["videoId"])
            if len(found) >= max_results:
                break
        except Exception as e:
            logger.warning("Search failed for '%s': %s", q, e)
            continue
    # fallback if none
    if not found:
        try:
            q = f'"{brand}" review'
            req = yt.search().list(part="id", q=q, type="video", maxResults=max_results, order="relevance", regionCode="US")
            resp = execute_with_retry(req, tries=tries, what=f"search('{q}')")
            found = [it["id"]["videoId"] for it in resp.get("items", [])]
        except Exception:
            pass
    # Dedupe & cap
    return list(dict.fromkeys(found))[:max_results]

# -------------------------
# API helpers
# -------------------------
def fetch_video_details(video_ids, tries=RETRIES):
    if not video_ids:
        return []
    try:
        req = yt.videos().list(part="snippet,statistics", id=",".join(video_ids), maxResults=len(video_ids))
        resp = execute_with_retry(req, tries=tries, what="videos.list")
    except Exception as e:
        logger.warning("videos.list failed: %s", e)
        return []
    out = []
    for it in resp.get("items", []):
        sn, st = it.get("snippet", {}), it.get("statistics", {})
        out.append({
            "video_id": it["id"],
            "title": sn.get("title", ""),
            "url": f"https://youtu.be/{it['id']}",
            "viewCount": int(st.get("viewCount", 0) or 0),
            "likeCount": int(st.get("likeCount", 0) or 0),
            "commentCount": int(st.get("commentCount", 0) or 0),
            "publishedAt": sn.get("publishedAt", ""),
            "channelTitle": sn.get("channelTitle", ""),
            "description": sn.get("description", ""),
        })
    out.sort(key=lambda v: v["viewCount"], reverse=True)
    return out

def looks_meaningful(text: str) -> bool:
    """Heuristics to remove spam/low-signal comments."""
    if len(text) < MIN_CHARS:
        return False
    words = text.split()
    if len(words) < MIN_WORDS:
        return False
    # too many non-alphanumeric characters
    non_alnum_ratio = sum(1 for ch in text if not ch.isalnum() and not ch.isspace()) / max(1, len(text))
    if non_alnum_ratio > 0.4:
        return False
    tlc = tolc(text)
    for pat in GENERIC_TRASH_PATTERNS:
        if re.search(pat, tlc):
            return False
    return True

def fetch_relevant_comments(video_id, brand: str, question: str, max_comments=5, min_similarity=MIN_SIMILARITY, tries=RETRIES):
    try:
        req = yt.commentThreads().list(part="snippet", videoId=video_id, order="relevance", maxResults=min(MAX_COMMENTS_PAGE, 100), textFormat="plainText")
        resp = execute_with_retry(req, tries=tries, what=f"commentThreads.list({video_id})")
    except HttpError as e:
        code = getattr(getattr(e, "resp", None), "status", None)
        if code == 403:
            logger.info("Comments disabled/forbidden for %s", video_id)
        else:
            logger.warning("Comments HTTP error for %s: %s", video_id, e)
        return []
    except Exception as e:
        logger.warning("Comments fetch failed for %s: %s", video_id, e)
        return []

    raw = []
    for it in resp.get("items", []):
        try:
            sn = it["snippet"]["topLevelComment"]["snippet"]
            txt = normalize(sn.get("textDisplay", ""))
            if not looks_meaningful(txt):
                continue
            # strict English only: if detection fails, skip
            try:
                if detect(txt) != "en":
                    continue
            except LangDetectException:
                continue
            raw.append({
                "text": txt,
                "author": sn.get("authorDisplayName", "Anonymous"),
                "likeCount": sn.get("likeCount", 0),
                "publishedAt": sn.get("publishedAt", ""),
            })
        except Exception:
            continue
    if not raw:
        return []

    # Must connect to the question: require overlap with question keywords
    q_keywords = set(extract_keywords(question))
    filtered = [c for c in raw if any(k in tolc(c["text"]) for k in q_keywords)]
    if not filtered:
        return []

    # Semantic rerank; require min_similarity and return up to max_comments
    try:
        q_emb = embedder.encode(question, convert_to_tensor=True)
        c_emb = embedder.encode([c["text"] for c in filtered], convert_to_tensor=True)
        sims = util.cos_sim(q_emb, c_emb)[0]
        pairs = list(zip(filtered, sims))
        pairs.sort(key=lambda x: float(x[1]), reverse=True)
        out = []
        for c, s in pairs:
            score = float(s)
            if score >= min_similarity:
                d = dict(c)
                d["similarity_score"] = score
                out.append(d)
                if len(out) >= max_comments:
                    break
        return out  # strict: no fallback if none meet threshold
    except Exception as e:
        logger.warning("Similarity calc failed: %s", e)
        return []

# -------------------------
# Processing
# -------------------------
def process_single_competitor(brand_name, website, questions, max_videos=5, max_comments=5):
    logger.info("Processing competitor: %s", brand_name)
    comp = {"brand": brand_name, "website": website, "total_questions": len(questions), "results": []}
    for idx, q in enumerate(questions, 1):
        logger.info("Q %d/%d: %s", idx, len(questions), (q[:100] + "..."))
        queries = build_search_queries(brand_name, website, q)
        video_ids = search_video_ids(queries, brand_name, max_results=max_videos)
        videos = fetch_video_details(video_ids)
        qres = {"question": q, "videos_found": len(videos), "videos": []}
        for v in videos:
            logger.info("Video: %s", (v["title"][:60] + "..."))
            comments = fetch_relevant_comments(v["video_id"], brand_name, q, max_comments=max_comments)
            qres["videos"].append({
                "video": {k: v[k] for k in ["video_id","title","url","viewCount","likeCount","commentCount","publishedAt","channelTitle"]},
                "top_comments": comments,
                "relevant_comments_count": len(comments),
            })
        comp["results"].append(qres)
    return comp

def process_all_competitors(competitors_data, followup_data, max_videos=5, max_comments=5):
    results = {
        "scraping_timestamp": datetime.now().isoformat(),
        "total_competitors": len(competitors_data.get("competitors", [])),
        "parameters": {
            "max_videos_per_question": max_videos,
            "max_comments_per_video": max_comments,
            "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
            "retries": RETRIES,
            "min_similarity": MIN_SIMILARITY,
            "min_words": MIN_WORDS,
            "min_chars": MIN_CHARS,
        },
        "competitors_data": [],
    }
    for comp in competitors_data.get("competitors", []):
        brand = comp["brand"].strip("*").strip()
        website = comp.get("website", "")
        questions = followup_data.get(f"**{brand}**") or followup_data.get(brand)
        if not questions:
            logger.warning("No questions for brand: %s", brand)
            continue
        cres = process_single_competitor(brand, website, questions, max_videos=max_videos, max_comments=max_comments)
        results["competitors_data"].append(cres)
    return results

# -------------------------
# IO helpers
# -------------------------
def save_results(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to: %s", output_file)
    total_videos = sum(len(q["videos"]) for comp in results["competitors_data"] for q in comp["results"])
    total_comments = sum(v["relevant_comments_count"] for comp in results["competitors_data"] for q in comp["results"] for v in q["videos"])
    print(f"\\n📊 Summary: {len(results['competitors_data'])} competitors • {total_videos} videos • {total_comments} relevant comments → {output_file}")

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------
# CLI
# -------------------------
def main():
    global HTTP_TIMEOUT_SECONDS, RETRIES, MIN_SIMILARITY, MAX_COMMENTS_PAGE, MIN_WORDS, MIN_CHARS, _http, yt

    parser = argparse.ArgumentParser(description="YouTube Competitor Q&A Comment Extractor (English-only & meaningful)")
    parser.add_argument("--competitors", required=True, help="Path to working_competitors.json")
    parser.add_argument("--followup", required=True, help="Path to followups.json")
    parser.add_argument("--output", default="youtube_analysis.json", help="Output file path")
    parser.add_argument("--max-videos", type=int, default=5, help="Max videos per question")
    parser.add_argument("--max-comments", type=int, default=5, help="Max comments per video")
    parser.add_argument("--timeout", type=int, default=HTTP_TIMEOUT_SECONDS, help="HTTP timeout (seconds)")
    parser.add_argument("--retries", type=int, default=RETRIES, help="Retries for transient errors")
    parser.add_argument("--min-sim", type=float, default=MIN_SIMILARITY, help="Min similarity for comments [0-1]")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS, help="Min words in a comment")
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS, help="Min characters in a comment")
    args = parser.parse_args()

    # Overrides
    if args.timeout != HTTP_TIMEOUT_SECONDS:
        HTTP_TIMEOUT_SECONDS = args.timeout
        _http = httplib2.Http(timeout=HTTP_TIMEOUT_SECONDS)
        yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, http=_http)
    RETRIES = args.retries
    MIN_SIMILARITY = args.min_sim
    MIN_WORDS = args.min_words
    MIN_CHARS = args.min_chars

    print("🎬 YouTube Competitor Q&A Comment Extractor (English-only & meaningful)\n")
    comps = load_json_file(args.competitors)
    follows = load_json_file(args.followup)

    if "competitors" not in comps:
        raise ValueError("Invalid competitors file: missing 'competitors' key.")

    tq = sum(len(v) for v in follows.values())
    print(f"📊 Config:\n • Competitors: {len(comps['competitors'])}\n • Total followup questions: {tq}\n • Videos/Q: {args.max_videos} • Comments/V: {args.max_comments}\n • Timeout: {HTTP_TIMEOUT_SECONDS}s • Retries: {RETRIES}\n • MinSim: {MIN_SIMILARITY} • MinWords: {MIN_WORDS} • MinChars: {MIN_CHARS}\n • Output: {args.output}\n")

    print("🔍 Running...\n")
    results = process_all_competitors(comps, follows, max_videos=args.max_videos, max_comments=args.max_comments)
    save_results(results, args.output)
    print("\n✅ Done.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # Interactive quick run
        try:
            cf = input("📁 Path to competitors.json: ").strip()
            ff = input("📁 Path to followup.json: ").strip()
            of = input("💾 Output file (default: youtube_analysis.json): ").strip() or "youtube_analysis.json"
            mv = int(input("🎯 Max videos per question (default: 5): ").strip() or "5")
            mc = int(input("💬 Max comments per video (default: 5): ").strip() or "5")
            to = int(input(f"⏱  HTTP timeout seconds (default: {HTTP_TIMEOUT_SECONDS}): ").strip() or str(HTTP_TIMEOUT_SECONDS))
            rt = int(input(f"🔁 Retries (default: {RETRIES}): ").strip() or str(RETRIES))
            ms = float(input(f"📐 Min similarity (default: {MIN_SIMILARITY}): ").strip() or str(MIN_SIMILARITY))
            mw = int(input(f"📝 Min words (default: {MIN_WORDS}): ").strip() or str(MIN_WORDS))
            mc2 = int(input(f"🔡 Min chars (default: {MIN_CHARS}): ").strip() or str(MIN_CHARS))
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        # apply overrides
        if to != HTTP_TIMEOUT_SECONDS:
            HTTP_TIMEOUT_SECONDS = to
            _http = httplib2.Http(timeout=HTTP_TIMEOUT_SECONDS)
            yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, http=_http)
        RETRIES = rt
        MIN_SIMILARITY = ms
        MIN_WORDS = mw
        MIN_CHARS = mc2
        comps = load_json_file(cf)
        follows = load_json_file(ff)
        results = process_all_competitors(comps, follows, max_videos=mv, max_comments=mc)
        save_results(results, of)
        print("\n✅ Done.")
    else:
        main()
