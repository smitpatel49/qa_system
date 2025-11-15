# FastAPI service that answers natural-language questions about "members"
# using data from the provided /messages API.

# Flow:
#   - fetch messages from upstream
#   - try to identify which member the question is about
#   - if the question names a member we don’t have: say “I don't know”
#   - otherwise, filter to relevant messages and rank them with TF–IDF
#   - extract a short answer from the top messages
#   - for numeric / when / where / favorite questions, if no clean
#     answer is found, return “I don't know based on the available messages.”

import os
import re
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Upstream API URL (can be overridden by env variable).
MEMBER_MESSAGES_API = os.getenv(
    "MEMBER_MESSAGES_API",
    "https://november7-730026606190.europe-west1.run.app/messages",
)

app = FastAPI(
    title="Member Question-Answering Service",
    description="Answers natural-language questions about member messages.",
    version="1.0.0",
)

# Keywords that help detect numeric “facts”
NUMERIC_FACT_KEYWORDS = [
    "car", "cars",
    "child", "children", "kid", "kids",
    "pet", "pets",
    "dog", "dogs",
    "cat", "cats",
]

# Preference-like words for “favorite” style questions
PREFERENCE_KEYWORDS = [
    "favorite", "favourite",
    "love", "loves",
    "like", "likes",
    "prefer", "prefers",
    "preference", "preferences",
]


# --------------------------------------------------------
# Upstream data loading
# --------------------------------------------------------

def fetch_messages() -> List[Dict[str, Any]]:
    """Fetch messages from the upstream API and normalize schema."""
    try:
        resp = requests.get(MEMBER_MESSAGES_API, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch member messages from upstream API: {exc}",
        )

    data = resp.json()
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        raise HTTPException(
            status_code=502,
            detail="Unexpected upstream format: list expected.",
        )

    normalized: List[Dict[str, Any]] = []
    for item in data:
        text = item.get("text") or item.get("message") or ""
        if not isinstance(text, str):
            text = str(text)

        normalized.append(
            {
                "member_id": item.get("member_id") or item.get("user_id") or item.get("memberId"),
                "member_name": item.get("member_name") or item.get("user_name") or item.get("memberName") or "",
                "text": text.strip(),
                "timestamp": item.get("timestamp"),
            }
        )

    normalized = [m for m in normalized if m["text"]]
    if not normalized:
        raise HTTPException(
            status_code=502,
            detail="No usable message text returned from upstream API.",
        )

    return normalized


def build_corpus(messages: List[Dict[str, Any]]) -> List[str]:
    """
    Build TF–IDF documents. Prefix the member name so queries
    like “How many cars does Vikram Desai have?” match better.
    """
    docs: List[str] = []
    for m in messages:
        name = m.get("member_name") or ""
        text = m.get("text") or ""
        docs.append(f"{name}: {text}".strip())
    return docs


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def classify_question_type(q: str) -> str:
    """Classify into numeric/when/where/favorite/other."""
    ql = q.lower()

    if any(x in ql for x in ["how many", "number of", "count of"]):
        return "numeric"

    if any(x in ql for x in ["when", "what date", "what day"]):
        return "when"

    if any(x in ql for x in ["where", "which city", "which country"]):
        return "where"

    if any(x in ql for x in ["favorite", "favourite", "what are", "list of"]):
        return "favorite"

    return "other"


def normalize_for_match(s: str) -> str:
    """
    Lowercase and strip everything that is not a–z into spaces.
    This helps with things like "Amira’s" vs "Amira".
    """
    return re.sub(r"[^a-z]+", " ", s.lower()).strip()


def extract_candidate_names(question: str) -> List[str]:
    """
    Grab capitalised word/phrase chunks as potential names, e.g.
    "Michael", "Vikram Desai", "Layla Kawaguchi".
    """
    parts = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", question)
    seen = set()
    out: List[str] = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def extract_destination_from_question(question: str) -> Optional[str]:
    """
    Try to pull out a destination like 'London', 'Dubai' from a phrase
    such as 'trip to London' or 'traveling to Dubai'.
    """
    m = re.search(
        r"\b(?:to|in|at)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
        question,
    )
    if m:
        return m.group(1)
    return None


# --------------------------------------------------------
# Extraction Logic
# --------------------------------------------------------

def try_extract_numeric_fact(question: str, context: str) -> Optional[str]:
    """Extract numbers from sentences that mention cars/pets/kids/etc."""
    ql = question.lower()
    if "how many" not in ql and "number of" not in ql and "count of" not in ql:
        return None

    sentences = re.split(r"(?<=[.!?])\s+", context)
    for s in sentences:
        sl = s.lower()
        if any(k in sl for k in NUMERIC_FACT_KEYWORDS):
            nums = re.findall(r"\b\d+(?:\.\d+)?\b", s)
            if nums:
                return nums[0]
    return None


def try_extract_answer(question: str, context: str) -> Optional[str]:
    """
    Convert a single context string into a short answer if possible.

    Important bit: for numeric / when / where / favorite questions, if we can’t
    extract a clean value, we return None and let the caller decide
    whether to say "I don't know" rather than echoing a random snippet.
    """
    q_type = classify_question_type(question)
    q = question.lower()
    ctx = context.strip()
    ctx_lower = ctx.lower()

    # ---------- NUMERIC ----------
    numeric = try_extract_numeric_fact(question, ctx)
    if numeric is not None:
        return numeric
    if q_type == "numeric":
        # Clearly a numeric question but no number → no answer here.
        return None

    # ---------- WHEN ----------
    if q_type == "when":
        # If the question itself mentions a destination (e.g. “trip to London”),
        # only accept dates from contexts that also mention that destination.
        dest = extract_destination_from_question(question)
        if dest and dest.lower() not in ctx_lower:
            return None

        date_regex = (
            r"\b("
            r"\d{4}-\d{2}-\d{2}"
            r"|"
            r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?"
            r"|"
            r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
            r"[a-z]*\s+\d{1,2}(?:,\s*\d{2,4})?"
            r")\b"
        )
        m = re.search(date_regex, ctx, flags=re.IGNORECASE)
        if m:
            return m.group(0)

        rel = re.search(
            r"\b(next|this|coming)\s+"
            r"(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            ctx,
            flags=re.IGNORECASE,
        )
        if rel:
            return rel.group(0)

        month_only = re.search(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
            ctx,
            flags=re.IGNORECASE,
        )
        if month_only:
            return month_only.group(0)

        return None

    # ---------- WHERE ----------
    if q_type == "where":
        loc = re.search(
            r"\b(?:to|in|at)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            ctx,
        )
        if loc:
            return loc.group(1)
        return None

    # ---------- FAVORITE ----------
    if q_type == "favorite":
        # Only treat as a valid preference context if we see preference-like language.
        if not any(pk in ctx_lower for pk in PREFERENCE_KEYWORDS):
            # Context doesn’t look like it’s about preferences at all.
            return None

        fav = re.search(
            r"(?:favorite|favourite)\s+([^\.!?]+)",
            ctx,
            flags=re.IGNORECASE,
        )
        if fav:
            return fav.group(1).strip(" .")

        # If that doesn’t work but we know this sentence is about preferences,
        # falling back to the first sentence is okay.
        first = re.split(r"[\.!?]", ctx)[0]
        return first.strip()

    # ---------- OTHER (open-ended) ----------
    if len(ctx) > 280:
        return ctx[:277] + "..."
    return ctx or None


# --------------------------------------------------------
# Main QA pipeline
# --------------------------------------------------------

def answer_question(question: str, k: int = 5) -> str:
    messages = fetch_messages()
    q_type = classify_question_type(question)

    # 1) Try to figure out which member the question is about.
    normalized_q = normalize_for_match(question)
    candidate_names = extract_candidate_names(question)
    candidate_norms = [normalize_for_match(n) for n in candidate_names if normalize_for_match(n)]

    member_filtered: List[Dict[str, Any]] = []
    for m in messages:
        raw_name = m.get("member_name") or ""
        norm_name = normalize_for_match(raw_name)
        if not norm_name:
            continue

        # Match if:
        #  - the normalized member name appears in the normalized question
        #  - OR any candidate name from the question appears inside the member name
        if norm_name and norm_name in normalized_q:
            member_filtered.append(m)
        elif any(c and c in norm_name for c in candidate_norms):
            member_filtered.append(m)

    # If the question clearly mentioned at least one proper name
    # (e.g. "Michael", "Amira’s") but we couldn't match it to any member,
    # then it's safer to say "I don't know" than to answer using
    # someone else's messages.
    if candidate_norms and not member_filtered:
        return "I don't know based on the available messages."

    base_space = member_filtered if member_filtered else messages

    # 2) For numeric questions, optionally filter to messages that mention
    #    cars/kids/pets/etc.
    ql = question.lower()
    wants_numeric_fact = (
        q_type == "numeric"
        or any(k in ql for k in NUMERIC_FACT_KEYWORDS)
    )

    numeric_filtered: List[Dict[str, Any]] = []
    if wants_numeric_fact:
        for m in base_space:
            tl = (m.get("text") or "").lower()
            if any(k in tl for k in NUMERIC_FACT_KEYWORDS):
                numeric_filtered.append(m)

    search_space = numeric_filtered if numeric_filtered else base_space
    if not search_space:
        return "I don't know based on the available messages."

    # 3) Rank by TF–IDF similarity.
    docs = build_corpus(search_space)
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_matrix = vectorizer.fit_transform(docs)
    query_vec = vectorizer.transform([question])
    similarities = cosine_similarity(query_vec, doc_matrix)[0]
    top_indices = similarities.argsort()[::-1][:k]

    # 4) Try to extract answers from the top-k contexts.
    answers: List[str] = []
    for idx in top_indices:
        ctx = search_space[int(idx)]["text"]
        candidate = try_extract_answer(question, ctx)
        if candidate:
            answers.append(candidate)

    # 5) If nothing extracted and the question is clearly fact-like,
    #    be honest instead of hallucinating.
    if not answers:
        if q_type in {"numeric", "when", "where", "favorite"}:
            return "I don't know based on the available messages."
        # For more open questions, returning the best snippet is still fine.
        best_idx = int(top_indices[0])
        return search_space[best_idx]["text"]

    return answers[0]


# --------------------------------------------------------
# FastAPI endpoints
# --------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ask")
def ask(
    q: str = Query(..., description="Natural-language question about members"),
) -> Dict[str, str]:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' must not be empty.")
    answer = answer_question(q.strip())
    return {"answer": answer}
