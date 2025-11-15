# Helper script for the “data insights”.
# (This doesn’t affect the main API at all.)

import os
import re
import collections
from datetime import datetime
from typing import Any, Dict, List

import requests

MEMBER_MESSAGES_API = os.getenv(
    "MEMBER_MESSAGES_API",
    "https://november7-730026606190.europe-west1.run.app/messages",
)


def fetch_messages() -> List[Dict[str, Any]]:
    resp = requests.get(MEMBER_MESSAGES_API, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        raise RuntimeError("Unexpected data format from upstream API.")
    return data


def parse_timestamp(ts: Any):
    """
    Try a couple of common timestamp formats + unix timestamps.
    If none of them match, just return None.
    """
    if not ts:
        return None

    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts))
        except Exception:
            return None

    if not isinstance(ts, str):
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            continue

    return None


def main() -> None:
    print(f"Fetching messages from {MEMBER_MESSAGES_API} ...")
    raw = fetch_messages()
    print(f"Total raw messages: {len(raw)}")

    cleaned: List[Dict[str, Any]] = []
    for item in raw:
        text = item.get("text") or item.get("message") or ""
        if not isinstance(text, str):
            text = str(text)
        cleaned.append(
            {
                "member_name": item.get("member_name") or item.get("memberName") or "",
                "text": text.strip(),
                "timestamp": item.get("timestamp"),
            }
        )

    # 1) Blank / empty messages
    blanks = [m for m in cleaned if not m["text"]]
    print(f"Empty / blank messages: {len(blanks)}")

    # 2) Timestamps that don’t parse cleanly
    unparseable_ts = [
        m for m in cleaned
        if m["timestamp"] and parse_timestamp(m["timestamp"]) is None
    ]
    print(f"Messages with unparseable timestamps: {len(unparseable_ts)}")

    # 3) Duplicate message bodies
    text_counter = collections.Counter(m["text"] for m in cleaned if m["text"])
    dupes = {txt: count for txt, count in text_counter.items() if count > 1}
    print(f"Distinct message texts that show up more than once: {len(dupes)}")
    print("Top 5 duplicate texts:")
    for txt, count in sorted(dupes.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        snippet = (txt[:77] + "...") if len(txt) > 80 else txt
        print(f"  {count}x  {snippet}")

    # 4) Very rough attempt to spot conflicting numeric “facts” per member.
    fact_tokens = ["car", "cars", "children", "kids", "pets", "dogs", "cats"]
    messages_by_member: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for m in cleaned:
        messages_by_member[m["member_name"]].append(m)

    conflicts = []
    for member, msgs in messages_by_member.items():
        vals = set()
        for m in msgs:
            text = m["text"].lower()
            if any(token in text for token in fact_tokens):
                nums = re.findall(r"\b\d+\b", text)
                if nums:
                    vals.add(tuple(nums))
        if len(vals) > 1:
            conflicts.append((member, vals))

    print(f"Members with potentially conflicting numeric facts: {len(conflicts)}")
    for member, vals in conflicts[:5]:
        print(f"  - {member}: {sorted(vals)}")

if __name__ == "__main__":
    main()
