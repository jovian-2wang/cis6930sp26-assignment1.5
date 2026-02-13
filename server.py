import os
import random
from typing import Any, Dict, List

from dotenv import load_dotenv
from datasets import load_dataset
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("emotion-mcp")

DATASET_NAME = os.getenv("HF_DATASET", "dair-ai/emotion")
SPLIT = os.getenv("HF_SPLIT", "train")
HF_TOKEN = os.getenv("HF_TOKEN")  # put real token in .env (DO NOT COMMIT)


def load_ds():
    ds = load_dataset(DATASET_NAME, split=SPLIT, token=HF_TOKEN)
    label_names = ds.features["label"].names  # index -> emotion string
    return ds, label_names


DS, LABEL_NAMES = load_ds()


def format_row(row: Dict[str, Any]) -> Dict[str, Any]:
    lbl = row["label"]
    return {
        "text": row["text"],
        "label": lbl,
        "emotion": LABEL_NAMES[lbl],
    }


@mcp.tool()
def get_sample(n: int) -> List[Dict[str, Any]]:
    """Get n random samples from the dataset."""
    if n <= 0:
        return []
    n = min(n, len(DS))
    idxs = random.sample(range(len(DS)), k=n)
    return [format_row(DS[i]) for i in idxs]


@mcp.tool()
def count_by_emotion(emotion: str) -> Dict[str, Any]:
    """Count samples for a specific emotion string (e.g., joy)."""
    if not emotion:
        return {"emotion": emotion, "count": 0, "error": "emotion must be non-empty"}

    emo = emotion.strip().lower()
    names_lower = [x.lower() for x in LABEL_NAMES]
    if emo not in names_lower:
        return {"emotion": emo, "count": 0, "error": f"Unknown emotion. Valid: {LABEL_NAMES}"}

    label_id = names_lower.index(emo)
    labels = DS["label"]
    count = sum(1 for x in labels if x == label_id)
    return {"emotion": emo, "label_id": label_id, "count": count}


@mcp.tool()
def search_text(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search for query substring in dataset text (case-insensitive)."""
    q = (query or "").strip()
    if not q:
        return {"query": q, "limit": limit, "matches": 0, "results": [], "error": "query must be non-empty"}

    limit = max(1, min(int(limit), 50))
    qlower = q.lower()

    results = []
    for i in range(len(DS)):
        text = DS[i]["text"]
        if qlower in text.lower():
            results.append({"index": i, **format_row(DS[i])})
            if len(results) >= limit:
                break

    return {"query": q, "limit": limit, "matches": len(results), "results": results}


@mcp.tool()
def analyze_emotion_distribution() -> Dict[str, Any]:
    """Return counts + percent for each emotion label."""
    total = len(DS)
    counts = {name: 0 for name in LABEL_NAMES}
    for lbl in DS["label"]:
        counts[LABEL_NAMES[lbl]] += 1

    stats = []
    for name in LABEL_NAMES:
        c = counts[name]
        pct = (c / total * 100.0) if total else 0.0
        stats.append({"emotion": name, "count": c, "percent": round(pct, 3)})

    stats.sort(key=lambda x: x["count"], reverse=True)
    return {"dataset": DATASET_NAME, "split": SPLIT, "total": total, "stats": stats}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
