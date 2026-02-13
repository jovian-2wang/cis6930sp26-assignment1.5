# cis6930sp26-assignment1.5

## Overview

This project implements an MCP server with 4 tools that process the `dair-ai/emotion` dataset:

- `get_sample(n)` – Return n random samples  
- `count_by_emotion(emotion)` – Count samples for a specific emotion  
- `search_text(query, limit)` – Search text in dataset  
- `analyze_emotion_distribution()` – Return emotion statistics  

The dataset is loaded once at server startup and cached locally.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uv sync
python server.py
