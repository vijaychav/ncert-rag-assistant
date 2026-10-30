"""Semantic search in plain NumPy.

Usage:
    python src/search.py
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
PASSAGES_PATH = Path("data/passages.json")
EMB_PATH = Path("data/embeddings.npy")


def search(query, model, passages, embeddings, k=5):
    """Return the top-k passages for a query as a list of (score, passage_dict).

    TODO (Step 4): implement this yourself.
    Hints:
    - Embed the query with the SAME model and SAME normalization as the passages.
    - (N, 384) and (384,) -> one matrix operation gives all N scores. No loop.
    - np.argsort gives you the order. Check the sort direction.
    - Keep each score linked to its passage.
    """
    raise NotImplementedError


# TODO (Step 5): add cosine, dot product, and L2 versions and compare their top-5.


def main():
    model = SentenceTransformer(MODEL_NAME)
    passages = json.loads(PASSAGES_PATH.read_text())
    embeddings = np.load(EMB_PATH)  # shape (N, 384)

    print("Type a question (empty line to quit).")
    while True:
        query = input("\n> ").strip()
        if not query:
            break
        for score, p in search(query, model, passages, embeddings):
            print(f"\n[{score:.3f}] {p['id']}\n{p['text'][:300]}")


if __name__ == "__main__":
    main()
