"""Semantic search in plain NumPy.

Usage:
    python src/search.py
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
PASSAGES_PATH = Path("/Users/vijayc/LLM/ncert-rag-assistant/data/passages.json")
EMBEDDINGS_PATH = Path("/Users/vijayc/LLM/ncert-rag-assistant/data/embeddings.npy")


def search(query, model, passages, embeddings, search_type, k=5):
    """Return the top-k passages for a query as a list of (score, passage_dict).

    Hints:
    - Embed the query with the SAME model and SAME normalization as the passages.
    - (N, 384) and (384,) -> one matrix operation gives all N scores. No loop.
    - np.argsort gives you the order. Check the sort direction.
    - Keep each score linked to its passage.
    """

    if search_type == 'COSINE':
        model.similarity_fn_name = SimilarityFunction.COSINE
    elif search_type == 'DOT':
        model.similarity_fn_name = SimilarityFunction.DOT_PRODUCT
    elif search_type == 'L2':
        model.similarity_fn_name = SimilarityFunction.EUCLIDEAN
    else:
        raise ValueError("Invalid Search type")
    
    k = min(k, len(embeddings))
    emb_query = model.encode_query(query, normalize_embeddings=True)
    similarity = model.similarity(emb_query, embeddings)[0].detach().cpu().numpy()
    top_k = np.argsort(similarity)[::-1][:k]
    # print(top_k)
    results = [(similarity[idx], passages[idx]) for idx in top_k]
    # print(results)
    return results


def main():
    model = SentenceTransformer(MODEL_NAME)
    passages = json.loads(PASSAGES_PATH.read_text())
    embeddings = np.load(EMBEDDINGS_PATH)  # shape (N, 384)

    print("Type a question (empty line to quit).")
    while True:
        query = input("\n> ").strip()
        if not query:
            break
        for search_type in ['COSINE', 'DOT', 'L2']:
            print(f"###########\n\nSearch_type: {search_type}")
            for score, p in search(query, model, passages, embeddings, search_type):
                print(f"\n[{score:.3f}] {p['id']}\n{p['text'][:300]}")


if __name__ == "__main__":
    main()