"""Embed all passages and save them to disk.

Usage:
    python src/embed.py                  
    python src/embed.py --no-normalize   
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
PASSAGES_PATH = Path("/Users/vijayc/LLM/ncert-rag-assistant/data/passages.json")
EMBEDDINGS_PATH = Path("/Users/vijayc/LLM/ncert-rag-assistant/data/")


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--no-normalize", action="store_true")
    # args = parser.parse_args()
    # normalize = not args.no_normalize
    normalize = True

    passages = json.loads(PASSAGES_PATH.read_text())
    texts = [p["text"] for p in passages]

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, normalize_embeddings=normalize, show_progress_bar=True)

    out_path = Path(EMBEDDINGS_PATH/"embeddings.npy" if normalize else EMBEDDINGS_PATH/"embeddings_raw.npy")
    np.save(out_path, emb)
    print(f"Saved embeddings with shape {emb.shape} to {out_path}")


if __name__ == "__main__":
    main()
