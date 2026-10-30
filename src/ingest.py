import json
import re
from pathlib import Path

import fitz  # PyMuPDF

RAW_DIR = Path("/Users/vijayc/LLM/ncert-rag-assistant/data/raw")
OUT_PATH = Path("/Users/vijayc/LLM/ncert-rag-assistant/data/passages.json")


def extract_pages(pdf_path):
    """Return a list of (page_number, text) for one PDF. Page numbers start at 1."""
    doc = fitz.open(pdf_path)
    return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


def clean_text(text):
    """Light cleanup only. Don't aim for perfection.

    
    - Join lines that were broken mid-sentence
    - Remove obvious page headers/footers you notice in the output
    - Collapse repeated whitespace
    """
    text = re.sub(r'\nn\n|\nReprint 2026-27\n|\n|\uf8e7', ' ', text)
    text = re.sub(r'(\w+ \d+(?:\.\d+)?\s*)','',text)
    text = re.sub(r'\b\d+\.','',text)
    text = re.sub(r'\s+',' ', text)
    
    return text


def split_into_passages(pages, chapter, target_words=150):
    """Split cleaned page text into passages of roughly 100-200 words.

    First try paragraph breaks. If extraction lost them, fall back to
    ~target_words chunks that end at a sentence boundary.

    pages: list of (page_number, cleaned_text)
    Return a list of dicts with keys: id, chapter, page, text.
    Use ids like f"ch{chapter}_p{page}_{n}".

    
    """
    out = []
    for page, page_text in pages:
        n = 0
        lines = page_text.split('. ')
        for line in lines:
            words = line.split(' ')
            for i in range(0, len(words), target_words):
                chunk = words[i: i+target_words]
                passage = ' '.join(chunk)

                if i + target_words > len(words):
                    passage += '.'

                out.append({
                    "id": f"ch{chapter}_p{page}_{n}",
                    "chapter": chapter,
                    "page": page,
                    "text": passage,
                })
                n += 1

    return out


def main():
    passages = []
    for pdf_path in sorted(RAW_DIR.glob("ch*.pdf")):
        # print(pdf_path)
        chapter = int(re.search(r"\d+", pdf_path.stem).group())
        # print(chapter)
        pages = [(num, clean_text(text)) for num, text in extract_pages(pdf_path)]
        # print(pages)
        passages.extend(split_into_passages(pages, chapter))

    # print(passages)
    OUT_PATH.write_text(json.dumps(passages, indent=2, ensure_ascii=False))
    print(f"Wrote {len(passages)} passages to {OUT_PATH}")


if __name__ == "__main__":
    main()