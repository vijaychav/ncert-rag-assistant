# NCERT RAG Assistant

A question-answering assistant over NCERT Science textbooks, built from scratch and measured at every step.

This project grows over time: semantic search → full RAG pipeline → retrieval
improvements → self-hosted serving → evaluation framework → multimodal input.

## Status

| Phase | Focus | Status |
|---|---|---|
| 2 | RAG from scratch + evaluation | In progress |
| 3 | Retrieval quality, tool use | Not started |
| 4 | Serving and inference | Not started |
| 5 | Evaluation depth | Not started |
| 6 | LoRA and polish | Not started |
| 7 | Multimodal | Not started |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

NCERT PDFs are not included in this repo. Download the chapters from
https://ncert.nic.in and place them in `data/raw/`, named `ch01.pdf`, `ch02.pdf`, etc.

## Usage

```bash
python src/ingest.py     # PDFs → data/passages.json
python src/embed.py      # passages → data/embeddings.npy
python src/search.py     # interactive search
```

## Results

Results are logged in `results/`.