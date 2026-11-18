# Private local RAG + college predictor (NEET.info demo)

Core task: a locally hosted RAG assistant for NEET.info-style counselling Q&A, with citations and refusals.
Track A: a rank/category/state college predictor. **All data is synthetic** (see `data/DATA_README.md`).
Nothing is sent to a cloud LLM; the main path is an open-weight model served by Ollama on the same machine.

## How it works (one paragraph)
`ingest` turns every CSV row into one chunk and every FAQ section into one chunk. A question first passes
**policy guards** (injection / planted facts, guarantees, forecasts, personal advice, years outside the data), then an
**entity check** (college or course not in the index), then a **relevance gate**. Named-college questions use exact
table filtering; FAQ-style questions use BM25 (optionally fused with embeddings). The model may only use the numbered
sources, and a **post-check** rejects any answer with a missing/invalid citation or a number that isn't in a cited source.
The predictor is plain code over the cut-off table; no LLM chooses colleges.

## Quick start (no GPU, no LLM: verifies retrieval, guards, predictor, eval)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/generate_synthetic.py      # (re)creates the synthetic CSVs
python -m src.ingest                   # builds index/chunks.jsonl
export LLM_BACKEND=extractive
python -m src.app ask "What was the closing rank for Meridian Institute of Medical Sciences, MBBS, General, All India, 2024?"
python -m src.app ask "What is the cut-off for Zenith Medical College MBBS General 2024?"   # -> refusal
python -m src.app predict --rank 6000 --category General --state "West Bengal"
python -m eval.run_eval                # score table -> eval/results_extractive.md
python -m pytest -q
```

## With the local LLM (the real demo)
```bash
ollama pull qwen2.5:7b-instruct        # ~4.7 GB (4-bit). Needs Ollama running: `ollama serve`
export LLM_BACKEND=ollama LLM_MODEL=qwen2.5:7b-instruct
python -m src.app chat                 # or: python -m src.app ask "..."
python -m eval.run_eval                # -> eval/results_ollama.md
```
Optional embeddings: `pip install -r requirements-dense.txt && USE_DENSE=1 python -m src.ingest` then run with `USE_DENSE=1`.

## Docker
```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b-instruct
docker compose run --rm app                              # interactive chat
docker compose run --rm app python -m eval.run_eval
```

## Model and hardware
- Model: Qwen2.5-7B-Instruct (Apache-2.0), 4-bit GGUF via Ollama. Embeddings (optional): BAAI/bge-small-en-v1.5.
- Assumed target: one RTX 6000-class workstation GPU, **assumed 48 GB VRAM** (RTX 6000 Ada). Estimates are in `docs/design_note.md`.
- **Actually run on:** `<FILL IN: e.g. laptop CPU / RTX 3060 12 GB / rented A10 24 GB for N hours>`; measured speed: `<FILL IN tokens/s>`.

## AI tools used (required by the brief)
- Claude (Anthropic): used to draft and review code, tests and docs. `<ADD any other tool you used>`
I reviewed and can explain every module.

## Layout
```
data/   synthetic CSVs + generator + FAQ      src/    ingest, retrieve, guards, generate, pipeline, predictor, app
eval/   questions.jsonl, make_eval.py, run_eval.py, results_*.md      tests/  pytest (LLM stubbed)
docs/   design_note.md     logs/events.jsonl  local audit log (questions, status, reasons, source ids)
```

## Known gaps (read before judging)
- The relevance gate is **heuristic** (word lists + vocabulary check). Unusual phrasings can be refused wrongly; it
  was tuned partly on the dev set, so the dev-set score is optimistic. See "Honest results" in the design note.
- Aggregate questions ("which colleges have the lowest...") are deliberately not answered by RAG; use the predictor.
- Dense retrieval, Docker/GPU compose and the Ollama path were **not exercised by the automated sandbox run**; only the
  extractive backend and stubbed-LLM tests were. Re-run `eval.run_eval` with Ollama and report those numbers.
- Synthetic data: results say nothing about real counselling outcomes.
