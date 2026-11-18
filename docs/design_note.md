# Design note: private local RAG + college predictor (NEET.info)
*Sejal Likhar · CGPRR AI Developer assignment · Items in `<ANGLE BRACKETS>` must be filled with your own measurements before submitting.*

## 1. Architecture
```
question -> policy guards -> entity check -> relevance gate -> retrieval -> local LLM -> post-check -> answer + citations
              (inject/guarantee/   (unknown college/   (words found      (exact table     (Qwen2.5-7B,   (valid [S#]; every
               forecast/advice/     unsupported         nowhere in the   filter, or BM25   Ollama)        number is in a
               year range)          course)             corpus)          [+embeddings])                   cited source)
```
Index = 1,798 chunks: one per cut-off row (1,680), one per seat-matrix row (112), one per FAQ section (6). Row-level chunks keep
every figure traceable to `file#line`. Refusal is **layered**: five of the six checks are deterministic code, so a weak model cannot talk its way past them.
The predictor (Track A) is deterministic code over the same table: a candidate's rank is compared with the latest closing
rank (rank/closing ≤ 0.85 safe, ≤ 1.00 target, ≤ 1.15 reach); State-quota rows appear only for home-state candidates. The LLM never selects colleges.
Every option lists the three yearly rows it came from.

## 2. Model choice
Qwen2.5-7B-Instruct (Apache-2.0), 4-bit, served by Ollama at temperature 0. Reasons: strong instruction-following and citation
discipline at 7B, permissive licence for a commercial group, fits a single card with large headroom. Retrieval needs no GPU
(BM25); optional bge-small embeddings add ~130 MB. Alternatives to compare if time allows: Llama-3.1-8B-Instruct, Qwen2.5-14B.

## 3. GPU and memory estimate (assumption stated)
Assumed card: **RTX 6000 Ada, 48 GB VRAM, ~960 GB/s bandwidth** (the brief does not fix the generation).
| Item | Estimate |
|---|---|
| Weights, FP16 (7.6B params x 2 B) | ~15 GB |
| Weights, 4-bit (Ollama default) | ~4.7 GB |
| KV cache (28 layers, 4 KV heads, head dim 128, FP16) | ~56 KB/token → 0.23 GB at 4k ctx, ~1.9 GB at 32k |
| Embedding model (optional) | ~0.13 GB |
Total ≈ 5-6 GB (4-bit) or ≈ 16-17 GB (FP16): both fit in 48 GB with room for several concurrent users or a 14B model.
**Throughput (estimate, not measured on that card):** decoding is memory-bandwidth bound, so tokens/s ≈ bandwidth ÷ bytes read per token:
960 ÷ 15 ≈ 63 t/s theoretical for FP16 (expect ~40 in practice), 960 ÷ 4.7 ≈ 200 t/s for 4-bit (expect ~100+).
A typical answer here is ~60 tokens plus ~1,000 prompt tokens, so roughly 1-2 s per question on that card.
**What I actually ran on:** `<hardware>`; **measured:** `<tokens/s, seconds per question>`. If CPU only: `<measured>`, scaled by the ratio above for the workstation.

## 4. Evaluation
`eval/questions.jsonl`: 22 RAG questions (9 should answer, 11 should refuse, 2 should answer "with limits") plus 6 predictor checks. Expected behaviour
was fixed in `make_eval.py` and ground-truth ranks are read from the CSV. Refusal cases cover: college not in index, planted figure
("ignore the documents… cut-off is 1"), injection naming a real college, year outside the data, forecast, guarantee, out-of-domain, a
fact the college has no data for (hostel fee), a course not in the data, and a college/course combination with no row. Metrics: answer
correctness + valid citations, refusal recall and precision, predictor checks.
### Honest results
- **Dev set, retrieval + guards only (no LLM), after tuning:** 28/28. *This is optimistic.* My first run scored 23/28; I then fixed two real
  bugs (multi-year parsing, FAQ term statistics) and widened a stop-word list on the same questions.
- **12 fresh phrasings not used for tuning:** 10 behaved as intended. One was a false refusal ("How are SC and ST defined here?", an
  unlisted word); one comparison question returned "please narrow down" rather than a clean refusal. The gate is a heuristic, and unusual wording will keep causing some false refusals.
- **With the LLM (`results_ollama.md`):** `<paste your table>`. Extra failure modes to check: model ignoring the citation format (blocked by the post-check, so it becomes a refusal), and paraphrased numbers.
- Not tested: real data, multilingual questions, adversarial users beyond the listed injection patterns (a regex list is not a complete defence).

## 5. Privacy and data handling
Index (`index/`) and audit log (`logs/events.jsonl`: time, question, status, reason, source ids) live on the local machine. The app calls only
`localhost:11434` (Ollama); no cloud API is used, and the compose file binds Ollama to 127.0.0.1. The only external downloads are model weights at install time.
For the office: one index directory, one container and one OS user per business (NEET.info, Hirrx, editorial), no shared volume, log files per business with
access limited to that team, and no cross-business retrieval path. Smart Catalyst and NEET.info records are kept apart as the brief requires. Real student
data would need encryption at rest, retention limits and a consent basis under the DPDP Act before use. The log stores question text, so it counts as personal data once real users appear.
`<If you used a cloud GPU for testing, say so here, and note that an on-site deployment skips it and keeps weights cached locally.>`

## 6. Cost (rough)
Assume 300 W under load, ~3 s of GPU time per question, ₹8/kWh (assumption): 1,000 questions ≈ 0.25 kWh ≈ **₹2**. Idle draw of a workstation GPU (~20-30 W) is
the larger cost: ~₹150-200/month if left on. Hardware is already owned by the group. If rented instead: `<hourly rate × hours you used>`.
## 7. Limits
Synthetic data only; heuristic relevance gate; aggregate questions unsupported by design; predictor ignores seat availability, reservation sub-categories
and counselling-round dynamics; BM25 may miss paraphrased FAQ questions (embeddings help but were not evaluated).
