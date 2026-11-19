# Early start and a split one-bit lookup

Code and data for the paper **Early Start with Exact Reuse and a Split
One-Bit Lookup for Retrieval-Grounded Voice Agents** (Suvendu Mohanty, F0 Labs
LLC, 2026).

A voice agent that answers from company knowledge usually transcribes the
caller and searches its knowledge only after it decides the caller has
finished. *Early start* begins both at the caller's first short silence
(128 ms) and keeps them only when the turn's request is identical: it
guesses *when* the caller may be done, never *what* they will say. The
*split lookup* keeps one bit per dimension of every passage on the box that
runs the models, embeds and shortlists there, and asks a model-free vector
database to score only the shortlist exactly, in one request.

The repository holds:

- the early-start policy, as a library with no I/O;
- the one-bit shortlist;
- the paper's context-stability measures;
- a reference split lookup built on open components;
- the measurement tools;
- every record the paper's tables were computed from;
- tests that regenerate those tables from the records.

## What is here

| path | what |
|---|---|
| `crates/early-start` | The early-start policy: the setting, the per-chunk silence rule (`SilenceClock`), the reuse rules, the turn record. |
| `crates/onebit` | One-bit codes of passage vectors and the shortlist made from them, with benchmarks. |
| `crates/ctxstab` | Context-stability measures: flip rate, repeat floor, reuse audit, settle lead. |
| `crates/split-lookup` | The reference split lookup on Qdrant and text-embeddings-inference: the `near` service and the `ingest` loader. |
| `crates/measure` | The measurement tools and the analysis behind every table: `report`, `ctxaudit`, `codescan`, `lastvoiced`, `delayline`, `brainstub`. |
| `data/` | The questions, every replayed call's records, every recorded lookup, the latency records, the shortlist measurements. See [`data/README.md`](data/README.md). |
| Release download | The 406 question recordings (16 kHz WAV). |

## Reproduce the paper's numbers

Everything runs offline from `data/`. Install Rust (1.88 or newer), then:

```sh
# Every number in Tables 1, 2, 3 and 5, regenerated and checked
cargo test -p measure --test paper

# The replayed calls: each arm's waits and server hops, each pair of arms
# compared question by question (Table 3, Figure 1, Sections 5.2 to 5.6)
cargo run --release -p measure --bin report -- data/calls \
  --questions data/questions.json --pages data/passage_pages.jsonl

# Context stability and the reuse audit (Tables 1 and 2)
cargo run --release -p measure --bin ctxaudit -- report \
  data/questions.json data/lookups/box_forms_run1.jsonl

# The repeat floor: the same lookups, run again
cargo run --release -p measure --bin ctxaudit -- again \
  data/questions.json data/lookups/box_forms_run1.jsonl data/lookups/box_forms_run2.jsonl

# The settle curve
cargo run --release -p measure --bin ctxaudit -- report \
  data/questions.json data/lookups/shared_forms.jsonl \
  --prefixes data/lookups/shared_prefixes.jsonl

# The split lookup against the existing one (Table 5)
cargo run --release -p measure --bin ctxaudit -- again \
  data/questions.json data/lookups/box_existing.jsonl data/lookups/box_split.jsonl
```

| paper | source |
|---|---|
| Table 1, Table 2 | `ctxaudit report` on `lookups/box_forms_run1.jsonl` |
| Repeat floor (1.5 to 2.0%) | `ctxaudit again` on any two of `lookups/box_forms_run{1,2,3}.jsonl` |
| Settle curve | `ctxaudit report` on `lookups/shared_forms.jsonl` with `--prefixes` |
| Table 3, Figure 1, Figure 3 (left), Sections 5.4 to 5.6 | `report` on `calls/` (Figure 1: each arm's `timeline_turns_judged_complete`) |
| Table 4 and the dense-leg recall | `shortlist/*.json`, the output of `codescan` |
| Table 5 | `ctxaudit again` and `report` on `lookups/box_existing.jsonl` and `box_split.jsonl`; `latency/split_distance.jsonl`; the `*-split` arms in `calls/` |
| Figure 3 (right) | `latency/placement.jsonl`, `latency/split_distance.jsonl` |
| Where a lookup's time goes | `latency/split_stages.jsonl`, `latency/existing_service_stages.json` |

`crates/measure/tests/expected/` holds the reports exactly as the experiment
printed them. `tests/paper.rs` recomputes each one from `data/` and requires
every value to agree, bootstrap intervals included (4,000 resamples, seed
20261119).

## Run the split lookup on your own passages

You need a Qdrant (1.16 was used) and a text-embeddings-inference server for
each model: one embedding model, one reranking model. The paper's runs used
`intfloat/multilingual-e5-large` for embeddings and
`jinaai/jina-reranker-v2-base-multilingual` for reranking. That reranker's
licence is CC BY-NC 4.0, which allows research and evaluation only. Any
reranker the server can run will do, for example `BAAI/bge-reranker-v2-m3`
(Apache-2.0).

```sh
docker run -p 6333:6333 qdrant/qdrant:v1.16.1

# passages.jsonl: {"id": "...", "text": "...", "page": "...", "vector": [...]?}
cargo run --release -p split-lookup --bin ingest -- \
  --passages passages.jsonl --embed-url http://127.0.0.1:8080

cargo run --release -p split-lookup --bin near -- \
  --embed-url http://127.0.0.1:8080 --rerank-url http://127.0.0.1:8081

curl -s http://127.0.0.1:8084/v1/search -H 'content-type: application/json' \
  -d '{"query": "How do I reset my password?", "top_k": 4, "score_threshold": 0.12}'
```

Each lookup answers with its passages and the mode it ran in, `split` or
`graph`. Graph mode uses the database's own approximate dense search. The
answer also says whether it fell back, and where the time went. To compare
the two modes, put `"mode": "graph"` or `"mode": "split"` in a request. For a
whole question set:

```sh
cargo run --release -p measure --bin ctxaudit -- forms questions.json graph.jsonl \
  --endpoint http://127.0.0.1:8084/v1/search --mode graph
cargo run --release -p measure --bin ctxaudit -- forms questions.json split.jsonl \
  --endpoint http://127.0.0.1:8084/v1/search --mode split
cargo run --release -p measure --bin ctxaudit -- again questions.json graph.jsonl split.jsonl
```

A split lookup that cannot shortlist runs as a graph lookup, and so does one
whose request the database refuses. Without an embedder, the keyword leg
answers alone. Without a reranker, the fused order stands. Each case is
flagged in the answer.

## Use early start in a turn loop

```rust
use early_start::{Asked, Chunk, EarlyStart, Phase, SilenceClock, Thresholds, TurnTiming, DEFAULT_FLOOR, turn_query};

let timing = TurnTiming::new(DEFAULT_FLOOR, EarlyStart::MEASURED)?; // 320 ms floor, start after 128 ms
let mut clock = SilenceClock::new(timing, Thresholds::default());

// For every 32 ms chunk the voice detector judged:
let s = clock.on_chunk(Chunk::heard(speech_probability), Phase::CallerSpeaking);
if s.start_early { /* transcribe the words so far; search turn_query(&words) */ }
if s.drop_early { /* the caller went on: abort that work */ }
if s.floor_reached { /* ask the end-of-turn model, exactly as before */ }

// At the commit:
if clock.take_early() { /* only silence followed: the early words are the turn's */ }
let keep = EarlyStart::MEASURED.reuse().allows(
    Asked { words: &early_words, query: &early_query },
    Asked { words: &turn_words, query: &turn_query(&turn_words) },
);
```

The end-of-turn decision is untouched. Early start only moves the transcript
and the search into silence the caller leaves anyway.

## What is not here

- **The voice server and the knowledge service the measurements ran on.**
  - The early-start policy is extracted from that voice server as a library.
    The voice server's early-start scenarios are its tests.
  - The split lookup is reimplemented here on Qdrant and
    text-embeddings-inference.
  - The measured knowledge service also resolved symptom phrasings to the
    steps they stand for, removed duplicates and attached parent sections.
    This reference does not.
- **The caller that replayed the recordings.** It speaks that voice server's
  own protocol. Its timing rule is `measure::voiced`, and
  [`data/README.md`](data/README.md) describes its records.
- **Passage texts and vectors.** The workspace also held pages from other
  sites.

## How this code was checked

- `cargo test --workspace`: unit tests for every crate, including the
  early-start scenarios of the voice server it was extracted from.
- `crates/measure/tests/paper.rs` regenerates every report from `data/`, as
  described above.
- The keyword encoder (`split_lookup::bm25`) reproduces the measured
  knowledge service's keyword vectors exactly, token ids and weights, for
  all 2,889 passages.
- `crates/split-lookup/tests/qdrant_live.rs` runs the lookup against a real
  Qdrant, in CI as well. It loads passages, builds the codes from the
  database, and checks that the split's dense search is the exact search.
- **On the paper's workspace**, loaded with `ingest`, the 406 questions were
  replayed from their saved vectors, without a reranker:
  - the codes take 369,792 bytes;
  - every question was embedded, and no lookup fell back;
  - the split's 32 candidates held on average 95.6% of graph mode's;
  - the right page was among the candidates for 99% of questions in both
    modes;
  - `codescan` on this copy reproduces Table 4 exactly.

  The paper measured final lists with a reranker (Table 5); no reranker ran
  in this check.

Two things this check showed, worth knowing before you compare lookups:

- **The fused order is not repeatable.** The identical request returned the
  same 32 fused candidates for 85 to 87% of questions, and in the same order
  for 7 to 8%. The database orders tied fusion scores arbitrarily; reranking
  settles the final order. Compare lookups after reranking, or by candidate
  set.
- **Graph mode's recall depends on the database.** On this fresh copy, the
  database's own dense search held 82.8% of the exact top 64 at its default
  rescoring (3x). The test box's copy of the same workspace held 99.3%.
  Measure yours with `codescan --qdrant-url`.

## Licence

Apache License 2.0, for the code and the data. The recordings were
synthesized with Qwen3-TTS (Apache-2.0) in a preset voice.

## Citation

```bibtex
@misc{mohanty2026earlystart,
  author = {Mohanty, Suvendu},
  title  = {Early Start with Exact Reuse and a Split One-Bit Lookup for Retrieval-Grounded Voice Agents},
  year   = {2026},
  note   = {F0 Labs LLC}
}
```
