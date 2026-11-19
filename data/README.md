# Data

Everything the paper's tables and figures were computed from, as measured.
Nothing here holds a passage's text or vector. Passage ids are opaque, and
call ids are random.

Page labels: pages of the company's own sites (`docs.bidirection.ai`, its help
centre, and `bidirection.ai`) keep their URLs. The workspace also held 357
passages from other sites it had scanned. Their pages are labelled
`other-site-01` to `other-site-18`, which keeps every "same page" comparison
intact.

## `questions.json`

406 held-out questions, written by a language model from 203 help-centre
passages spread over all 27 pages: half short (about 7 words), half long
(about 15).

| field | |
|---|---|
| `qid` | the question's id |
| `style` | `short` or `long` |
| `question` | the text the caller said |
| `gold_id` | the passage it was written from |
| `gold_page` | that passage's page: the right answer's page |

## `passage_pages.jsonl`

Every passage of the workspace (2,889): `{"id", "page"}`.

## `calls/`: the replayed calls

One arm is one run of the voice server with one setting, the questions said
into real calls (10 a call, 0.7 s after each reply has played). Each arm has
three files:

**`<arm>.json`**: the arm's settings.

| field | |
|---|---|
| `early_start` | `numbers` (the prior behaviour) or `first-silence:<onset ms>:<rule>` |
| `silence_floor_ms` | silence before the end-of-turn model is asked |
| `mic_frame_ms` | the caller's audio frame: 32 or 256 ms |
| `kb_delay_ms` | the knowledge service's distance, each way (a delay line) |
| `lookup` | `existing` or `split` |
| `knowledge_searches` | every search the knowledge service ran during the arm. Round 2 and 3 arms were counted by the run script; round 1 arms (`*-f32`, `*-f256`) were counted from the knowledge service's log between the arm's start and end. |
| `stub_ttft_ms` | the stand-in brain's time to its first token |
| `started`, `ended` | UTC |

**`<arm>.replay.jsonl`**: one line per question the caller said.

| field | |
|---|---|
| `qid`, `call`, `n` | the question, the call, its place in the call |
| `frame_ms` | the caller's audio frame |
| `speech_ms`, `tail_ms` | the recording up to its last voiced sample, and after it |
| `wait_ms` | the caller's wait: the capture time of the recording's last voiced sample (`measure::voiced`) to the arrival of the reply's first audio; `null` when no reply came |
| `talked_over` | the agent's audio arrived while the caller was still speaking |
| `heard` | what the server heard, one entry per turn it made of the question |
| `reply`, `reply_audio_ms` | the reply's words and length |
| `began_unix_ms`, `at_unix_ms` | when the caller began the question, and when its turn ended (the reply played out) |

**`<arm>.turns.jsonl`**: one line per turn the server answered
(`early_start::TurnRecord`), with `at`, the time the server logged it.
Hops are in milliseconds; `null` is a hop the turn skipped.

| field | |
|---|---|
| `smart_turn_ms` | the end-of-turn model |
| `stt_ms`, `kb_ms` | the transcript and the knowledge lookup, from the turn's commit (0 when ready before it) |
| `brain_ttft_ms`, `tts_ttfa_ms` | the brain's first token; the speech engine's first audio |
| `total_ms` | the silence floor reached, to the first agent audio |
| `kb` | `none`, `hit`, `no-match`, `stale`, `fail-open` |
| `early_words` | `at-commit`, `kept` (decoded early, ready at the commit), `waited` (still decoding at the commit, waited for) |
| `early_search` | `none`, `kept`, `refused` (the reuse rule said no), `lost` |
| `early_lead_ms` | how long before the commit the early start began |
| `judged_complete` | the end-of-turn model's last verdict |
| `passages` | the ids of the passages the reply was grounded on, in order |

A question's line is joined to the last answered turn on the same call
between the question's start and 300 ms after its turn ended
(`measure::calls`).

**`pairs.txt`**: the pairs of arms the paper compares (`a=b`: a's wait minus
b's, question by question, on the questions both heard as one turn).

The arms:

| arm | round | early start | frame | distance | floor | lookup |
|---|---|---|---|---|---|---|
| `base-f32`, `early-f32` | 1 | prior / 128 ms | 32 ms | 0 | 320 ms | existing |
| `base-f256`, `early-f256` | 1 | prior / 128 ms | 256 ms | 0 | 320 ms | existing |
| `base-s`, `early-s` | 2 | prior / 128 ms | 32 ms | 0 | 320 ms | existing |
| `base-s-d60`, `early-s-d60` | 2 | prior / 128 ms | 32 ms | 60 ms | 320 ms | existing |
| `base-s-d120`, `early-s-d120`, `early64-s-d120` | 2 | prior / 128 ms / 64 ms | 32 ms | 120 ms | 320 ms | existing |
| `base-s-fl192`, `early-s-fl192` | 2 | prior / 128 ms | 32 ms | 0 | 192 ms | existing |
| `base-s-d120-split`, `early-s-d120-split` | 3 | prior / 128 ms | 32 ms | 120 ms | 320 ms | split |

Round 1 said the first 200 questions; rounds 2 and 3 the first 100.

## `lookups/`: recorded text lookups

One line per lookup (`measure::audit::Looked`): `qid`, `form` (one of
`as_written`, `no_end_mark`, `plain`, `last_word_missing`, or
`prefix_<k>_of_<n>` for a question's first `k` of `n` words), `n_words`,
`ids` and `pages` of the passages returned (top 4, reranker score at least
0.12), their `scores`, and the lookup's `ms`.

| file | what |
|---|---|
| `box_forms_run1.jsonl` ... `run3` | every question in its four forms, three times, on the test box (Tables 1 and 2, the repeat floor) |
| `box_existing.jsonl`, `box_split.jsonl` | the same lookups through the existing lookup and the split lookup (Table 5) |
| `box_oversampling3_run*.jsonl`, `box_oversampling16_run*.jsonl` | the existing lookup with the database rescoring 3 and 16 times the candidates |
| `shared_forms.jsonl`, `shared_prefixes.jsonl` | the four forms, and every beginning of the first 100 long questions, against the shared index on a laptop (the settle curve) |

## `latency/`

| file | what |
|---|---|
| `placement.jsonl` | one text lookup's latency by where its two models run, at 0, 20, 60 and 120 ms each way |
| `split_distance.jsonl` | the existing and the split lookup at 20, 60 and 120 ms each way |
| `split_stages.jsonl` | every split lookup's stages on the GPU box: embedding, shortlist (us), the database's request, reranking |
| `existing_service_stages.json` | the existing knowledge service's stage times, summed over every search it ran |

## `shortlist/`: the one-bit shortlist (`codescan` output)

| file | what |
|---|---|
| `shared_index.json` | Table 4: every way of shortlisting, at every size, on the shared index's vectors |
| `shared_index_dense_leg_oversampling*.json` | the shared index's own dense search against the split's, by rescoring factor |
| `box_copy.json` | the same on the test box's copy of the workspace |
| `fresh_copy.json` | the same on a fresh copy of the workspace, loaded with this repository's `ingest`, measured with this repository's `codescan` |

## The recordings

The 406 questions as the caller said them: 16 kHz mono 16-bit WAV, one file
per question (`<qid>.wav`), synthesized once with Qwen3-TTS 12 Hz 1.7B
CustomVoice in its preset voice "Vivian" (not a real person's voice).
They are a download of the repository's release, with a SHA-256 list.
