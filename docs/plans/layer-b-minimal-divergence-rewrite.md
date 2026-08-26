# Layer B minimum-divergence rewrite — synthesis and implementation plan

Status: **Phase 1 implemented** on `feat/layer-b-minimal-divergence-rewrite`. This
document is the design record: problem, root cause, alternatives considered
(with scores), the chosen approach and why, expected results, and the delivery
plan.

## 1. Problem

Layer B (`service/scripts/rewrite_text.py`) reduces watermark detection
successfully, but it optimizes almost entirely for removal strength, not for
preserving the original text. The default path (`--strength paraphrase`,
`--candidates 1`, `--max-loops 1`) generates one aggressive whole-document
rewrite that reaches roughly 0.92 bigram-Jaccard divergence against the
original — most wording and sentence structure changed, making the output
hard to reuse even when detection clears successfully.

The objective should instead be a constrained optimization:

```
minimize   divergence(original, candidate)
subject to detector(candidate) == not_watermarked
```

## 2. Root cause

Read in full in `service/scripts/rewrite_text.py` (pre-change):

1. **Every prompt requests substantial rewriting.** The only prompts
   available (`paraphrase`, `humanize`, `structural`, `backtranslate`, `code`)
   all instruct wholesale rewording. There was no tier between "do nothing"
   and "rewrite most of it."
2. **The loop stops at the first passing candidate**, not the least-divergent
   one (`if passed_i is True: break`). With `--candidates 3`, candidate 1 wins
   even if candidates 2 or 3 would also pass with much lower divergence.
3. **Passing candidates were never compared.** Selection was always "the last
   attempt made" (`selected_idx = len(attempts) - 1`), never a comparison
   across the batch.
4. **The prompt was built once, outside the loop.** Raising `--max-loops`
   retried the same aggressive prompt stochastically — no escalation.
5. **The no-detector fallback has the opposite objective** (correctly, for a
   blind attack): it explicitly selects the *most* divergent candidate. That
   logic is right for its own case but is not reusable for a minimum-
   divergence search.
6. **Defaults perform no real search**: `DEFAULT_CANDIDATES = DEFAULT_MAX_LOOPS
   = 1`, so the common path is exactly one aggressive rewrite, returned as-is.

None of this is a bug in the sense of wrong output — Layer B does what it was
built to do (defeat detection). It simply was never built to search for the
*cheapest* way to defeat detection.

## 3. Solutions considered and scored

Seven designs were independently researched and scored 1–5 on: (a)
divergence-reduction potential, (b) robustness across MarkLLM schemes and
Gumbel (their score/threshold semantics differ — Gumbel's `threshold` is a
p-value, not on the same scale as its own `-log10(p)` score), (c)
implementation complexity in this stdlib-first, no-bundled-model codebase,
(d) runtime/detector-call cost, (e) risk of quality/factual drift or new
failure modes, (f) fit with the existing CLI/agent-orchestrated architecture.

| Approach | a | b | c | d | e | f | Total |
|---|---|---|---|---|---|---|---|
| **Ladder + full-round min-divergence selection (chosen, Phase 1)** | 4 | 4 | 3 | 2 | 3 | 5 | **21/30** |
| Generate-then-revert diff-minimization (candidate Phase 2) | 5 | 4 | 3 | 2 | 3 | 4 | 21/30 |
| Binary-search continuous edit-budget | 4 | 3 | 4 | 2 | 3 | 4 | 20/30 |
| Token/word-substitution attack | 4 | 2 | 2 | 3 | 3 | 4 | 18/30 |
| Learned strength predictor | 1 | 2 | 3 | 3 | 3 | 4 | 16/30 |
| Segment-level blending across candidates | 4 | 3 | 2 | 2 | 2 | 3 | 16/30 |
| Localized/targeted span rewriting | 3 | 2 | 2 | 3 | 2 | 3 | 15/30 |

Why the other five were set aside (not because they're wrong, but because
each has a specific, verified blocker):

- **Binary-search edit-budget**: LLMs can't reliably obey a literal "change
  ≤N% of words" instruction, so the search still has to fall back on
  measuring realized divergence — at which point it's paying ~8–12 sequential,
  non-parallelizable detector round-trips to approximate what the ladder gets
  more cheaply.
- **Token/word-substitution**: well-matched to green/red-list schemes like
  KGW, but this repo's Gumbel scheme is a distortion-free design whose
  statistic is an additive sum spread thinly across every token — likely
  needs to edit close to a full-rewrite's worth of words on a strongly-marked
  document to flip detection, undermining the low-divergence premise.
- **Learned strength predictor**: earns almost no credit on the actual
  objective (divergence) — it only saves API calls on the resistant-document
  tail, and a hand-tuned heuristic likely captures most of that value without
  a table-maintenance/staleness burden.
- **Segment-level blending**: alignment between the original and a candidate
  is least reliable exactly on the aggressive rewrites the approach most
  needs to draw from (the paradox is structural, not an implementation
  detail), and detector calls scale with segment count, not a bounded knob.
- **Localized span rewriting**: blocked twice over — MarkLLM exposes no
  per-token signal at all through this codebase's detector wrapper, and even
  Gumbel's real per-token signal is likely too weakly contrasted to isolate
  "hot" spans, per the scheme's own math (significance only emerges from
  aggregating many weakly-biased tokens).

**Generate-then-revert** tied the winner and is not rejected — it's staged as
an optional Phase 2, gated on Phase 1's benchmark results (see §6). Its
strength (fine-grained, detector-agnostic, provably no worse than its seed)
is real; its risk (MarkLLM has no incremental scoring, so every hunk-revert
trial re-scores the whole document from scratch) makes it a second-stage
optimizer to prototype and measure, not a first release.

## 4. Chosen solution: discrete strength ladder + full-round minimum-divergence selection

**What it is.** A new `"minimal"` prompt tier (change as little as possible;
preserve sentence structure, word order, facts, numbers, names, URLs,
technical identifiers), an opt-in `--minimal-select` flag that generates and
evaluates every candidate in a round instead of stopping at the first pass
and selects the least-divergent passer, and an opt-in `--ladder
minimal,humanize,paraphrase` flag that escalates to the next strength only
when the current level produced zero passing candidates. Every attempt at
every level is generated from the **original** text, never from a prior
candidate, so divergence never accumulates across escalation.

**How, precisely** (implemented in `service/scripts/rewrite_text.py`):

- `rewrite()` gained `minimal_select: bool = False` and `ladder: list[str] |
  None = None`. When both are unset, behavior is byte-for-byte identical to
  before (verified: all 29 pre-existing tests in `tests/test_rewrite_text.py`
  pass unmodified).
- The single `prompt = build_prompt(...)` built once outside the loop became
  a per-level `level_prompt` built inside a new outer `for level_strength in
  strengths:` loop, where `strengths = ladder or [strength]`.
- The inner generate/evaluate loop is unchanged in shape; the early-stop
  `break` on first pass is now conditional on `not minimal_select`.
- A new `_select_min_divergence(original, passing)` helper picks the
  least-divergent candidate among a level's passers, preferring ones that
  satisfy a new `_length_ratio_ok(original, candidate, lo=0.5, hi=2.0)`
  eligibility guard (falling back to the ungated passing set only if the
  guard would eliminate everyone — a passing rewrite beats none).
- `minimal_select=True` with no detector configured raises `SystemExit`
  directly from `rewrite()` (not just from CLI argument validation), so any
  programmatic caller gets the same fail-fast guarantee.
- Acceptance criterion is the detector's plain `is_watermarked` boolean, not
  a generic `score <= threshold - margin` formula — deliberately deferred.
  Verified in code: Gumbel's `score` is `-log10(p_value)` while its
  `threshold` field is the raw p-value in `(0, 1)` — a different domain from
  the score — and MarkLLM's `threshold` can be `None` when a scheme's config
  doesn't expose one. A shared numeric margin needs a per-detector
  normalization step first; that's out of scope here.
- Each attempt record gained `"level"` (which ladder strength produced it)
  and `"entity_drift"` — a **reporting-only** dict (not an enforced gate)
  noting whether the original's URLs/numbers survived, via new stdlib regex
  helpers `_URL_RE`/`_NUMBER_RE`/`_entities`. This is intentionally scoped to
  URLs and numbers only; it is not named-entity recognition and does not
  verify proper names or other technical identifiers, despite the prompts
  asking the model to preserve those — an honest limitation, not silently
  overclaimed.
- The pre-existing `_select_candidate()` helper (dead code — never called by
  `rewrite()`, only exercised by `test_select_candidate_prefers_more_divergent`)
  was deliberately **left untouched, not repurposed**. Its selection
  direction (prefers *more* divergent) is correct for its own documented
  purpose and would be wrong if reused for this feature; building a new,
  correctly-directed helper (`_select_min_divergence`) was the safer,
  non-destructive choice over silently inverting an existing function a test
  depends on.

**Why this design over the runner-up.** It is strictly cheaper
(`levels × candidates × max_loops` detector calls, a small bounded constant)
than generate-then-revert (`O(hunk count)`, scaling with document length and
unbounded by any existing knob), reuses nearly all of the existing loop
machinery rather than adding a new diff/revert subsystem, and is fully
backward compatible — zero risk to any existing caller. Generate-then-revert
remains the better *ceiling* on achievable divergence reduction, which is
exactly why it's staged as a benchmark-gated Phase 2 rather than abandoned.

## 5. Expected results

- **Divergence**: initial candidates at the `minimal` tier are expected to
  move from ~0.92 (today's default) toward roughly the 0.3–0.5 range for
  documents that don't need heavy rewriting to clear detection, per the
  motivating discussion (`discussions/162`) — this is a hypothesis to be
  confirmed by the benchmark in Phase 1b, not a guarantee: statistical
  watermarks are specifically designed to be robust to a handful of edits, so
  a solidly-watermarked document may still need `humanize` or `paraphrase`
  before anything passes.
- **Cost**: worst case scales as `levels × candidates × max_loops` generate+
  detect calls versus today's 1; with `candidates=3, max_loops=1` and a
  3-level ladder that's up to 9 calls if every level fails until the last.
  Against MarkLLM without a resident worker (`WATERMARKS_MARKLLM_PORT`),
  each `detect()` can cost up to its configured timeout (180–600s default),
  so this cost profile needs to be surfaced honestly in the benchmark, not
  assumed acceptable.
- **Backward compatibility**: zero behavior change for any caller that
  doesn't pass the new flags — confirmed by the full existing test suite.
- **Quality**: `entity_drift` gives visibility into URL/number preservation
  per candidate without silently gating on it (a hard gate risks the search
  never finding a pass on documents where an LLM validly rephrases a number's
  presentation); factual/name preservation remains prompt-instructed only,
  not mechanically verified — documented as a known limitation, not fixed
  here.

## 6. Delivery plan

| Delivery | Status | Notes |
|---|---|---|
| `minimal` prompt + `--minimal-select` full-round selection | **Done** | `service/scripts/rewrite_text.py`, all 29 existing tests pass unmodified |
| `--ladder` strength escalation | **Done** | Same file; escalates only on zero-pass levels, always rewrites from the original |
| Corrected selection helpers (`_length_ratio_ok`, `_entity_drift`, `_select_min_divergence`) | **Done** | New, non-destructive — `_select_candidate()` left untouched per §4 |
| Unit/regression tests for the new behavior | Assigned to test agent | Extends `tests/test_rewrite_text.py`; must not modify `rewrite_text.py` |
| README / SKILL.md / removal-matrix.md / .env.example docs | Assigned to docs agent | Grounded in the shipped code, not the design conversation |
| Benchmark extension (`minimal`/`ladder` variants) | Assigned to benchmark agent | Additive to `bench_synthid_text.py`'s existing `<strength>:<candidates>` variant grammar |
| Benchmark run: clear rate, median/P90 divergence, cost, drift | Not started | Gates Phase 2 — run against multiple MarkLLM schemes, not just `synthid`, to avoid overfitting to one scheme |
| Generate-then-revert prototype (Phase 2) | Not started | Only build if the benchmark shows meaningful recoverable divergence left after Phase 1; corrected acceptance rule required: accept a hunk revert only when `is_watermarked == False AND divergence(original, candidate) < divergence(original, current)` — bigram-Jaccard is non-additive and hunk boundaries change bigrams, so a revert is not automatically an improvement |
| Production revert refinement (budgets, caching, tests, docs) | Not started | Behind `--revert-repair`, `--max-detector-calls`, `--max-revert-hunks`; every evaluator (Gumbel included) needs a call budget — Gumbel's per-call cost is cheap but not literally free, and still rescans the full document per trial |
| Normalized detector score margins | Deferred | Separate later work; needs a per-detector normalization scheme before a generic `score <= threshold - margin` rule is safe (see §4) |

Explicitly deferred, not planned: score-margin acceptance criteria, learned
strength predictors, localized/span attacks, and any new heavy semantic
dependency (embedding/NLI-based divergence metrics) — each was scored and
set aside in §3 for a specific, verified reason, not for lack of
consideration.

## 7. Test requirements for generate-then-revert (Phase 2, when built)

Recorded here so the gate is unambiguous when Phase 2 starts:

- Repair is skipped entirely when the seed candidate does not pass detection.
- Only strictly-lower-divergence reversions are accepted (see the corrected
  acceptance rule in §6).
- Accepted outputs remain detector-negative.
- Detector-call budgets (`--max-detector-calls`, `--max-revert-hunks`) are
  never exceeded.
- A failed or unavailable detection call retains the last confirmed-passing
  output — never accepts an unverified reversion.
- Identical trial texts are cached (avoids redundant detector calls when a
  revert order revisits the same candidate text).
- Hunk order and tie-breaking are deterministic given the same input.
- URLs, numbers, identifiers, and formatting survive at least as well as the
  seed candidate.
- Final divergence is never greater than the seed candidate's divergence.
- The feature is disabled by default (`--revert-repair` opt-in).
