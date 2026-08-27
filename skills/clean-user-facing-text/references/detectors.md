# AI text detectors: what exists and what is legitimate to claim

Background for the scoring and levers in `SKILL.md`. Current as of mid-2026.
The short version: detectors are uncalibrated proxies, and a humanizer can
honestly report that measurable signals moved. It cannot honestly claim a
verdict, and none of it applies to content the user is not authorized to clean
(see `responsible-use.md`).

## Two mechanically different families

A pass that beats one family is often invisible to the other, so any claim must
say which family it refers to.

**Family A: statistical / zero-shot detectors.** They score the text's
probability profile under a reference model, with no training on labeled data.

- Perplexity: AI text has uniformly low perplexity because LLMs emit the most
  probable next token. Countered by genuinely raising perplexity through
  specific, concrete diction.
- Burstiness: the variance of perplexity and sentence length across a document.
  AI text is too even. Countered by varied sentence and paragraph lengths.
- DetectGPT (Mitchell et al., 2023): perturbs the text and checks whether the
  original sits at a local maximum of model log-probability. Countered by
  rebuilding sentence logic, not just swapping words.
- Binoculars (Hans et al., 2024, arXiv:2401.12070): ratio of an observer LLM's
  log-perplexity to cross-perplexity between two LLMs. High accuracy and
  notably robust to styled noise and simple adversarial edits: the
  high-perplexity choices must be coherent human choices.

**Family B: trained neural classifiers.** A transformer fine-tuned on millions
of human-vs-AI samples learns the LLM fingerprint directly.

- Pangram (arXiv:2402.14873): tokenize, embed, classify; hardened with
  hard-negative mining and synthetic mirrors that include generic
  humanizer/paraphraser output, so paraphrase-style edits are trained in as the
  AI class.
- GPTZero (3.2b and successors): end-to-end deep classifiers plus sentence-level
  and mixed-content detection, with explicit shields against paraphrasing and
  homoglyph attacks.
- Originality, Turnitin, Copyleaks: same family; specifics undisclosed.

Against Family B, perplexity and burstiness tricks and synonym swaps are weak.
The levers that matter are matching a real human distribution (voice matching
against a sample the user owns) and, when the user can run the actual detector,
iterating with detector feedback.

## Watermarking is a separate axis

Watermarks are inserted at generation time by the model owner; they are not a
property of "AI-ness" the way perplexity is.

- Kirchenbauer green/red list (2023): hash the preceding token to bias a
  green subset; detection counts the green-token surplus.
- SynthID-Text (Google DeepMind, 2024): tournament sampling from the same
  red-green family, less perceptible.
- Both key on the preceding token, so any meaning-preserving rewrite that
  changes token order degrades detection: paraphrasing, edits, and
  back-translation all work; a full architectural rewrite removes it entirely.

Vendor status notes: Google retired SynthID text watermarking on the Generative
Language API (mid-2026), so current API output is no longer watermarked.
Anthropic's text-watermark detection API is not public, so no local tool can
verify a Claude watermark.

## What the vendored scorer measures

`score_stylometry.py` is a zero-LLM estimator: sentence-length burstiness CV,
weighted AI-cadence phrase density per 100 words, MATTR lexical diversity, and a
dampened composite (0-1) with the same thresholds as the service pipeline. It
deliberately covers Family-A signals only.

It does **not** measure: trained classifier output, secret-key watermarks,
semantic quality, fact accuracy, or authenticity. Also note the small-sample
guard: below 30 words it reports `insufficient_length` and no score; below 100
words the score is dampened. Treat score movement as evidence about lexical and
cadence signals, never as a verdict.

## Legitimate claims versus not

Legitimate to report, honestly labeled:

- Before and after scores from the vendored estimator, with its confidence
  level and findings ("the lexical and cadence signals moved").
- What changed: phrase list cleared, burstiness CV moved, artifacts removed.
- What the user must verify themselves: semantic equality, fact preservation,
  voice, required disclosures.

Not legitimate to claim:

- "Passed GPTZero / Turnitin / Originality" without the user running that exact
  detector on the output, and even then only as that detector's own report.
- "Undetectable", "no AI trace", or "certified human".
- Removal of a vendor secret-key watermark (impossible to verify locally).
- That a rewrite proves human authorship.

For high-stakes cases, the only honest workflow is detector feedback: the user
runs the actual detector(s) they care about, and only claims what those reports
show. The rewrite itself should aim at genuine, coherent human prose first;
scoring is a check that the measurable signals moved, not a substitute for
quality.

## Sources

- Mitchell et al., DetectGPT (2023).
- Hans et al., Binoculars (arXiv:2401.12070, 2024).
- Pangram detector notes (arXiv:2402.14873, 2024).
- Sadasivan et al., "Can AI-Generated Text be Reliably Detected?"
  (arXiv:2303.11156, 2023): paraphrase attacks break watermarking, neural, and
  zero-shot detectors; recursive paraphrasing defeats even watermark and
  retrieval-based defenses; the paper also establishes a theoretical
  impossibility result, which is why all of the above is the honest limit.
- Kirchenbauer et al., watermarking (2023); Google DeepMind SynthID-Text (2024).
