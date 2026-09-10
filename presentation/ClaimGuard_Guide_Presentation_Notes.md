# ClaimGuard Guide Presentation — Speaker Notes

Generated directly from the notes embedded in `ClaimGuard_Guide_Presentation.pptx` — the two files cannot drift out of sync since this file is derived from the .pptx, not authored separately.

## Slide 1

**Title:** ClaimGuard

**Speaker notes:**

Good [morning/afternoon]. I'm going to walk you through ClaimGuard — the hallucination-detection and self-correction pipeline I've been building. This is a full progress presentation: what the system is, how I evaluated it end-to-end, and importantly, what I found when I tested it honestly. I want to say up front that this is not a 'we solved hallucination detection' talk — it's a 'here's what we built, here's what happened when we tested it rigorously, and here's what that tells us' talk. That's deliberate, and I'll explain why it's actually the more useful outcome scientifically.


## Slide 2

**Title:** Why Hallucination Detection Matters

**Speaker notes:**

The starting problem is simple: LLMs are fluent, but fluency and factual correctness are two different things. A model can produce a confident, well-formed sentence that's simply wrong. The natural fix people reach for is retrieval-augmented verification — go get evidence, check the claim against it. But that's not automatically a solved problem either, because the hard part isn't just retrieving SOME evidence — it's correctly judging whether the evidence you got actually supports the claim. That judgment step is where a lot of the interesting failure modes in this project ended up living, and it's the thread that runs through the whole talk.


## Slide 3

**Title:** Research Problem

**Speaker notes:**

So the research question is: can an evidence-grounded pipeline reliably detect and correct hallucinated responses? The pipeline has six stages, and I want to flag early that every one of these stages is a place where things can go wrong — a bad retrieval, a reranker that reorders things unhelpfully, an overconfident verifier, a decision policy that flags the wrong thing, or a correction step that actually makes the answer worse. The point of building this carefully was to be able to isolate WHICH of these stages actually drives the end-to-end result, rather than just reporting one aggregate number.


## Slide 4

**Title:** Research Gap

**Speaker notes:**

The gap I wanted to address is that a lot of prior work reports retrieval recall or verifier accuracy as separate numbers, but doesn't test how those components interact once you chain them together with a decision policy and a correction loop. My working hypothesis going in was that reliable detection needs BOTH relevant evidence AND a verifier whose confidence actually tracks whether the evidence supports the claim — not just one or the other. I want to be careful here: this is the hypothesis I tested, not something I'm claiming is universally proven. The rest of the talk is basically the test of this hypothesis.


## Slide 5

**Title:** ClaimGuard Architecture

**Speaker notes:**

This is the full architecture. A user query goes to Qwen3-8B to generate a candidate answer. That answer is embedded with BGE-large and searched against a FAISS index to get the top-20 evidence candidates. A BGE reranker narrows that to the top 5. The DeBERTa-v3-large verifier scores each candidate for entailment or contradiction. The Step-16 decision policy turns those scores into ACCEPT, CORRECT, or ABSTAIN. If CORRECT, a bounded correction loop regenerates the answer using the retrieved evidence and re-verifies, up to a fixed attempt budget. I'll come back to almost every one of these boxes later in the talk, because the failure analysis traces back through this exact chain.


## Slide 6

**Title:** Datasets and Their Roles

**Speaker notes:**

Four datasets, each with a specific role. FEVER and HaluEval are for developing the verifier — training and dev. RAGTruth is the primary held-out test for the FULL pipeline — 2,700 responses that were never touched during training. TruthfulQA is a secondary, evaluation-only adversarial stress test. The important point on this slide is that RAGTruth and TruthfulQA are held out — not just informally, but with actual guards in the code that raise an error if anyone tries to build a training pool from them. I'll show that mechanism on the next slide.


## Slide 7

**Title:** Data Integrity and Contamination Control

**Speaker notes:**

This slide is here because I think it matters for how much you should trust everything that follows. We found that 100 HaluEval summarization records exactly overlapped 100 RAGTruth test source documents. Since each RAGTruth source has 6 model responses, that expanded to 600 affected response records. We excluded 200 pool records from training — never touched RAGTruth test itself. Beyond that, there are structural guards: if any script tries to build a training pool that includes RAGTruth test or TruthfulQA, it raises an error. And there are tests that literally run the real inference code with a sentinel gold-label value and confirm that value never appears in any model call argument. This is what lets me trust the negative results I'm about to show you — they're not an artifact of leakage.


## Slide 8

**Title:** Binary NLI Verifier

**Speaker notes:**

The verifier is a fine-tuned DeBERTa-v3-large, reduced to a binary entailment/contradiction head — 435 million parameters. Important detail: we didn't fabricate a neutral class. We investigated whether a legitimate 'not enough info' class could be built and found no scientifically valid source for it within this project's data, so the verifier stays strictly binary and the 'insufficient evidence' case is instead handled downstream by the decision policy's ABSTAIN outcome. On its own dev set — gold evidence paired with the right claim — it looks very strong: 96.2% accuracy, 95.85% macro F1. But I want to flag immediately: this is a GOLD-PREMISE evaluation. It tells you the verifier is good AT THE NLI TASK ITSELF. It does not tell you how the full pipeline performs once retrieval is uncertain — that's a completely different number, coming up later.


## Slide 9

**Title:** Evidence Retrieval Pipeline

**Speaker notes:**

Retrieval: BGE-large embeddings into a FAISS flat index, top-20 candidates, then a BGE reranker narrows to the top 5. On a 2,000-claim held-out evaluation set, FAISS alone gets 28% recall at 1, climbing to 73.25% at 20. Adding the reranker actually LOWERS recall at 1 — 24.45% versus 28% — while improving recall at 5 and 10. This is a real, repeated finding across multiple steps of this project, not noise. So I want to be careful about the framing: the reranker is not 'better' or 'worse' in some universal sense — it trades off top-1 precision for slightly broader coverage in the top 5-10. Whether that trade-off matters downstream is exactly what the ablation study later tests directly.


## Slide 10

**Title:** Retrieval + Reranking + Verification

**Speaker notes:**

Once retrieval, reranking, and the verifier are chained together, the picture already starts to shift. The verifier's own SELECTED evidence — the one it scores highest — only actually matches gold evidence 31.8% of the time. Overall classification accuracy in this pipeline setting is 88.6%, but that's dominated by the entailment class — entailment F1 is 92.85%, contradiction F1 drops to 71.92%. And here's the key number: look at the confidence chart. Gold SUPPORT evidence gets 98.26% mean confidence — great. Gold REFUTE evidence correctly gets pushed down to 8.07%. But NON-gold evidence — evidence that's just topically similar but not actually the right supporting sentence — still gets 69.68% mean confidence. That's uncomfortably close to genuine support. This is the first concrete sign of the verifier discrimination problem that becomes central later in the talk.


## Slide 11

**Title:** From Verification to Correction

**Speaker notes:**

The verifier's score feeds a deterministic decision policy that outputs ACCEPT, CORRECT, or ABSTAIN. If CORRECT, the correction loop regenerates the answer using the retrieved evidence and re-verifies — bounded to a small number of attempts so it can never run forever. Here's the important observation that sets up the rest of the talk: across the real RAGTruth and TruthfulQA evaluations, the attempt-0 ABSTAIN rate is exactly 0%. The policy never once says 'I'm not sure.' It always picks ACCEPT or CORRECT, even when — as we'll see — it's frequently wrong. That's the headline observation for this slide: the system is rarely uncertain, even when it should be.


## Slide 12

**Title:** RAGTruth: End-to-End Evaluation

**Speaker notes:**

This is the central result of the whole project, and I'm not going to soften it: on RAGTruth's 2,700-response held-out test set, ClaimGuard's attempt-0 accuracy is 55.9%, final accuracy after correction is 58.0% — and the trivial majority-class baseline, just always guessing 'not hallucinated', gets 65.1%. The full pipeline underperforms a baseline that does no work at all. And on the correction side: only 25 hallucinated responses were successfully fixed, while 923 originally-correct responses were degraded by the correction step. I want to say this plainly and not bury it: ClaimGuard does not demonstrate reliable hallucination detection on RAGTruth. The rest of the talk is about why, because that diagnosis is where the actual scientific value of this project is.


## Slide 13

**Title:** RAGTruth Failure: Evidence-Domain Mismatch

**Speaker notes:**

Here's the diagnosis. ClaimGuard's retrieval corpus is built from FEVER and Wikipedia. RAGTruth's actual source material is news articles, business listings, and QA passages — a genuinely different domain. So there's little real evidence overlap, which means retrieval returns evidence that's topically adjacent but not actually the right supporting text. The verifier then scores that irrelevant-but-topical evidence overconfidently, leading to incorrect decisions. Two diagnostic details matter here: zero of the 300 diagnostic responses had a MECHANICAL retrieval failure — retrieval never returned empty-handed. But we could NOT directly measure retrieval RELEVANCE failure, because RAGTruth doesn't have gold-relevance annotations the way FEVER does. That's an honest, disclosed gap — I did not invent a relevance metric to paper over it.


## Slide 14

**Title:** TruthfulQA Evaluation

**Speaker notes:**

TruthfulQA: 790 questions, 37 categories, evaluation-only. On the multiple-choice metrics — which measure Qwen3's own answer-likelihood calibration, independent of the pipeline — MC1 is 34.05%, MC2 is 55.05%, MC0 is 44.18%. On the free-form side, where the full pipeline runs, attempt-0 accepts 80% of answers outright and the final accept rate climbs to 95.3%. Now, the important thing I want to be very explicit about: I am NOT presenting those ACCEPT rates as a truthfulness score. There is no approved automatic judge for scoring free-form generated text in this project — no human raters, no GPT-judge access — and I was explicit that I would not invent a semantic-similarity heuristic to fill that gap. So this is a disclosed limitation, not a hidden one, and I'd actually call that methodological honesty a strength rather than a weakness of the project.


## Slide 15

**Title:** What Actually Causes the Failure?

**Speaker notes:**

So which component actually matters? I ran controlled ablations to find out. Condition A is the full pipeline with reranking, on a 300-response subset — 58.67% accuracy. Condition B removes the reranker entirely — 58.67% accuracy, identical to four decimal places on macro F1 too. So the reranker changes WHICH evidence gets selected for every single response, but it changes NOTHING about the final classification outcome. Condition C uses only the single top FAISS candidate, no top-5 — and that's measurably worse, 51% accuracy. So candidate POOL SIZE matters, ordering doesn't. Conditions D and E reuse the full 2,700-response data: D is no-correction-at-all, E is verify-only. Verify-only is worse than the full pipeline, meaning correction does contribute some net signal — but as we'll see next, it's not fixing the root cause.


## Slide 16

**Title:** Evidence-Supported Failure Model

**Speaker notes:**

This is the synthesis diagram. Domain-mismatched corpus leads to weak or irrelevant retrieved evidence, which the verifier then scores overconfidently. Combined with near-zero abstention, that produces wrong decisions, and wrong decisions feed a correction loop that sometimes does more harm than good. The numbers on the right back this up: 38.3% of the diagnostic subset was a case where the verifier was confidently — meaning above 90% confidence — WRONG. 33.7% of corrections were harmful. Retrieval's MECHANICAL failure rate is 0% — it always returns something. And reranking, as we just saw, is not a meaningful bottleneck. I want to be careful about how I label this diagram: it's an evidence-supported failure model, built from measurements across several steps of this project. It is explicitly NOT a proven causal DAG — I have correlational, controlled-ablation evidence for each link, not a formal causal identification.


## Slide 17

**Title:** Key Findings

**Speaker notes:**

Five findings to take away. One, ClaimGuard did not beat the majority baseline on its primary benchmark — that's the headline negative result. Two, the FEVER/Wikipedia corpus is a poor match for RAGTruth's actual domains. Three, the verifier is substantially overconfident specifically on irrelevant evidence. Four, the decision policy essentially never abstains. Five, correction-caused harm is a downstream consequence of the earlier failures, not an independent bug in the correction logic itself. Putting that together, the conclusion I'd draw is that this project demonstrates the importance of evidence relevance and calibrated uncertainty for this kind of pipeline — not that the current implementation has solved hallucination detection. I'm not claiming that, and I don't want to leave that impression.


## Slide 18

**Title:** Limitations and Future Research

**Speaker notes:**

None of these limitations are hidden — they're listed explicitly. The biggest one, which I keep coming back to, is the domain-mismatched retrieval corpus. On the future work side, I want to be careful: none of these eight directions have been attempted or solved in this project. They're the concrete, evidence-based next steps that fall out of the failure diagnosis — things like domain-matched retrieval, verifier calibration specifically for out-of-domain evidence, an actual abstention mechanism that fires when it should, and gating correction so it only triggers on reliable signals. This is a roadmap, not a claim of partial completion.


## Slide 19

**Title:** Reproducibility & Project Status

**Speaker notes:**

On reproducibility: inference is deterministic and that was re-checked after every major run, not assumed once and forgotten. Dataset isolation and contamination checks are enforced structurally, as I showed earlier. There are over 400 tests passing throughout the project. Evaluation configurations are frozen — nothing was tuned against the RAGTruth or TruthfulQA results themselves. No model weights were changed once evaluation started. And the full set of experiment artifacts is preserved, so every number in this talk traces back to a saved file, not something recomputed on the fly. The project's status right now is that the experimental cycle is complete — the next phase is architectural redesign based on the failure diagnosis, not more tuning of the current system.


## Slide 20

**Title:** ClaimGuard: From Building a Detector to

**Speaker notes:**

To close: this project went from building a hallucination detector to understanding, in a rigorous and evidence-backed way, why it fails. Three things I'd want you to remember: evidence relevance is fundamental — you can't verify against evidence that isn't actually about the claim. Confidence has to reflect evidence quality, not just pattern-match to something that looks similar. And correction should be gated by reliable uncertainty, not triggered every time the policy has any doubt at all. The next step is to redesign the evidence and uncertainty layer based on exactly what this failure analysis identified — and I want to be clear that's future work, not something already started. Happy to take questions.


## Slide 21

**Title:** Appendix — Anticipated Guide Questions (1/5)

**Speaker notes:**

Reference slide — use only if the guide asks one of these questions. Not part of the timed core presentation.


## Slide 22

**Title:** Appendix — Anticipated Guide Questions (2/5)

**Speaker notes:**

Reference slide — use only if the guide asks one of these questions. Not part of the timed core presentation.


## Slide 23

**Title:** Appendix — Anticipated Guide Questions (3/5)

**Speaker notes:**

Reference slide — use only if the guide asks one of these questions. Not part of the timed core presentation.


## Slide 24

**Title:** Appendix — Anticipated Guide Questions (4/5)

**Speaker notes:**

Reference slide — use only if the guide asks one of these questions. Not part of the timed core presentation.


## Slide 25

**Title:** Appendix — Anticipated Guide Questions (5/5)

**Speaker notes:**

Reference slide — use only if the guide asks one of these questions. Not part of the timed core presentation.

