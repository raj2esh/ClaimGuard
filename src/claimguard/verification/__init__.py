"""ClaimGuard verification: integrated retrieval + reranking + binary
verifier pipeline (Step 15).

This is the FIRST integrated experiment combining Step 13's retrieval,
Step 14's reranking, and Step 12's binary verifier - NOT the final
ClaimGuard end-to-end pipeline (no generator, no correction loop). See
`pipeline.py` for the orchestration and `verify.py` for the verifier
checkpoint loading/scoring.
"""
