## Analysis

Written 2026-10-02. A development-only diagnostic for P2.6 step 2: there is no decision
block, and nothing here chooses. `reports/m2-retrieval-gaps.md` (step 2) has the reading,
and `gaps.md` beside this file has the slices.

- The 100-per-branch row reproduces E3's recorded hybrid run, so the system measured is the
  shipped one.
- A bigger pool finds gold the fusion then buries past rank 100. Never-pooled gold falls
  from 14.0% to 9.0% at 300 per branch, but gold within the fused top 100 rises only from
  82.8% to 84.3%.
- The 9–14% never pooled sits mostly in broad and citation-derived queries, out of reach of
  a larger pool. That is a first-stage model problem (P2.6 step 4).
