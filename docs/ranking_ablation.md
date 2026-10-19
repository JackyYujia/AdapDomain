# Ranking-only offline replay

`scripts/replay_ranking_ablation.py` compares three candidate-card ranking rules: harmonic CEFF, a Tetris-style demand/residual dot product, and uniform random selection among feasible cards. All three use the same Cube, Vector, and HBM-bandwidth admission limits and the same phase-conditioned slowdown checks. This deliberately isolates the ranking rule; it does not reproduce the full controller, queueing model, or live request goodput.

For the available archive, the phase-arrival trace belongs to the earlier cloud-burst phase study, while the FP16 stage and interference tables describe the later 1x1024 prefill / batch-16 decode study. The command below therefore requires `--acknowledge-cross-cohort` and labels its output `exploratory_cross_cohort`. It reuses the archived trace as an **arrival pattern only**. A trace count is converted to `ceil(count / profile batch)` phase jobs; prefill holds a slot for its p95 stage latency and decode for `128 x p95 ITL`. These fixed holding times and lack of a request-level prefill-to-decode linkage make the output a sensitivity check, not an empirical SLO or throughput result for the manuscript.

From the `AdapDomain` repository root:

```powershell
python scripts/replay_ranking_ablation.py `
  --profiles '..\SOCC_2026_new3latest_20.44\adapdomain_spe_wiley\data\live_llama7b_20261016\stage_profiles.csv' `
  --interference '..\SOCC_2026_new3latest_20.44\adapdomain_spe_wiley\data\live_llama7b_20261016\colocation_interference.csv' `
  --supplement-zip '..\SOCC_2026_new3latest_20.44\adapdomain_spe_wiley\submission_ready_20261010\Supplementary_AdapDomain_20261010.zip' `
  --acknowledge-cross-cohort `
  --output data\results\ranking_ablation_exploratory_20261019.json
```

The preliminary run gives identical admission percentages for all three rankers, although their choice of a shared card when an idle card is available differs. It does not support a claim that harmonic ranking improves end-to-end goodput. For a manuscript-ready ablation, supply a phase/request trace from the same FP16 serving configuration, retain all three identical guards, and report paired outcomes over the same seeds. The script accepts such a CSV through `--trace` (columns `phase,arrival_ms,count`); its `same_cohort_claim_not_verified` status intentionally leaves source verification to the experimenter.

## Risk-matched scorer comparison reported in the manuscript

The primary comparison is CEFF plus X/H/Q/F/U against Tetris-style `q dot R` plus the **same** X/H/Q/F/U, with the same capacity, SLO, and memory hard filters. The raw dot-product variant without risks is a diagnostic, not the main Tetris baseline. Cosine similarity and arithmetic, minimum, and geometric fit are additional same-risk shape controls. `score_with_shared_risks` implements the paper's six weights as a pure function, so a future replay can pass the exact same per-candidate risk vector to each mode. K=2 versus K=3 changes only the scored shape dimensions; the HBM feasibility guard should stay fixed in both.

The existing trace preview does **not** yet use the full risk-aware scorer: the archive has no matching per-candidate queue state or physical-HBM pressure ledger. Its current result must not be relabeled as the proposed Full/Tetris+risk ablation. For a fair combined-score comparison, record the risk ledger from each policy's replay state, compute all five terms by the same functions, and keep their coefficients fixed. Because a raw dot product and CEFF have different numeric ranges, specify a common shape-term scaling or calibrate each coefficient on held-out data before comparing them against the shared risk penalty; report the chosen rule.

The author confirmed that Tables K-O are completed measured results, and their supplied summaries are now reported in the manuscript. The older checked archive does not contain the matched Tetris/cosine/alternative-fit trial records or the LLaMA scorer-substitution runs; the exploratory script above does not generate the manuscript tables. Before submission, add the matched per-seed outputs and aggregation code to a new supplementary revision and update the manuscript's Data Availability Statement to that revision. The existing 19.2 requests/s FP16 Full value and its 12.7% violation rate can be checked in the live-serving summary, but they do not independently verify the other Table N rows or paired p-values. The old high-density ablation lists 5.87% Full violations, while a separate matched scheduler table lists 5.83%; confirm which trial cohort supplies the 0.166 fragmentation value.

Candidate-table audit:

- K: BFD-2D, No-shape, and Full have earlier summary rows. The new Tetris, cosine, and alternative-fit means/CI are author-supplied measured summaries, with trial-level records absent from the checked archive. The 5.87% Full violation rate comes from the older ablation table, while fragmentation 0.166 appears in a matched-control table whose Full violation rate is 5.83%; confirm trial identity in the new release.
- L: the five reported ratios average to 1.11, consistent with a 0.90 Tetris+risk aggregate. Include the matched seed/trial outputs behind the per-workload CIs and paired p-values.
- M: include the K=2 and K=3 scorer-substitution records with the HBM guard held constant; the checked older archive does not contain them.
- N: the FP16 Full value of 19.2 requests/s gives `1 - 19.2/22 = 12.7%`. Include the request-level or per-seed records behind the new ranking rows. Overlapping marginal CIs do not determine a paired-test result.
- O: include precision-specific matched scorer comparisons behind the +4% and +7% increments; the checked older precision summaries compare whole-card and phase-aware policies instead.
