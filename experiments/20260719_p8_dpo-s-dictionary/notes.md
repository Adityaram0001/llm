# 20260719_p8_dpo-s-dictionary — DPO, phase 8 Part C

**What:** first DPO run of the project. Policy AND frozen reference both start from
`20260719_p8_sft-s-dictionary` (Part A's full-FT SFT model). Preference data =
`data/dpo/dictionary_pairs/{train,val}.jsonl` (2740 train / 144 val triples): chosen = the
already-validated phase-7 SFT pairs reused as-is; rejected = a NEW data-factory generation
(`tools/data_factory/tasks/dpo_dictionary_pairs.yaml`, DeepSeek v4-flash, $0.20/2884 pairs)
rotating three deliberate failure modes (wrong_fact / verbose / off_format). lr 5e-6, beta 0.1,
batch_size 16, max_len 640, bf16.

## Stopped early at step 91/172 (53% of 1 epoch) — on purpose, not a crash

Two things justified stopping rather than pushing to the nominal end:

1. **Wall-clock degraded steeply and non-linearly** (per-step time roughly doubled every ~25
   steps: ~4.3s/step early on -> ~15-30s/step by step 90), NOT explained by batch shape/length —
   checked directly: rejected-side padded width averages ~460 tokens with no upward trend across
   the visited step range. Most likely a sustained-heavy-MPS-workload effect (DPO runs FOUR
   forward passes/step — policy x {chosen,rejected}, reference x {chosen,rejected} — on
   long/variably-shaped sequences, vs SFT's one short one) combined with M4 thermal throttling;
   not root-caused further this session (flagged in DECISIONS.md D-053 as a follow-up before any
   bigger/longer DPO run).
2. **The training signal had already saturated hard.** By step 75's eval: val_loss 0.672 -> 0.106,
   val reward_accuracy 79% -> 95.8%, val reward_margin 0.04 -> 8.97 (beta=0.1, so a margin of ~9
   means an average policy/reference log-ratio GAP of ~90 nats — a very large drift for less than
   half an epoch). Continuing mostly deepens an already-large divergence rather than teaching
   something new; interrupting (SIGINT, which `DPOTrainer` catches -> saves `latest.pt` + a real
   registry row) was the right call over letting an increasingly-slow run balloon toward
   multi-hour without checking in first (CLAUDE.md's own rule).

## Result: DPO changed behavior fast, and partly by exploiting a length shortcut

| metric | base | SFT (pre-DPO) | DPO | read |
|---|---:|---:|---:|---|
| stop-rate | 0% | 82% | **99%** | DPO further suppressed the off_format ("doesn't answer") failure mode |
| mean answer length (tok) | 64.0 | 33.9 | **16.5** | answers got MUCH shorter under DPO |
| dict MC accuracy (chance 25%) | 26.5% | 29.5% | 33.0% | small further nudge, still a noisy small probe |
| dict cloze exact-match | 0% | 0% | 0% | unaffected either way |
| pretrain val ppl (forgetting) | 34.93 | 40.10 (+14.8%) | **44.82 (+28.3% vs base)** | +11.8% further forgetting in just 91 steps |
| reward_accuracy vs frozen SFT ref (val) | - | 0.0%* | **95.8%** | *degenerate tie (policy==reference), not "SFT prefers rejected" |
| reward_margin vs frozen SFT ref (val) | - | 0.0000 | **8.9694** | real, large preference shift |

**The honest catch (why answers got so short):** a reference-FREE check — does the model, on its
own, already assign higher raw summed log-prob to chosen over rejected? — says **YES, 95.8% of
the time, even for the PRE-DPO SFT model** (mean gap +623.7 nats). That's not evidence SFT
already understood the wrong_fact/verbose/off_format distinction; it's a **length confound**:
chosen responses average ~59 tokens, rejected average ~461 (the verbose failure mode dominates
that mean), and an un-normalized summed log-prob is mechanically larger (less negative) for a
shorter sequence almost regardless of content. DPO's actual reward (relative to the SAME-length
reference evaluation of the SAME y) is length-invariant by construction — that's WHY the
reference-relative reward, not the raw log-prob, is the right metric — but the training gradient
itself has no such protection, and the qualitative samples (`samples/step_000050.txt`) show it:
answers got terse and vague ("A 'wit.'" for 'go-between'), consistent with the model partly
learning "shorter closes the reward gap" rather than uniformly learning all three failure modes.
This is a well-known real-world RLHF/DPO pitfall (verbosity/length bias in preference data), not
a bug in this implementation — a genuine finding, not a caveat to bury.

## Repro
```
python scripts/build_dpo_pairs.py                      # (re)join chosen+rejected -> data/dpo/
python scripts/dpo.py --config configs/dpo_s_dictionary.yaml
python scripts/eval_dpo.py --dpo-run experiments/20260719_p8_dpo-s-dictionary
```
See D-053. Derivation: `notebooks/09_dpo_from_scratch.ipynb`.

## Follow-ups (not done this session, flagged for later)
- Root-cause the MPS per-step slowdown before a longer/bigger DPO run (bisect: is it the extra
  reference forward passes, the long sequences, or genuinely thermal?).
- Mitigate the length confound: cap the verbose failure mode's length multiplier tighter (closer
  to 2-3x chosen instead of the observed up to ~10x), or length-normalize the DPO loss
  (`logp / len`, a documented DPO variant), before trusting reward_margin as a pure quality signal.
- A completed (non-interrupted) 1-epoch run, once the above are addressed, for a cleaner
  end-of-epoch number to sit next to Parts A/B in the report.
