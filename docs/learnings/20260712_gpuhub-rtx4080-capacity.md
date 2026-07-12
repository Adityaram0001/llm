# How far can gpuhub's GPU tiers go? Measuring capacity instead of guessing (RTX 4080 vs RTX 5090)

*Session 2026-07-12, hands-on cloud GPU work with the coding model, following the docs research
and D-027/D-028/D-029 setup. Question: now that the pipeline works end-to-end, what are the
actual limits of these GPU tiers — how big a model, how long a sequence, how large a batch —
and which one should future runs actually use? Started on the $0.25/hr RTX 4080 dry-run tier,
then repeated the same measurement on a real $0.46/hr RTX 5090 for comparison (see the "RTX 5090
comparison" section below the initial RTX 4080 results) — the two-GPU comparison changes the
conclusion, so read to the end rather than stopping at the first cost table.*

## Why measure this at all, instead of just trusting the spec sheet

The instance listing says "32GB VRAM" and that alone tells you almost nothing useful. What
actually determines whether a GPU can do a job for us is **tokens/sec at the point where
throughput stops improving** — the "sweet spot" — plus how memory scales up to that point. This
project already learned this lesson once on the Mac (D-008): the spec sheet (16GB unified
memory, "12.7GB `recommended_max_memory`") was nowhere near where things actually broke (a
throughput cliff at ~1GB `mps_alloc`). Trusting a number you haven't measured on your *actual*
model, on *this* hardware, is exactly the mistake D-018 built `scripts/find_batch_size.py` to
stop us from making. So: rent the cheap GPU, spend a few minutes and a few cents actually
sweeping it, and write down what's real.

## The metric that matters: tokens/sec at the plateau, not peak batch size

`find_batch_size.py` doubles the micro-batch size (1, 2, 4, 8, ...) and, for each one, times a
few forward+backward passes on **random data** (no real corpus needed — this measures compute
and memory, not learning). It stops either at a hard CUDA OOM, or once tok/s stops improving by
more than 5% over the running best ("plateaued"). The number you actually want out of this is
the **tok/s at the plateau**, and the micro-batch size where that happens — that's what you set
in the real training config (`batch.micro_batch`), with `grad_accum` chosen so
`micro_batch * grad_accum * seq_len` hits your target effective batch size (a fixed
hyperparameter — D-018 is explicit that this should never be tuned dynamically mid-run, only
calibrated once per hardware).

## What we found: S-tier (9.71M params, seq_len=512)

```
micro_batch=   1  tokens/sec=    6,298
micro_batch=   2  tokens/sec=   12,463
micro_batch=   4  tokens/sec=   25,178
micro_batch=   8  tokens/sec=   49,484
micro_batch=  16  tokens/sec=   98,757
micro_batch=  32  tokens/sec=  198,088   <- sweet spot
micro_batch=  64  tokens/sec=  131,732   <- WORSE, not a plateau
micro_batch= 128  tokens/sec=   87,258   <- worse again
micro_batch= 256  OOM
```

Up to 32, tokens/sec scales almost perfectly *linearly* with batch size (doubling batch roughly
doubles throughput) — a classic sign that at small batch sizes, the GPU is spending more time on
per-launch overhead (kernel dispatch, small-tensor bookkeeping) than actual compute, so bigger
batches amortize that fixed cost. Past 32, though, throughput doesn't plateau flat — it **drops**,
repeatably (confirmed twice, and again in an isolated single-batch-size script with explicit
`torch.cuda.empty_cache()` between runs, to rule out memory fragmentation as a false cause: 32→
200,707, 64→136,835, 128→87,091 tok/s). Peak memory at those three points was 5.87GB, 11.66GB,
23.25GB — clean linear scaling with batch size, so the memory side behaves exactly as expected;
it's specifically *throughput* that regresses.

**Best working theory:** this is a tiny model (d_model=192, 15 layers) — at small batch it's
launch-overhead-bound, but somewhere past 32 the working set for all the intermediate
activations across 15 layers stops fitting comfortably in the GPU's fast on-chip cache/SM
scheduling sweet spot, so larger batches start paying more in memory-bandwidth/cache-miss cost
than they gain in parallelism. This is a genuinely different pattern from the Mac's cliff (D-008
was a sudden 3-15x collapse at a memory threshold; this is a gradual real regression, no cliff,
until an actual OOM at 256). Filed as an observation, not fully explained — if a future session
wants to dig further, profiling with `torch.profiler` or checking SM occupancy would be the next
step, but it wasn't necessary to unblock anything here: we just use the measured sweet spot.

## Sequence length scaling: same total "budget," different split — confirmed across ALL three tiers

`model_s.yaml` (and `model_m.yaml`/`model_l.yaml`) hard-code `max_seq_len: 512`, and
`GPT.forward()` rejects longer sequences with a `ValueError` — so testing seq_len 1024/2048
needed a temporarily widened `max_seq_len` in throwaway benchmark-only model configs (**this is
a pure throughput/memory measurement, unrelated to phase 5 Wave B's planned RoPE-extrapolation
ablation**, which is specifically about evaluating a model *trained* at 512 on *longer* sequences
than it was trained on — that ablation will need an actual code change to `GPT.forward()`'s
length guard, not just a config bump; flagged again below).

Full matrix, all three tiers × all three sequence lengths, sweet-spot micro-batch only:

| Tier | seq_len=512 | seq_len=1024 | seq_len=2048 | Sweet-spot tokens/step |
|---|---|---|---|---|
| S (9.71M) | mb=32 → 198,088 tok/s | mb=16 → 198,406 tok/s | mb=8 → 191,693 tok/s | **16,384** |
| M (34.62M) | mb=32 → 72,611 tok/s | mb=16 → 70,695 tok/s | mb=8 → 72,535 tok/s | **16,384** |
| L (104.80M) | mb=16 → 42,499 tok/s | mb=8 → 42,655 tok/s | mb=4 → 40,450 tok/s | **8,192** |

The pattern generalizes cleanly, not just an S-tier coincidence: **each tier has its own roughly
fixed "sweet-spot tokens-per-forward-pass" constant** (S and M both settle at ~16,384; L settles
at ~8,192, half that — consistent with L's bigger per-token compute cost saturating the GPU at a
smaller batch×seq_len product). Within a tier, tok/s stays close to flat across all three sequence
lengths (S: 198K/198K/192K; M: 72.6K/70.7K/72.5K; L: 42.5K/42.7K/40.5K) as long as micro-batch is
adjusted to keep batch×seq_len at that tier's constant. **Practical reading: going to a longer
default context window costs essentially nothing in total throughput on this hardware** — you
just need a proportionally smaller micro-batch to stay at the sweet spot. This is genuinely
useful and non-obvious: naive intuition says "longer sequences = more compute = slower," but at
these model sizes the GPU's real bottleneck is the total tokens-per-step "work packet" size, not
sequence length specifically.

## Cross-tier: how throughput drops as the model grows (seq_len=512, sweet-spot only)

| Tier | Params | Sweet-spot micro_batch | tok/s | Peak mem at sweet spot |
|---|---|---|---|---|
| S | 9.71M | 32 | 198,088 | 5.87 GB |
| M | 34.62M | 32 | 72,611 | (not isolated-measured; sweep-reported ~0.29GB is unreliable, see caveat below) |
| L | 104.80M | 16 | 42,499 | (same caveat) |

Roughly: tok/s drops faster than params grow (S→M is 3.6x more params but 2.7x less throughput;
M→L is 3x more params but 1.7x less throughput) — consistent with bigger models getting
relatively more efficient per-parameter (better compute/memory-traffic ratio per FLOP as the
matrices get bigger), which is the normal, expected direction for this kind of scaling.

**Caveat on the `mem=` column in the sweep's default output:** it stayed suspiciously flat
(~0.09/0.29/0.86GB) across every micro-batch size within a tier, which contradicts the isolated
S-tier check's real numbers (5.87 → 11.66 → 23.25GB). This is almost certainly `find_batch_size.py`
reading `torch.cuda.memory_allocated()` at a point in each loop iteration that doesn't reflect
peak usage, rather than `torch.cuda.max_memory_allocated()` after a `reset_peak_memory_stats()`
call (which is what the isolated check used, and which gave numbers matching the OOM message's
real usage). **Trust the isolated-check pattern (linear scaling with batch) over the sweep
script's own memory column** — worth fixing `find_batch_size.py` itself in a future session
(track peak memory properly), but wasn't blocking today's throughput question.

## What this means for planning future runs (the actually useful part)

Using the sweet-spot tok/s as the expected *real* training throughput is justified by one more
finding: the raw fwd+bwd-only benchmark at micro_batch=16 (98,757 tok/s) matched the **actual
full training run's** measured throughput (99,554 tok/s, from `20260712_p4_s-smoke_cloud4080`)
almost exactly — meaning on this GPU, the overhead of the optimizer step, data loading from the
memmap, periodic eval/sampling/checkpointing, and wandb logging is negligible next to the
forward+backward compute itself. That's not obvious in advance — CPU-bound data loading or a
slow optimizer step could easily have eaten a big chunk of that, the way it might on weaker
hardware. So the numbers below are not wild extrapolation; they're one already-validated data
point extended with reasonable confidence, not blind faith:

| Task | Tokens | Time | Cost @ $0.25/hr |
|---|---|---|---|
| One S-tier ablation run (phase 5 protocol: 50-100M tokens) | 75M | ~6.3 min | **~$0.03** |
| M-tier confirmation run (illustrative, 1B tokens) | 1B | ~3.8 hr | **~$0.96** |
| L-tier hero run (D-015's ~2.1B Chinchilla budget) | 2.1B | ~13.7 hr | **~$3.43** |

**The L-tier number is the headline finding.** D-008 originally estimated the 100M-tier hero run
at 1.5-3 weeks on the Mac; D-010 planned an RTX 5090 burst run at "$10-20 overnight" to fix that.
This $0.25/hr *dry-run* tier — not even the 5090 — could plausibly finish the entire hero run in
under 14 hours for roughly **$3-4**, if the sweet-spot micro-batch is used (not the Mac-derived
`micro_batch=16` default currently sitting in the S-tier configs — D-022's Mac-tuned default
isn't gpuhub's optimum; **update train configs to the GPU-measured sweet spot before a real cloud
run**, per D-018's own rule to recalibrate per hardware rather than reuse an old number). This
doesn't retire the RTX 5090 plan — an M/L-tier config hasn't been calibrated on a 5090 yet, and a
short real validation run (not just the synthetic benchmark) should happen before committing a
multi-hour run at any tier — but it does mean the "cheap tier" is worth taking seriously as more
than just a dry-run sandbox.

## RTX 5090 comparison — same methodology, and it changes the conclusion

Same session, same day: switched from the RTX 4080 dry-run instance to a real RTX 5090
($0.46/hr vs $0.25/hr) and ran the identical 9-sweep matrix (3 tiers × 3 seq_lens). This answers
the question the 4080 numbers alone couldn't: is the cheap tier actually the *right* choice, or
just the cheap one?

**The 4080's "throughput regresses past the sweet spot" pattern did NOT reproduce on the 5090.**
Instead, on the 5090, sweeps mostly either kept climbing to the tested ceiling or hit a real CUDA
OOM — never the gradual regression seen on the 4080. Example, S-tier @ seq_len=512:
```
4080:  ...  mb=32 -> 198,088 (sweet spot)  mb=64 -> 131,732 (WORSE)  mb=128 -> 87,258 (worse again)
5090:  ...  mb=32 -> 264,778             mb=64 -> 511,538 (better!) mb=128 -> 627,326 (better still, still climbing)
```
This confirms the caution flagged in D-031's "revisit if" — the 4080's regression is a
real quirk of *that* GPU (plausibly its doubled, non-stock VRAM configuration interacting badly
with larger batches — see the note in the "What we found" section above about it being a modified
card), not a general property of these small models. The "sweet-spot tokens-per-step is roughly
constant per tier" finding still held on the 5090 too, just at a much higher constant — S-tier
locked to ~65,536 tokens/step (4x the 4080's ~16,384) at 512/1024/2048: 627,326 / 607,058 /
569,295 tok/s. L-tier locked to ~16,384 tokens/step (matching the 4080's own S/M-tier constant,
coincidentally — not a cross-GPU pattern, just how the numbers landed): 127,033 / 122,598 /
114,984 tok/s. (M-tier's 512 sweep stopped early at the 5%-plateau threshold with mb=64 only
0.6% behind mb=32 — likely didn't reach its true ceiling; treat that one data point as an
underestimate, unlike the cleaner S/L patterns.)

**The actual conclusion — and it's a clean one: the 5090 is strictly better, not just faster.**
Despite costing 84% more per hour, it's fast enough that every tier is BOTH faster AND cheaper
per completed run:

| Task | 4080 ($0.25/hr) | 5090 ($0.46/hr) | Speedup | $ saved |
|---|---|---|---|---|
| S-tier ablation (75M tok) | 0.11hr / $0.03 | 0.03hr / $0.02 | 3.17x | $0.01 |
| M-tier (1B tok, illustrative) | 3.83hr / $0.96 | 1.28hr / $0.59 | 2.98x | $0.37 |
| L-tier hero run (2.1B tok) | 13.73hr / $3.43 | 4.59hr / $2.11 | 2.99x | $1.32 |

**This updates the earlier framing** (this doc's own cost table above, and D-030's "cheap tier
worth taking seriously" conclusion): the 4080 tier is still useful as a near-free dry-run/
debugging sandbox (a smoke test costs a literal penny either way), but for any real run —
ablation sweeps, confirmation runs, the hero run — **the 5090 wins outright once it's available**,
not as a "pay more for convenience" tradeoff but a genuine free lunch: more than 3x the
throughput for less than 2x the price. The L-tier hero run at ~4.6 hours / ~$2.11 is dramatically
better than either D-008's original "1.5-3 weeks on Mac" or D-010's "$10-20 overnight on a 5090"
estimates — both were reasonable guesses at the time, made before any real measurement existed;
this is the "measure, don't assume" lesson (D-018) paying off concretely.

**One process lesson from this comparison, unrelated to the GPU itself:** setting up the 5090
instance initially hit the *exact same* "python/pip not on PATH" bug from D-029 — because the fix
to `gpuhub_setup.sh` was made locally on the Mac but never committed/pushed to GitHub, and the
convenience one-liner (`bash <(curl -fsSL .../gpuhub_setup.sh)`) pulls from GitHub, not the local
working copy. Fell back to `scp`-ing the locally-fixed script directly (same as the very first
setup), which worked immediately. Lesson: a fix that only exists in an uncommitted local file
isn't really "fixed" for any workflow that fetches from the remote — commit fixes to scripts
promptly, especially ones designed to be pulled by URL.

## RTX PRO 6000 "extreme" test — confirms it's not worth it, and corrects the 5090 comparison

At the user's request to "test to the extreme," a third GPU — RTX PRO 6000, $0.91/hr, 96GB VRAM
— got the most thorough test of the three: 5 sequence lengths (512 through 8192, not just 3) and,
critically, **the sweep was run with early-stopping disabled and no artificial batch-size cap**,
so every single sweep ran to a real CUDA OOM rather than stopping at a heuristic "good enough"
point. All 120 raw data points are in `docs/results/cloud_gpu_benchmarks.csv` alongside the
other two GPUs' data (233 rows total).

**Expected result, now confirmed rather than just predicted: RTX PRO 6000 isn't worth it for
this project.** D-018 reasoned from VRAM math alone that our ~10-105M-param models would never
need PRO 6000's extra VRAM over the 5090's — now measured directly: PRO 6000 has *higher* raw
throughput at every tier (S: 644,000 vs 5090's 627,326 tok/s; M: 246,864 vs 216,199; L: 153,490
vs 127,033) but costs ~2x as much per hour, so it's the *most expensive* option per completed run
at every single tier — even pricier than the cheap 4080 tier:

| Tier (budget) | RTX 4080 | RTX 5090 | RTX PRO 6000 |
|---|---|---|---|
| S (75M tok) | $0.026 | $0.015 | $0.029 |
| M (1B tok) | $0.956 | $0.591 | $1.024 |
| L (2.1B tok) | $3.431 | $2.112 | $3.458 |

**Unexpected and more important: the "throughput regresses past the sweet spot" pattern (first
seen on the 4080) reproduced cleanly on the PRO 6000 too, at every tier** — once tested properly.
S-tier @512 climbs to 644,000 tok/s at micro_batch=128, then drops to 555,119 (mb=256) and
537,031 (mb=512) before a real OOM at mb=1024. Same shape as the 4080's curve, just ~3.3x higher.
**This means the earlier finding "the 5090 doesn't show this regression" (from the RTX 5090
comparison section above) was based on an incomplete test, not a real hardware difference.** The
5090 sweep used a hard `--max-micro-batch 128` cap AND left early-stopping active — a more
conservative methodology than this PRO 6000 run's "push to real OOM" approach. Concrete evidence:
the 5090's M-tier@512 sweep stopped at micro_batch=64 (214,834 tok/s) because that was ~0.6%
below micro_batch=32's 216,199 (triggering the "plateaued" early-stop) — but PRO 6000's uncapped
sweep of the same tier/seq_len kept rising well past that exact point, all the way to a true peak
of 246,864 at the same mb=64. So the 5090's recorded M/L-tier numbers are probably a **lower
bound**, not its true ceiling — worth a same-methodology re-test if a precise number matters, but
it doesn't change the qualitative ranking: PRO 6000 already loses on cost even against the 5090's
conservative numbers, so a corrected (higher) 5090 number only strengthens "5090 is the best
value" further. (S-tier's 5090 numbers are probably fine as-is — PRO 6000's true S-tier peaks
landed at the *exact same* micro-batch as the 5090's capped sweep reported: 128/64/32 at
512/1024/2048 — the cap happened to sit right at the natural optimum for that tier.)

**Lesson for next time:** when comparing hardware, use the *identical* sweep methodology for
every candidate. A partial measurement isn't just "less complete" than a full one — it can
actively mislead a comparison if the gap between "how far GPU A was pushed" and "how far GPU B
was pushed" isn't the same. Measuring beats assuming, but measuring *inconsistently* across a
comparison can reintroduce the same error in a subtler form.

**The tokens-per-step sweet-spot constant (from the seq_len-scaling section above) held up even
more cleanly here**, now confirmed across 5 seq_lengths instead of 3: S-tier's constant is
~65,536 (matching the 5090 exactly); M-tier settles at ~32,768 (a cleaner number than either
other GPU alone showed); **L-tier's constant is ~16,384 tokens/step across all five tested
lengths (512 through 8192) without a single exception** — the strongest confirmation of this
pattern across the whole investigation.

## The corrected 5090 re-test — the user's hunch was right about something real

After the PRO 6000 test exposed the methodology gap, the natural next question (raised by the
user, not assumed) was: "maybe PRO 6000 only pulls ahead at longer context — the short-sequence
tests wouldn't have shown that." That's a genuinely different, falsifiable hypothesis from "the
5090 numbers were just undertested" — so it needed its own check, not just a re-run with better
settings. Re-ran the 5090 with the identical extreme methodology used for PRO 6000 (every sweep
to real OOM) and compared throughput at matching tier/seq_len pairs:

| Tier | seq=512 | seq=1024 | seq=2048 | seq=4096 | seq=8192 |
|---|---|---|---|---|---|
| S | PRO6000 +2.2% | +6.3% | +6.4% | +11.6% | +19.1% |
| M | +14.4% | +16.2% | +17.5% | +20.5% | +25.2% |
| L | +20.4% | +21.5% | +23.3% | +25.9% | +30.3% |

**The hypothesis was confirmed, cleanly, at every tier: PRO 6000's throughput edge over the 5090
grows monotonically with sequence length.** Best explanation: memory bandwidth. Longer sequences
push proportionally more memory traffic per token through attention, and PRO 6000 — a larger,
more complete Blackwell die built for workstation/datacenter use — most plausibly has higher
memory bandwidth than the consumer-tier 5090. This is a real architectural difference, not
measurement noise (it's monotonic and consistent across all three model sizes).

**But — and this is the part worth internalizing — a real architectural advantage doesn't
automatically translate into a real cost advantage.** Even at the widest gap measured (L-tier @
8192, PRO 6000 30.3% faster), the cost still favors the 5090: $3.14 vs $4.77 for the L-tier hero
budget. PRO 6000's ~98% price premium is a bigger number than its largest measured speed
advantage (30.3%) at every combination tested. **The practical recommendation doesn't change —
RTX 5090 remains the right default — but now that's backed by a real long-context data point,
not an assumption that short-sequence results generalize.**

One more thing this re-test caught: the "sweet-spot tokens-per-step is one sharp constant"
framing (from the seq_len-scaling section above) turns out to oversimplify the 5090 specifically
— its S-tier peak is a **broad, flat plateau** (several micro-batches from ~32,768 to ~65,536
tokens/step all within ~1% of each other) rather than one clear winner, unlike the 4080's and
PRO 6000's sharper single-point peaks. The L-tier constant (16,384 tokens/step) still held
cleanly and identically on both GPUs across all 5 seq_lens — so the finding is solid where it
matters most (the tier that'll actually get used for real runs), just noisier at the small-model
end where measurement noise is a bigger fraction of the signal.

**The meta-lesson, worth carrying forward:** the user's pushback here wasn't "I don't trust your
numbers" — it was a specific, falsifiable alternative hypothesis ("maybe it's a long-context
effect specifically"), which is exactly the kind of question that's worth an actual test rather
than a confident-sounding guess either way. It turned out to be right. Good instinct to keep
having.

## Loose end for a future session (not fixed today, just flagged)

`GPT.forward()` hard-rejects any sequence longer than `model_config.max_seq_len` — phase 5 Wave
B's plan ("train at 512, eval ppl at 1024/2048" to show RoPE/ALiBi's length-extrapolation
advantage over learned/sinusoidal encodings) will need that guard relaxed for eval-only forward
passes (or a config path that keeps training capped but allows longer eval sequences) before that
ablation can run. Not a blocker for anything today — Wave B hasn't started — just worth knowing
before that session starts cold.

## Related
- [[20260711_gpu-vocab-datamix]] — the original RTX 5090 vs PRO 6000 reasoning ($/FLOP not VRAM)
  this session's numbers extend with a second, cheaper real data point.
- D-008 (Mac throughput cliff), D-018 (calibrate-once-per-hardware rule), D-022 (Mac's real
  measured numbers), D-027/D-028/D-029 (gpuhub provider choice and live pipeline validation this
  builds on).
