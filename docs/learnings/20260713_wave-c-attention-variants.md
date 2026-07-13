# Wave C — attention variants (MHA / GQA / MQA / MLA): the complete deep dive

*Discussion session 2026-07-13, after running Wave C (4 runs) and implementing MLA + the
incremental KV-cache decode path. This note consolidates everything from the 4 runs' `notes.md`,
`docs/results/ablation_log.md`, `docs/results/wave_c_inference_bench.csv`, `D-038`, and
`notebooks/06_mla_explained.ipynb` into one revision-ready place — with the arithmetic worked out,
the subtleties made explicit, and the scale-extrapolation the individual files don't contain.
No code or specs were changed (discussion-session rule); items needing future action are already
tracked in D-038's "Revisit if" and are re-flagged at the end.*

---

## TL;DR (the one-screen revision box)

Wave C asked: **the four ways of doing attention differ mainly in how much KV cache they need at
inference — does the cheaper cache cost you model quality?** At S-tier (10M params, 98.3M tokens,
seq 512) the answer is **essentially no**:

| variant | val loss | Δ vs MHA control | cache B/tok/layer | cache vs MHA | attn params |
|---|---|---|---|---|---|
| GQA-2 | **3.5107** | −0.0205 | 512 | 2.0× smaller | 2.21M |
| MLA | 3.5146 | −0.0166 | **320** | **3.2× smaller** | 3.23M |
| MHA (control) | 3.5312 | — | 1024 | — | 2.95M |
| MQA | 3.5498 | +0.0186 | 256 | 4.0× smaller | 1.84M |

- **Quality is flat.** All four land in a 0.039 band — only ~2.6× the seed-noise floor (0.0150,
  D-035). The *type* of attention barely moves loss at this scale. **So the decision is made on
  cache, not quality.**
- **MHA is dominated.** It has the biggest cache and (here) the 3rd-best loss. Nothing recommends
  it once you can do GQA/MLA.
- **MLA reproduces DeepSeek-V2's headline at 10M params:** near-MQA cache (3.2× smaller than MHA)
  with near-MHA quality — because it keeps all 4 content heads (rebuilt from a latent) instead of
  collapsing them the way MQA does.
- **Honest catch:** at 10M params the KV cache does **not** buy decode *speed* (decode is
  launch-overhead bound, not memory-bound), and MLA is ~26% slower/token than MHA. The cache's
  payoff at this scale is purely **memory** — which is what unlocks long context / large batch.
- **Verdict fed to phase 9:** default to **GQA** (free 2× cache cut); reach for **MLA** when
  KV-cache memory is the binding constraint; **MQA** only if cache is the single overriding need.

---

## 1. Why Wave C exists: the KV cache is the inference bottleneck

Training and inference stress different resources. In **training** every token in a sequence is
processed in parallel, so attention is compute-bound and the four variants differ mostly in a few
projection matrices — a rounding error. The variants were invented for **inference**, specifically
**autoregressive decoding**, where you emit one token at a time and each new token must attend to
*every* previous token.

**The naïve way** (no cache): to produce token *t* you re-run the whole prefix of length *t*
through all layers. That's O(*t*) work per token, **O(T²) total** for a T-token generation. You'd
recompute the same keys and values over and over.

**The cached way:** the keys and values of past tokens never change once computed (a token's K/V
depends only on that token and its position), so **cache them**. Then each new token is one
forward pass over a single position that reads the cached K/V — O(1) compute per step (ignoring the
attention read itself), O(T) total. This is a huge win, and it's why every real LLM decodes with a
KV cache. `GPT.generate()` was rewritten this session to do exactly this (prefill the prompt once,
then feed 1 token/step against a per-layer cache).

**But the cache costs memory**, and that memory is the problem the variants attack:

```
KV cache size = 2 (K and V) × n_layers × n_kv_heads × head_dim × dtype_bytes × seq_len × batch
                └──────────────── per token, per layer ─────────────────┘
```

At large scale this dominates. Worked example at our **L-tier capstone** shape (`model_l.yaml`:
d_model 576, 24 layers, 9 heads, head_dim 64), if it used plain MHA and ran at a 2048-token chat
context:

- per token per layer = 2 × 9 × 64 × 2 B = **2304 B**
- × 24 layers = **55,296 B/token ≈ 54 KiB/token**
- × 2048 context × batch 1 = **108 MiB for one conversation's cache**
- × batch 16 = **1.7 GiB** — on a 16 GB Mac or even the 32 GB RTX 5090 that is a serious slice of
  memory *just for the cache*, before weights and activations.

That 1.7 GiB is the number GQA/MLA shrink. Everything below is about *how*, and *whether it hurts*.

---

## 2. The four mechanisms, precisely

### 2.1 The MHA → GQA → MQA spectrum: one knob, `n_kv_heads`

Ordinary **multi-head attention** (MHA) gives every one of the `n_heads` query heads its own key
and value head. The insight behind GQA/MQA (Shazeer '19; Ainslie et al. '23) is that **query heads
can share K/V heads** with little quality loss — most of the cache is redundant across heads.

- **MHA:** `n_kv_heads == n_heads`. Full diversity, full cache.
- **GQA** (grouped-query): `1 < n_kv_heads < n_heads`. Query heads split into groups; each group
  shares one K/V head. The industry default (LLaMA-2-70B, Mistral). "2 groups" = `n_kv_heads=2`.
- **MQA** (multi-query): `n_kv_heads == 1`. *All* query heads share a single K/V head. Smallest
  cache, least diversity.

In code this is genuinely one knob. The K/V projections output `n_kv_heads * head_dim`, and just
before the attention matmul the K/V are `repeat_interleave`d up to `n_heads` so the shapes line up:

```python
if self.n_kv_heads != self.n_heads:
    rep = self.n_heads // self.n_kv_heads
    k = k.repeat_interleave(rep, dim=1)   # cache stores n_kv_heads; matmul sees n_heads
    v = v.repeat_interleave(rep, dim=1)
```

The saving is real because you **cache before the repeat** — the cache holds `n_kv_heads` copies,
not `n_heads`. (That `repeat_interleave` is why the divisibility constraint `n_heads % n_kv_heads
== 0` exists — and it's what forced the whole wave to `n_heads=4`; see §7.)

### 2.2 MLA (DeepSeek-V2 §2): compress the representation, not the head count

GQA/MQA shrink the *number* of heads. **MLA shrinks the representation instead** — it keeps all
`n_heads` distinct heads but caches a single **low-rank latent** from which per-head K/V are
*rebuilt on demand*. Full flow for one token `h_t ∈ ℝ^{d_model}` (our head-dim-preserving sizing:
d_c = nope_head_dim = 32, d_r = rope_head_dim = 32, d_v = v_head_dim = 64, r_kv = kv_lora_rank =
128, r_q = q_lora_rank = 192, H = 4 heads):

```
QUERY (not cached — queries are transient):
   c_q  = RMSNorm(W_DQ · h_t)              ∈ ℝ^{r_q=192}
   q    = W_UQ · c_q  → per head [ q_nope (32) | q_rope (32) ]

KEY/VALUE (the latent is what gets CACHED):
   c_kv = W_DKV · h_t                       ∈ ℝ^{r_kv=128}     ◀── CACHED
   [k_nope (32) | v (64)] = W_UKV · RMSNorm(c_kv)   per head
   k_rope = RoPE(W_KR · h_t)                ∈ ℝ^{d_r=32}, ONE head, shared  ◀── CACHED

ASSEMBLE per head:
   q = [q_nope | q_rope]   (dim 64)   attn = softmax(q·kᵀ / √64)
   k = [k_nope | k_rope]   (dim 64)   out  = attn · v          (dim 64)
   v = v                   (dim 64)   → concat H heads → W_O → ℝ^{d_model}
```

Per token, MLA stores only `c_kv` (128 values) + one shared `k_rope` (32 values) = **160 values**,
versus MHA's `2 × H × head_dim = 2 × 4 × 64 = 512 values`. Everything else (`k_nope`, `v`) is
**recomputed** from `c_kv` at read time via `W_UKV`. That recomputation is the compute-for-memory
trade at the heart of MLA (§5).

### 2.3 Why the nope/rope split? (the decoupling — the trickiest part)

This is the piece worth slowing down on, because it's non-obvious and it's the crux of the paper.

RoPE encodes position by **rotating** q and k by an angle proportional to the token's **absolute**
position. Crucially, the rotation is applied *after* the projection: `k_rotated = RoPE(W_K · h, pos)`.

Now suppose we wanted to cache a compressed KV latent `c_kv` and rebuild a *RoPE'd* key from it. The
rebuilt key would be `RoPE(W_UK · c_kv, pos)` — which **depends on `pos`**. But the whole point of
caching `c_kv` is that it's a compact, **position-agnostic** summary we compute *once* and reuse.
If position had to be baked into the cached quantity, we couldn't reuse a single latent — the
compression breaks.

**DeepSeek's fix is to decouple.** Split each query/key into two parts:

- a **content (nope) part** — *no* RoPE, so it's reconstructable from the position-free latent;
- a **rope part** — carries *all* the positional signal. For keys, it's a **single shared head**
  computed straight from `h_t` (`W_KR`) and cached already-rotated.

So position lives only in the small 32-dim `k_rope` slice (cached directly, per token, once), and
the big content slice stays position-free and therefore compressible into `c_kv`. That's why
`MLAAttention` runs its **own** RoPE on just the rope slice and ignores the outer model's
`pos_encoding` entirely.

**Concrete "aha":** in the shape trace (notebook §2b), `k_rope` has shape `(B, 1, T, 32)` — head
dimension **1** — and is broadcast across all 4 heads. All heads share the exact same positional
key; only their *content* keys differ. That single shared positional signal is 32 values/token;
the 4 distinct content keys are rebuilt from the 128-value latent, never cached.

---

## 3. The cache-bytes arithmetic (worked, and extrapolated to scale)

All numbers bf16 (2 bytes/value). Per-token-per-layer is the fundamental quantity; multiply by
`n_layers`, `seq_len`, `batch` for a footprint.

| variant | formula | per tok/layer | × 15 layers (S-tier) | measured empirically |
|---|---|---|---|---|
| MHA | 2·n_kv(4)·64·2 | 1024 B | 15.0 KiB | 15360 B ✓ |
| GQA-2 | 2·2·64·2 | 512 B | 7.5 KiB | 7680 B ✓ |
| MQA | 2·1·64·2 | 256 B | 3.8 KiB | 3840 B ✓ |
| **MLA** | (r_kv 128 + d_r 32)·2 | **320 B** | 4.7 KiB | 4800 B ✓ |

The **empirical** column comes from building a real cache by decoding 256 tokens and reading the
tensor's `.nbytes()` (`scripts/bench_inference.py`) — it matches the analytical formula **exactly**,
which is the point of measuring both: it proves our mental model of "what's in the cache" is right.
(A subtlety worth remembering: this only matched after casting the bench model to **bf16** — a
random fp32 model gave 2× the bytes. Cache size is a dtype question as much as an architecture one;
fp8 KV cache, which we can't do on MPS, would halve every number again.)

**Where MLA sits:** 320 B is **between MQA (256) and GQA (512)**. But unlike MQA — which gets there
by throwing away head diversity (1 shared K/V head) — MLA keeps all 4 content heads. So MLA buys
near-MQA cache *without* MQA's mechanism. That's the whole selling point in one sentence.

**Scale extrapolation (why this matters for the capstone).** The per-layer number scales with
`n_kv_heads·head_dim` (MHA/GQA/MQA) or `r_kv + d_r` (MLA), and the total scales with `n_layers`.
At L-tier (24 layers, 9 heads, head_dim 64) running a 2048 chat context, batch 1:

| L-tier variant | per tok/layer | KiB/token (×24) | cache @2048 ctx |
|---|---|---|---|
| MHA (`model_l.yaml` default) | 2304 B | 54.0 KiB | **108 MiB** |
| GQA-3 (n_kv=3) | 768 B | 18.0 KiB | 36 MiB (3× smaller) |
| MLA-analog (r_kv 256, d_r 64) | 640 B | 15.0 KiB | 30 MiB (3.6× smaller) |

The capstone currently defaults to **MHA at L-tier**. Wave C says that's the wrong default for a
long-context chat model — the cache saving from GQA/MLA is large and (per this wave) free on
quality. This is a real phase-9 recipe input, already noted in D-038 and tied to RW-5's open
"capstone max_seq_len + pos_encoding" decision.

---

## 4. Reading the results carefully

### 4.1 The flat-quality finding, and how to state it honestly

Final val losses span **3.5107 → 3.5498**, a 0.039 range. The seed-noise floor (D-035) is 0.0150 —
so the total spread is ~2.6× the noise floor. This means the differences are **real but small**,
and you must resist over-reading the *ordering*:

- **Defensible claims:** MQA is really worse than MHA (+0.0186, > floor). MLA and GQA are really
  ≥ MHA (each ~−0.017/−0.021, just past floor). The *type* of attention moves loss by at most ~0.04
  at this scale.
- **NOT defensible:** "GQA (3.5107) beats MLA (3.5146)." They're 0.0039 apart — *far* inside the
  noise floor. GQA and MLA are a statistical tie; treat them as equal-quality.

The disciplined reading (per the phase spec's "judge at fixed compute" rule): **since quality is
flat, cache is the tiebreaker, and MHA loses the tiebreak badly.**

### 4.2 Why might GQA/MLA slightly *beat* MHA? (honest speculation)

It's tempting to invent a story ("fewer K/V heads regularize"). Resist it. −0.02 at a 0.015 noise
floor is *barely* significant on a single seed; it could easily be noise that a 3-seed repeat would
wash out. The honest statement is "no worse than MHA," not "better than MHA." If we ever needed to
claim a real quality *win* for GQA/MLA, we'd re-run each at 3 seeds — we did not, because the
decision (cache) doesn't depend on it.

### 4.3 Reading the tradeoff plot (`wave_c_attention_variants.png`, right panel)

x = cache bytes/token/layer (cheaper left), y = final val loss (better down). The four points tell
the story at a glance:

- **MHA** sits far **right** (expensive cache) and mid-height — **dominated**: GQA and MLA are both
  down-and-left of it (cheaper *and* at-least-as-good). A dominated point should never be chosen.
- **MLA and GQA** cluster **low** (best quality) at **low-mid** cache — the good region.
- **MQA** is **far left** (cheapest) but **highest** (worst quality) — the corner you pick only if
  cache is everything.

The left panel (loss curves) is deliberately boring: all four overlap almost perfectly for the
entire run. That overlap *is* the finding — "attention type barely matters for quality here."

---

## 5. The honest inference-speed story (cache ≠ speed at this scale)

This is the most counter-intuitive part and the one most worth internalizing, because it corrects a
common misconception: **"KV cache makes generation faster"** and **"smaller cache ⇒ faster
decode."** Neither held at our scale, and the bench shows why.

Measured decode throughput (`wave_c_inference_bench.csv`, RTX 5090, bf16, generate 128 tokens):

| variant | cached tok/s (@512) | no-cache tok/s (@512) | cached/nocache |
|---|---|---|---|
| MHA | 116.0 | 133.2 | **0.87×** (cache *slower!*) |
| GQA-2 | 108.9 | 124.1 | 0.88× |
| MQA | 108.5 | 124.8 | 0.87× |
| MLA | 85.8 | 94.3 | 0.91× |

Two surprises to explain:

**(a) The cache made decode *slower*, not faster.** Two reasons, both about our tiny model:
1. **Launch-overhead bound.** At 10M params, a single-token forward pass is dominated by kernel
   *launch latency*, not matmul time — the GPU is almost idle either way. The cached path does
   *more* kernel launches per step (the cache `torch.cat`, the per-step bookkeeping), so it loses.
2. **O(T²) cat.** Our cache appends with `torch.cat`, which **reallocates and copies the whole
   cache every step** — O(T) copy per step, O(T²) total. Real serving pre-allocates the cache to
   max length and writes in place (O(1)/step). We deliberately kept the simple version for
   readability; the O(T²) cat is exactly what eats the theoretical win at tiny compute.

   *(Caveat on the no-cache column: it caps context at `max_seq_len=512` (`idx[:, -512:]`), so at
   ctx > 512 it isn't even doing the same computation — it silently truncates. The only
   apples-to-apples comparison is at ctx ≤ 512, and even there cached loses, for reasons 1–2.)*

   **The lesson:** the KV cache's speed benefit needs *either* a model big enough that per-token
   matmul dominates launch overhead, *or* a pre-allocated cache. Its **memory** benefit, by
   contrast, is unconditional — which is why Wave C's robust result is the bytes table, not tok/s.

**(b) MLA is ~26% slower/token than MHA** (85.8 vs 116.0 cached). MLA does *extra* matmuls at
decode: it re-expands `k_nope` and `v` from the cached latent every step via `W_UKV`, where MHA just
reads K/V from the cache. So **MLA trades compute for cache memory** — you pay in FLOPs to save
bytes. Two production optimizations we skipped (both noted in notebook §4) would largely erase this:

- **Weight absorption.** The score `q^C·k^C = (W_UQ c_q)·(W_UK c_kv) = c_q·(W_UQᵀ W_UK)·c_kv` — the
  two up-projections can be **folded into one matrix** acting directly on the latents, so you never
  materialize per-head K at decode. It's inference-time linear algebra, not a different model.
- **Pre-allocated cache** (kills the O(T²) cat above).

So the fair summary is: **MLA's decode is slower in our un-optimized implementation; a real serving
stack would make it competitive while keeping the cache win.** We measured the honest, unoptimized
number and said so, rather than quoting the paper's optimized claim.

---

## 6. Subtleties and misconceptions worth pinning down

**"Smaller cache ⇒ fewer parameters."** *False, and MLA is the counterexample.* Attn params:
MQA 1.84M < GQA 2.21M < MHA 2.95M < **MLA 3.23M**. MLA has the **largest** attention parameter count
(the down- and up-projection matrices) yet the 2nd-**smallest** cache. Params (a *training/storage*
cost, paid once) and KV cache (an *inference/runtime* cost, paid per token per concurrent request)
are **different budgets**. MLA spends the cheap one (params) to save the expensive-at-serving one
(cache). At our 10M scale the extra params even make MLA the biggest *model* of the wave — an
artifact of tiny scale; at DeepSeek's scale the low-rank projections are a rounding error and MLA is
roughly param-matched to MHA.

**"MLA is just MQA with extra steps."** *No.* MQA caches one shared K and one shared V head — every
query head sees identical K/V. MLA caches a latent from which **all H content heads are rebuilt
distinctly** — the heads are as diverse as MHA's; only the *positional* key is shared. That's why
MLA keeps MHA-quality where MQA doesn't. Same-ish cache size, fundamentally different expressiveness.

**"The cache stores the RoPE'd keys."** For MHA/GQA/MQA, *yes* — we cache K/V **post-RoPE** (RoPE is
position-fixed, so it's computed once). For MLA, we cache the **position-free latent** `c_kv` plus
the **already-rotated** shared `k_rope` — the content keys are rebuilt *without* RoPE at read time.
This asymmetry is the entire reason the nope/rope decoupling exists (§2.3).

**"All four are interchangeable since quality is flat."** Only *for quality, at this scale, on this
corpus*. They differ enormously on cache (4× span) and non-trivially on decode compute (MLA ~26%
slower). "Flat quality" is a statement about one axis; the choice is made on the others.

---

## 7. The methodological lesson: why the wave runs at n_heads=4

The baseline S-tier model has **`n_heads=3`**. GQA "2 groups" needs `n_heads` divisible by the group
count, and **3 is prime** — the only valid `n_kv_heads` at 3 heads are 1 (MQA) and 3 (MHA). GQA-2 is
**mathematically undefined** there. To study the full MHA→GQA→MQA→MLA spectrum with head geometry
held fixed, the entire wave was run at **`n_heads=4, head_dim=64`** (attention inner dim 256, o_proj
256→192), and the **4-head MHA run became the wave's internal control** — *not* the 3-head
`p4_s_baseline`.

Two lessons here, both generalizable:

1. **The single-variable discipline forced the base config.** Comparing MHA-3 vs GQA-4 vs MLA-4
   would confound "attention type" with "head count." Keeping *everything* fixed except the one
   variable (n_kv_heads / attention type) required a common head count, which required abandoning
   the 3-head baseline *for this wave only*. When an ablation variable interacts with a structural
   constraint, you change the base and re-establish a matched control — you don't smuggle two
   changes into one comparison.
2. **This is why the deltas are vs the MHA-4 control, and why they're small.** The MHA-4 control
   (val 3.5312) is itself slightly different from the 3-head p4_s_baseline (3.5037) — extra head,
   more attn params, different init draw. Wave C's deltas are *within* the 4-head family, which is
   the correct, clean comparison; comparing any Wave C run to the 3-head 3.5037 would be the
   confounded mistake.

(The noise floor 0.0150 was measured on the *3-head* baseline. Strictly, the 4-head family could
have a slightly different noise floor; we reused 0.0150 as the best available estimate and kept the
verdicts conservative — none of them hinge on sub-0.02 differences except the GQA-vs-MLA tie, which
we correctly declined to call.)

---

## 8. What Wave C does *not* establish (and when to revisit)

- **S-tier only.** Every quality number is 10M params / 98.3M tokens / seq 512. The MLA-vs-GQA
  quality gap (currently a tie) could open *or* close at M/L scale or with longer training. The
  flat-quality finding is expected to *hold* (bigger models tolerate cache compression at least as
  well), but "expected" isn't "measured." **Before choosing an attention variant for the phase-9
  capstone, re-run this comparison at the capstone's tier/context.** (D-038 "Revisit if".)
- **Speed numbers are un-optimized and tiny-scale.** The tok/s table reflects an O(T²)-cat cache
  and no MLA absorption, on a launch-bound 10M model. It says nothing about MLA's *optimized*
  throughput at scale. If MLA is chosen for the capstone, implement absorption + a pre-allocated
  cache first. (D-038 "Revisit if".)
- **No long-context quality probe.** Wave B did length-extrapolation for positional encodings;
  Wave C did not test whether the cache variants degrade at long context (they shouldn't — it's the
  same math — but untested). MLA's decoupled RoPE means its extrapolation behavior ≈ RoPE's, which
  Wave B already characterized (degrades gracefully, ppl 33→46 over 512→2048).
- **These are not new rework rows.** Both revisit items are already captured in D-038; per the
  discussion-session rule, this note logs no new decisions or RW rows.

---

## 9. Connections (where Wave C plugs into the rest of the project)

- **Wave A (norms/activations):** QK-norm was Wave A's real win. It's *orthogonal* to attention
  type — QK-norm normalizes q/k before the score, and would apply on top of GQA/MLA. A capstone
  recipe could stack qk_norm + GQA/MLA. (Our MLA impl doesn't currently apply qk_norm to the
  assembled q/k; if MLA is chosen for the capstone that combination is worth an explicit test.)
- **Wave B (positional):** MLA *is* a positional-encoding decision in disguise — its decoupled RoPE
  is the only positional mechanism in an MLA model. Wave B's "RoPE degrades gracefully, ALiBi
  extrapolates best" applies to the rope slice. An MLA-with-ALiBi variant isn't standard and we
  didn't build it.
- **Wave F (MoE/MTP) — the spec's "C before F" ordering:** MoE changes the *FFN*, MLA changes
  *attention*; DeepSeek-V2/V3 use **both together**. Getting MLA correct and cached first means
  Wave F can layer MoE onto a known-good attention block. The KV-cache infrastructure built here
  (per-layer cache objects, cached `generate`) is also what a MoE decode path will reuse.
- **Phase 8/9 (fine-tuning, capstone chat):** the incremental KV-cache decode path is the single
  most reusable artifact from this wave — real chat inference *needs* it. The `generate(use_cache=
  True)` prefill-then-step loop is the same shape a chat server uses.
- **Papers:** Shazeer '19 (MQA), Ainslie et al. '23 (GQA), **DeepSeek-V2 '24 §2 (MLA)**. Reading
  order in `docs/TECHNIQUES.md`: GQA → DeepSeek-V2 (MLA) → DeepSeekMoE → DeepSeek-V3.

---

## 10. Learning checkpoints (the notebook's questions, answered)

1. **What's in an MHA cache vs an MLA cache, per token per layer?** MHA: post-RoPE K and V at
   `n_kv_heads` heads = `2·n_kv·head_dim` values (S-tier MHA: 512 values = 1024 B). MLA: the
   position-free latent `c_kv` (r_kv=128) + one shared already-rotated `k_rope` (d_r=32) = 160
   values = 320 B. MLA rebuilds per-head `k_nope`/`v` from `c_kv` at read time.
2. **Why does RoPE force the nope/rope split?** Because RoPE bakes *absolute position* into the key
   *after* projection. A key rebuilt from a position-free cached latent would then have to be
   re-rotated per position, making the cached quantity position-dependent and un-reusable. Splitting
   off a small position-free content slice (reconstructable from the latent) and a small dedicated
   rope slice (carrying all position, cached directly) keeps the bulk compressible.
3. **Why is MLA's cache smaller than GQA's here but its decode compute larger?** Smaller cache: it
   stores a 160-value latent+key vs GQA-2's `2·2·64=256` values. Larger compute: it *rebuilds*
   per-head K/V from the latent every decode step (extra `W_UKV` matmul), whereas GQA just reads
   cached K/V. Memory-for-compute, absorbable in production.

---

## 11. Key takeaways to carry forward

1. **The variants are an inference-memory story, not a quality story.** At S-tier they're
   quality-equivalent (spread ≈ 2.6× noise floor); choose on cache.
2. **MHA is dominated.** Biggest cache, no quality edge. Don't default to it.
3. **GQA is the free lunch:** 2× cache cut, zero quality cost, one-line code. The sane default.
4. **MLA is the specialist:** near-MQA cache with near-MHA quality, via a compressed latent +
   decoupled RoPE. Worth its complexity only when KV-cache memory is the binding constraint —
   and only with absorption + a pre-allocated cache to recover decode speed.
5. **MQA is the cache extreme:** cheapest, and the first to cost quality.
6. **Cache ≠ speed at small scale.** The cache's guaranteed win is *memory*; its *speed* win needs
   scale or a pre-allocated cache. Measure the honest number, don't quote the paper's.
7. **Params ≠ cache.** MLA has the most params and nearly the least cache — different budgets.
8. **Single-variable discipline can force the base config** (n_heads=4 here). Change the base,
   re-establish a matched control, never confound two variables.

**Open questions for later:** does the flat-quality result hold at M/L tier and 2k context? What's
MLA's *optimized* (absorbed, pre-allocated) decode throughput at scale? Does qk_norm + MLA stack
cleanly? All deferred to the phase-9 capstone attention decision (D-038, RW-5).
