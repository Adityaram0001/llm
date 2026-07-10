# TECHNIQUES — research-paper catalog

Every technique this lab will (or won't) implement, with source paper, why it matters,
MPS feasibility, and where it lands in the plan. Priority: ★★★ do, ★★ do if time, ★ stretch.

## Architecture

| Technique | Paper | What it does | MPS OK? | Priority / where |
|---|---|---|---|---|
| Pre-norm | GPT-2; Xiong'20 | norm before sublayer → stable deep training | ✅ | ★★★ baseline (P3) |
| RMSNorm | Zhang & Sennrich '19 | drops mean-centering; cheaper, same quality | ✅ | ★★★ P5-A |
| SwiGLU FFN | Shazeer '20 (GLU variants) | gated FFN; consistent quality win, used by Llama/DeepSeek | ✅ | ★★★ P5-A |
| QK-norm | Henry '20; used in OLMo2 | normalizes Q,K → prevents logit blow-ups | ✅ | ★★ P5-A |
| RoPE | Su '21 (RoFormer) | rotate Q/K by position → relative positions, extrapolation | ✅ | ★★★ P5-B |
| ALiBi | Press '21 | linear attention-score penalty by distance | ✅ | ★★ P5-B |
| NoPE | Kazemnejad '23 | no positional encoding at all (causal mask leaks position) | ✅ | ★★ P5-B (fun) |
| MQA | Shazeer '19 | 1 KV head → tiny KV cache, slight quality loss | ✅ | ★★★ P5-C |
| GQA | Ainslie '23 | KV-head groups; Llama-2/3 default | ✅ | ★★★ P5-C |
| **MLA** | **DeepSeek-V2 '24** | low-rank-compress KV into a latent vector; big cache cut, quality kept; decoupled RoPE part | ✅ (careful impl) | ★★★ P5-C — flagship study |
| Weight tying | Press & Wolf '16 | share embedding & output matrix (~25% of a 100M model!) | ✅ | ★★★ P3 decision |
| **DeepSeekMoE** | **DeepSeek-MoE '24** | many fine-grained experts + shared expert | ✅ small scale | ★★ P5-F |
| Aux-loss-free balancing | **DeepSeek-V3 '24** | per-expert bias instead of aux loss for load balance | ✅ | ★★ P5-F |
| **Multi-Token Prediction** | Gloeckle '24; **DeepSeek-V3** | predict t+1..t+k with extra head(s) → denser signal | ✅ | ★★ P5-F |
| Sliding-window attn | Mistral '23 | local attention window | ✅ | ★ stretch |
| Mamba/SSM | Gu & Dao '23 | linear-time sequence model | ⚠️ no fast kernel | ★ read-only, no impl |

## Optimization & training

| Technique | Paper | What it does | MPS OK? | Priority / where |
|---|---|---|---|---|
| AdamW (tuned) | Loshchilov '17 | decoupled weight decay; THE baseline | ✅ | ★★★ P4 |
| Cosine + warmup | std practice | the default schedule | ✅ | ★★★ P4 |
| **WSD schedule** | MiniCPM '24; DeepSeek-V3 uses variant | warmup-stable-decay → resume-friendly, checkpoints along "stable" reusable | ✅ | ★★★ P5-D |
| Lion | Chen '23 | sign-based optimizer, less memory | ✅ | ★★ P5-D |
| **Muon** | Jordan '24 (nanoGPT speedruns; Kimi/Moonshot K2 scaled it) | orthogonalized momentum for 2D weights; big speedup claims | ✅ (Newton–Schulz is just matmuls) | ★★★ P5-D — flagship study |
| Grad clipping | std | stability | ✅ | ★★★ P4 |
| z-loss | PaLM '22 | penalize huge logits → stability | ✅ | ★★ P5-D |
| bf16 mixed precision | std | 2× memory/speed | ✅ MPS supports bf16 | ★★★ P4 |
| Grad accumulation | std | big effective batch on small RAM | ✅ | ★★★ P4 |
| Gradient checkpointing | Chen '16 | trade compute for memory | ✅ | ★★ P5-E |
| Sequence packing | T5 etc. | no padding waste | ✅ | ★★★ P1/P4 (concat+chunk) |
| Curriculum / data ordering | various | order data easy→hard | ✅ | ★ P5-G variant |
| Scaling laws (mini) | Kaplan '20; **Chinchilla '22** | loss vs params/tokens; fit our own curve | ✅ | ★★ P5-G |
| Depth-μP / μTransfer | Yang '21 | tune small, transfer HPs to big | ✅ math only | ★ stretch read |
| FP8 training | DeepSeek-V3 | 8-bit matmuls | ❌ no MPS support | read-only note |
| FlashAttention | Dao '22 | IO-aware exact attention | ❌ CUDA kernel; use `F.scaled_dot_product_attention` (has fused path on MPS) | concept only |

## Fine-tuning & alignment (phase 8)

| Technique | Paper | Notes |
|---|---|---|
| SFT + loss masking | InstructGPT '22 | mask prompt tokens from loss — implement by hand ★★★ |
| LoRA (from scratch) | Hu '21 | implement the BA decomposition ourselves, then compare vs full FT ★★★ |
| DPO | Rafailov '23 | preference tuning without RL; needs preference pairs from data factory ★★ |
| GRPO | DeepSeek-Math '24 / R1 | group-relative policy optimization; needs a verifiable-reward task (e.g. word→definition match) ★ stretch |
| Distillation | Hinton '15 | teacher = our L model or API model logits→ student S ★★ |

## Reading list ordering (paper club, one per phase-5 wave)

1. Attention Is All You Need (refresher) → 2. GPT-2 → 3. Chinchilla → 4. Llama-1/2 (arch summary)
→ 5. RoFormer → 6. GQA → 7. **DeepSeek-V2** (MLA) → 8. **DeepSeekMoE** → 9. **DeepSeek-V3**
(MTP, aux-free, WSD-ish schedule, fp8 section as culture) → 10. Muon writeup → 11. LoRA →
12. DPO → 13. **DeepSeek-R1** (GRPO, reasoning) as closing read.
