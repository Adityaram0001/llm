"""DPO (Direct Preference Optimization, Rafailov et al. '23) — implemented from scratch.

**The RLHF problem DPO sidesteps.** Classic RLHF trains an explicit reward model on preference
pairs, then does PPO against it. DPO's insight: under a Bradley-Terry preference model, the
*optimal* RLHF policy has a closed form in terms of the reward, so the reward can be substituted
out algebraically — training the policy directly on preference pairs with a loss that is
*implicitly* a reward-model objective. No separate reward model, no RL loop. The full derivation
is worked step by step in `notebooks/10_dpo_from_scratch.ipynb`; this module is the ~20 lines of
code that fall out the other end.

**The loss.** For a preference pair `(y_w chosen, y_l rejected)` given prompt `x`:

    L(θ) = -log σ( β · [ (logπ_θ(y_w|x) - logπ_ref(y_w|x)) - (logπ_θ(y_l|x) - logπ_ref(y_l|x)) ] )

`π_θ` is the trainable policy (initialized from the SFT checkpoint); `π_ref` is a FROZEN copy of
that same SFT checkpoint — it anchors the policy so it can't just inflate `logπ(y_w)` for any
old reason, only relative to where it already was. `β` controls how sharply the loss penalizes
preference violations; effectively an inverse KL-penalty strength (larger β = trust the reference
less, allow bigger policy shifts). `logπ(y|x)` here means the SUM of token log-probs over the
assistant span only — reuses the exact same loss mask Part A built for SFT.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llmlab.model import GPT


def sequence_logprobs(model: GPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum of token log-probs over the supervised (non-ignored) positions of each row.

    `y[i, t] == IGNORE_INDEX` (-1) marks a non-assistant / pad position (see `sft_loader.py`,
    `dpo_loader.py`) — excluded from the sum so this is exactly "log-probability the model
    assigns to the assistant response," matching what SFT trains on.
    """
    logits, _ = model(x)  # targets=None -> no internal loss, just logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    mask = y != -1
    safe_y = y.clamp(min=0)  # gather needs a valid index even at masked (-1) positions
    token_lp = log_probs.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
    return (token_lp * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict]:
    """The DPO loss + diagnostics, batched (all four args are shape `[B]`).

    Returns `(loss, metrics)` where metrics has:
      - `reward_chosen` / `reward_rejected`: β·(policy - ref) log-ratio per side — DPO's
        *implicit* reward (no reward model was ever trained, but this quantity plays that role).
      - `reward_margin`: mean(reward_chosen - reward_rejected) — how much the policy currently
        prefers chosen over rejected; this is what the loss pushes up.
      - `reward_accuracy`: fraction of the batch where reward_chosen > reward_rejected — the
        preference-pair analogue of classification accuracy.
      - `kl_chosen` / `kl_rejected`: mean(policy_lp - ref_lp) per side — a cheap, standard DPO
        diagnostic for "how far has the policy drifted from the reference," NOT an exact
        sequence-level KL divergence (that would need the full distribution, not one sample), but
        the direction and magnitude are the right signal to watch for over-optimization.
    """
    reward_chosen = beta * (policy_chosen_lp - ref_chosen_lp)
    reward_rejected = beta * (policy_rejected_lp - ref_rejected_lp)
    logits = reward_chosen - reward_rejected
    loss = -F.logsigmoid(logits).mean()

    rc, rr = reward_chosen.detach(), reward_rejected.detach()
    pc, pr = policy_chosen_lp.detach(), policy_rejected_lp.detach()
    metrics = {
        "reward_chosen": float(rc.mean()),
        "reward_rejected": float(rr.mean()),
        "reward_margin": float((rc - rr).mean()),
        "reward_accuracy": float((rc > rr).float().mean()),
        "kl_chosen": float((pc - ref_chosen_lp).mean()),
        "kl_rejected": float((pr - ref_rejected_lp).mean()),
    }
    return loss, metrics
