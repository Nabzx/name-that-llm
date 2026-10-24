# ADR-0001: Build on published checkpoints, add our own members

- Status: Accepted
- Date: 2026-10-24
- Issue: #12 (decision in #6)

## Context

Other Kagglers have published fine-tuned checkpoints for this task. Their best public notebook scores 0.639 on the LB and the top team about 0.70 (#2, #24). Our own n-gram SVC reaches 0.48 CV (#26). Training a 14B two-stage LoRA ourselves does not fit on two T4s in the time we have (#5). The event rules allow public pretrained models. On 24 Oct the organisers confirmed by private ticket that other participants' published weights are allowed if the report shows we understand them and what we built on top.

## Decision

Build on published checkpoints and add our own members:

- Published: Nishant Dahal's Qwen3-14B LoRA, Qwen2.5-14B LoRA and DeBERTa-v3-large, and prvsiyan's two Qwen3-8B seeds. All are run through our own loader (#27, #28), which reads two checkpoint layouts.
- Ours: the public pool build (84,470 passages, #19, #26), the four-view n-gram SVC on that pool (0.4838 CV, #63), the blend and half-strength balancing (#20, #23), and lab witnesses (#34).
- Blend weights: published members keep their authors' weights or join at a fixed 10% share, one LB probe each. Only members with honest out-of-fold predictions get fitted weights (#20).

## Consequences

- The LB score can reach the public stack's level, which half the event score rewards.
- The report must explain each published member and say plainly what is ours.
- Our original work is smaller in score than the published members. The witnesses add about +0.003 to +0.005 on our SVC (#34).

## Alternatives considered

- Train everything ourselves on Kaggle. Cleanest story, but capped at about 0.58 to 0.61 by T4 time.
- Lab witnesses alone. Too weak on their own (0.166 CV, #34).
