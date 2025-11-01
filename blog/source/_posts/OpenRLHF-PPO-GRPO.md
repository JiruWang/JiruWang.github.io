---
title: PPO vs. GRPO in OpenRLHF
date: 2024-10-30 14:30:00
categories: [GRPO, PPO, RL-Algorithms]
mathjax: true
---

# RL Theory

### policy based

- Policy gradient (objective: $\nabla J(\pi_{\theta})$):

  $$
  \nabla J(\pi_{\theta}) = \frac{1}{N} \sum_{n=0}^{N-1} \sum_{t=0}^{T_{n} -1} Adv_{(T_n)}\nabla log \pi_{\theta}(a_{t}|s_{t})
  $$

- $ Adv $ (**How much better to perform action $ a_t $ than action?**):

  State-Value Function:

  - $V_{\pi}(s_t)$, the avg reward from (t->T-1), with sampled action $\pi (.|s_t)$

  Action-Value Function:

  - $Q_{\pi}(s_t, a_t)$, the avg reward from (t->T-1), under determined action $a_t$

  Advantage--TD-error:

  \\[
  Q_{\pi}(s_t, a_t) - V_{\pi}(s_{t}) = \mathbb{E}[r_t + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s_{t})] \\\\
  S_{t+1} \sim P(.|s_t, a_t)
  \\]

### actor-critic