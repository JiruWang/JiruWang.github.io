---
title: PPO vs. GRPO in OpenRLHF
date: 2024-10-30 14:30:00
categories: [GRPO, PPO, RL-Algorithms]

mathjax: true
---

# GRPO 

- ### Policy Gradient

  - Trajectory Level Expectation

  $$
  argmax(\pi_{\theta}) J_(\pi_{\theta)} = E_{\tau\sim \pi_{\theta}}[R(\tau)] = \sum_{\tau}R(\tau)P(\tau|\pi_{\theta}) \\
  $$

  - Derivate

  $$
  \nabla J_(\pi_{\theta)} = E_{\tau\sim \pi_{\theta}}[R(\tau)\nabla log(P({\tau|\pi_\theta}))]
  $$

  - Step Level

  $$
  \nabla J_(\pi_{\theta)} = E_{\tau\sim \pi_{\theta}}[R(\tau)\nabla \sum_{t=0}^{t=T_n}log(\pi_\theta({a_t|s_t}))]
  $$

  - Policy Gradient

$$
\nabla J(\pi_{\theta}) = \frac{1}{N} \sum_{n=0}^{N-1} \sum_{t=0}^{T_{n} -1} R_{(T_n)}\nabla log \pi_{\theta}(a_{t}|s_{t})  \\
$$

​		R is the **value func** of the whole trajectory, $\pi_{\theta}$ is $P$ of each action for each step 

- ### GRPO 

  There are three parts in objective func of GRPO.  

   **Preliminary**: the original objective func of policy gradient RL  is the expection of value func on all sampled trajectories: $ E_\tau\sim \pi_{\theta}  [R(\tau)] $, so the objective of GRPO is
  $$
  J = (\frac{1}{G} \sum_{i=1}^{G} ) (\frac{1}{|O_i|} \sum_{t=1}^{|o_i|} ) \{ min [  r_{i,t}(\theta) Adv_{(i,t)} , clip (r_{i,t}(\theta), i-\epsilon, 1+\epsilon) Adv_{(i,t)}] - \beta(KL)\}
  $$
  

  Implention of OpenRLHF

  **step1: compute $\pi_{\theta_{old} }$**:[bs, seq_len]

  ```python
  experience_maker.py
  1. rollout_samples.sequences: list(output.prompt_token_ids) + list(output.outputs[0].token_ids)  # (experience_maker.py: _generate_vllm)
  2. output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids)["logits"]  # size:  [micro_bs, seq_len, voc_size] (actor.py: forward)
  log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature) # [micro_bs, seq_len], compute the log_softmax value of labeled token
  """
  logits = torch.tensor([[[1.0, 2.0, 3.0],  
                          [0.5, 1.5, 2.5]]]) 
  rolled_sequences = torch.tensor([[2, 1]])  # label
  logits_labels = torch.tensor([[3.0, 1.5]])
  
  logsumexp = log(exp(1.0) + exp(2.0) + exp(3.0)) ≈ 3.407
  logsumexp = log(exp(0.5) + exp(1.5) + exp(2.5)) ≈ 2.907
  
  log_probs = logits_labels - logsumexp_values
  """
  action_log_probs_list: len(list) = [bs]
  3. base_action_log_probs_list (reference model)
  """""""""""
  samples_list.action_log_probs： \theta_old
  samples_list.base_action_log_probs
  """""""""""
  
  ```

  **Step2: compute Adv over $\pi_{old}$**,[bs, seq_len]
  $$
  R = Adv_{i} = \frac{r_i - mean(r_1, r_2, ...., r_G)}{std(r_1, r_2, ...., r_G)}
  $$
  

  ```python
  experience_maker.py
  1. group_reward_std: group_reward_stds = (rewards.std(-1, keepdim=True).repeat(1, args.n_samples_per_prompt).reshape(-1)[indices].split(exp_len)) # [rollout_bs, n_samples_per_prompt]
  2. adv
  rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9) # [rollout_bs, n_samples]
  rewards = rewards.reshape(-1)[indices].split(exp_len) # flattened
      for experience, reward in zip(experiences, rewards):
      # rewards:  scalar->sequence. a)For each sample, reward is set on the index of 				last token of output sequence. b)kl_loss is computed token by token among ref and 		actor model
              reward = compute_reward(
                  reward,
                  self.kl_ctl.value,
                  experience.kl,
                  action_mask=experience.action_mask,
                  reward_clip_range=args.reward_clip_range,
              ) # [seq_len]
  	# reward is equal for all tokens ：tensor([[-0.7071, -0.7071, -0.7071,  ..., -0.7071, -0.7071, -0.7071]])
                  args.gamma = 1.0
                  experience.returns = self.get_cumulative_returns(
                      reward,
                      experience.action_mask,
                      args.gamma,
                  )
                  experience.advantages = deepcopy(experience.returns)
              else:
                  raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")
  
  ```

  **Step3: Importance sampling -- off policy**
  $$
  r_{i,t}(\theta) = \frac{\pi_\theta}{\pi_{\theta_{old} } }
  $$    
  ​	$\pi_{\theta}$ -> train_batch, $\pi_{\theta_{old} }$ -> rollout batch.  rollout batch > train_batch, GRPO is off-policy.

  ​	Use policy  $\pi_{\theta_{old} }$  to approximate expection of $\pi_{\theta}$ which haven't been computed.

  **Step4: loss func**

  ```python
  loss.py
  def forward(
          self,
          log_probs: torch.Tensor,
          old_log_probs: torch.Tensor,
          advantages: torch.Tensor,
          action_mask: Optional[torch.Tensor] = None,
          rollout_log_probs: Optional[torch.Tensor] = None,
      ) -> torch.Tensor:
          if self.policy_loss_type == "ppo":
              log_ratio = log_probs - old_log_probs
              ratio = log_ratio.exp()
          elif self.policy_loss_type == "gspo":
              # GSPO: https://arxiv.org/pdf/2507.18071
              if self.enable_vllm_is_correction:
                  log_ratio = log_probs - rollout_log_probs
              else:
                  log_ratio = log_probs - old_log_probs
              ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
              ratio = ratio.exp().unsqueeze(-1) * action_mask
          else:
              raise ValueError(f"Invalid policy loss type: {self.policy_loss_type}")
  
          surr1 = ratio * advantages
          surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages
  
          if self.dual_clip is None:
              # Standard PPO
              loss = -torch.min(surr1, surr2)
          else:
              # Standard PPO clipping
              clip1 = torch.min(surr1, surr2)
              # Dual-clip: additional lower bound for negative advantages
              clip2 = torch.max(clip1, self.dual_clip * advantages)
              # Apply dual-clip: use clip2 for negative advantages, clip1 for positive advantages
              loss = -torch.where(advantages < 0, clip2, clip1)
  
          # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
          vllm_kl = None
          if self.enable_vllm_is_correction and self.policy_loss_type == "ppo":
              vllm_is = torch.exp(old_log_probs - rollout_log_probs).clamp(max=self.vllm_is_truncated_threshold).detach()
              loss = vllm_is * loss
              vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)
  
          loss = (
              masked_mean(loss, action_mask, dim=None)
              if self.token_level_loss
              else masked_mean(loss, action_mask, dim=-1).mean()
          )
          clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
          ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
          return loss, clip_ratio, ppo_kl, vllm_kl
  ```

  



