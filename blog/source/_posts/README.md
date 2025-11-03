---
title: Falcon-MoE:Building Mixture-of-Experts from Falcon-H1 with Expert-aware GRPO Post-Tuning on math reasoning tasks
date: 205-10-30 14:30:00
categories: Openrlhf
mathjax: true
tags: [Falcon, MoE, Math-Reasoning, GRPO]
---


<div align="center">
  <div>
    <a href="https://huggingface.co/wmere/models" target="_blank">🤗 Model Weights</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#installation">⚙️ Installation Guide</a> | <a href="#Post-Training">🚧 Post-Training</a>
  </div>
</div>


<h2 id="moe">🎉 Introduction</h2>

1. **Architecture Development**: Implementing a high-performance MoE architecture plugin for decoder-only models.
2. **Training Analysis**: Conducting a detailed analysis of expert and router behavior to identify key bottlenecks and underlying issues during GRPO training.
3. **Algorithm Innovation**: Designing a novel Expert-aware GRPO Post-Tuning algorithm to optimize MoE models efficiently.

<h2 id="features">🔥 Features</h2>

1. **MoE architecture**: class FalconH1MoEForCausalLM is implemented for transformers and VLLM.
2. **Router Dump**: User can save routers' weight and gradient during training process for analysis purpose by setting --hook True, --dump_dir  ./


<h2 id="quick-start">🚀 QuickStart</h2>

huggingface: 
```python
# python>=3.12
import torch
from openrlhf.moe_utils import FalconH1MoEForCausalLM
from transformers import AutoTokenizer
model_dir = "wmere/falcon-h1-0.5b-moe"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = FalconH1MoEForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model.to("cuda:6")

input_text = "Solve the following mathematical problem step by step. Please reason carefully and put your final answer within \\boxed{}. Problem: A library cabinet houses five ancient scrolls. The first scroll is 4080 years old. If each scroll is older than the last by half as many years as the last scroll’s age, how old is the fifth scroll? Step by step reasoning: "
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda:6")

pred = model.generate(**inputs, max_length=2048, temperature=0.0)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```
vllm:
```python
from vllm import LLM, SamplingParams
from datasets import load_dataset
from math_verify import parse

import pandas as pd
import os

model_path = "wmere/falcon-h1-0.5b-moe"
sampling_params = SamplingParams(temperature=0.8, max_tokens=512)

dataset = load_dataset("gsm8k", "main")
test_set = dataset['test'].shuffle(seed=42).select(range(1))

model = LLM(model=model_path,enforce_eager=True, max_model_len=2048, gpu_memory_utilization=0.8,swap_space=2)
prompts = [test_set[i]['question'] for i in range(len(test_set))]
output = model.generate(prompts, sampling_params)
print(output[0].text)
```
<h2 id="installation">⚙️ Installation</h2>
follow the guide from openrlhf: https://github.com/OpenRLHF/OpenRLHF?tab=readme-ov-file#installation

<h2 id="performance">📊 Model Performance</h2>
setting: openrlhf/moe_utils/gsm8k.yaml

| Model | GSM8K |
| :---- | :---- |
| FalconH1-base-0.5B | 0.60 |
| FalconH1-base-3B | 0.67 |
| FalconH1-base-MoE-0.5B(2/4) | 0.65 |

<img width="600" height="400" alt="image" src="/images/router.png" />
https://wandb.ai/wmere39-uni/openrlhf_train_ppo/panel/nhe19qxdy?xAxisMin=0   


<h2 id="Post-Training">🚧  Post-Training</h2>
GRPO training falcon-moe-0.5 using a filtered gsm8k dataset:


```
ray start
ray job submit --address="http://127.0.0.1:8267" \
   --runtime-env-json='{"working_dir": "MoE-Tuning"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --init_kl_coef 1e-3 \
    --vllm_gpu_memory_utilization 0.6 \
    --gamma 1.0 \
    --use_kl_loss \
    --enforce_eager \
    --kl_estimator k3 \
    --advantage_estimator group_norm \
    --pretrain 'wmere/falcon-h1-0.5b-moe' \
    --remote_rm_url openrlhf/trainer/ppo_utils/hard_reward_label.py \
    --eval_n_samples_per_prompt 1 \
    --ckpt_path /home/jiru/save_falcon_0_5_moe \
    --save_steps 1000000000 \
    --micro_train_batch_size 1 \
    --train_batch_size 4 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 4 \
    --n_samples_per_prompt 2 \
    --max_epochs 1 \
    --num_episodes 10 \
    --label_key answer \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --eval_steps 1 \
    --actor_learning_rate 5e-1 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data data/train_base.json \
    --input_key input \
    --max_samples 10000000 \
    --packing_samples \
    --normalize_reward \
    --adam_offload \
    --vllm_sync_backend nccl \
    --gradient_checkpointing \
   --use_wandb  API KEY \
```

If you want save router's behavor, 

```
 --dump_step 10 \
--hook True \
 --dump_dir /home/jiru/save_falcon_0_5_moe/dump_moe \
```
