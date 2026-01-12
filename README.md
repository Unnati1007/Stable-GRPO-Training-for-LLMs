# Stable-GRPO-Training-for-LLMs

**Reward-Aligned Large Language Model Training using GRPO and LoRA**

This repository presents a **stable, parameter-efficient reinforcement learning fine-tuning pipeline** for large language models. It improves reasoning quality and structured output generation using **Group Relative Policy Optimization (GRPO)** combined with **Low-Rank Adaptation (LoRA)**, implemented entirely with **Tunix**.

---

## 📌 Project Overview

The goal of this project is to enable **stable RL-based fine-tuning of LLMs** while minimizing reward variance and preventing policy collapse. The pipeline supports **multi-objective reward optimization**, **KL-regularized updates**, and a **fully reproducible training workflow** with checkpointing.

This approach is designed for research and experimentation in reward-aligned language modeling, especially where reasoning quality, numerical correctness, and structured outputs matter.

---

## ✨ Key Features

* Parameter-efficient **LoRA fine-tuning**
  *(Rank = 64, Alpha = 64)*
* **Group Relative Policy Optimization (GRPO)** for stable, low-variance RL updates
* **Multi-objective reward design**, including:

  * Answer correctness
  * Numerical consistency
  * Output format adherence
* **KL-regularized optimization** to control policy drift
* Fully **checkpointed and reproducible** training pipeline
* Combined **quantitative and qualitative evaluation**

---

## ⚙️ Core Hyperparameters

| Hyperparameter          | Value                          |
| ----------------------- | ------------------------------ |
| Learning Rate           | `3e-6` (warmup + cosine decay) |
| GRPO β (KL coefficient) | `0.08`                         |
| GRPO ε (clipping)       | `0.2`                          |
| Generations per prompt  | `4`                            |
| Gradient clipping       | `0.1`                          |
| Max generation tokens   | `512`                          |

---

## 🔁 Training Workflow

1. Generate multiple responses per prompt
2. Compute group-relative rewards
3. Optimize the policy using GRPO
4. Save LoRA checkpoints periodically

```python
grpo_trainer.train(train_dataset)
```

The use of multiple generations per prompt enables better exploration and more reliable reward estimation during training.

---

## 📊 Evaluation Metrics

The model is evaluated using both automatic metrics and qualitative inspection:

* **Exact answer accuracy**
* **Partial numerical accuracy** (±10%)
* **Output format accuracy**
* **Qualitative reasoning inspection**

This combination ensures that improvements are meaningful, interpretable, and aligned with task objectives.



## 🧠 Why This Approach?

* **LoRA** drastically reduces memory usage and limits overfitting during fine-tuning
* **GRPO** provides stable, low-variance reinforcement learning updates compared to standard PPO
* **KL regularization** and **gradient clipping** ensure controlled and safe policy optimization
* **Multiple generations per prompt** improve exploration and reward reliability

Together, these design choices enable efficient and stable reward-aligned LLM training.



## 🧰 Tech Stack

* JAX
* Flax (nnx)
* Optax
* Orbax
* Tunix
* LoRA
* GRPO

Built with ❤️ by Unnati Jadon and Keshav Sharma
