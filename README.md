# AI-tasks-using-unsloth.ai

A hands-on collection of five end-to-end training recipes using **Unsloth** and **Hugging Face** tooling. Each part is a self-contained Colab/Notebook script that demonstrates a different training strategy on small, fast models you can run on a single T4/L4 GPU.

## Why this repo?
To make it easy to learn/teach modern LLM training workflows—from smoke-testing and supervised finetuning, through **LoRA**, preference optimization (**DPO/GRPO**), to continued pretraining—without heavy infra.

## 📦 Repository layout
```
.
├── Part 1 Fine tuning using SmolLM2/         # Full SFT sanity run on AG News (fp32)
├── Part 2 LoRA Finetuning/                   # LoRA + 4-bit SFT on AG News
├── Part 3 RL where both input and preferred… # Preference learning (DPO) + merge to FP16 + eval
├── Part 4 RL with GRPO/                      # GRPO toy math tasks + custom reward
└── Part 5 Continued PreTraining/             # Small Hindi continued pretraining example
```
Each folder contains a single notebook/script you can run as-is on Colab.

## 🛠️ Prerequisites
- Python 3.10+  
- A GPU is recommended (Colab T4 works). Scripts print CUDA/device info and fall back to CPU if needed.  
- Hugging Face account for some workflows (optional but useful).

## 🚀 Quick start (Colab)
Open the notebook in the part you want and run the cells top-to-bottom. Every script prints a ✅ “Step complete” banner after each stage so you always know where you are.  
**Tip:** Unsloth must be imported before transformers/peft/datasets so its patches activate.

## 🧩 Parts overview

### Part 1 — Full finetuning SmolLM2 (SFT sanity check)
**Model:** `HuggingFaceTB/SmolLM2-135M` (loaded via `FastLanguageModel`)  
**Data:** `ag_news` (falls back to `dbpedia_14` if needed)  
**Flow:** Environment check + smoke generation → Deterministic splits → SFT training → Eval  
**Install & run:**
```bash
!pip -q install --upgrade "unsloth>=2025.10.0" "transformers==4.57.1"   "accelerate>=1.10.0" "datasets>=2.20.0" "trl>=0.23.0" "peft>=0.17.1" sentencepiece
from unsloth import FastLanguageModel
```
**Why fp32?** Avoids GradScaler issues on T4 for first-run sanity.

### Part 2 — LoRA finetuning (4-bit) on AG News
**Model:** `unsloth/smollm2-135m` with 4-bit quantization + LoRA adapters.  
**Flow:** Install → Device checks → LoRA attach → Train → Save → Infer  
**Install:**
```bash
!pip -q install --upgrade unsloth datasets accelerate bitsandbytes wandb huggingface_hub   "transformers==4.57.1" "trl>=0.10.0"
```
**Outputs:** `outputs_lora_agnews/`, `/content/SmolLM2-135M-AGNews-LoRA`

### Part 3 — Preference optimization with DPO + merge
**Model:** `unsloth/smollm2-135m` (LoRA 4-bit, merged FP16).  
**Data:** `Dahoas/full-hh-rlhf` / `Dahoas/synthetic-instruct-gptj-pairwise` / `Anthropic/hh-rlhf`  
**Flow:** Reduce → DPOTrainer → Train → Merge → Eval  
**Trainer args:** `lr=5e-5`, `batch=2`, `grad_accum=8`.  
**Outputs:** `preference_rl_model/`, `preference_rl_model_merged/`

### Part 4 — RL with GRPO (toy math, custom reward)
**Model:** `unsloth/smollm2-135m` + LoRA (4-bit optional).  
**Flow:** Prompt build → Reward fn → GRPOTrainer → Eval  
**Install:**
```bash
!pip -q install -U "unsloth>=2025.11.2" "transformers==4.56.2" "trl==0.22.2"   "peft>=0.17.1" accelerate datasets bitsandbytes
```
**Config:** `num_generations=4`, adjust batch multiple, prompt length.

### Part 5 — Continued pretraining (Hindi mini-corpus)
**Model:** `unsloth/llama-3-8b-Instruct-bnb-4bit` + LoRA.  
**Data:** Tiny Hindi corpus (replace with your dataset).  
**Flow:** Tokenize → Train causal LM → Save → Hindi generation  
**Install:**
```bash
!pip install -q unsloth accelerate bitsandbytes datasets transformers peft
```

## 📋 Common options & tips
- Import Unsloth before other libs.  
- Use fp32 on first run, then fp16/bf16.  
- LoRA targets: `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`  
- Set `tokenizer.pad_token = tokenizer.eos_token` if missing.  
- For memory: `gradient_checkpointing="unsloth"`.  
- For inference: `FastLanguageModel.for_inference(model)`.

## 📈 Expected footprints (T4)
| Part | Duration | Memory |
|------|-----------|--------|
| 1 | 8–10 min | 6–8 GB |
| 2 | 5–8 min | 4–6 GB |
| 3 | 10–15 min | +1–2 min merge |
| 4 | 10–20 min | varies |
| 5 | few mins | small |

## 🧪 Minimal smoke test
```python
from unsloth import FastLanguageModel
import torch
model, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/smollm2-135m",
    max_seq_length=1024,
    dtype=torch.float16,
    load_in_4bit=False,
    device_map="auto",
)
text = "You are a helpful assistant. Say hello in one short sentence."
out = model.generate(**tok(text, return_tensors="pt").to(model.device), max_new_tokens=24, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

## 🧰 Troubleshooting
- `ValueError: chat_template` → Use plain `"User: ...
Assistant:"` prompts.  
- OOM/CUDA → Reduce batch size or sequence length.  
- “Frozen” training → Lower `logging_steps`.  
- Disable W&B:  
```python
import os
os.environ["WANDB_DISABLED"] = "true"
```
