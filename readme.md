# Evaluating Text-to-Image Alignment with VQAScore and CLIPScore

_A reproduction and extension of the 2024 DeepMind VQAScore framework_

This project implements a full evaluation pipeline for measuring text–image semantic alignment using **VQAScore** and **CLIPScore**. We reproduce key ideas from the paper:

> **“VQAScore: A Semantic Metric for Evaluating Text-to-Image Alignment” (DeepMind, 2024)**  
> https://arxiv.org/abs/2404.01291

Our implementation extends the paper with improved scoring logic, stronger CLIP backbones, template sensitivity experiments, and negation-focused analysis using a dataset of **~1600 prompts** sampled from **GenAI-Bench**.

---

## 🚀 Project Overview

This repository contains an end-to-end pipeline for:

- ✔ Generating images using Stable Diffusion
- ✔ Computing **probabilistic VQAScore** using Yes/No likelihood (corrected implementation)
- ✔ Computing **CLIPScore** using **ViT-L/14**
- ✔ Running full semantic alignment evaluation on ~1600 prompts
- ✔ Studying:
  - category-wise trends (simple / compositional / negation)
  - question template sensitivity
  - VQA robustness to negation
- ✔ Producing plots, correlations, and summary statistics
- ✔ Providing an interactive demo to explore metrics on any image/prompt pair

---

## 📂 Repository Structure

```

.
├── notebooks/
│ ├── 00_setup.ipynb # Environment, installs, Drive mount
│ ├── 01_load_dataset.ipynb # Load GenAI-Bench and categorize prompts
│ ├── 02_data_prep.ipynb # Image generation with Stable Diffusion
│ ├── 03_run_metrics_baseline.ipynb # Compute VQAScore + CLIPScore
│ ├── 04_prompt_template_study.ipynb # Template variation & negation experiments
│ ├── 05_analysis_and_plots.ipynb # Score distributions, correlations, histograms
│ ├── 06_demo_app.ipynb # Interactive inference demo
│
├── data/
│ ├── genai_bench_subset.csv # ~1600 selected prompts
│ ├── templates.json # Prompt templates for VQA evaluation
│ └── generated_images/ # SD1.5 image outputs (skipped uploading as the folder is big)
│
├── results/
│ ├── raw_scores.csv # VQA + CLIP scores for all prompts
│ ├── template_scores.csv # Per-template evaluation
│ ├── template_variance_summary.csv # Template sensitivity stats
│ ├── negation_results.csv # Negation-specific scores
│ └── plots/ # All visualization outputs
│
└── README.md

```

---

## 🔍 Methodology

### **Image Generation**

- Model: **Stable Diffusion v1.5**
- Inputs: Prompts sampled from **GenAI-Bench**
- Output: 1600 generated images for evaluation

---

## 🧠 Evaluation Metrics

### **1. VQAScore (Corrected Implementation)**

We replaced the common but flawed fallback:

```

exp(-loss)

```

with a more rigorous probability formulation:

- Compute **negative log-likelihoods** of “Yes” and “No”
- Apply **softmax over the two logits**
- Final score = **P(Yes | image, question)** ∈ [0, 1]

👉 This matches the DeepMind paper’s intended behavior and produces realistic, interpretable scores.

### **2. CLIPScore (Upgraded)**

- Backbone upgraded from **ViT-B/32 → ViT-L/14**
- Results in more expressive similarity signals
- But still limited in detecting structure, relations, and negation

---

## 📊 Results Summary

### **Global Findings**

- **VQAScore** ranges widely (~0.1–0.8) → captures semantic correctness
- **CLIPScore** stays narrow (~0.28–0.34) → insensitive to semantics
- Weak–moderate correlation between metrics
- Negation prompts show:
  - CLIPScore barely changes
  - VQAScore significantly decreases when semantics break

### **Template Sensitivity**

- Most prompts: low variance (0.01–0.04) across templates
- Few prompts show higher sensitivity
- Overall: VQAScore is robust to question phrasing

### **Category-wise Correlations**

```

simple 0.33 Spearman
compositional 0.27
negation 0.11

```

👉 Negation is the hardest semantic challenge for CLIP-based metrics.

---

## 🧪 Representative Plots (saved under `results/plots/`)

- Global score distributions
- VQAScore vs CLIPScore scatter
- CLIP/VQA stats on negation
- Category-wise correlations
- Template variance histogram

---

## 🎛 Interactive Demo (Phase 6)

The notebook `06_demo_app.ipynb` allows:

- Uploading any image
- Typing a prompt
- Choosing a template
- Computing:
  - **VQAScore** (probability)
  - **CLIPScore** (similarity)

Perfect for presentations and qualitative exploration.

---

## 🧩 Key Takeaways

- **High VQAScore** → strong semantic agreement with the prompt
- **Moderate CLIPScore** → loose embedding similarity but not correctness
- VQAScore is significantly more reliable for semantic evaluation
- Negation showcases the largest discrepancy between metrics
- Template choice has mild but notable impact on VQAScore

---

## ⚙️ Compute Requirements

- Google Colab Pro (A100 or L4)
- Works with manual batch sizing for large runs
- VQA model: InstructBLIP Vicuna-7B/instructblip-flan-t5-xl (FP16 recommended)

---

## 📘 References

- VQAScore Paper: https://arxiv.org/abs/2404.01291
- GenAI-Bench Dataset: https://huggingface.co/datasets/BaiqiL/GenAI-Bench
- t2v_metrics Repository: https://github.com/linzhiqiu/t2v_metrics
- Stable Diffusion 1.5: https://huggingface.co/runwayml/stable-diffusion-v1-5
