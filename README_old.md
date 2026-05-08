# Face Generation using Conditional DDPM

A from-scratch implementation of Denoising Diffusion Probabilistic Models (DDPM) for conditional face generation using the CelebA dataset.

## Overview

This project trains a conditional UNet-based diffusion model to generate realistic human faces conditioned on 10 facial attributes. Sampling is done using DDIM for fast inference. An interactive Gradio GUI allows real-time attribute control.

## Features

- DDPM training with cosine noise schedule
- DDIM sampling (100 steps vs 1000 for DDPM)
- Classifier-Free Guidance (CFG) for attribute control
- 10 controllable facial attributes
- FID score evaluation for quantitative comparison
- Interactive Gradio web interface

## Attributes

`Smiling`, `Male`, `Eyeglasses`, `Blond_Hair`, `Young`, `Mustache`, `Pale_Skin`, `Heavy_Makeup`, `Bald`, `Bangs`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train:**
```bash
cd code
python train.py
```

**Sample:**
```bash
python sample.py --checkpoint checkpoints/best.pth --sampler ddim --steps 100 --cfg_scale 7.0 --attrs Smiling --num_images 16
```

**Compare CFG scales:**
```bash
python sample.py --checkpoint checkpoints/best.pth --mode compare_guidance --attrs Smiling
```

**Evaluate FID:**
```bash
python fid_eval.py --checkpoint checkpoints/best.pth --num_samples 1000
```

**Run Gradio app:**
```bash
python app.py
```

## Model Architecture

- **Backbone:** Conditional UNet with channel multipliers (1, 2, 4, 8)
- **Time embedding:** Sinusoidal encoding → 256 dimensions
- **Attribute conditioning:** 10-dim vector → 128 dimensions, injected at each UNet block
- **CFG dropout:** 0.1 during training

