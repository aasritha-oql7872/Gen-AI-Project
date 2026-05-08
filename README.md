# Face Generation using Conditional DDPM

**GitHub:** https://github.com/aasritha-oql7872/Gen-AI-Project

---

## Overview

This project is an implementation of Denoising Diffusion Probabilistic Models (DDPM) from scratch. Involves generating realistic face images by conditioning on face attributes. The project uses the CelebA dataset, and some attributes used are Smiling, Male, Eyeglasses, Blond Hair, and more.

---

## Extra Criteria Pursued

**1. Hyperparameter Tuning:** Used FID score as a metric and evaluated on different sampling parameters.

| CFG Scale | FID Score |
|-----------|-----------|
| 3.0 | 63.39 |
| 5.0 | 67.12 |
| 7.0 | 77.47 |
| 10.0 | 121.68 |

**2. Gradio GUI:** Built an interactive web interface where we can toggle different attributes like facial attributes, CFG scale, DDIM sampling rate, etc.

---

## Model Architecture

We are using a Conditional U-Net model:
1. Initial channel size of 64, which is scaled using multipliers of 1, 2, 4, 8 across stages -> [64, 128, 256, 512]
2. Sinusoidal timestep encoding projected to 256 dimensions using MLP
3. 10-dimensional attribute vector projected to 128 dimensions using MLP
4. Time and attribute embeddings are concatenated and projected to a combined embedding
5. Every encoder and decoder layer has 2 ResNet blocks with GroupNorm and SiLU activations
6. Self-Attention applied at lower resolutions.
7. Skip connections between encoder and decoder
8. Middle block: ResNet Block -> Self-Attention -> ResNet Block
9. Downsampling using strided convolution, upsampling using nearest-neighbour interpolation + conv
10. Adopted Classifier Free Guidance during training, where attributes are dropped at a probability of 0.1


---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download CelebA from Kaggle:
```bash
kaggle datasets download -d jessicali9530/celeba-dataset
unzip celeba-dataset.zip -d data/
```

### 3. Train

```bash
cd code
python train.py
```

Resumes automatically from `checkpoints/latest.pth` if it exists.

### 4. Sample Images

```bash
python sample.py --checkpoint checkpoints/best.pth --sampler ddim --steps 100 --cfg_scale 3.0 --num_images 16 --attrs Smiling
```

Available attributes: `Smiling`, `Male`, `Eyeglasses`, `Blond_Hair`, `Young`, `Mustache`, `Pale_Skin`, `Heavy_Makeup`, `Bald`, `Bangs`

Compare CFG scales:
```bash
python sample.py --checkpoint checkpoints/best.pth --mode compare_guidance --attrs Smiling
```

### 5. Evaluate FID Score

```bash
python fid_eval.py --checkpoint checkpoints/best.pth --num_samples 1000 --cfg_scale 3.0
```

### 6. Run Gradio App

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | CelebA (202,599 images) |
| Image Size | 128×128 |
| Batch Size | 32 |
| Optimizer | Adam (lr=2e-4) |
| Epochs | 100 |
| Noise Schedule | Cosine |
| GPU | NVIDIA A100 |
