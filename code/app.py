"""
Gradio app for interactive face generation.
Run: python app.py
"""

import torch
import gradio as gr
from torchvision.utils import make_grid
import numpy as np

import config
from model import UNet
from diffusion import DiffusionProcess


# ---- Load model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)
ckpt = torch.load("checkpoints/best.pth", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Model loaded (epoch {ckpt['epoch']})")

diffusion = DiffusionProcess(device=device)


def generate(smiling, male, eyeglasses, blond_hair, young,
             mustache, pale_skin, heavy_makeup, bald, bangs,
             sampler, num_steps, cfg_scale, num_images):
    """Generate images with selected attributes."""

    # Build attribute vector
    attr_values = [smiling, male, eyeglasses, blond_hair, young,
                   mustache, pale_skin, heavy_makeup, bald, bangs]
    attrs = torch.tensor([attr_values], dtype=torch.float32, device=device)
    attrs = attrs.repeat(int(num_images), 1)

    shape = (int(num_images), config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)

    with torch.no_grad():
        if sampler == "DDPM (1000 steps)":
            samples = diffusion.ddpm_sample(model, shape, attrs, cfg_scale=cfg_scale)
        else:
            samples = diffusion.ddim_sample(model, shape, attrs,
                                            num_steps=int(num_steps), cfg_scale=cfg_scale)

    # Convert to image grid
    samples = (samples + 1) / 2  # [-1,1] → [0,1]
    grid = make_grid(samples, nrow=4).permute(1, 2, 0).cpu().numpy()
    grid = (grid * 255).astype(np.uint8)
    return grid


# ---- Gradio UI ----
with gr.Blocks(title="Diffusion Face Generator") as demo:
    gr.Markdown("# Face Generation with DDPM/DDIM")
    gr.Markdown("Select attributes, choose a sampler, and generate faces.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Attributes")
            smiling = gr.Checkbox(label="Smiling", value=True)
            male = gr.Checkbox(label="Male")
            eyeglasses = gr.Checkbox(label="Eyeglasses")
            blond_hair = gr.Checkbox(label="Blond Hair")
            young = gr.Checkbox(label="Young", value=True)
            mustache = gr.Checkbox(label="Mustache")
            pale_skin = gr.Checkbox(label="Pale Skin")
            heavy_makeup = gr.Checkbox(label="Heavy Makeup")
            bald = gr.Checkbox(label="Bald")
            bangs = gr.Checkbox(label="Bangs")

            gr.Markdown("### Sampler Settings")
            sampler = gr.Radio(
                ["DDIM (fast)", "DDPM (1000 steps)"],
                value="DDIM (fast)",
                label="Sampler"
            )
            num_steps = gr.Slider(10, 500, value=50, step=10, label="DDIM Steps")
            cfg_scale = gr.Slider(1.0, 15.0, value=5.0, step=0.5, label="Guidance Scale")
            num_images = gr.Slider(1, 16, value=4, step=1, label="Number of Images")

            btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            output = gr.Image(label="Generated Faces")

    btn.click(
        fn=generate,
        inputs=[smiling, male, eyeglasses, blond_hair, young,
                mustache, pale_skin, heavy_makeup, bald, bangs,
                sampler, num_steps, cfg_scale, num_images],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch(share=True)
