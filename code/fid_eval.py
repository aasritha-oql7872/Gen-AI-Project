
# FID Score


import argparse
import os
import torch
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch_fidelity
import config
from model import UNet
from diffusion import DiffusionProcess
from dataset import CelebADataset


def generate_images(model, diffusion, num_samples, cfg_scale, ddim_steps, device, output_dir):
   
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    batch_size = 16
    num_batches = (num_samples + batch_size - 1) // batch_size
    img_idx = 0

    # Random attributes for diversity
    for i in range(num_batches):
        current_batch = min(batch_size, num_samples - img_idx)
        shape = (current_batch, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)

        # Random attribute combinations 
        attrs = (torch.rand(current_batch, config.NUM_CLASSES, device=device) > 0.5).float()

        with torch.no_grad():
            samples = diffusion.ddim_sample(model, shape, attrs,
                                            num_steps=ddim_steps, cfg_scale=cfg_scale)

        samples = (samples + 1) / 2  # [-1,1] → [0,1]

        for j in range(current_batch):
            save_image(samples[j], os.path.join(output_dir, f"{img_idx:05d}.png"))
            img_idx += 1

        print(f"  Generated {img_idx}/{num_samples} images")

    print(f"Generated images saved to {output_dir}/")


def save_real_images(num_samples, output_dir):
   
    os.makedirs(output_dir, exist_ok=True)

    dataset = CelebADataset(split="valid")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    img_idx = 0
    for images, _ in loader:
        images = (images + 1) / 2  # [-1,1] → [0,1]
        for j in range(images.shape[0]):
            if img_idx >= num_samples:
                break
            save_image(images[j], os.path.join(output_dir, f"{img_idx:05d}.png"))
            img_idx += 1
        if img_idx >= num_samples:
            break

    print(f"Real images saved to {output_dir}/ ({img_idx} images)")


def compute_fid(real_dir, fake_dir):
    
    metrics = torch_fidelity.calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=torch.cuda.is_available(),
        fid=True,
        verbose=True,
    )
    return metrics["frechet_inception_distance"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=config.CFG_SCALE)
    parser.add_argument("--ddim_steps", type=int, default=config.DDIM_STEPS)
    parser.add_argument("--real_dir", type=str, default="fid_real")
    parser.add_argument("--fake_dir", type=str, default="fid_fake")
    parser.add_argument("--skip_real", action="store_true",
                        help="Skip saving real images if already saved")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = UNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    diffusion = DiffusionProcess(device=device)

    # Save real images
    if not args.skip_real:
        print(f"\nSaving {args.num_samples} real images...")
        save_real_images(args.num_samples, args.real_dir)

    # Generate fake images
    print(f"\nGenerating {args.num_samples} fake images (CFG={args.cfg_scale}, steps={args.ddim_steps})...")
    generate_images(model, diffusion, args.num_samples, args.cfg_scale,
                    args.ddim_steps, device, args.fake_dir)

    # Compute FID
    print("\nComputing FID...")
    fid = compute_fid(args.real_dir, args.fake_dir)
    print(f"\nFID Score: {fid:.2f}")

    return fid


if __name__ == "__main__":
    main()
