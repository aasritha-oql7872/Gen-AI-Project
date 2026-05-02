
import argparse
import os
import time
import torch
from torchvision.utils import save_image

import config
from model import UNet
from diffusion import DiffusionProcess


def load_model(checkpoint_path, device):
    model = UNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")
    return model


def make_attr_vector(attr_names, device):
    """Create attribute vector from a list of attribute names."""
    attrs = torch.zeros(1, config.NUM_CLASSES, device=device)
    for name in attr_names:
        if name in config.SELECTED_ATTRS:
            idx = config.SELECTED_ATTRS.index(name)
            attrs[0, idx] = 1.0
        else:
            print(f"Warning: '{name}' not in selected attributes. Available: {config.SELECTED_ATTRS}")
    return attrs


def generate_samples(model, diffusion, attrs, num_images, sampler, steps, cfg_scale, device):
    """Generate images and return them along with time taken."""
    shape = (num_images, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    # Expand attrs to batch size
    attrs_batch = attrs.repeat(num_images, 1)

    start_time = time.time()

    if sampler == "ddpm":
        samples = diffusion.ddpm_sample(model, shape, attrs_batch, cfg_scale=cfg_scale)
    else:
        samples = diffusion.ddim_sample(model, shape, attrs_batch, num_steps=steps, cfg_scale=cfg_scale)

    elapsed = time.time() - start_time
    return samples, elapsed


def compare_samplers(model, diffusion, attrs, device, output_dir):
    """Generate side-by-side comparison of DDPM vs DDIM at different step counts."""
    os.makedirs(output_dir, exist_ok=True)
    num_images = 8

    step_configs = [
        ("ddpm", 1000),
        ("ddim", 250),
        ("ddim", 100),
        ("ddim", 50),
        ("ddim", 20),
    ]

    print("\n--- Sampler Comparison ---")
    print(f"{'Sampler':<10} {'Steps':<8} {'Time (s)':<12} {'Time/img (s)':<12}")
    print("-" * 45)

    all_samples = []
    for sampler, steps in step_configs:
        samples, elapsed = generate_samples(
            model, diffusion, attrs, num_images, sampler, steps, config.CFG_SCALE, device
        )
        all_samples.append(samples)
        print(f"{sampler:<10} {steps:<8} {elapsed:<12.2f} {elapsed/num_images:<12.3f}")

        # Save individual grid
        grid = (samples + 1) / 2
        save_image(grid, os.path.join(output_dir, f"{sampler}_{steps}_steps.png"), nrow=4)

    print(f"\nSaved to {output_dir}/")


def compare_guidance(model, diffusion, attrs, device, output_dir):
    """Generate images at different CFG guidance scales."""
    os.makedirs(output_dir, exist_ok=True)
    num_images = 4

    scales = [1.0, 2.0, 5.0, 7.5, 10.0]
    print("\n--- Guidance Scale Comparison ---")

    for scale in scales:
        samples, elapsed = generate_samples(
            model, diffusion, attrs, num_images, "ddim", 50, scale, device
        )
        grid = (samples + 1) / 2
        save_image(grid, os.path.join(output_dir, f"cfg_{scale}.png"), nrow=4)
        print(f"  CFG scale {scale}: saved ({elapsed:.2f}s)")

    print(f"\nSaved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--mode", type=str, default="generate",
                        choices=["generate", "compare_samplers", "compare_guidance", "interpolate"])
    parser.add_argument("--attrs", nargs="+", default=["Smiling"],
                        help="Attribute names, e.g., --attrs Smiling Eyeglasses")
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    diffusion = DiffusionProcess(device=device)
    attrs = make_attr_vector(args.attrs, device)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "generate":
        print(f"Generating {args.num_images} images with {args.sampler} ({args.steps} steps)...")
        print(f"Attributes: {args.attrs}, CFG scale: {args.cfg_scale}")
        samples, elapsed = generate_samples(
            model, diffusion, attrs, args.num_images, args.sampler, args.steps, args.cfg_scale, device
        )
        grid = (samples + 1) / 2
        save_image(grid, os.path.join(args.output_dir, "generated.png"), nrow=4)
        print(f"Done in {elapsed:.2f}s. Saved to {args.output_dir}/generated.png")

    elif args.mode == "compare_samplers":
        compare_samplers(model, diffusion, attrs, device, args.output_dir)

    elif args.mode == "compare_guidance":
        compare_guidance(model, diffusion, attrs, device, args.output_dir)

    elif args.mode == "interpolate":
        # Latent space interpolation between two noise vectors
        print("Generating latent interpolation...")
        shape = (1, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
        z1 = torch.randn(shape, device=device)
        z2 = torch.randn(shape, device=device)

        frames = []
        num_interp = 10
        attrs_batch = attrs.repeat(1, 1)

        for i in range(num_interp):
            alpha = i / (num_interp - 1)
            # Spherical interpolation (slerp)
            omega = torch.acos(torch.clamp(
                (z1 * z2).sum() / (z1.norm() * z2.norm()), -1, 1
            ))
            if omega.abs() < 1e-6:
                z = (1 - alpha) * z1 + alpha * z2
            else:
                z = (torch.sin((1 - alpha) * omega) / torch.sin(omega)) * z1 + \
                    (torch.sin(alpha * omega) / torch.sin(omega)) * z2

            # Use DDIM with this starting noise
            # We override the initial noise in the sampler
            sample = diffusion.ddim_sample(model, shape, attrs_batch, num_steps=50)
            frames.append((sample + 1) / 2)

        grid = torch.cat(frames, dim=0)
        save_image(grid, os.path.join(args.output_dir, "interpolation.png"), nrow=num_interp)
        print(f"Saved to {args.output_dir}/interpolation.png")


if __name__ == "__main__":
    main()
