"""
Training loop for conditional DDPM on CelebA.
Run: python train.py
"""

import os
import torch
from torchvision.utils import save_image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from model import UNet
from diffusion import DiffusionProcess
from dataset import get_dataloader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def train():
    # ---- Setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)

    # Init wandb (optional)
    if HAS_WANDB:
        wandb.init(project="ddpm-celeba", config={
            "schedule": config.SCHEDULE,
            "timesteps": config.TIMESTEPS,
            "lr": config.LEARNING_RATE,
            "base_channels": config.BASE_CHANNELS,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "cfg_drop_prob": config.CFG_DROP_PROB,
            "cfg_scale": config.CFG_SCALE,
        })

    # ---- Model, optimizer, diffusion ----
    model = UNet().to(device)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    diffusion = DiffusionProcess(device=device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Resume from checkpoint if exists ----
    start_epoch = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest.pth")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # ---- Dataloader ----
    dataloader = get_dataloader("train")
    print(f"Dataset size: {len(dataloader.dataset):,} images")

    # ---- Fixed attributes for consistent sample visualization ----
    # Generate samples with same attributes each time to track progress
    fixed_attrs = torch.zeros(config.NUM_SAMPLES, config.NUM_CLASSES, device=device)
    # First 4: smiling
    fixed_attrs[:4, 0] = 1.0
    # Next 4: smiling + glasses
    fixed_attrs[4:8, 0] = 1.0
    fixed_attrs[4:8, 2] = 1.0
    # Next 4: male
    fixed_attrs[8:12, 1] = 1.0
    # Last 4: blonde
    fixed_attrs[12:16, 3] = 1.0

    # ---- Training loop ----
    best_loss = float("inf")

    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, attrs) in enumerate(dataloader):
            images = images.to(device)
            attrs = attrs.to(device)

            # Classifier-free guidance: randomly drop attributes 10% of the time
            drop_mask = torch.rand(images.shape[0], device=device) < config.CFG_DROP_PROB
            attrs[drop_mask] = 0.0

            # Sample random timesteps
            t = torch.randint(0, config.TIMESTEPS, (images.shape[0],), device=device)

            # Compute loss
            loss = diffusion.compute_loss(model, images, t, attrs)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / num_batches

        # Log to wandb
        if HAS_WANDB:
            wandb.log({"loss": avg_loss, "lr": scheduler.get_last_lr()[0], "epoch": epoch})

        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        # ---- Save checkpoint ----
        if (epoch + 1) % config.SAVE_EVERY == 0 or avg_loss < best_loss:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }
            torch.save(ckpt, checkpoint_path)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, os.path.join(config.CHECKPOINT_DIR, "best.pth"))
            print(f"  Checkpoint saved (epoch {epoch})")

        # ---- Generate samples ----
        if (epoch + 1) % config.SAMPLE_EVERY == 0:
            model.eval()
            print("  Generating samples...")
            shape = (config.NUM_SAMPLES, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)

            samples = diffusion.ddim_sample(model, shape, fixed_attrs, num_steps=50)

            # Save grid (undo normalization from [-1,1] to [0,1])
            samples = (samples + 1) / 2
            save_image(samples, os.path.join(config.SAMPLE_DIR, f"epoch_{epoch:04d}.png"), nrow=4)
            print(f"  Samples saved to {config.SAMPLE_DIR}/epoch_{epoch:04d}.png")

            if HAS_WANDB:
                wandb.log({"samples": wandb.Image(os.path.join(config.SAMPLE_DIR, f"epoch_{epoch:04d}.png"))})

    print("Training complete!")
    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    train()
