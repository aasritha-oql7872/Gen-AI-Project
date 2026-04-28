"""
DDPM
"""

import math
import torch
import torch.nn.functional as F
import config


# ---- Noise Schedules ----

def linear_schedule(timesteps):
    return torch.linspace(config.BETA_START, config.BETA_END, timesteps)


def cosine_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionProcess:
    

    def __init__(self, schedule=config.SCHEDULE, timesteps=config.TIMESTEPS, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        # Compute schedule
        if schedule == "cosine":
            betas = cosine_schedule(timesteps)
        else:
            betas = linear_schedule(timesteps)

        # Precompute all the constants we need
        self.betas = betas.to(device)
        self.alphas = (1.0 - betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        # For forward process (adding noise)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

        # For DDPM reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(device)

    def _extract(self, tensor, t, shape):
        """Grab the value at index t for each item in batch, reshape for broadcasting."""
        out = tensor.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    # ---- Forward Process ----

    def q_sample(self, x_0, t, noise=None):
        """Add noise to clean image x_0 at timestep t.
        Returns: noisy image x_t, and the noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    # ---- Training Loss ----

    def compute_loss(self, model, x_0, t, attrs):
        """Simple noise prediction loss (MSE)."""
        x_t, noise = self.q_sample(x_0, t)
        predicted_noise = model(x_t, t, attrs)
        return F.mse_loss(predicted_noise, noise)

    # ---- DDPM Sampling ----

    @torch.no_grad()
    def ddpm_sample(self, model, shape, attrs, cfg_scale=config.CFG_SCALE):
        """Full DDPM sampling: 1000 steps from noise to image."""
        device = self.device
        B = shape[0]
        img = torch.randn(shape, device=device)
        null_attrs = torch.zeros_like(attrs)

        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            # Classifier-free guidance
            noise_cond = model(img, t, attrs)
            noise_uncond = model(img, t, null_attrs)
            predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            # DDPM reverse step
            beta_t = self._extract(self.betas, t, img.shape)
            sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, img.shape)
            sqrt_one_minus_alpha_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, img.shape)

            mean = sqrt_recip_alpha_t * (img - beta_t / sqrt_one_minus_alpha_t * predicted_noise)

            if i > 0:
                variance = self._extract(self.posterior_variance, t, img.shape)
                noise = torch.randn_like(img)
                img = mean + torch.sqrt(variance) * noise
            else:
                img = mean

        return img.clamp(-1, 1)

    # ---- DDIM Sampling ----

    @torch.no_grad()
    def ddim_sample(self, model, shape, attrs, num_steps=config.DDIM_STEPS,
                    cfg_scale=config.CFG_SCALE, eta=0.0):
        """DDIM sampling: fewer steps, deterministic when eta=0."""
        device = self.device
        B = shape[0]
        img = torch.randn(shape, device=device)
        null_attrs = torch.zeros_like(attrs)

        # Create subsequence of timesteps: e.g., [999, 979, 959, ..., 19, 0]
        times = torch.linspace(self.timesteps - 1, 0, num_steps + 1).long().to(device)

        for idx in range(num_steps):
            t_now = times[idx]
            t_next = times[idx + 1]

            t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

            # Classifier-free guidance
            noise_cond = model(img, t_batch, attrs)
            noise_uncond = model(img, t_batch, null_attrs)
            predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            # Get alpha values for current and next timestep
            alpha_bar_t = self._extract(self.alphas_cumprod, t_batch, img.shape)
            alpha_bar_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

            # Predict x_0
            predicted_x0 = (img - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            predicted_x0 = predicted_x0.clamp(-1, 1)

            # Compute sigma (eta=0 → deterministic DDIM, eta=1 → like DDPM)
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_next)
            )

            # Direction pointing to x_t
            direction = torch.sqrt(1 - alpha_bar_next - sigma ** 2) * predicted_noise

            # DDIM step
            noise = torch.randn_like(img) if t_next > 0 and eta > 0 else 0.0
            img = torch.sqrt(alpha_bar_next) * predicted_x0 + direction + sigma * noise

        return img.clamp(-1, 1)
