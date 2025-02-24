import torch
import numpy as np

class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        """
        Linear noise scheduler for DDPM.

        Args:
            timesteps (int): Number of diffusion timesteps.
            beta_start (float): Starting value of beta (noise variance).
            beta_end (float): Ending value of beta.
        """
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # Cumulative product of alphas
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # sqrt(alpha_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # sqrt(1 - alpha_cumprod)

    def add_noise(self, x_start, noise, timesteps):
        """
        Add noise to the input data at the given timesteps.

        Args:
            x_start (torch.Tensor): Original data (batch_size, *shape).
            noise (torch.Tensor): Noise to add (same shape as x_start).
            timesteps (torch.Tensor): Timesteps for each sample in the batch (batch_size,).

        Returns:
            noisy_data (torch.Tensor): Noisy data at the given timesteps.
        """
        # Gather precomputed values for the given timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(x_start.device)[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[timesteps]

        # Expand dimensions to match x_start shape
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, *([1] * (x_start.dim() - 1)))
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, *([1] * (x_start.dim() - 1)))

        # Add noise to the data
        noisy_data = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_data

    def sample_timesteps(self, batch_size, device):
        """
        Sample random timesteps for a batch.

        Args:
            batch_size (int): Number of timesteps to sample.
            device (torch.device): Device to place the timesteps on.

        Returns:
            timesteps (torch.Tensor): Random timesteps (batch_size,).
        """
        return torch.randint(0, self.timesteps, (batch_size,), device=device)