"""Rectified-flow training targets used for InfoVis.

Convention (same as FLUX.2 inference): t = 0 is the clean latent, t = 1 is noise.

    x_t = (1 - t) * x_data + t * noise
    v   = noise - x_data            # regression target for the transformer

t is drawn from a logit-normal distribution and then shifted toward noise
according to the image token count (the "dynamic shift" of FLUX).
"""

import torch


def _shift_mu(num_tokens, x1=256, y1=0.5, x2=4096, y2=1.15):
    """Linear map from token count to shift parameter mu (base 0.5 at 256 tokens, 1.15 at 4096)."""
    m = (y2 - y1) / (x2 - x1)
    return m * num_tokens + (y1 - m * x1)


def sample_timesteps(latents, dynamic_shift=True):
    """Sample one t per latent in `latents` (B, C, H, W)."""
    t = torch.sigmoid(torch.randn(latents.shape[0]))
    if dynamic_shift:
        # Token count of the patchified latent grid, divided by 4 as in the original implementation.
        tokens = (latents.shape[-2] // 2) * (latents.shape[-1] // 2)
        mu = torch.full_like(t, _shift_mu(tokens))
        t = torch.exp(mu) / (torch.exp(mu) + (1 / t - 1))
    return t.to(latents)


def flow_matching_targets(x_data, dynamic_shift=True):
    """Return (t, x_t, v_target) for clean latents `x_data`."""
    noise = torch.randn_like(x_data)
    t = sample_timesteps(x_data, dynamic_shift)
    t_ = t.view(-1, *([1] * (x_data.ndim - 1)))
    x_t = (1 - t_) * x_data + t_ * noise
    return t, x_t, noise - x_data

