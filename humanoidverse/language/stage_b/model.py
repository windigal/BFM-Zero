from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .frequency import DCTFutureCodec


@dataclass(slots=True)
class BFMTextOpModelConfig:
    clip_model_name: str = "openai/clip-vit-base-patch32"
    prompt_dim: int = 256
    history_len: int = 4
    future_len: int = 16
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    diffusion_steps: int = 5
    max_text_length: int = 77
    freeze_text_encoder: bool = True
    cond_mask_prob: float = 0.1
    normalize_output: bool = True
    objective_type: str = "diffusion"
    sampling_steps: int | None = None
    target_representation: str = "dct"
    dct_keep_coeffs: int | None = 4
    flow_sigma_min: float = 0.0
    flow_time_sampling: str = "logit_normal"
    flow_logit_mean: float = 0.0
    flow_logit_std: float = 1.0
    flow_sampler: str = "euler"


@dataclass(slots=True)
class BFMTextOpLossConfig:
    target_weight: float = 1.0
    future_weight: float = 1.0
    cosine_weight: float = 0.25
    boundary_weight: float = 0.05
    smooth_weight: float = 0.05
    flow_weight: float = 1.0


def project_prompt(z: torch.Tensor) -> torch.Tensor:
    return math.sqrt(z.shape[-1]) * torch.nn.functional.normalize(z, dim=-1)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5).pow(2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999).float()


def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = buffer.to(timesteps.device)[timesteps]
    return out.reshape(timesteps.shape[0], *([1] * (len(x_shape) - 1)))


class GaussianDiffusionSchedule(nn.Module):
    def __init__(self, timesteps: int) -> None:
        super().__init__()
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=alphas.dtype), alphas_cumprod[:-1]], dim=0)

        self.num_timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        posterior_mean = (
            _extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        posterior_log_variance = _extract(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return posterior_mean, posterior_log_variance


class FlowMatchingSchedule(nn.Module):
    def __init__(
        self,
        *,
        sigma_min: float = 0.0,
        time_sampling: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.time_sampling = str(time_sampling).lower().strip()
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.time_sampling == "uniform":
            timesteps = torch.rand(batch_size, device=device)
        elif self.time_sampling == "logit_normal":
            noise = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
            timesteps = torch.sigmoid(noise)
        elif self.time_sampling == "mode":
            timesteps = torch.distributions.Beta(2.0, 2.0).sample((batch_size,)).to(device)
        else:
            raise ValueError(
                f"Unsupported flow_time_sampling={self.time_sampling!r}. "
                "Expected one of: 'uniform', 'logit_normal', 'mode'."
            )
        return timesteps.clamp(1e-5, 1.0 - 1e-5)

    def make_training_sample(
        self,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = target.shape[0]
        device = target.device
        source_noise = torch.randn_like(target)
        timesteps = self.sample_time(batch_size=batch_size, device=device)
        timestep_view = timesteps.reshape(batch_size, *([1] * (target.ndim - 1)))
        x_t = (1.0 - timestep_view) * source_noise + timestep_view * target
        if self.sigma_min > 0:
            x_t = x_t + self.sigma_min * torch.randn_like(target)
        velocity_target = target - source_noise
        return x_t, timesteps, source_noise, velocity_target


class BFMTextOpModel(nn.Module):
    def __init__(self, cfg: BFMTextOpModelConfig) -> None:
        super().__init__()
        try:
            from transformers import AutoTokenizer, CLIPTextModel
            from transformers.utils import logging as hf_logging
        except ImportError as exc:
            raise ImportError(
                "transformers is required for Stage B. Install with `uv sync` or `pip install transformers`."
            ) from exc

        self.cfg = cfg
        self.objective_type = cfg.objective_type.lower().strip()
        if self.objective_type not in {"diffusion", "flow"}:
            raise ValueError(
                f"Unsupported objective_type={cfg.objective_type!r}. Expected 'diffusion' or 'flow'."
            )
        self.diffusion = GaussianDiffusionSchedule(cfg.diffusion_steps) if self.objective_type == "diffusion" else None
        self.flow = (
            FlowMatchingSchedule(
                sigma_min=cfg.flow_sigma_min,
                time_sampling=cfg.flow_time_sampling,
                logit_mean=cfg.flow_logit_mean,
                logit_std=cfg.flow_logit_std,
            )
            if self.objective_type == "flow"
            else None
        )
        self.target_representation = cfg.target_representation.lower().strip()
        if self.target_representation not in {"raw", "dct"}:
            raise ValueError(
                f"Unsupported target_representation={cfg.target_representation!r}. Expected 'raw' or 'dct'."
            )
        self.future_codec = (
            DCTFutureCodec(future_len=cfg.future_len, keep_coeffs=int(cfg.dct_keep_coeffs))
            if self.target_representation == "dct"
            else None
        )
        self.target_len = self.future_codec.keep_coeffs if self.future_codec is not None else cfg.future_len
        self.normalize_target_output = cfg.normalize_output and self.target_representation == "raw"

        previous_hf_verbosity = hf_logging.get_verbosity()
        progress_bar_enabled = hf_logging.is_progress_bar_enabled()
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.clip_model_name)
            self.text_encoder = CLIPTextModel.from_pretrained(cfg.clip_model_name)
        finally:
            hf_logging.set_verbosity(previous_hf_verbosity)
            if progress_bar_enabled:
                hf_logging.enable_progress_bar()

        if cfg.freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

        text_hidden = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_hidden, cfg.hidden_dim)
        self.history_proj = nn.Linear(cfg.prompt_dim, cfg.hidden_dim)
        self.noisy_future_proj = nn.Linear(cfg.prompt_dim, cfg.hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim),
        )

        total_tokens = 2 + cfg.history_len + self.target_len
        self.position_embedding = nn.Embedding(total_tokens, cfg.hidden_dim)
        self.token_type_embedding = nn.Embedding(4, cfg.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
            norm=nn.LayerNorm(cfg.hidden_dim),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.prompt_dim),
        )

    def train(self, mode: bool = True) -> BFMTextOpModel:
        super().train(mode)
        if self.cfg.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        if self.cfg.freeze_text_encoder:
            with torch.no_grad():
                outputs = self.text_encoder(**tokenized)
        else:
            outputs = self.text_encoder(**tokenized)
        text_emb = outputs.pooler_output
        empty_mask = torch.tensor([not text.strip() for text in texts], device=device, dtype=torch.bool)
        if empty_mask.any():
            text_emb = text_emb.clone()
            text_emb[empty_mask] = 0
        return text_emb

    def _mask_text(self, text_emb: torch.Tensor, force_uncond: bool = False) -> torch.Tensor:
        if force_uncond:
            return torch.zeros_like(text_emb)
        if self.training and self.cfg.cond_mask_prob > 0:
            keep_mask = (torch.rand((text_emb.shape[0], 1), device=text_emb.device) >= self.cfg.cond_mask_prob).float()
            return text_emb * keep_mask
        return text_emb

    def encode_future_target(self, future: torch.Tensor) -> torch.Tensor:
        if self.future_codec is None:
            if future.shape[-2] != self.cfg.future_len:
                raise ValueError(f"Expected future length {self.cfg.future_len}, got shape={tuple(future.shape)}")
            return future
        return self.future_codec.encode(future)

    def decode_future_target(self, target: torch.Tensor) -> torch.Tensor:
        if self.future_codec is None:
            if target.shape[-2] != self.cfg.future_len:
                raise ValueError(f"Expected target length {self.cfg.future_len}, got shape={tuple(target.shape)}")
            return target
        return self.future_codec.decode(target)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        texts: list[str],
        history: torch.Tensor,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        if x_t.ndim != 3 or history.ndim != 3:
            raise ValueError("x_t and history must be rank-3 tensors [B, T, D]")
        if x_t.shape[1] != self.target_len or history.shape[1] != self.cfg.history_len:
            raise ValueError(
                f"Expected target_len={self.target_len}, history_len={self.cfg.history_len}, "
                f"got x_t={tuple(x_t.shape)}, history={tuple(history.shape)}"
            )

        batch_size = x_t.shape[0]
        device = x_t.device
        text_emb = self._mask_text(self.encode_text(texts, device=device), force_uncond=force_uncond)
        time_token = self.time_proj(timestep_embedding(timesteps, self.cfg.hidden_dim)).unsqueeze(1)
        text_token = self.text_proj(text_emb).unsqueeze(1)
        hist_tokens = self.history_proj(history)
        fut_tokens = self.noisy_future_proj(x_t)

        tokens = torch.cat([time_token, text_token, hist_tokens, fut_tokens], dim=1)
        position_ids = torch.arange(tokens.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
        type_ids = torch.cat(
            [
                torch.zeros((1, 1), dtype=torch.long, device=device),
                torch.ones((1, 1), dtype=torch.long, device=device),
                torch.full((1, self.cfg.history_len), 2, dtype=torch.long, device=device),
                torch.full((1, self.target_len), 3, dtype=torch.long, device=device),
            ],
            dim=1,
        ).expand(batch_size, -1)
        tokens = tokens + self.position_embedding(position_ids) + self.token_type_embedding(type_ids)
        decoded = self.transformer(tokens)
        pred = self.output_proj(decoded[:, -self.target_len :])
        if self.normalize_target_output and self.objective_type == "diffusion":
            pred = project_prompt(pred)
        return pred

    def predict_model_output(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        texts: list[str],
        history: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        if guidance_scale == 1.0:
            return self(x_t=x_t, timesteps=timesteps, texts=texts, history=history, force_uncond=False)
        cond = self(x_t=x_t, timesteps=timesteps, texts=texts, history=history, force_uncond=False)
        uncond = self(x_t=x_t, timesteps=timesteps, texts=texts, history=history, force_uncond=True)
        return uncond + guidance_scale * (cond - uncond)

    def reconstruct_target_from_model_output(
        self,
        model_output: torch.Tensor,
        *,
        source_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.objective_type == "diffusion":
            return model_output
        if source_noise is None:
            raise ValueError("source_noise is required to reconstruct Stage B flow targets.")
        pred_target = source_noise + model_output
        if self.normalize_target_output:
            pred_target = project_prompt(pred_target)
        return pred_target

    def prepare_training_inputs(
        self,
        target_target: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        if self.objective_type == "diffusion":
            if self.diffusion is None:
                raise RuntimeError("diffusion schedule is missing while objective_type='diffusion'")
            timesteps = torch.randint(
                low=0,
                high=self.diffusion.num_timesteps,
                size=(target_target.shape[0],),
                device=target_target.device,
            )
            source_noise = torch.randn_like(target_target)
            x_t = self.diffusion.q_sample(target_target, timesteps, noise=source_noise)
            return {
                "x_t": x_t,
                "timesteps": timesteps,
                "source_noise": source_noise,
                "objective_target": target_target,
            }
        if self.flow is None:
            raise RuntimeError("flow schedule is missing while objective_type='flow'")
        x_t, timesteps, source_noise, velocity_target = self.flow.make_training_sample(target_target)
        return {
            "x_t": x_t,
            "timesteps": timesteps,
            "source_noise": source_noise,
            "objective_target": velocity_target,
        }

    def predict_target(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        texts: list[str],
        history: torch.Tensor,
        *,
        guidance_scale: float = 1.0,
        source_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_output = self.predict_model_output(
            x_t=x_t,
            timesteps=timesteps,
            texts=texts,
            history=history,
            guidance_scale=guidance_scale,
        )
        pred_target = self.reconstruct_target_from_model_output(model_output, source_noise=source_noise)
        return model_output, pred_target

    def _sample_flow_euler(
        self,
        *,
        texts: list[str],
        history: torch.Tensor,
        guidance_scale: float,
        noise: torch.Tensor | None,
        num_steps: int,
    ) -> torch.Tensor:
        shape = (history.shape[0], self.target_len, self.cfg.prompt_dim)
        x_t = noise if noise is not None else torch.randn(shape, device=history.device, dtype=history.dtype)
        dt = 1.0 / max(num_steps, 1)
        for step in range(max(num_steps, 1)):
            timestep = torch.full(
                (history.shape[0],),
                float(step) * dt,
                device=history.device,
                dtype=history.dtype,
            )
            velocity = self.predict_model_output(
                x_t=x_t,
                timesteps=timestep,
                texts=texts,
                history=history,
                guidance_scale=guidance_scale,
            )
            x_t = x_t + velocity * dt
        return project_prompt(x_t) if self.normalize_target_output else x_t

    def _sample_flow_rk4(
        self,
        *,
        texts: list[str],
        history: torch.Tensor,
        guidance_scale: float,
        noise: torch.Tensor | None,
        num_steps: int,
    ) -> torch.Tensor:
        shape = (history.shape[0], self.target_len, self.cfg.prompt_dim)
        x_t = noise if noise is not None else torch.randn(shape, device=history.device, dtype=history.dtype)
        dt = 1.0 / max(num_steps, 1)

        def velocity_fn(state: torch.Tensor, timestep_value: float) -> torch.Tensor:
            timestep = torch.full(
                (history.shape[0],),
                timestep_value,
                device=history.device,
                dtype=history.dtype,
            )
            return self.predict_model_output(
                x_t=state,
                timesteps=timestep,
                texts=texts,
                history=history,
                guidance_scale=guidance_scale,
            )

        for step in range(max(num_steps, 1)):
            t_val = float(step) * dt
            k1 = velocity_fn(x_t, t_val)
            k2 = velocity_fn(x_t + 0.5 * dt * k1, t_val + 0.5 * dt)
            k3 = velocity_fn(x_t + 0.5 * dt * k2, t_val + 0.5 * dt)
            k4 = velocity_fn(x_t + dt * k3, t_val + dt)
            x_t = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return project_prompt(x_t) if self.normalize_target_output else x_t

    @torch.no_grad()
    def sample(
        self,
        texts: list[str],
        history: torch.Tensor,
        guidance_scale: float = 1.0,
        noise: torch.Tensor | None = None,
        sampling_steps: int | None = None,
        flow_sampler: str | None = None,
    ) -> torch.Tensor:
        if self.objective_type == "flow":
            num_steps = sampling_steps if sampling_steps is not None else (self.cfg.sampling_steps or self.cfg.diffusion_steps)
            sampler = (flow_sampler or self.cfg.flow_sampler).lower().strip()
            if sampler == "rk4":
                return self._sample_flow_rk4(
                    texts=texts,
                    history=history,
                    guidance_scale=guidance_scale,
                    noise=noise,
                    num_steps=num_steps,
                )
            return self._sample_flow_euler(
                texts=texts,
                history=history,
                guidance_scale=guidance_scale,
                noise=noise,
                num_steps=num_steps,
            )

        if self.diffusion is None:
            raise RuntimeError("diffusion schedule is missing while objective_type='diffusion'")
        batch_size = history.shape[0]
        shape = (batch_size, self.target_len, self.cfg.prompt_dim)
        x_t = noise if noise is not None else torch.randn(shape, device=history.device, dtype=history.dtype)
        for step in reversed(range(self.diffusion.num_timesteps)):
            timesteps = torch.full((batch_size,), step, device=history.device, dtype=torch.long)
            _, pred_target = self.predict_target(
                x_t=x_t,
                timesteps=timesteps,
                texts=texts,
                history=history,
                guidance_scale=guidance_scale,
                source_noise=None,
            )
            mean, log_variance = self.diffusion.q_posterior(pred_target, x_t, timesteps)
            if step == 0:
                x_t = mean
            else:
                noise_step = torch.randn_like(x_t)
                x_t = mean + torch.exp(0.5 * log_variance) * noise_step
        return project_prompt(x_t) if self.normalize_target_output else x_t

    def save_checkpoint(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = dict(payload)
        checkpoint["model_config"] = asdict(self.cfg)
        checkpoint["model_state"] = self.state_dict()
        torch.save(checkpoint, path)


def bfm_textop_losses(
    pred_target: torch.Tensor,
    target_target: torch.Tensor,
    pred_future: torch.Tensor,
    target_future: torch.Tensor,
    history: torch.Tensor,
    target_history: torch.Tensor,
    cfg: BFMTextOpLossConfig,
    *,
    use_reconstruction_loss: bool,
) -> dict[str, torch.Tensor]:
    target_l1 = torch.nn.functional.smooth_l1_loss(pred_target, target_target)
    if use_reconstruction_loss:
        future_l1 = torch.nn.functional.smooth_l1_loss(pred_future, target_future)
    else:
        future_l1 = torch.zeros((), device=pred_future.device)
    cosine = 1.0 - torch.nn.functional.cosine_similarity(
        pred_future.reshape(-1, pred_future.shape[-1]),
        target_future.reshape(-1, target_future.shape[-1]),
        dim=-1,
    ).mean()
    pred_boundary = pred_future[:, 0] - history[:, -1]
    target_boundary = target_future[:, 0] - target_history[:, -1]
    boundary = torch.nn.functional.smooth_l1_loss(pred_boundary, target_boundary)
    if pred_future.shape[1] > 1:
        pred_delta = pred_future[:, 1:] - pred_future[:, :-1]
        target_delta = target_future[:, 1:] - target_future[:, :-1]
        smooth = torch.nn.functional.mse_loss(pred_delta, target_delta)
    else:
        smooth = torch.zeros((), device=pred_future.device)
    total = (
        cfg.target_weight * target_l1
        + cfg.future_weight * future_l1
        + cfg.cosine_weight * cosine
        + cfg.boundary_weight * boundary
        + cfg.smooth_weight * smooth
    )
    return {
        "loss": total,
        "target_loss": target_l1.detach(),
        "future_loss": future_l1.detach(),
        "cosine_loss": cosine.detach(),
        "boundary_loss": boundary.detach(),
        "smooth_loss": smooth.detach(),
    }
