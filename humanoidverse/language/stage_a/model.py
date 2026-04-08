from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class StageAModelConfig:
    clip_model_name: str = "openai/clip-vit-base-patch32"
    z_dim: int = 256
    hidden_dim: int = 512
    num_chunks: int = 8
    num_decoder_layers: int = 2
    num_decoder_heads: int = 8
    dropout: float = 0.1
    max_text_length: int = 77
    freeze_text_encoder: bool = True
    use_query_latent_decoder: bool = False
    info_nce_temperature: float = 0.07
    learnable_logit_scale: bool = True
    max_logit_scale: float = 100.0


class StageATextToLatentModel(nn.Module):
    def __init__(self, cfg: StageAModelConfig) -> None:
        super().__init__()
        try:
            from transformers import AutoTokenizer, CLIPTextModel
            from transformers.utils import logging as hf_logging
        except ImportError as exc:
            raise ImportError(
                "transformers is required for Stage A. Install with `uv sync` or `pip install transformers`."
            ) from exc

        self.cfg = cfg
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

        text_hidden_dim = self.text_encoder.config.hidden_size
        self.memory_proj = nn.Linear(text_hidden_dim, cfg.hidden_dim)
        self.clip_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.z_dim),
        )
        self.chunk_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.z_dim),
        )
        self.contrastive_text_proj = nn.Linear(cfg.hidden_dim, cfg.z_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_decoder_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        if cfg.use_query_latent_decoder:
            self.latent_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_decoder_layers)
            self.latent_queries = nn.Parameter(torch.randn(cfg.num_chunks + 1, cfg.hidden_dim) * 0.02)
        else:
            self.pool_proj = nn.Sequential(
                nn.LayerNorm(text_hidden_dim),
                nn.Linear(text_hidden_dim, cfg.hidden_dim),
                nn.GELU(),
            )
            self.chunk_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_decoder_layers)
            self.chunk_queries = nn.Parameter(torch.randn(cfg.num_chunks, cfg.hidden_dim) * 0.02)
        if cfg.learnable_logit_scale:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / cfg.info_nce_temperature)))
        else:
            self.register_buffer("logit_scale", torch.tensor(math.log(1.0 / cfg.info_nce_temperature)))

    def project_latent(self, z: torch.Tensor) -> torch.Tensor:
        return math.sqrt(z.shape[-1]) * torch.nn.functional.normalize(z, dim=-1)

    def train(self, mode: bool = True) -> StageATextToLatentModel:
        super().train(mode)
        if self.cfg.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    def encode_text(self, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_length,
            return_tensors="pt",
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        if self.cfg.freeze_text_encoder:
            with torch.no_grad():
                outputs = self.text_encoder(**tokenized)
        else:
            outputs = self.text_encoder(**tokenized)
        memory = self.memory_proj(outputs.last_hidden_state)
        encoded = {
            "memory": memory,
            "memory_key_padding_mask": ~tokenized["attention_mask"].bool(),
        }
        if not self.cfg.use_query_latent_decoder:
            encoded["pooled"] = self.pool_proj(outputs.pooler_output)
        return encoded

    def forward(self, texts: list[str], device: torch.device | None = None) -> dict[str, torch.Tensor]:
        model_device = device or next(self.parameters()).device
        text_features = self.encode_text(texts, model_device)
        if self.cfg.use_query_latent_decoder:
            query = self.latent_queries.unsqueeze(0).expand(len(texts), -1, -1)
            decoded = self.latent_decoder(
                tgt=query,
                memory=text_features["memory"],
                memory_key_padding_mask=text_features["memory_key_padding_mask"],
            )
            clip_token = decoded[:, 0]
            chunk_tokens = decoded[:, 1:]
            clip_pred = self.project_latent(self.clip_head(clip_token))
            chunk_pred = self.project_latent(self.chunk_head(chunk_tokens))
            contrastive_text = torch.nn.functional.normalize(self.contrastive_text_proj(clip_token), dim=-1)
        else:
            clip_pred = self.project_latent(self.clip_head(text_features["pooled"]))
            query = self.chunk_queries.unsqueeze(0).expand(len(texts), -1, -1)
            decoded = self.chunk_decoder(
                tgt=query,
                memory=text_features["memory"],
                memory_key_padding_mask=text_features["memory_key_padding_mask"],
            )
            chunk_pred = self.project_latent(self.chunk_head(decoded))
            contrastive_text = torch.nn.functional.normalize(self.contrastive_text_proj(text_features["pooled"]), dim=-1)

        return {
            "clip_pred": clip_pred,
            "chunk_pred": chunk_pred,
            "contrastive_text": contrastive_text,
            "logit_scale": self.logit_scale.exp().clamp(max=self.cfg.max_logit_scale),
        }


def stage_a_losses(
    outputs: dict[str, torch.Tensor],
    target_clip: torch.Tensor,
    target_chunks: torch.Tensor,
    temperature: float,
    clip_weight: float,
    chunk_weight: float,
    info_nce_weight: float,
    smooth_weight: float,
    clip_cosine_weight: float = 0.0,
    chunk_cosine_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    clip_l1 = torch.nn.functional.smooth_l1_loss(outputs["clip_pred"], target_clip)
    chunk_l1 = torch.nn.functional.smooth_l1_loss(outputs["chunk_pred"], target_chunks)
    clip_cosine = (1.0 - torch.nn.functional.cosine_similarity(outputs["clip_pred"], target_clip, dim=-1)).mean()
    flat_pred_chunks = outputs["chunk_pred"].reshape(-1, outputs["chunk_pred"].shape[-1])
    flat_target_chunks = target_chunks.reshape(-1, target_chunks.shape[-1])
    chunk_cosine = (1.0 - torch.nn.functional.cosine_similarity(flat_pred_chunks, flat_target_chunks, dim=-1)).mean()
    clip_loss = clip_l1 + clip_cosine_weight * clip_cosine
    chunk_loss = chunk_l1 + chunk_cosine_weight * chunk_cosine

    target_clip_norm = torch.nn.functional.normalize(target_clip, dim=-1)
    if "logit_scale" in outputs:
        logits = outputs["contrastive_text"] @ target_clip_norm.T * outputs["logit_scale"]
    else:
        logits = outputs["contrastive_text"] @ target_clip_norm.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    info_nce = 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.T, labels)
    )

    if outputs["chunk_pred"].shape[1] > 1:
        pred_delta = outputs["chunk_pred"][:, 1:] - outputs["chunk_pred"][:, :-1]
        target_delta = target_chunks[:, 1:] - target_chunks[:, :-1]
        smooth_loss = torch.nn.functional.mse_loss(pred_delta, target_delta)
    else:
        smooth_loss = torch.zeros((), device=target_chunks.device)

    total = (
        clip_weight * clip_loss
        + chunk_weight * chunk_loss
        + info_nce_weight * info_nce
        + smooth_weight * smooth_loss
    )
    return {
        "loss": total,
        "clip_loss": clip_loss.detach(),
        "chunk_loss": chunk_loss.detach(),
        "info_nce_loss": info_nce.detach(),
        "smooth_loss": smooth_loss.detach(),
        "clip_cosine_loss": clip_cosine.detach(),
        "chunk_cosine_loss": chunk_cosine.detach(),
    }
