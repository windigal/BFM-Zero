from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .model import BFMTextOpModel, BFMTextOpModelConfig


def load_bfm_textop_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[BFMTextOpModel, dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model_cfg = BFMTextOpModelConfig(**payload["model_config"])
    model = BFMTextOpModel(model_cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload


@dataclass(slots=True)
class BFMTextOpGenerator:
    model: BFMTextOpModel
    guidance_scale: float = 1.0

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str = "cpu", guidance_scale: float = 1.0) -> "BFMTextOpGenerator":
        model, _ = load_bfm_textop_checkpoint(checkpoint_path=checkpoint_path, device=device)
        return cls(model=model, guidance_scale=guidance_scale)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def history_len(self) -> int:
        return self.model.cfg.history_len

    @property
    def future_len(self) -> int:
        return self.model.cfg.future_len

    @torch.no_grad()
    def sample_future(self, texts: list[str], history: torch.Tensor) -> torch.Tensor:
        history = history.to(self.device)
        future_target = self.model.sample(texts=texts, history=history, guidance_scale=self.guidance_scale)
        return self.model.decode_future_target(future_target)

    @torch.no_grad()
    def rollout_text_stream(self, texts: list[str], history: torch.Tensor) -> torch.Tensor:
        history = history.to(self.device)
        futures: list[torch.Tensor] = []
        current_history = history
        for text in texts:
            future = self.sample_future(texts=[text] * current_history.shape[0], history=current_history)
            futures.append(future)
            current_history = future[:, -self.history_len :]
        return torch.cat(futures, dim=1)
