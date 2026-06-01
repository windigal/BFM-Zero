from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import nn


@lru_cache(maxsize=None)
def build_orthonormal_dct_matrix(length: int) -> torch.Tensor:
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    t = torch.arange(length, dtype=torch.float64)
    k = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    mat = torch.cos(math.pi / length * (t + 0.5) * k)
    mat[0] /= math.sqrt(2.0)
    mat *= math.sqrt(2.0 / length)
    return mat.to(torch.float32)


class DCTFutureCodec(nn.Module):
    def __init__(self, future_len: int, keep_coeffs: int) -> None:
        super().__init__()
        if future_len <= 0:
            raise ValueError(f"future_len must be > 0, got {future_len}")
        if keep_coeffs <= 0 or keep_coeffs > future_len:
            raise ValueError(
                f"keep_coeffs must be in [1, future_len], got keep_coeffs={keep_coeffs}, future_len={future_len}"
            )
        basis = build_orthonormal_dct_matrix(future_len)
        self.future_len = future_len
        self.keep_coeffs = keep_coeffs
        self.register_buffer("dct_basis", basis, persistent=False)
        self.register_buffer("kept_basis", basis[:keep_coeffs], persistent=False)

    def encode(self, future: torch.Tensor) -> torch.Tensor:
        if future.shape[-2] != self.future_len:
            raise ValueError(f"Expected future length {self.future_len}, got shape={tuple(future.shape)}")
        coeffs = torch.einsum("kf,...fd->...kd", self.kept_basis.to(device=future.device, dtype=future.dtype), future)
        return coeffs

    def decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.shape[-2] != self.keep_coeffs:
            raise ValueError(f"Expected coeff length {self.keep_coeffs}, got shape={tuple(coeffs.shape)}")
        future = torch.einsum(
            "fk,...kd->...fd",
            self.kept_basis.transpose(0, 1).to(device=coeffs.device, dtype=coeffs.dtype),
            coeffs,
        )
        return future

