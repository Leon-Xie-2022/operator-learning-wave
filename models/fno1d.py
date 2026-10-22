"""1D Fourier Neural Operator for terminal-state wave-equation learning.

The model learns the operator `(u0, c) -> u(., T)` for the conservative
variable-coefficient wave equation.

Canonical data-loader contract:
- input:  `(batch, N, 2)` with channels `[u0, c]`
- output: `(batch, N, 1)` with channel `[uT]`

The normalized coordinate `x` is appended internally before the lifting layer,
so the lifted feature at each grid point is `[u0(x), c(x), x]`.

The PDE uses homogeneous Dirichlet endpoints. By default the network enforces
that structure at the output by multiplying the projected field by
`4*x*(1-x)`, which is zero at `x=0` and `x=1` and order-one in the interior.
The optional real-space padding is kept as a compatibility knob for standard
FNO experiments, but it is disabled by default because it is not the main
boundary treatment for this fixed-endpoint problem.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierLayer1D(nn.Module):
    """Spectral convolution plus the local FNO branch `Wv(x)`."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.width = width
        self.modes = modes

        scale = 1.0 / (width * width)
        self.weights_real = nn.Parameter(scale * torch.randn(width, width, modes))
        self.weights_imag = nn.Parameter(scale * torch.randn(width, width, modes))
        self.local = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, n_grid = x.shape
        x_fft = torch.fft.rfft(x, dim=-1)
        kept_modes = min(self.modes, x_fft.shape[-1])
        x_fft_trunc = x_fft[:, :, :kept_modes]

        weight_real = self.weights_real[:, :, :kept_modes]
        weight_imag = self.weights_imag[:, :, :kept_modes]

        out_real = torch.einsum("bim,oim->bom", x_fft_trunc.real, weight_real) - torch.einsum(
            "bim,oim->bom", x_fft_trunc.imag, weight_imag
        )
        out_imag = torch.einsum("bim,oim->bom", x_fft_trunc.real, weight_imag) + torch.einsum(
            "bim,oim->bom", x_fft_trunc.imag, weight_real
        )

        out_fft = torch.zeros(batch_size, channels, x_fft.shape[-1], device=x.device, dtype=x_fft.dtype)
        out_fft[:, :, :kept_modes] = torch.complex(out_real, out_imag)
        spectral = torch.fft.irfft(out_fft, n=n_grid, dim=-1)
        return spectral + self.local(x)


class FNO1D(nn.Module):
    """1D FNO for `(u0, c) -> u(x, T)` on a fixed normalized grid.

    Forward contract
    ----------------
    `forward(x)` accepts the FNO batch tensor returned by
    `data.dataset_loader.WaveEquationDataset`:

    - `x.shape == (batch, N, 2)`
    - `x[..., 0]` is the initial displacement `u0`
    - `x[..., 1]` is the wave speed `c`
    - return shape is `(batch, N, 1)`

    Parameters
    ----------
    modes:
        Number of low Fourier coefficients retained in each spectral layer.
    width:
        Hidden channel width after lifting.
    num_layers:
        Number of Fourier layers.
    padding:
        Optional one-sided real-space padding used in standard FNO baselines.
        It is disabled by default; homogeneous Dirichlet structure is handled
        by the output envelope instead.
    enforce_dirichlet:
        If True, multiply the raw prediction by `4*x*(1-x)` so the endpoints
        are exactly zero for unnormalized Dirichlet targets.
    """

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        num_layers: int = 4,
        padding: int = 0,
        enforce_dirichlet: bool = True,
    ):
        super().__init__()
        if padding < 0:
            raise ValueError(f"padding must be non-negative, got {padding}")

        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.padding = padding
        self.enforce_dirichlet = enforce_dirichlet

        self.lifting = nn.Linear(3, width)
        self.fourier_layers = nn.ModuleList([FourierLayer1D(width, modes) for _ in range(num_layers)])
        self.projection = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.lifting.weight)
        nn.init.zeros_(self.lifting.bias)
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"FNO1D expects input shape (batch, N, 2), got {tuple(x.shape)}")
        if x.shape[1] < 3:
            raise ValueError(f"FNO1D expects at least three grid points, got N={x.shape[1]}")
        if x.shape[-1] != 2:
            raise ValueError(f"FNO1D expects channels [u0, c], got {x.shape[-1]} channels")

    def _append_grid(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        grid = torch.linspace(0.0, 1.0, x.shape[1], device=x.device, dtype=x.dtype)
        grid = grid.view(1, -1, 1).repeat(x.shape[0], 1, 1)
        return torch.cat([x, grid], dim=-1)

    def _apply_dirichlet_envelope(self, raw_output: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        envelope = 4.0 * grid * (1.0 - grid)
        return raw_output * envelope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map FNO loader input `(batch, N, 2)` to prediction `(batch, N, 1)`."""
        x = self._append_grid(x)
        grid = x[..., -1:]
        x = self.lifting(x).permute(0, 2, 1)

        if self.padding > 0:
            x = F.pad(x, (0, self.padding))

        for idx, layer in enumerate(self.fourier_layers):
            x = layer(x)
            if idx < len(self.fourier_layers) - 1:
                x = F.gelu(x)

        if self.padding > 0:
            x = x[..., :-self.padding]

        output = self.projection(x.permute(0, 2, 1))
        if self.enforce_dirichlet:
            output = self._apply_dirichlet_envelope(output, grid)
        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_complexity(self) -> dict:
        return {
            "modes": self.modes,
            "width": self.width,
            "num_layers": self.num_layers,
            "padding": self.padding,
            "enforce_dirichlet": self.enforce_dirichlet,
            "total_parameters": self.count_parameters(),
        }


def create_fno1d(
    modes: int = 16,
    width: int = 64,
    num_layers: int = 4,
    padding: int = 0,
    enforce_dirichlet: bool = True,
) -> FNO1D:
    return FNO1D(
        modes=modes,
        width=width,
        num_layers=num_layers,
        padding=padding,
        enforce_dirichlet=enforce_dirichlet,
    )
