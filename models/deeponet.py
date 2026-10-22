"""Fixed-grid DeepONet baseline for 1D wave-equation operator learning.

This is the supervised baseline used in the paper. It keeps the grid fixed:
the branch network reads the discretized input functions [u0; c], and the
trunk network reads the normalized grid where u(., T) is evaluated.

Canonical data-loader contract:
- branch: `(batch, 2*N)` containing `[u0; c]`
- trunk: `(batch, N)` containing normalized grid coordinates
- target: `(batch, N)` containing `uT`

Canonical model contract:
- `forward(u0, c, x) -> (batch, N)`
- `forward_branch_trunk(branch, trunk) -> (batch, N)` adapts loader batches
  without changing the architecture.

The merge is the standard DeepONet inner product, with an optional scalar bias.
When requested, the final output is multiplied by 4*x*(1-x) so the endpoint
values are exactly zero.
"""

import torch
import torch.nn as nn


class BranchNet(nn.Module):
    """MLP encoder for the concatenated input functions."""

    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super(BranchNet, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, hidden_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TrunkNet(nn.Module):
    """MLP encoder for spatial coordinates."""

    def __init__(self, hidden_size=128, num_layers=3):
        super(TrunkNet, self).__init__()

        layers = []
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, hidden_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2:
            batch_size, N = x.shape
            x = x.unsqueeze(-1)
        elif x.dim() == 3 and x.shape[-1] == 1:
            batch_size, N, _ = x.shape
        else:
            raise ValueError(f"Unexpected trunk input shape: {x.shape}")

        x_flat = x.view(-1, 1)
        out_flat = self.net(x_flat)
        return out_flat.view(batch_size, N, -1)


class DeepONet1D(nn.Module):
    """
    Fixed-grid DeepONet for 1D terminal-state operator learning.

    Learns the mapping (u0, c) -> u(., T) using a separable branch-trunk architecture.

    Branch input: concatenated [u0; c] of shape (batch, 2*N)
    Trunk input: spatial grid x of shape (batch, N)
    Output: u(x, T) of shape (batch, N)

    The raw operator is represented as:
        u_raw(x) = sum_i branch_i(u0,c) * trunk_i(x) + bias

    When enforce_dirichlet=True, the final prediction is
        u(x) = 4*x*(1-x)*u_raw(x),
    which exactly enforces the homogeneous endpoint values for unnormalized
    terminal-state targets.
    """

    def __init__(
        self,
        N,
        branch_hidden=128,
        trunk_hidden=128,
        num_layers=3,
        use_output_bias=True,
        enforce_dirichlet=True,
    ):
        super(DeepONet1D, self).__init__()

        self.N = N
        self.branch_hidden = branch_hidden
        self.trunk_hidden = trunk_hidden
        self.num_layers = num_layers
        self.use_output_bias = use_output_bias
        self.enforce_dirichlet = enforce_dirichlet

        self.branch = BranchNet(2 * N, branch_hidden, num_layers)
        self.trunk = TrunkNet(trunk_hidden, num_layers)

        assert branch_hidden == trunk_hidden, "Branch and trunk hidden sizes must match"
        self.output_bias = nn.Parameter(torch.zeros(1)) if use_output_bias else None

    def _validate_fields(self, u0, c, x):
        if u0.dim() != 2 or c.dim() != 2:
            raise ValueError(
                f"DeepONet1D expects u0 and c with shape (batch, N), got {tuple(u0.shape)} and {tuple(c.shape)}"
            )
        if u0.shape != c.shape:
            raise ValueError(f"u0 and c must have identical shape, got {tuple(u0.shape)} and {tuple(c.shape)}")
        if u0.shape[1] != self.N:
            raise ValueError(f"DeepONet1D was initialized with N={self.N}, got input N={u0.shape[1]}")

        if x.dim() == 3 and x.shape[-1] == 1:
            x_batch, x_n = x.shape[0], x.shape[1]
        elif x.dim() == 2:
            x_batch, x_n = x.shape
        else:
            raise ValueError(f"DeepONet1D expects x with shape (batch, N) or (batch, N, 1), got {tuple(x.shape)}")

        if x_batch != u0.shape[0] or x_n != self.N:
            raise ValueError(
                f"x must match u0/c batch and grid shape ({u0.shape[0]}, {self.N}), got {tuple(x.shape)}"
            )

    def _split_branch_input(self, branch):
        if branch.dim() != 2:
            raise ValueError(f"DeepONet branch input must have shape (batch, 2*N), got {tuple(branch.shape)}")
        expected_features = 2 * self.N
        if branch.shape[1] != expected_features:
            raise ValueError(f"Expected branch feature size {expected_features}, got {branch.shape[1]}")
        return branch[:, :self.N], branch[:, self.N:]

    def _apply_dirichlet_envelope(self, raw_output, x):
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        elif x.dim() != 2:
            raise ValueError(f"Unexpected coordinate shape for Dirichlet envelope: {x.shape}")
        return raw_output * (4.0 * x * (1.0 - x))

    def forward(self, u0, c, x):
        """Evaluate the operator on field tensors and grid coordinates."""
        self._validate_fields(u0, c, x)

        branch_input = torch.cat([u0, c], dim=-1)
        branch_out = self.branch(branch_input)
        trunk_out = self.trunk(x)

        output = torch.einsum('bh,bnh->bn', branch_out, trunk_out)
        if self.output_bias is not None:
            output = output + self.output_bias
        if self.enforce_dirichlet:
            output = self._apply_dirichlet_envelope(output, x)

        return output

    def forward_branch_trunk(self, branch, trunk):
        """
        Forward pass using the canonical DeepONet loader batch interface.

        Parameters:
        -----------
        branch : torch.Tensor
            Concatenated input functions [u0; c], shape (batch, 2*N)
        trunk : torch.Tensor
            Normalized spatial coordinates, shape (batch, N) or (batch, N, 1)

        Returns:
        --------
        torch.Tensor
            Predicted u(x,T) of shape (batch, N)
        """
        u0, c = self._split_branch_input(branch)
        return self.forward(u0, c, trunk)

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_complexity(self):
        """Get model complexity information."""
        return {
            'N': self.N,
            'branch_hidden': self.branch_hidden,
            'trunk_hidden': self.trunk_hidden,
            'num_layers': self.num_layers,
            'use_output_bias': self.use_output_bias,
            'enforce_dirichlet': self.enforce_dirichlet,
            'total_parameters': self.count_parameters()
        }


# Convenience function for creating DeepONet models
def create_deeponet1d(
    N,
    branch_hidden=128,
    trunk_hidden=128,
    num_layers=3,
    use_output_bias=True,
    enforce_dirichlet=True,
):
    """Create a DeepONet model with the project defaults."""
    return DeepONet1D(
        N=N,
        branch_hidden=branch_hidden,
        trunk_hidden=trunk_hidden,
        num_layers=num_layers,
        use_output_bias=use_output_bias,
        enforce_dirichlet=enforce_dirichlet,
    )
