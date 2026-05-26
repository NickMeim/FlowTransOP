from __future__ import annotations

import torch


def activation_from_name(name: str) -> torch.nn.Module:
    activations = {
        "LeakyReLU": torch.nn.LeakyReLU(0.01),
        "ReLU": torch.nn.ReLU(),
        "ELU": torch.nn.ELU(),
        "Sigmoid": torch.nn.Sigmoid(),
    }
    try:
        return activations[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


class ElementWiseLinear(torch.nn.Module):
    def __init__(self, dim: int, leak: float = 0.1, bias: bool = True, drop: float = 0.2):
        super().__init__()
        self.weight = torch.nn.Parameter(1e-1 * torch.randn(dim))
        self.bias = torch.nn.Parameter(1e-3 * torch.randn(dim)) if bias else None
        self.drop = torch.nn.Dropout(drop)
        self.activation = torch.nn.LeakyReLU(leak)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return self.drop(self.activation(x))


class SimpleEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channel: int,
        hidden_layers: list[int],
        latent_dim: int,
        dropRate: float = 0.1,
        dropIn: float = 0,
        bn: float = 0.6,
        activation: torch.nn.Module | None = None,
        normalizeOutput: bool = False,
        bias: bool = True,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self.bias = bias
        self.normalizeOutput = normalizeOutput
        self.num_hidden_layers = len(hidden_layers)
        self.bn = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(in_channel, hidden_layers[0], bias=bias, dtype=dtype))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=bn, dtype=dtype))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias, dtype=dtype))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=bn, dtype=dtype))
        self.linear_latent = torch.nn.Linear(hidden_layers[-1], latent_dim, bias=False, dtype=dtype)
        self.activation = activation or torch.nn.ELU()
        self.dropout = torch.nn.Dropout(dropRate)
        self.drop_in = dropIn
        if dropIn > 0:
            self.drop_input = torch.nn.Dropout(dropIn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_in > 0:
            x = self.drop_input(x)
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        z_latent = self.linear_latent(x)
        if self.normalizeOutput:
            z_latent = torch.nn.functional.normalize(z_latent)
        return z_latent


class VarDecoder(torch.nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_layers: list[int],
        out_dim: int,
        dropRate: float = 0.1,
        dropIn: float = 0,
        bn: float = 0.6,
        activation: torch.nn.Module | None = None,
        bias: bool = True,
        loss: str = "gauss",
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        if loss not in ["nb", "gauss"]:
            raise ValueError("loss must be 'nb' or 'gauss'")
        self.loss = loss
        self.bias = bias
        self.num_hidden_layers = len(hidden_layers)
        self.linear_layers = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        self.linear_layers.append(torch.nn.Linear(latent_dim, hidden_layers[0], bias=bias, dtype=dtype))
        self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[0], momentum=bn, dtype=dtype))
        for i in range(1, len(hidden_layers)):
            self.linear_layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=bias, dtype=dtype))
            self.bn.append(torch.nn.BatchNorm1d(num_features=hidden_layers[i], momentum=bn, dtype=dtype))
        self.out_var = torch.nn.Linear(hidden_layers[-1], out_dim, bias=False, dtype=dtype)
        self.out_mu = torch.nn.Linear(hidden_layers[-1], out_dim, bias=False, dtype=dtype)
        self.activation = activation or torch.nn.ELU()
        self.dropout = torch.nn.Dropout(dropRate)
        self.dropIn = dropIn
        if dropIn > 0:
            self.drop_input = torch.nn.Dropout(dropIn)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dropIn > 0:
            x = self.drop_input(x)
            x = self.drop_input(x)
        for i in range(self.num_hidden_layers):
            x = self.linear_layers[i](x)
            x = self.bn[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        gene_means = torch.nn.functional.softplus(self.relu(self.out_mu(x))).add(1e-3)
        gene_vars = torch.nn.functional.softplus(self.relu(self.out_var(x))).add(1e-3)
        return gene_means, gene_vars


class Flow(torch.nn.Module):
    def __init__(self, dim: int, h: int, dtype: torch.dtype = torch.float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, h, dtype=dtype),
            torch.nn.ELU(),
            torch.nn.Linear(h, h, dtype=dtype),
            torch.nn.ELU(),
            torch.nn.Linear(h, h, dtype=dtype),
            torch.nn.ELU(),
            torch.nn.Linear(h, dim, dtype=dtype),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((t, x_t), -1))

    def step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        h = t_end - t_start
        return x_t + h * self(x_t + self(x_t, t_start) * h / 2, t_start + h / 2)
