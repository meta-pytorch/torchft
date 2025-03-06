<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./media/torchft_logo_dark.svg">
    <img width="55%" src="./media/torchft_logo.svg" alt="torchft">
  </picture>
</p>

<h3 align="center">
Easy Per Step Fault Tolerance for PyTorch
</h3>

<p align="center">
  | <a href="https://pytorch.org/torchft/"><b>Documentation</b></a>
  | <a href="https://github.com/pytorch-labs/torchft/blob/main/media/fault_tolerance_poster.pdf"><b>Poster</b></a>
  | <a href="https://docs.google.com/document/d/1OZsOsz34gRDSxYXiKkj4WqcD9x0lP9TcsfBeu_SsOY4/edit"><b>Design Doc</b></a>
  |
</p>
<p align="center">
  <a href="https://pypi.org/project/torchft-nightly/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/torchft-nightly"></a>
</p>

---

> ⚠️ WARNING: This is an alpha prototype for PyTorch fault tolerance and may have bugs
> or breaking changes as this is actively under development. We'd love to collaborate
> and contributions are welcome. Please reach out if you're interested in torchft
> or want to discuss fault tolerance in PyTorch

This repository implements techniques for doing a per-step fault tolerance so
you can keep training if errors occur without interrupting the entire training
job.

This is based off of the large scale training techniques presented at PyTorch
Conference 2024.

[![](./media/fault_tolerance_poster.png)](./media/fault_tolerance_poster.pdf)

## Design

torchft is designed to allow for fault tolerance when using training with replicated weights such as in DDP or HSDP (FSDP with DDP).

torchft implements a lighthouse server that coordinates across the different
replica groups and then a per replica group manager and fault tolerance library
that can be used in a standard PyTorch training loop.

This allows for membership changes at the training step granularity which can
greatly improve efficiency by avoiding stop the world training on errors.

![](./media/torchft-overview.png)

## Prerequisites

Before proceeding, ensure you have the following installed:

- Rust (with necessary dependencies)
- `protobuf-compiler` and the corresponding development package for Protobuf.

Note that the Rust versions available in many conda environments may be outdated. To install the latest version of Rust, we recommend downloading it directly from the official website as shown in the below command:
```sh
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```

To install the required packages on a Debian-based system (such as Ubuntu) using apt, run:

```sh
sudo apt install protobuf-compiler libprotobuf-dev
```

or for a Red Hat-based system, run:

```sh
sudo dnf install protobuf-compiler protobuf-devel
```

## Installation

```sh
pip install .
```

This uses pyo3+maturin to build the package, you'll need maturin installed.

If the installation command fails to invoke `cargo update` due to an inability to fetch the manifest, it may be caused by the `proxy`, `proxySSLCert`, and `proxySSLKey` settings in your .`gitconfig` file affecting the `cargo` command. To resolve this issue, try temporarily removing these fields from your `.gitconfig` before running the installation command.

To install in editable mode w/ the Rust extensions you can use the normal pip install command:

```sh
pip install -e .
```

## Usage

### Lighthouse

The lighthouse is used for fault tolerance across replicated workers (DDP/FSDP)
when using synchronous training.

You can start a lighthouse server by running:

```sh
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

### Example Training Loop (DDP)

See [train_ddp.py](./train_ddp.py) for the full example.

Invoke with:

```sh
TORCHFT_LIGHTHOUSE=http://localhost:29510 torchrun --master_port 29501 --nnodes 1 --nproc_per_node 1 train.py
```

train.py:

```py
from torchft import Manager, DistributedDataParallel, Optimizer, ProcessGroupGloo

manager = Manager(
    pg=ProcessGroupGloo(),
    load_state_dict=...,
    state_dict=...,
)

m = nn.Linear(2, 3)
m = DistributedDataParallel(manager, m)
optimizer = Optimizer(manager, optim.AdamW(m.parameters()))

for i in range(1000):
    batch = torch.rand(2, 2, device=device)

    optimizer.zero_grad()

    out = m(batch)
    loss = out.sum()

    loss.backward()

    optimizer.step()
```

### Example Parameter Server

torchft has a fault tolerant parameter server implementation built on it's
reconfigurable ProcessGroups. This does not require/use a Lighthouse server.

See [parameter_server_test.py](./torchft/parameter_server_test.py) for an example.

## Contributing

We welcome PRs! See the [CONTRIBUTING](./CONTRIBUTING.md) file.

## License

torchft is BSD 3-Clause licensed. See [LICENSE](./LICENSE) for more details.
