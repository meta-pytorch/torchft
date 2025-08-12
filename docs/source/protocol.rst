:github_url: https://github.com/meta-pytorch/torchft/protocol

Protocol
========

TorchFT maintains a replicated state machine with the **DDP** (Distributed Data Parallel) replicas, termed **state\_dict**. The **state\_dict** contains information about the maximum number of steps taken by a replica. This is incremented each time the parameters across all replicas are updated after a collective operation e.g. **allreduce**. When a new node joins or falls behind, we have it first recover from a node that has taken the largest number of steps (**max\_step**) i.e. it is the most up to date. This mechanism guarantees all replicas have a consistent state of model parameters – because there should always at least be a majority with the largest step since it's only possible to increment a step by having a majority agree on it.

So it is critical that the **state\_dict** remains consistent across all replicas i.e. replicas with the same **max\_step** have the same model parameters so that recovering replicas can recover their parameters from any node with the consistent **max\_step**.
