# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import hashlib
import logging
import os
import sys
from datetime import timedelta
from itertools import chain

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
os.environ["NCCL_HOSTID"] = str(REPLICA_GROUP_ID)

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader

from torchft import (
    Manager,
    Optimizer,
    process_group,
    ProcessGroupGloo,
    ProcessGroupNCCL,
    ProcessGroupXCCL,
)
from torchft.checkpointing.pg_transport import PGTransport
from torchft.data import DistributedBatchSampler, SkipDistributedSampler

logging.basicConfig(level=logging.INFO)

NUM_EPOCHS = 1
BATCH_SIZE = 4
MODEL_SHARDING_SIZE = 2
TOTAL_BATCH_SIZE = BATCH_SIZE * 6 * MODEL_SHARDING_SIZE
INIT_CHECKPOINT_PATH = "./tmp/train_fsdp2_fix_batch/ckpt-init"


def maybe_set_all_reduce_hook(model_parts: list[torch.nn.Module], replicate_pg) -> None:
    def all_reduce_hook(output):
        dist.all_reduce(output, group=replicate_pg, op=ReduceOp.AVG)

    def apply_set_all_reduce_hook(m):
        if isinstance(m, FSDPModule):
            m.set_all_reduce_hook(all_reduce_hook)

    for model_part in model_parts:
        model_part.apply(apply_set_all_reduce_hook)


def is_first_dp(manager):
    return manager.participating_rank() == 0 and dist.get_rank() == 0


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        final_dim = 10
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, final_dim),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.classifier(x)
        return x


def setup_logger():
    # Use UnbufferedFileHandler to avoid losing logs in case of failure.
    class UnbufferedFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
            os.fsync(self.stream.fileno())

    loss_logger = logging.getLogger("loss")
    loss_logger.setLevel(logging.INFO)
    loss_logger.propagate = False
    file_handler = UnbufferedFileHandler(
        "./tmp/train_fsdp2_fix_batch/loss.txt", encoding="utf-8"
    )
    loss_logger.addHandler(file_handler)
    return loss_logger


@record
def main() -> None:
    loss_logger = setup_logger()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./cifar", train=True, download=True, transform=transform
    )

    if torch.cuda.is_available():
        local_rank = os.environ.get("LOCAL_RANK")
        device = torch.device(
            f"cuda:{local_rank}" if local_rank is not None else "cuda"
        )
        print(f"Using CUDA device: {device}")
        pg = ProcessGroupNCCL(timeout=timedelta(seconds=30))
    elif torch.xpu.is_available():
        device = "xpu"
        pg = ProcessGroupXCCL(timeout=timedelta(seconds=30))
    else:
        device = "cpu"
        pg = ProcessGroupGloo(timeout=timedelta(seconds=5))

    transport = PGTransport(
        pg,
        timeout=timedelta(seconds=10),
        device=device,
    )

    def dataloader_fn(replica_world_size, replica_rank, current_batches_committed):
        sampler = SkipDistributedSampler(
            dataset=trainset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=0,
            drop_last=True,
            skip_samples=current_batches_committed * BATCH_SIZE * MODEL_SHARDING_SIZE,
        )
        batch_sampler = DistributedBatchSampler(
            sampler=sampler,
            batch_size=BATCH_SIZE,
            drop_last=True,
            num_replicas=replica_world_size * MODEL_SHARDING_SIZE,
            rank=replica_rank * MODEL_SHARDING_SIZE
            + dist.get_rank() % MODEL_SHARDING_SIZE,
            even_batches=True,
        )

        dataloader = DataLoader(trainset, num_workers=0, batch_sampler=batch_sampler)
        print(
            f"num_batches remaining: {len(dataloader)}, dataset length: {len(trainset)},"
            f"sampler length: {len(sampler)}, replica_world_size: {replica_world_size},"
            f"replica_rank: {replica_rank}, batches_committed: {current_batches_committed}"
        )

        return dataloader

    manager = Manager(
        pg=pg,
        min_replica_size=1,
        load_state_dict=None,
        state_dict=None,
        replica_id=f"train_ddp_{REPLICA_GROUP_ID}",
        timeout=timedelta(seconds=30),
        checkpoint_transport=transport,
        dataloader_fn=dataloader_fn,
    )

    m = Net()
    criterion = nn.CrossEntropyLoss()
    if os.path.exists(INIT_CHECKPOINT_PATH):
        print(f"Loading initial model from {INIT_CHECKPOINT_PATH}")
        init_state_dict = torch.load(INIT_CHECKPOINT_PATH)
        m.load_state_dict(init_state_dict["model"])
    else:
        print("No initial model found, training from random.")
    torch.cuda.set_device(int(local_rank))

    # Apply FSDP sharding
    for layer in chain(m.cnn, m.classifier):
        fully_shard(layer, reshard_after_forward=True)
    m = fully_shard(m, reshard_after_forward=True)

    # Create optimizer by sharding model parameters
    base_optimizer = optim.AdamW(m.parameters())
    optimizer = Optimizer(manager, base_optimizer)

    replicate_pg = process_group.ManagedProcessGroup(manager)
    maybe_set_all_reduce_hook(model_parts=[m], replicate_pg=replicate_pg)

    def load_state_dict(state_dict):
        # It's necessary to ensure that `set_model_state_dict` does not trigger `optim.step`,
        # as this operation may occur concurrently with both forward and backward iterations.
        print("Received checkpoint!")
        set_model_state_dict(m, state_dict["model"])
        set_optimizer_state_dict(m, base_optimizer, state_dict["optim"])

    def state_dict():
        # It's necessary to ensure that `get_model_state_dict` does not trigger `optim.step`,
        # as this operation may occur concurrently with both forward and backward iterations.
        ret = {
            "model": get_model_state_dict(m),
            "optim": get_optimizer_state_dict(m, base_optimizer),
        }
        print("Setup checkpoint to send!")
        return ret

    manager.register_state_dict_fn("set_state_dict_fns", load_state_dict, state_dict)

    print(m)
    num_params = sum(p.numel() for p in m.parameters())
    print(f"Total number of parameters: {num_params}")

    for epoch in range(NUM_EPOCHS):
        while (
            batches := manager.get_batch_samples(
                epoch=epoch,
                batch_size=BATCH_SIZE,
                total_batch_size=TOTAL_BATCH_SIZE // MODEL_SHARDING_SIZE,
            )
        ) is not None:
            optimizer.zero_grad()
            total_loss = 0.0
            accumulation_steps = len(batches)
            for i in range(accumulation_steps):
                inputs, labels = batches[i]
                inputs = inputs.to(device)
                labels = labels.to(device)
                out = m(inputs)
                loss = criterion(out, labels)
                # For fsdp2, synchronization must be performed every time; `no_sync` cannot be used.
                # This is because if `all_reduce_hook` is executed in `_fsdp_collectives.py`, it applies
                # to the temporary `reduce_output` variable. If synchronization is only performed in the
                # last step, each shard will lose the gradients of the previous `accumulation_steps - 1`
                # steps from other shards.
                loss.backward()
                total_loss += loss.item()

            if accumulation_steps > 1:
                for group in base_optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            if isinstance(param.grad, DTensor):
                                param.grad.data._local_tensor.div_(accumulation_steps)
                            else:
                                param.grad.data.div_(accumulation_steps)

            # If errored, the optimizer step will be a no-op, and the parameter will not be updated.
            # Although it is possible to use new pg to compute old batches, it is still safe.
            if not optimizer.step():
                # For fsdp2, the model may be updated in should_commit. We must wait for all model shard
                # to finish loading before proceeding; otherwise, inconsistencies may occur.
                dist.barrier()
                # The first call to `get_batch_samples` will return empty and mark the dataloader as dirty.
                # The manager server will force synchronization for `_step` being 0. If `_step` doesn't
                # increment here, it will cause synchronization checkpoints twice because `_step` was 0 in
                # the first two rounds. The second checkpoint will run in parallel with the computation,
                # leading to pollution. Therefore, it's necessary to avoid having `_step` 0 in two
                # consecutive training rounds.
                if manager._step == 0:
                    manager._step += 1
                continue

            loss_tensor = torch.tensor(total_loss, device=device)
            # Perform allreduce within replica group. Then perform allreduce across replica groups.
            dist.all_reduce(loss_tensor, op=ReduceOp.AVG)
            manager.allreduce(loss_tensor).wait()
            avg_loss = loss_tensor.item()
            avg_loss /= accumulation_steps
            if is_first_dp(manager):
                loss_logger.info(f"{manager.current_step() - 1} {avg_loss}")
                if manager.current_step() % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}, step = {manager.current_step() - 1}, "
                        f"batch_committed: {manager.batches_committed()}, Loss: {avg_loss:.4f}"
                    )
        print(
            f"Epoch {epoch + 1} completed, batches_committed {manager.batches_committed()}."
        )
        manager.next_epoch()
    print("Training completed.")


def save_init_model():
    m = Net()
    state_dict_to_save = {
        "model": m.state_dict(),
    }
    if not os.path.exists(INIT_CHECKPOINT_PATH):
        torch.save(state_dict_to_save, INIT_CHECKPOINT_PATH)
        print("Initial model saved.")
    else:
        print("Init model already exists.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "save_init_model":
        save_init_model()
    else:
        main()
