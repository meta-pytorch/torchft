# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from datetime import timedelta

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
os.environ["NCCL_HOSTID"] = str(REPLICA_GROUP_ID)

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader

from torchft import (
    DistributedDataParallel,
    Manager,
    Optimizer,
    ProcessGroupGloo,
    ProcessGroupNCCL,
    ProcessGroupXCCL,
)
from torchft.checkpointing.pg_transport import PGTransport
from torchft.data import DistributedBatchSampler, SkipDistributedSampler

logging.basicConfig(level=logging.INFO)

NUM_EPOCHS = 1
BATCH_SIZE = 4
TOTAL_BATCH_SIZE = BATCH_SIZE * 6
CHECKPOINT_ENABLED = False
INIT_CHECKPOINT_PATH = "./tmp/train_ddp_fix_batch/ckpt-init"


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
        "./tmp/train_ddp_fix_batch/loss.txt", encoding="utf-8"
    )
    loss_logger.addHandler(file_handler)
    return loss_logger


def main() -> None:
    loss_logger = setup_logger()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./cifar", train=True, download=True, transform=transform
    )

    def load_state_dict(state_dict):
        print("Received checkpoint!")
        m.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optim"])

    def state_dict():
        ret = {
            "model": m.state_dict(),
            "optim": optimizer.state_dict(),
        }
        print("Setup checkpoint to send!")
        return ret

    if torch.cuda.is_available():
        device = "cuda"
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
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "xpu"
            if torch.xpu.is_available()
            else "cpu"
        ),
    )

    def dataloader_fn(replica_world_size, replica_rank, current_batches_committed):
        sampler = SkipDistributedSampler(
            dataset=trainset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=0,
            drop_last=True,
            skip_samples=current_batches_committed * BATCH_SIZE,
        )
        batch_sampler = DistributedBatchSampler(
            sampler=sampler,
            batch_size=BATCH_SIZE,
            drop_last=True,
            num_replicas=replica_world_size,
            rank=replica_rank,
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
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"train_ddp_{REPLICA_GROUP_ID}",
        timeout=timedelta(seconds=30),
        checkpoint_transport=transport,
        dataloader_fn=dataloader_fn,
        accumulation_grad=True,
    )

    m = Net().to(device)
    if os.path.exists(INIT_CHECKPOINT_PATH):
        # Load from the same model to ensure that each experiment has the same initial state.
        print(f"Loading initial model from {INIT_CHECKPOINT_PATH}")
        init_state_dict = torch.load(INIT_CHECKPOINT_PATH)
        m.load_state_dict(init_state_dict["model"])
    else:
        print("No initial model found, training from random.")

    m = DistributedDataParallel(manager, m)
    optimizer = Optimizer(manager, optim.AdamW(m.parameters()))
    criterion = nn.CrossEntropyLoss()

    print(m)
    num_params = sum(p.numel() for p in m.parameters())
    print(f"Total number of parameters: {num_params}")

    for epoch in range(NUM_EPOCHS):
        while (
            batches := manager.get_batch_samples(
                epoch=epoch, batch_size=BATCH_SIZE, total_batch_size=TOTAL_BATCH_SIZE
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
                if i == accumulation_steps - 1:
                    loss.backward()
                else:
                    with manager.no_sync():
                        loss.backward()
                total_loss += loss.item()

            # If errored, the optimizer step will be a no-op, and the parameter will not be updated.
            # Although it is possible to use new pg to compute old batches, it is still safe.
            if not optimizer.step():
                # The first call to `get_batch_samples` will return empty and mark the dataloader as dirty.
                # The manager server will force synchronization for `_step` being 0. If `_step` doesn't
                # increment here, it will cause synchronization checkpoints twice because `_step` was 0 in
                # the first two rounds. The second checkpoint will run in parallel with the computation,
                # leading to pollution. Therefore, it's necessary to avoid having `_step` 0 in two
                # consecutive training rounds.
                if manager._step == 0:
                    manager._step += 1
                continue

            # allreduce the loss across all replicas for logging
            loss_tensor = torch.tensor(total_loss, device=device)
            # manager all reduce will divide by replica world size * accumulation steps
            manager.allreduce(loss_tensor).wait()
            avg_loss = loss_tensor.item()
            if manager.participating_rank() == 0:
                loss_logger.info(f"{manager.current_step()} {avg_loss}")
                if manager.current_step() % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}, step = {manager.current_step()}, "
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
