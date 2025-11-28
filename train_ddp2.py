# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from datetime import timedelta

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchft.data import SkipDistributedSampler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
os.environ["NCCL_HOSTID"] = str(REPLICA_GROUP_ID)

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.distributed.elastic.multiprocessing.errors import record

from torchft import (
    DistributedDataParallel,
    Manager,
    Optimizer,
    ProcessGroupGloo,
    ProcessGroupNCCL,
    ProcessGroupXCCL,
)
from torchft.checkpointing.pg_transport import PGTransport

logging.basicConfig(level=logging.INFO)

NUM_EPOCHS = 10
BATCH_SIZE = 16
TOTAL_BATCH_SIZE = BATCH_SIZE * 6
CHECKPOINT_ENABLED = True
CHECKPOINT_PATH = "./tmp/train_ddp2_checkpoint/ckpt"


def save_model(m, optimizer, manager):
    state_dict_to_save = {
        "model": m.state_dict(),
        "optim": optimizer.state_dict(),
        "torchft": manager.state_dict(),
    }
    # Save the checkpoint path by step and save the latest step to latest file
    step_checkpoint_path = f"{CHECKPOINT_PATH}_step_{manager.current_step()}"
    torch.save(state_dict_to_save, step_checkpoint_path)
    latest_path = f"{CHECKPOINT_PATH}_latest"
    with open(latest_path, "w") as f:
        f.write(step_checkpoint_path)
    # Delete the older checkpoints
    for filename in os.listdir("./tmp/train_ddp2_checkpoint/"):
        if filename.startswith("ckpt_step_"):
            step_str = filename.split("_")[-1]
            try:
                step_num = int(step_str)
                if step_num < manager.current_step() - 1000:
                    os.remove(os.path.join("./tmp/train_ddp2_checkpoint/", filename))
            except ValueError:
                continue


def load_model(m, optimizer, manager):
    if os.path.exists(f"{CHECKPOINT_PATH}_latest"):
        with open(f"{CHECKPOINT_PATH}_latest", "r") as f:
            latest_checkpoint_path = f.read().strip()
        print(f"Loading checkpoint from {latest_checkpoint_path}")
        loaded_state_dict = torch.load(latest_checkpoint_path, weights_only=True)
        m.load_state_dict(loaded_state_dict["model"])
        optimizer.load_state_dict(loaded_state_dict["optim"])
        manager.load_state_dict(loaded_state_dict["torchft"])


@record
def main() -> None:
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))

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
        print("Setup checkpoint to send!")
        return {
            "model": m.state_dict(),
            "optim": optimizer.state_dict(),
        }

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
            num_replicas=replica_world_size,
            rank=replica_rank,
            shuffle=True,
            seed=0,
            drop_last=True,
            skip_samples=current_batches_committed * BATCH_SIZE,
        )

        # drop_last to ensure all replicas have the same number of batches
        dataloader = DataLoader(
            trainset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            sampler=sampler,
            drop_last=True,
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
    )

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
            # We add a useless 1GB intermediate layer so we spend more time in dist
            # communication so injected failures are more likely to cause issues
            # if they exist.
            target_size = 1_000_000_000
            self.useless = nn.Embedding(target_size // final_dim // 4, final_dim)

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
            x += self.useless.weight[0]
            return x

    m = Net().to(device)
    m = DistributedDataParallel(manager, m)
    optimizer = Optimizer(manager, optim.AdamW(m.parameters()))
    criterion = nn.CrossEntropyLoss()
    if CHECKPOINT_ENABLED:
        load_model(m, optimizer, manager)

    print(m)
    num_params = sum(p.numel() for p in m.parameters())
    print(f"Total number of parameters: {num_params}")

    loss_writer = SummaryWriter(log_dir="./tmp/loss_train_ddp2")
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
                continue

            # all reduce the loss across all replicas
            total_loss = total_loss / BATCH_SIZE
            loss_tensor = torch.tensor(total_loss, device=device)
            # manager all reduce will divide by replica world size * accumulation steps
            manager.allreduce(loss_tensor).wait()
            avg_loss = loss_tensor.item()
            if manager.participating_rank() == 0:
                loss_writer.add_scalar(
                    "Training Loss", avg_loss, global_step=manager.batches_committed()
                )
                if manager.current_step() % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}, step = {manager.current_step()}, batch_committed {manager.batches_committed()}, Loss: {avg_loss:.4f}"
                    )
            if (
                CHECKPOINT_ENABLED
                and manager.current_step() % 200 == 0
                and manager.participating_rank() == 0
            ):
                save_model(m, optimizer, manager)
        print(
            f"Epoch {epoch + 1} completed, batches_committed {manager.batches_committed()}."
        )
        manager.next_epoch()
    loss_writer.close()


if __name__ == "__main__":
    main()
