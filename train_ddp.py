# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from datetime import timedelta

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.distributed.elastic.multiprocessing.errors import record

from torchft import (
    DistributedDataParallel,
    Manager,
    ProcessGroupBabyNCCL,
    ProcessGroupGloo,
)

logging.basicConfig(level=logging.INFO)


@record
def main() -> None:
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
    NUM_REPLICA_GROUPS = int(os.environ.get("NUM_REPLICA_GROUPS", 2))

    def load_state_dict(state_dict):
        m.load_state_dict(state_dict["model"])

    def state_dict():
        return {
            "model": m.state_dict(),
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pg = (
        ProcessGroupBabyNCCL(
            timeout=timedelta(seconds=5),
        )
        if torch.cuda.is_available()
        else ProcessGroupGloo(timeout=timedelta(seconds=5))
    )

    manager = Manager(
        pg=pg,
        min_replica_size=1,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"allreduce_example_{REPLICA_GROUP_ID}",
        timeout=timedelta(seconds=10),
    )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    m = Net().to(device)
    m = DistributedDataParallel(manager, m)
    
    print(f"Model created: {m}")
    
    # Create a dummy tensor for allreduce
    dummy_tensor = torch.randn(100, device=device)
    print(f"Rank {manager.pg.rank()}: Performing allreduce operation")
    
    # Call allreduce instead of training
    result = manager.allreduce(dummy_tensor)
    
    print(f"Rank {manager.pg.rank()}: AllReduce completed successfully")
    print(f"Result tensor shape: {result.shape}")
    
    # Optionally you can perform additional operations with the result
    # For example, calculate and print mean of the reduced tensor
    print(f"Mean of reduced tensor: {result.mean().item()}")


if __name__ == "__main__":
    main()