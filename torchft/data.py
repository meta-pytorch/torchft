# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data
====

This module provides helper classes to implement fault tolerant data loaders.

We recommend using torchdata's StatefulDataLoader to checkpoint each replica's
dataloader frequently to avoid duplicate batches.
"""

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils import data

import math
from collections.abc import Iterator
from typing import Optional, TypeVar

_T_co = TypeVar("_T_co", covariant=True)

class SkipDistributedSampler(Sampler[_T_co]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        skip_samples: int = 0,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.skip_samples = skip_samples
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.skip_samples - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil((len(self.dataset) - self.skip_samples) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            indices = indices[self.skip_samples: len(indices)]
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[self.skip_samples : self.skip_samples + self.total_size]
        if len(indices) != self.total_size:
            raise AssertionError(
                f"Number of indices ({len(indices)}) does not match total_size ({self.total_size})"
            )

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise AssertionError(
                f"Number of subsampled indices ({len(indices)}) does not match num_samples ({self.num_samples})"
            )

        # pyrefly: ignore  # bad-return
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

# pyre-fixme[24]: expected generic parameter
class DistributedSampler(data.distributed.DistributedSampler):
    """
    DistributedSampler extends the standard PyTorch DistributedSampler with a
    `num_replica_groups` that is used to shard the data across the fault
    tolerance replica groups.

    torchft doesn't know how many replica groups ahead of time so we need to set
    this to be the max number.

    This sampler is inherently lossy when used with torchft. torchft
    occasionally drops batches on rejoining and if a replica group is down that
    group examples will never be used. This can lead to imbalances if using a
    small dataset.

    This will shard the input dataset into ``num_replicas*num_replica_group``
    number of shards.

    Each shard rank is calculated via: ``rank + num_replicas*replica_rank``

    num_replicas and replica_rank must be the same on all workers.
    """

    def __init__(
        self,
        dataset: data.Dataset,
        replica_rank: int,
        num_replica_groups: int,
        group_rank: Optional[int] = None,
        num_replicas: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        """
        Args:
            data: the dataset to use
            replica_rank: the group ID (0-num_replica_groups) to use for this shard of data.
            num_replica_groups: the max number of global replica groups
            rank: the local group rank
            num_replicas: the local group world size
        """
        if group_rank is None:
            group_rank = dist.get_rank()
        if num_replicas is None:
            num_replicas = dist.get_world_size()

        self.global_rank: int = group_rank + num_replicas * replica_rank
        self.global_world_size: int = num_replicas * num_replica_groups

        super().__init__(
            dataset,
            rank=self.global_rank,
            num_replicas=self.global_world_size,
            # pyre-fixme[6]: got object
            **kwargs,
        )
