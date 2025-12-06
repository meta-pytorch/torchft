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

import math
from collections.abc import Iterator
from typing import Iterable, Optional, TypeVar, Union

import torch
import torch.distributed as dist
from torch.utils import data
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, Sampler

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
                (len(self.dataset) - self.skip_samples - self.num_replicas)
                / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.skip_samples) / self.num_replicas
            )  # type: ignore[arg-type]
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
            indices = indices[self.skip_samples : len(indices)]
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


class DistributedBatchSampler(Sampler[list[int]]):
    r"""Wraps a BatchSampler to distribute batches across multiple processes in distributed training.

    Each process gets a subset of batches based on its rank and the total number of replicas.
    This is useful for distributed training where each process should work on different batches
    to avoid data duplication.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        num_replicas (int): Number of processes participating in distributed training.
        rank (int): Rank of the current process within num_replicas.
            Should be in range [0, num_replicas - 1].
        even_batches (bool): If ``True``, ensures all ranks get exactly the same number
            of batches by potentially dropping some batches. If ``False``, some ranks
            may get one extra batch. Default: ``True``.

    Example:
        >>> # For a dataset with indices 0-20, batch_size=2, num_replicas=2
        >>> # All batches would be: [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], ...]
        >>>
        >>> # With even_batches=False (original behavior):
        >>> # rank=0 gets batches: [[0,1], [4,5], [8,9], [12,13], [16,17], [20]] (6 batches)
        >>> # rank=1 gets batches: [[2,3], [6,7], [10,11], [14,15], [18,19]] (5 batches)
        >>> sampler_rank0 = DistributedBatchSampler(
        ...     SequentialSampler(range(21)), batch_size=2, drop_last=False,
        ...     num_replicas=2, rank=0, even_batches=False
        ... )
        >>> list(sampler_rank0)
        [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17], [20]]
        >>>
        >>> # With even_batches=True (default behavior):
        >>> # Both ranks get exactly 5 batches (drops the last batch [20])
        >>> # rank=0 gets batches: [[0,1], [4,5], [8,9], [12,13], [16,17]] (5 batches)
        >>> # rank=1 gets batches: [[2,3], [6,7], [10,11], [14,15], [18,19]] (5 batches)
        >>> sampler_rank0_even = DistributedBatchSampler(
        ...     SequentialSampler(range(21)), batch_size=2, drop_last=False,
        ...     num_replicas=2, rank=0, even_batches=True
        ... )
        >>> list(sampler_rank0_even)
        [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17]]
    """

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool,
        num_replicas: int = 1,
        rank: int = 0,
        even_batches: bool = True,
    ) -> None:
        # Validate batch_size
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )

        # Validate drop_last
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )

        # Validate num_replicas
        if not isinstance(num_replicas, int) or num_replicas <= 0:
            raise ValueError(
                f"num_replicas should be a positive integer value, but got num_replicas={num_replicas}"
            )

        # Validate rank
        if not isinstance(rank, int) or rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank should be an integer in range [0, {num_replicas - 1}], but got rank={rank}"
            )

        # Validate even_batches
        if not isinstance(even_batches, bool):
            raise ValueError(
                f"even_batches should be a boolean value, but got even_batches={even_batches}"
            )

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.even_batches = even_batches

        # Create a BatchSampler to generate all batches
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self) -> Iterator[list[int]]:
        if self.even_batches:
            # When even_batches=True, ensure all ranks get the same number of batches
            # by potentially dropping some batches
            all_batches = list(self.batch_sampler)
            total_batches = len(all_batches)

            # Calculate how many batches each rank should get to make them even
            batches_per_rank = total_batches // self.num_replicas

            # Only consider the first batches_per_rank * num_replicas batches
            # This ensures even distribution
            total_even_batches = batches_per_rank * self.num_replicas

            batch_idx = 0
            for batch in all_batches:
                if batch_idx >= total_even_batches:
                    # Stop yielding once we've exhausted the even batches
                    break
                # Only yield batches that belong to current rank
                if batch_idx % self.num_replicas == self.rank:
                    yield batch
                batch_idx += 1
        else:
            # Original behavior when even_batches=False
            batch_idx = 0
            for batch in self.batch_sampler:
                # Only yield batches that belong to current rank
                if batch_idx % self.num_replicas == self.rank:
                    yield batch
                batch_idx += 1

    def __len__(self) -> int:
        # Calculate total number of batches from BatchSampler
        total_batches = len(self.batch_sampler)  # type: ignore[arg-type]

        if self.even_batches:
            # When even_batches=True, all ranks get exactly the same number of batches
            return total_batches // self.num_replicas
        else:
            # Original behavior when even_batches=False
            # Each rank gets approximately total_batches // num_replicas batches
            # The remaining batches are distributed among the first few ranks
            batches_per_rank = total_batches // self.num_replicas
            remaining_batches = total_batches % self.num_replicas

            # Current rank gets one extra batch if it's among the first 'remaining_batches' ranks
            if self.rank < remaining_batches:
                return batches_per_rank + 1
            else:
                return batches_per_rank


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
