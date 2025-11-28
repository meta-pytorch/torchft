# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from torch.utils.data import Dataset

from torchft.data import DistributedSampler, SkipDistributedSampler


class DummyDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> int:
        return idx


class TestData(TestCase):
    def test_distributed_sampler(self) -> None:
        dataset = DummyDataset(1000)
        sampler = DistributedSampler(
            dataset,
            replica_rank=1,
            num_replica_groups=2,
            group_rank=3,
            num_replicas=4,
        )
        self.assertEqual(sampler.global_rank, 3 + 1 * 4)
        self.assertEqual(sampler.global_world_size, 2 * 4)

        sampler_iter = iter(sampler)
        self.assertEqual(next(sampler_iter), 500)

    def test_skip_distributed_sampler(self):
        dataset_length = 100
        dataset = DummyDataset(dataset_length)

        # Case 1: sample is not skipped
        for drop_last in [True, False]:
            num_replicas = 7
            for rank in range(num_replicas):
                sampler = SkipDistributedSampler(
                    dataset=dataset,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=False,
                    drop_last=drop_last,
                )
                cur = rank
                for idx in sampler:
                    self.assertEqual(
                        idx, (cur % dataset_length), f"idx={idx}, cur={cur}"
                    )
                    cur += num_replicas
                # If drop_last is True, read ceil((100-7)/7)*7=98 samples totally.
                # If drop_last is False, read ceil(100/7)*7=105 samples totally.
                if drop_last:
                    self.assertEqual(cur, 98 + rank, f"rank={rank}, cur={cur}")
                else:
                    self.assertEqual(cur, 105 + rank, f"rank={rank}, cur={cur}")

        # Case 2: sample is skipped
        for drop_last in [True, False]:
            num_replicas = 7
            skip_samples = 10
            for rank in range(num_replicas):
                sampler = SkipDistributedSampler(
                    dataset=dataset,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=False,
                    drop_last=drop_last,
                    skip_samples=skip_samples,
                )
                cur = rank
                for idx in sampler:
                    expected = (
                        ((cur + skip_samples) % dataset_length + skip_samples)
                        if (cur + skip_samples) >= dataset_length
                        else (cur + skip_samples)
                    )
                    self.assertEqual(idx, expected, f"idx={idx}, expected={expected}")
                    cur += num_replicas
                # If drop_last is True, read ceil((100-10-7)/7)*7=84 samples totally.
                # If drop_last is False, read ceil((100-10)/7)*7=91 samples totally.
                if drop_last:
                    self.assertEqual(cur, 84 + rank, f"rank={rank}, cur={cur}")
                else:
                    self.assertEqual(cur, 91 + rank, f"rank={rank}, cur={cur}")

        # Case 3: drop last is False and padding size is larger than number of indices
        # If skip_samples is 90, and num_replicas is 31, then the indices is [90, 92, ..., 99].
        # It means only 10 samples are left, so padding size is 21 which is larger than 10.
        num_replicas = 31
        skip_samples = 90
        expected = list(range(90, 100))
        expected = (expected * 4)[:31]
        for rank in range(num_replicas):
            sampler = SkipDistributedSampler(
                dataset=dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                drop_last=False,
                skip_samples=skip_samples,
            )
            cnt = 0
            for idx in sampler:
                self.assertEqual(
                    idx, expected[rank], f"idx={idx}, rank={rank}, expected={expected}"
                )
                cnt += 1
            self.assertTrue(cnt, 1)
