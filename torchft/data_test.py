# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from torchft.data import (
    DistributedBatchSampler,
    DistributedSampler,
    SkipDistributedSampler,
)


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
                # print(f"---- sample is not skipped, drop_last={drop_last}, rank={rank} ----")
                sampler = SkipDistributedSampler(
                    dataset=dataset,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=False,
                    drop_last=drop_last,
                )
                cur = rank
                for idx in sampler:
                    # print("idx = ", idx)
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
                # print(f"---- sample is skipped, drop_last={drop_last}, rank={rank} ----")
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
                    # print("idx = ", idx)
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
            # print(f"---- sample is skipped, drop_last={drop_last}, rank={rank} ----")
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
                # print("idx = ", idx)
                self.assertEqual(
                    idx, expected[rank], f"idx={idx}, rank={rank}, expected={expected}"
                )
                cnt += 1
            self.assertTrue(cnt, 1)

    def test_distributed_batch_sampler(self):
        # Test 1: Basic functionality with dataset 0-20, batch_size=2, num_replicas=2
        dataset_size = 21
        batch_size = 2
        num_replicas = 2

        # Test with even_batches=True (default behavior) - all ranks get same number of batches
        sampler_rank0 = DistributedBatchSampler(
            SequentialSampler(DummyDataset(dataset_size)),
            batch_size=batch_size,
            drop_last=False,
            num_replicas=num_replicas,
            rank=0,
            even_batches=True,
        )

        sampler_rank1 = DistributedBatchSampler(
            SequentialSampler(DummyDataset(dataset_size)),
            batch_size=batch_size,
            drop_last=False,
            num_replicas=num_replicas,
            rank=1,
            even_batches=True,
        )

        batches_rank0 = list(sampler_rank0)
        batches_rank1 = list(sampler_rank1)

        # With even_batches=True, both ranks get exactly 5 batches (drops the last batch [20])
        expected_rank0_even = [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17]]
        expected_rank1_even = [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19]]

        assert (
            batches_rank0 == expected_rank0_even
        ), f"Expected {expected_rank0_even}, got {batches_rank0}"
        assert (
            batches_rank1 == expected_rank1_even
        ), f"Expected {expected_rank1_even}, got {batches_rank1}"
        assert len(sampler_rank0) == 5, f"Expected length 5, got {len(sampler_rank0)}"
        assert len(sampler_rank1) == 5, f"Expected length 5, got {len(sampler_rank1)}"

        # Test with even_batches=False - some ranks may get extra batches
        sampler_rank0_uneven = DistributedBatchSampler(
            SequentialSampler(DummyDataset(dataset_size)),
            batch_size=batch_size,
            drop_last=False,
            num_replicas=num_replicas,
            rank=0,
            even_batches=False,
        )

        sampler_rank1_uneven = DistributedBatchSampler(
            SequentialSampler(DummyDataset(dataset_size)),
            batch_size=batch_size,
            drop_last=False,
            num_replicas=num_replicas,
            rank=1,
            even_batches=False,
        )

        batches_rank0_uneven = list(sampler_rank0_uneven)
        batches_rank1_uneven = list(sampler_rank1_uneven)

        # With even_batches=False, rank0 gets 6 batches, rank1 gets 5 batches
        expected_rank0_uneven = [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17], [20]]
        expected_rank1_uneven = [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19]]

        assert (
            batches_rank0_uneven == expected_rank0_uneven
        ), f"Expected {expected_rank0_uneven}, got {batches_rank0_uneven}"
        assert (
            batches_rank1_uneven == expected_rank1_uneven
        ), f"Expected {expected_rank1_uneven}, got {batches_rank1_uneven}"
        assert (
            len(sampler_rank0_uneven) == 6
        ), f"Expected length 6, got {len(sampler_rank0_uneven)}"
        assert (
            len(sampler_rank1_uneven) == 5
        ), f"Expected length 5, got {len(sampler_rank1_uneven)}"

        # Test 2: Verify no data loss and no overlap (using even_batches=False for completeness)
        all_indices_distributed = []
        for batch in batches_rank0_uneven + batches_rank1_uneven:
            all_indices_distributed.extend(batch)

        normal_sampler = BatchSampler(
            SequentialSampler(DummyDataset(dataset_size)), batch_size, False
        )
        all_indices_normal = []
        for batch in normal_sampler:
            all_indices_normal.extend(batch)

        assert sorted(all_indices_distributed) == sorted(
            all_indices_normal
        ), "Data completeness check failed"
        assert len(set(all_indices_distributed)) == len(
            all_indices_distributed
        ), "Overlap detected"

        # Test 3: drop_last=True
        sampler_rank0_drop = DistributedBatchSampler(
            SequentialSampler(DummyDataset(dataset_size)),
            batch_size=batch_size,
            drop_last=True,
            num_replicas=num_replicas,
            rank=0,
        )

        sampler_rank1_drop = DistributedBatchSampler(
            SequentialSampler(DummyDataset(dataset_size)),
            batch_size=batch_size,
            drop_last=True,
            num_replicas=num_replicas,
            rank=1,
        )

        batches_rank0_drop = list(sampler_rank0_drop)
        batches_rank1_drop = list(sampler_rank1_drop)

        # With drop_last=True, we should get 10 total batches (dropping the last incomplete batch)
        # rank0 should get batches 0,2,4,6,8 -> [[0,1], [4,5], [8,9], [12,13], [16,17]]
        # rank1 should get batches 1,3,5,7,9 -> [[2,3], [6,7], [10,11], [14,15], [18,19]]
        expected_rank0_drop = [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17]]
        expected_rank1_drop = [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19]]

        assert (
            batches_rank0_drop == expected_rank0_drop
        ), f"Expected {expected_rank0_drop}, got {batches_rank0_drop}"
        assert (
            batches_rank1_drop == expected_rank1_drop
        ), f"Expected {expected_rank1_drop}, got {batches_rank1_drop}"

        # Test 4: num_replicas=3
        dataset_size = 20
        num_replicas = 3

        samplers = []
        batches = []
        for rank in range(num_replicas):
            sampler = DistributedBatchSampler(
                SequentialSampler(DummyDataset(dataset_size)),
                batch_size=2,
                drop_last=False,
                num_replicas=num_replicas,
                rank=rank,
                even_batches=False,
            )
            samplers.append(sampler)
            batches.append(list(sampler))

        # Total batches should be 10: [[0,1], [2,3], ..., [18,19]]
        # rank0 gets: [0,3,6,9] -> [[0,1], [6,7], [12,13], [18,19]] (4 batches)
        # rank1 gets: [1,4,7] -> [[2,3], [8,9], [14,15]] (3 batches)
        # rank2 gets: [2,5,8] -> [[4,5], [10,11], [16,17]] (3 batches)
        expected_batches = [
            [[0, 1], [6, 7], [12, 13], [18, 19]],  # rank0
            [[2, 3], [8, 9], [14, 15]],  # rank1
            [[4, 5], [10, 11], [16, 17]],  # rank2
        ]

        for rank, (expected, actual) in enumerate(zip(expected_batches, batches)):
            assert actual == expected, f"Rank {rank}: Expected {expected}, got {actual}"

        # Verify lengths
        assert (
            len(samplers[0]) == 4
        ), f"Rank 0 length: expected 4, got {len(samplers[0])}"
        assert (
            len(samplers[1]) == 3
        ), f"Rank 1 length: expected 3, got {len(samplers[1])}"
        assert (
            len(samplers[2]) == 3
        ), f"Rank 2 length: expected 3, got {len(samplers[2])}"

        # Test 5: even_batches functionality
        # Test even_batches=True with different scenarios
        dataset_size = 23  # This will create 12 total batches with batch_size=2
        batch_size = 2
        num_replicas = 3

        samplers_even = []
        batches_even = []
        for rank in range(num_replicas):
            sampler = DistributedBatchSampler(
                SequentialSampler(DummyDataset(dataset_size)),
                batch_size=batch_size,
                drop_last=False,
                num_replicas=num_replicas,
                rank=rank,
                even_batches=True,
            )
            samplers_even.append(sampler)
            batches_even.append(list(sampler))

        # With 12 total batches and 3 ranks, each rank should get exactly 4 batches
        for rank in range(num_replicas):
            assert (
                len(batches_even[rank]) == 4
            ), f"Rank {rank} should get 4 batches, got {len(batches_even[rank])}"
            assert (
                len(samplers_even[rank]) == 4
            ), f"Rank {rank} __len__ should return 4, got {len(samplers_even[rank])}"

        # Test even_batches=False with same scenario
        samplers_uneven = []
        batches_uneven = []
        for rank in range(num_replicas):
            sampler = DistributedBatchSampler(
                SequentialSampler(DummyDataset(dataset_size)),
                batch_size=batch_size,
                drop_last=False,
                num_replicas=num_replicas,
                rank=rank,
                even_batches=False,
            )
            samplers_uneven.append(sampler)
            batches_uneven.append(list(sampler))

        # With 12 total batches and 3 ranks, each rank gets exactly 4 batches (evenly divisible)
        for rank in range(num_replicas):
            assert (
                len(batches_uneven[rank]) == 4
            ), f"Rank {rank} should get 4 batches, got {len(batches_uneven[rank])}"

        # Test with 13 total batches (not evenly divisible)
        dataset_size = 25  # This will create 13 total batches with batch_size=2

        samplers_even_13 = []
        batches_even_13 = []
        for rank in range(num_replicas):
            sampler = DistributedBatchSampler(
                SequentialSampler(DummyDataset(dataset_size)),
                batch_size=batch_size,
                drop_last=False,
                num_replicas=num_replicas,
                rank=rank,
                even_batches=True,
            )
            samplers_even_13.append(sampler)
            batches_even_13.append(list(sampler))

        # With 13 total batches and 3 ranks, even_batches=True gives each rank 4 batches (drops 1 batch)
        for rank in range(num_replicas):
            assert (
                len(batches_even_13[rank]) == 4
            ), f"Rank {rank} should get 4 batches with even_batches=True, got {len(batches_even_13[rank])}"

        samplers_uneven_13 = []
        batches_uneven_13 = []
        for rank in range(num_replicas):
            sampler = DistributedBatchSampler(
                SequentialSampler(DummyDataset(dataset_size)),
                batch_size=batch_size,
                drop_last=False,
                num_replicas=num_replicas,
                rank=rank,
                even_batches=False,
            )
            samplers_uneven_13.append(sampler)
            batches_uneven_13.append(list(sampler))

        # With 13 total batches and 3 ranks, even_batches=False: rank0 gets 5, rank1 gets 4, rank2 gets 4
        assert (
            len(batches_uneven_13[0]) == 5
        ), f"Rank 0 should get 5 batches with even_batches=False, got {len(batches_uneven_13[0])}"
        assert (
            len(batches_uneven_13[1]) == 4
        ), f"Rank 1 should get 4 batches with even_batches=False, got {len(batches_uneven_13[1])}"
        assert (
            len(batches_uneven_13[2]) == 4
        ), f"Rank 2 should get 4 batches with even_batches=False, got {len(batches_uneven_13[2])}"

        # Test 6: Parameter validation
        base_sampler = SequentialSampler(DummyDataset(10))

        # Test invalid batch_size
        try:
            DistributedBatchSampler(base_sampler, -1, False, 2, 0)
            assert False, "Should raise ValueError for negative batch_size"
        except ValueError:
            pass

        try:
            DistributedBatchSampler(base_sampler, 0, False, 2, 0)
            assert False, "Should raise ValueError for zero batch_size"
        except ValueError:
            pass

        # Test invalid drop_last
        try:
            DistributedBatchSampler(base_sampler, 2, "false", 2, 0)
            assert False, "Should raise ValueError for non-bool drop_last"
        except ValueError:
            pass

        # Test invalid num_replicas
        try:
            DistributedBatchSampler(base_sampler, 2, False, 0, 0)
            assert False, "Should raise ValueError for zero num_replicas"
        except ValueError:
            pass

        try:
            DistributedBatchSampler(base_sampler, 2, False, -1, 0)
            assert False, "Should raise ValueError for negative num_replicas"
        except ValueError:
            pass

        # Test invalid rank
        try:
            DistributedBatchSampler(base_sampler, 2, False, 2, -1)
            assert False, "Should raise ValueError for negative rank"
        except ValueError:
            pass

        try:
            DistributedBatchSampler(base_sampler, 2, False, 2, 2)
            assert False, "Should raise ValueError for rank >= num_replicas"
        except ValueError:
            pass

        # Test invalid even_batches
        try:
            DistributedBatchSampler(base_sampler, 2, False, 2, 0, "true")
            assert False, "Should raise ValueError for non-bool even_batches"
        except ValueError:
            pass
