import logging
import pickle
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Generator, List, Tuple, TypeVar, Union, cast

import torch
from torch.distributed import Work
from torch.distributed.tensor import DTensor, _DTensorSpec
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from torchft.checkpointing.transport import CheckpointTransport
from torchft.process_group import ProcessGroup

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _TensorMeta:
    """
    This is the metadata for a tensor that is used to transfer checkpoints.
    It contains the shape, the dtype, the storage offset and the stride of the
    tensor.

    This must be pickleable so that it can be sent over the wire.
    """

    shape: torch.Size
    dtype: torch.dtype
    storage_offset: int
    stride: Tuple[int, ...]
    nbytes: int


@dataclass
class _DTensorMeta:
    """
    This is the metadata for a DTensor that is used to transfer checkpoints.
    It contains the metadata for the local tensor and the spec of the DTensor.

    This must be pickleable so that it can be sent over the wire.
    """

    local: _TensorMeta
    spec: _DTensorSpec


@dataclass
class _StateDictMeta:
    """
    This is the metadata for a state dict that is used to transfer checkpoints.
    It contains the step, the pytree spec of the state dict and the metadata for
    each tensor in the state dict.

    This must be pickleable so that it can be sent over the wire.

    Args:
        step: the step of the checkpoint to verify consistency
        treespec: the pytree spec of the state dict
        non_tensor_leaves: the metadata for each tensor in the state dict and any
            non-tensor leaves in the state dict
    """

    step: int
    treespec: TreeSpec
    non_tensor_leaves: List[Union[object, _TensorMeta, _DTensorMeta]]


@contextmanager
def _timeit(name: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    yield
    dur = time.perf_counter() - start
    logger.info(f"{name} took {dur}s")


def _prepare_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, _TensorMeta]:
    return (
        _cast_tensor(tensor, torch.uint8),
        _TensorMeta(
            shape=tensor.shape,
            dtype=tensor.dtype,
            storage_offset=cast(int, tensor.storage_offset()),
            stride=tensor.stride(),
            nbytes=tensor.untyped_storage().nbytes(),
        ),
    )


def _prepare_state_dict(
    state_dict: object,
    step: int,
    device: torch.device,
) -> Tuple[_StateDictMeta, List[torch.Tensor]]:
    leaves, treespec = tree_flatten(state_dict)

    non_tensor_leaves = []
    tensors = []
    for v in leaves:
        if isinstance(v, DTensor):
            tensor, tensor_meta = _prepare_tensor(v._local_tensor)

            tensors.append(tensor)

            non_tensor_leaves.append(
                _DTensorMeta(
                    local=tensor_meta,
                    spec=v._spec,
                )
            )
        elif isinstance(v, torch.Tensor):
            tensor, tensor_meta = _prepare_tensor(v)
            tensors.append(tensor)
            non_tensor_leaves.append(tensor_meta)
        else:
            non_tensor_leaves.append(v)

    return (
        _StateDictMeta(
            step=step,
            treespec=treespec,
            non_tensor_leaves=non_tensor_leaves,
        ),
        tensors,
    )


def _cast_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Casts the underlying storage to a tensor of the given dtype.

    The returned tensor will be of size ``storage.nbytes``.

    This works for all datatypes and supports strided/offset tensors with the
    caveat that the cast tensor may be larger than the original tensor due to
    the differences in striding.
    """
    storage = tensor.untyped_storage()
    ret = torch.tensor(storage, dtype=dtype, device=tensor.device)
    assert ret.untyped_storage() is storage, "storage should be the same"
    return ret


class PGTransport(CheckpointTransport[T]):
    """
    This is a checkpoint transport that uses the process group to transfer checkpoints.
    This allows for fast recovery of workers by fetching the current weights
    from an existing worker.
    Args:
        state_dict: a callable that returns the state dict to be transferred
    """

    def __init__(
        self, pg: ProcessGroup, timeout: timedelta, device: torch.device
    ) -> None:
        self._work: List[Work] = []
        self._pg = pg
        self._timeout = timeout
        self._device = device

    def metadata(self) -> str:
        return "<n/a>"

    def disallow_checkpoint(self) -> None:
        pass

    def send_checkpoint(
        self, dst_ranks: List[int], step: int, state_dict: T, timeout: timedelta
    ) -> None:
        with _timeit("preparing state_dict"):
            meta, tensors = _prepare_state_dict(state_dict, step, device=self._device)

        work = []

        with _timeit("send pickle"):
            buf = pickle.dumps(meta)
            len_t = torch.tensor([len(buf)], dtype=torch.int64, device=self._device)
            buf_t = torch.frombuffer(buf, dtype=torch.uint8).to(self._device)
            for dst_rank in dst_ranks:
                work.append(self._pg.send([len_t], dst_rank, tag=1))
                work.append(self._pg.send([buf_t], dst_rank, tag=2))

        with _timeit("send tensors"):
            for i, t in enumerate(tensors):
                t = t.to(self._device)
                for dst_rank in dst_ranks:
                    work.append(self._pg.send([t], dst_rank, tag=3 + i))

                # allow 3 concurrent transfers at a time to avoid OOMs
                while len(work) > (3 * len(dst_ranks)):
                    work.pop(0).wait(timeout)

            for w in work:
                w.wait(timeout)

    def recv_checkpoint(
        self, src_rank: int, metadata: str, step: int, timeout: timedelta
    ) -> T:
        len_t = torch.zeros(1, dtype=torch.int64, device=self._device)
        self._pg.recv([len_t], src_rank, tag=1).wait(timeout)
        length = cast(int, len_t.item())

        assert length > 0, f"invalid metadata length {length=}"

        buf = torch.empty(length, dtype=torch.uint8, device=self._device)
        self._pg.recv([buf], src_rank, tag=2).wait(timeout)

        meta: _StateDictMeta = pickle.loads(buf.cpu().numpy().tobytes())
        assert meta.step == step

        i: int = 0

        def recv(v: _TensorMeta) -> torch.Tensor:
            nonlocal i

            t = torch.empty(v.nbytes, dtype=torch.uint8, device=self._device)
            # TODO: parallelize receives
            self._pg.recv([t], src_rank, tag=3 + i).wait(timeout)
            i += 1

            # TODO: allow in place receives to avoid having to copy to cpu to
            # avoid OOMs
            t = t.cpu()

            return torch.as_strided(
                t.view(v.dtype),
                size=v.shape,
                stride=v.stride,
                storage_offset=v.storage_offset,
            )

        values = []
        for v in meta.non_tensor_leaves:
            if isinstance(v, _TensorMeta):
                values.append(recv(v))
            elif isinstance(v, _DTensorMeta):
                tensor = recv(v.local)
                # pyre-fixme[29]: DTensor is not a function
                values.append(DTensor(tensor, v.spec, requires_grad=False))
            else:
                values.append(v)

        return tree_unflatten(values, meta.treespec)
