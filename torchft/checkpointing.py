# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpointing
==============

This module implements methods for checkpointing and resuming training from a checkpoint.
"""

import io
import logging
import pickle
import socket
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from http.server import BaseHTTPRequestHandler
from typing import Callable, Generator, Generic, List, Optional, Tuple, TypeVar

import torch
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_flatten, tree_unflatten

from torchft.http import _IPv6HTTPServer
from torchft.process_group import ProcessGroup
from torchft.rwlock import RWLock

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


"""
def _save(obj: object, f: io.BufferedIOBase) -> None:
    torch.save(obj, f)


def _load(f: io.BufferedIOBase) -> object:
    data = f.read()
    reader = io.BytesIO(data)
    return torch.load(reader, weights_only=False)
"""


try:
    from torch.distributed._serialization import _streaming_load, _streaming_save
except ImportError:
    from torchft._serialization import _streaming_load, _streaming_save

_save = _streaming_save


def _load(f: io.BufferedIOBase) -> object:
    return _streaming_load(f, weights_only=False, map_location="cpu")


class CheckpointTransport(Generic[T], ABC):
    @abstractmethod
    def metadata(self) -> str:
        """
        Returns a string that will be used by the remote CheckpointTransport to fetch the checkpoint.
        """
        ...

    @abstractmethod
    def send_checkpoint(
        self, dst_ranks: List[int], step: int, state_dict: T, timeout: timedelta
    ) -> None:
        """
        Sends the checkpoint, only called when there is a rank that is behind.

        This may be async.

        Args:
            dst_ranks: the ranks to send to
            step: the step number to send
            state_dict: the state dict to send
            timeout: the timeout to wait for the checkpoint to be sent
        """
        ...

    def disallow_checkpoint(self) -> None:
        """
        Called after send_checkpoint to wait for the checkpoint to be sent.

        Once this returns, the state_dict may be mutated so no further data should be sent.
        """
        ...

    @abstractmethod
    def recv_checkpoint(
        self, src_rank: int, metadata: str, step: int, timeout: timedelta
    ) -> T:
        """
        Receives the checkpoint from the given rank.

        Args:
            src_rank: the rank to receive the checkpoint from
            metadata: the metadata returned by the remote CheckpointTransport
            step: the step number to receive
            timeout: the timeout to wait for the checkpoint
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """
        Called to shutdown the checkpoint transport.

        Args:
            wait: whether to wait for the transport to shutdown
        """


@dataclass
class _TensorMeta:
    shape: torch.Size
    dtype: torch.dtype
    storage_offset: int
    stride: int
    nbytes: int


@dataclass
class _DTensorMeta:
    local: _TensorMeta
    spec: object


@dataclass
class _StateDictMeta:
    step: int
    spec: object
    non_tensors: List[object]
    tensor_metas: List[_TensorMeta]


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
            storage_offset=tensor.storage_offset(),
            stride=tensor.stride(),
            nbytes=tensor.untyped_storage().nbytes(),
        ),
    )


def _prepare_state_dict(
    state_dict: object,
    step: int,
    device: str,
) -> Tuple[_StateDictMeta, List[torch.Tensor]]:
    start = time.perf_counter()
    values, spec = tree_flatten(state_dict)

    non_tensors = []
    tensors = []
    tensor_metas = []
    for v in values:
        if isinstance(v, DTensor):
            tensor, tensor_meta = _prepare_tensor(v._local_tensor)

            tensor_metas.append(tensor_meta)
            tensors.append(tensor)

            non_tensors.append(
                _DTensorMeta(
                    local=tensor_meta,
                    spec=v._spec,
                )
            )
        elif isinstance(v, torch.Tensor):
            tensor, tensor_meta = _prepare_tensor(v)
            tensors.append(tensor)
            non_tensors.append(tensor_meta)
            tensor_metas.append(tensor_meta)
        else:
            non_tensors.append(v)

    total_size = sum(t.nbytes for t in tensors)

    dur = time.perf_counter() - start
    logger.info(
        f"prepared state_dict {total_size=} {len(tensors)=} {len(non_tensors)=} in {dur}s"
    )

    return (
        _StateDictMeta(
            step=step,
            spec=spec,
            non_tensors=non_tensors,
            tensor_metas=tensor_metas,
        ),
        tensors,
    )


def _cast_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
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
        self._work = []
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

                # allow 3 concurrent transfers at a time
                while len(work) > (3 * len(dst_ranks)):
                    work.pop(0).wait()

            for w in work:
                w.wait()

    def recv_checkpoint(
        self, src_rank: int, metadata: str, step: int, timeout: timedelta
    ) -> T:
        len_t = torch.zeros(1, dtype=torch.int64, device=self._device)
        self._pg.recv([len_t], src_rank, tag=1).wait()
        length = len_t.item()

        assert length > 0, f"invalid metadata length {length=}"

        buf = torch.empty(length, dtype=torch.uint8, device=self._device)
        self._pg.recv([buf], src_rank, tag=2).wait()

        meta = pickle.loads(buf.cpu().numpy().tobytes())
        assert meta.step == step

        i = 0

        values = []
        for v in meta.non_tensors:
            if isinstance(v, _TensorMeta):
                t = torch.empty(v.nbytes, dtype=torch.uint8, device=self._device)
                self._pg.recv([t], src_rank, tag=3 + i).wait()
                i += 1
                t = t.cpu()

                tensor = torch.as_strided(
                    _cast_tensor(t, v.dtype),
                    size=v.shape,
                    stride=v.stride,
                    storage_offset=v.storage_offset,
                )
                values.append(tensor)
            elif isinstance(v, _DTensorMeta):
                t = torch.empty(v.local.nbytes, dtype=torch.uint8, device=self._device)
                self._pg.recv([t], src_rank, tag=3 + i).wait()
                i += 1
                t = t.cpu()

                tensor = torch.as_strided(
                    _cast_tensor(t, v.local.dtype),
                    size=v.local.shape,
                    stride=v.local.stride,
                    storage_offset=v.local.storage_offset,
                )
                values.append(DTensor(tensor, v.spec, requires_grad=False))
            else:
                values.append(v)

        return tree_unflatten(values, meta.spec)


class CheckpointServer(CheckpointTransport[T]):
    """
    This is an HTTP server that can be used to transfer checkpoints
    between workers.

    This allows for fast recovery of workers by fetching the current weights
    from an existing worker.

    Args:
        state_dict: a callable that returns the state dict to be transferred
    """

    def __init__(self, timeout: timedelta, num_chunks: int = 10) -> None:
        self._checkpoint_lock = RWLock(timeout=timeout.total_seconds())
        self._disallowed = False
        self._step = -1
        self._timeout = timeout
        self._state_dict: Optional[T] = None
        self._num_chunks = num_chunks
        self._spec: Optional[object] = None
        self._chunks: Optiona[List[object]] = None
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )

        # We don't allow checkpoints until the first send_checkpoint to avoid
        # serving the default step=-1 invalid checkpoint.
        self.disallow_checkpoint()

        ckpt_server = self

        class RequestHandler(BaseHTTPRequestHandler):
            # set request socket timeout to avoid hanging forever
            timeout = self._timeout.total_seconds()

            def do_GET(self):
                try:
                    # validate socket timeout is actually set
                    assert self.connection.gettimeout() == self.timeout

                    sock = self.wfile._sock
                    sock.setsockopt(
                        socket.SOL_SOCKET, socket.SO_SNDBUF, 2097152
                    )  # set send buffer size to 2MB

                    with ckpt_server._checkpoint_lock.r_lock():
                        step = ckpt_server._step

                        parts = self.path.split("/")
                        assert len(parts) == 4
                        if parts[1] != "checkpoint":
                            self.send_response(400)
                            self.send_header("Content-type", "text/plain")
                            self.end_headers()
                            self.err(
                                f"invalid checkpoint requested, serving {step} but got {self.path}"
                            )
                            return

                        step = int(parts[2])

                        key = parts[3]
                        if key == "full":
                            self.send_response(200)
                            self.send_header("Content-type", "application/octet-stream")
                            self.end_headers()

                            state_dict = ckpt_server._state_dict

                            _save(state_dict, self.wfile)
                            return

                        if key == "metadata":
                            self.send_response(200)
                            self.send_header("Content-type", "application/octet-stream")
                            self.end_headers()

                            _save(ckpt_server._spec, self.wfile)
                        else:
                            self.send_response(200)
                            self.send_header("Content-type", "application/octet-stream")
                            self.end_headers()

                            chunk = ckpt_server._chunks[int(key)]
                            _save(chunk, self.wfile)

                except Exception as e:
                    logger.exception(
                        f"Exception in checkpoint server when handling {self.path=}: {e}",
                    )
                    self.send_response(500, str(e))
                    self.end_headers()

            def err(self, msg: str) -> None:
                logger.error(msg)
                self.wfile.write(msg.encode())

        server_address = ("", 0)
        self._server = _IPv6HTTPServer(server_address, RequestHandler)
        logger.info(f"Started CheckpointServer on {self.address()}...")

        self._thread = threading.Thread(
            target=self._serve,
            args=(),
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def load_from_address(cls, address: str, timeout: timedelta) -> T:
        """
        Loads a checkpoint from the given address.

        Args:
            address: the HTTP address to load the checkpoint from
        """
        logger.info(f"fetching checkpoint from {address}")

        start = time.perf_counter()

        with urllib.request.urlopen(address, timeout=timeout.total_seconds()) as f:
            sock = f.fp.raw._sock
            sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, 2097152
            )  # set receive buffer size to 2MB
            data = _load(f)

        dur = time.perf_counter() - start

        logger.info(f"done fetching checkpoint from {address} in {dur}s")

        return data

    def address(self) -> str:
        """
        Returns the HTTP address to fetch a checkpoint from this server. Step must be appended to the end of the address.

        Format: http://host:port/checkpoint/1234

        Returns:
            an HTTP address
        """
        port = self._server.socket.getsockname()[1]
        return f"http://{socket.gethostname()}:{port}/checkpoint/"

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception as e:
            logger.exception("got exception in checkpoint server")

    def disallow_checkpoint(self) -> None:
        """
        Disallows serving the checkpoint.

        All requests will block until allow_checkpoint is called.
        """
        if not self._disallowed:
            self._disallowed = True
            self._checkpoint_lock.w_acquire()

    def allow_checkpoint(self, step: int) -> None:
        """
        Allows serving the checkpoint with the specified step number.

        Args:
            step: the step number to serve
        """
        self._step = step

        if self._disallowed:
            self._disallowed = False
            self._checkpoint_lock.w_release()

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the server.
        """
        if not wait:
            # hack for nonblocking shutdown of socketserver threads
            # pyre-fixme[16]: no attribute `__shutdown_request`.
            self._server.__shutdown_request = True
        if wait:
            self._server.shutdown()
            self._thread.join()

    def metadata(self) -> str:
        return self.address()

    def send_checkpoint(
        self, dst_ranks: List[int], step: int, state_dict: T, timeout: timedelta
    ) -> None:
        self._state_dict = state_dict

        from torch.utils._pytree import tree_flatten

        values, spec = tree_flatten(state_dict)

        with (
            torch.cuda.stream(self._stream)
            if self._stream is not None
            else nullcontext()
        ):
            logger.info("transferring to CPU")
            start = time.perf_counter()
            values = _to_cpu(values, pin_memory=False)
            if self._stream is not None:
                self._stream.synchronize()
            logger.info(f"done transferring to CPU in {time.perf_counter() - start}s")

        self._spec = spec
        self._chunks = _split_chunks(values, self._num_chunks)

        self.allow_checkpoint(step)

    def recv_checkpoint(
        self, src_rank: int, metadata: str, step: int, timeout: timedelta
    ) -> T:
        base_url = f"{metadata}{step}"
        if self._num_chunks == 0:
            return self.load_from_address(f"{base_url}/full", timeout)
        else:
            urls = [f"{base_url}/metadata"] + [
                f"{base_url}/{i}" for i in range(self._num_chunks)
            ]

            with ThreadPoolExecutor(max_workers=len(urls)) as executor:
                futures = [
                    executor.submit(self.load_from_address, url, timeout)
                    for url in urls
                ]

                spec, *chunks = [future.result() for future in futures]

            values = _merge_chunks(chunks, self._num_chunks)

            from torch.utils._pytree import tree_flatten, tree_unflatten

            return tree_unflatten(values, spec)


def _to_cpu(values: List[object], pin_memory: bool) -> List[object]:
    out = []
    for v in values:
        if isinstance(v, torch.Tensor):
            if v.device.type == "cuda":
                if pin_memory:
                    cpu = torch.empty(*tuple(v.size()), dtype=v.dtype, pin_memory=True)
                    cpu.copy_(v, non_blocking=True)
                    out.append(cpu)
                else:
                    out.append(v.cpu())
            else:
                out.append(v)
        else:
            out.append(v)
    return out


def _split_chunks(values: List[object], num_chunks: int) -> List[object]:
    return [values[i::num_chunks] for i in range(num_chunks)]


def _merge_chunks(chunks: List[List[object]], num_chunks: int) -> List[object]:
    max_len = max(len(lst) for lst in chunks)
    output_list = []
    for i in range(max_len):
        for lst in chunks:
            if i < len(lst):
                output_list.append(lst[i])
    return output_list


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    transport = CheckpointServer(timedelta(seconds=60), num_chunks=0)
    metadata = transport.metadata()
    print(f"fetching from {metadata}")

    device = torch.device("cpu")

    state_dict = {}
    CHUNK_SIZE = 64_000_000  # 64MB
    TOTAL_SIZE = 5_000_000_000  # 1GB
    for i in range(0, TOTAL_SIZE, CHUNK_SIZE):
        state_dict[f"chunk/{i}"] = torch.zeros(
            CHUNK_SIZE // 4, dtype=torch.float32, device=device
        )

    transport.send_checkpoint(
        dst_ranks=[0], step=1, state_dict=state_dict, timeout=timedelta(seconds=60)
    )

    import time

    print("starting")
    start = time.perf_counter()
    transport.recv_checkpoint(
        src_rank=1, metadata=metadata, step=1, timeout=timedelta(seconds=60)
    )
    end = time.perf_counter()
    print(f"took {end - start} seconds")
