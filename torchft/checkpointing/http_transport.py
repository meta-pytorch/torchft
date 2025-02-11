# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import socket
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from http.server import BaseHTTPRequestHandler
from typing import Generator, List, Optional, TypeVar, cast

import torch
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from torchft.checkpointing._rwlock import RWLock
from torchft.checkpointing._serialization import _streaming_load, _streaming_save
from torchft.checkpointing.transport import CheckpointTransport
from torchft.http import _IPv6HTTPServer

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def _time(desc: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{desc} took {end - start}s")


class HTTPTransport(CheckpointTransport[T]):
    """
    This is an HTTP server that can be used to transfer checkpoints
    between workers.

    This allows for fast recovery of workers by fetching the current weights
    from an existing worker.

    Args:
        timeout: the timeout for HTTP requests
        num_chunks: the number of chunks to split the checkpoint into (0 for no chunking)
    """

    def __init__(self, timeout: timedelta, num_chunks: int) -> None:
        self._checkpoint_lock = RWLock(timeout=timeout.total_seconds())
        self._disallowed = False
        self._step = -1
        self._timeout = timeout
        self._state_dict: Optional[T] = None
        self._num_chunks = num_chunks
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )

        # staged checkpoint information
        self._spec: Optional[TreeSpec] = None
        self._chunks: Optional[List[List[object]]] = None

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

                    with ckpt_server._checkpoint_lock.r_lock():
                        step = ckpt_server._step

                        parts = self.path.split("/")
                        assert len(parts) == 4
                        if parts[1] != "checkpoint":
                            self.send_error(
                                400,
                                f"invalid url format, expected /checkpoint/step/key but got {self.path}",
                            )
                            return

                        step = int(parts[2])
                        if step != ckpt_server._step:
                            self.send_error(
                                400,
                                f"invalid checkpoint requested, serving {ckpt_server._step} but got {step=}",
                            )
                            return

                        key = parts[3]
                        if key == "full":
                            self.send_response(200)
                            self.send_header("Content-type", "application/octet-stream")
                            self.end_headers()

                            state_dict = ckpt_server._state_dict

                            _streaming_save(state_dict, self.wfile)
                            return

                        if key == "metadata":
                            self.send_response(200)
                            self.send_header("Content-type", "application/octet-stream")
                            self.end_headers()

                            _streaming_save(ckpt_server._spec, self.wfile)
                        else:
                            chunk = ckpt_server._chunks[int(key)]

                            self.send_response(200)
                            self.send_header("Content-type", "application/octet-stream")
                            self.end_headers()

                            _streaming_save(chunk, self.wfile)
                except Exception as e:
                    logger.exception(
                        f"Exception in checkpoint server when handling {self.path=}: {e}",
                    )
                    self.send_error(500, str(e))

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
    def _load_from_address(cls, address: str, timeout: timedelta) -> object:
        """
        Loads a checkpoint from the given address.

        Args:
            address: the HTTP address to load the checkpoint from
        """
        msg = f"fetching checkpoint from {address}"
        logger.info(msg)

        with _time(msg), urllib.request.urlopen(
            address, timeout=timeout.total_seconds()
        ) as f:
            # We have to set weights_only to False as there are some non-tensor
            # states like lr_scheduler.
            # pyre-fixme[16]: needs torch>=2.7
            return cast(T, _streaming_load(f, weights_only=False))

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
        values, spec = tree_flatten(state_dict)

        with (
            torch.cuda.stream(self._stream)
            if self._stream is not None
            else nullcontext()
        ):
            with _time("transferring state_dict to CPU"):
                values = _to_cpu(values, pin_memory=False)
                if self._stream is not None:
                    self._stream.synchronize()

        # Unflatten so non-chunked transfer uses CPU tensors
        self._state_dict = tree_unflatten(values, spec)

        # Save spec for chunked
        self._spec = spec
        self._chunks = _split_chunks(values, self._num_chunks)

        self.allow_checkpoint(step)

    def recv_checkpoint(
        self, src_rank: int, metadata: str, step: int, timeout: timedelta
    ) -> T:
        base_url = f"{metadata}{step}"
        if self._num_chunks == 0:
            return cast(T, self._load_from_address(f"{base_url}/full", timeout))
        else:
            urls = [f"{base_url}/metadata"] + [
                f"{base_url}/{i}" for i in range(self._num_chunks)
            ]

            with ThreadPoolExecutor(max_workers=len(urls)) as executor:
                futures = [
                    executor.submit(self._load_from_address, url, timeout)
                    for url in urls
                ]

                spec, *chunks = [future.result() for future in futures]
                spec = cast(TreeSpec, spec)
                chunks = cast(List[List[object]], chunks)

            values = _merge_chunks(chunks, self._num_chunks)

            return tree_unflatten(values, spec)


def _to_cpu(values: List[T], pin_memory: bool) -> List[T]:
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


def _split_chunks(values: List[T], num_chunks: int) -> List[List[T]]:
    return [values[i::num_chunks] for i in range(num_chunks)]


def _merge_chunks(chunks: List[List[T]], num_chunks: int) -> List[T]:
    max_len = max(len(lst) for lst in chunks)
    output_list = []
    for i in range(max_len):
        for lst in chunks:
            if i < len(lst):
                output_list.append(lst[i])
    return output_list
