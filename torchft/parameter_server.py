# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parameter Servers
==================

This module provides a prototype implementation of a fault tolerant parameter server bulit on the reconfigurable ProcessGroups.
"""

import json
import logging
import socket
import threading
import urllib.request
import uuid
from abc import ABC, abstractmethod
from http.server import BaseHTTPRequestHandler

from torch.distributed import TCPStore

from torchft.http import _IPv6HTTPServer
from torchft.process_group import ProcessGroup

logger: logging.Logger = logging.getLogger(__name__)


class ParameterServer(ABC):
    """
    This implements a threaded parameter server using the torchft reconfigurable
    ProcessGroups.
    """

    def __init__(self, port: int, store_port: int = 0) -> None:
        """
        Create a new ParameterServer.

        Args:
            port: the port to bind the HTTP server to.
            store_port: the port to bind the TCPStore server to.
        """
        self.store = TCPStore(
            host_name="0.0.0.0",
            port=store_port,
            is_master=True,
            wait_for_workers=False,
        )

        ps = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/new_session":
                    self.send_response(400)
                    self.send_header("Content-type", "text/plain")
                    self.end_headers()
                    self.err(f"invalid path, got {self.path}")
                    return

                try:
                    self.send_response(200)
                    self.send_header(
                        "Content-type", "application/json"
                    )  # TODO: correct mime type
                    self.end_headers()

                    session_id = str(uuid.uuid4())

                    store_addr = (
                        f"{socket.getfqdn()}:{ps.store.port}/session/{session_id}"
                    )

                    logger.info(f"creating new session {session_id}")

                    data = (
                        json.dumps(
                            {
                                "session_id": session_id,
                                "store_addr": store_addr,
                            }
                        )
                        + "\n"
                    )
                    data = data.encode()

                    self.wfile.write(data)

                    # close the connection up front so client will know json is
                    # complete
                    self.finish()
                    self.connection.close()

                    # hijack thread for the session
                    ps._handle_session(session_id, store_addr)
                except Exception:
                    logger.exception(
                        f"got exception in request handler for {self.path}"
                    )
                    raise

        server_address = ("", port)
        self._server = _IPv6HTTPServer(server_address, RequestHandler)
        self._server.daemon_threads = True
        logger.info(f"Started ParameterServer on {self.address()}...")

        self._thread = threading.Thread(
            target=self._serve,
            args=(),
            daemon=True,
        )
        self._thread.start()

    def address(self) -> str:
        """
        Returns the HTTP address to create a new session on this server.

        Format: http://host:port/new_session

        Returns:
            an HTTP address
        """
        port = self._server.socket.getsockname()[1]
        return f"http://{socket.getfqdn()}:{port}/new_session"

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception as e:
            logger.exception("got exception in checkpoint server")

    @classmethod
    @abstractmethod
    def new_process_group(cls) -> ProcessGroup:
        """
        Create a new non-configured ProcessGroup for the ParameterServer to
        configure when setting up server and client connections.

        Must be implemented by subclasses.

        Returns:
            a new ProcessGroup
        """
        ...

    @classmethod
    def new_session(cls, address: str) -> ProcessGroup:
        """
        Creates a new session on the parameter server and returns a ProcessGroup
        configured for that server.

        Client is rank 1, server is rank 0.
        """
        with urllib.request.urlopen(address) as f:
            data = json.load(f)

        session_id = data["session_id"]
        store_addr = data["store_addr"]

        logger.info(f"connecting to session {session_id} at {store_addr}")

        pg = cls.new_process_group()
        # client is always rank 1
        pg.configure(store_addr, replica_id="0", rank=1, world_size=2)

        return pg

    def _handle_session(self, session_id: str, store_addr: str) -> None:
        pg = self.new_process_group()
        # paramter server is always rank 0
        pg.configure(store_addr, replica_id="0", rank=0, world_size=2)

        self.forward(session_id, pg)

    @abstractmethod
    def forward(self, session_id: str, pg: ProcessGroup) -> None:
        """
        This method will be called once per session in a dedicated thread. To
        support multiple operations on a single session you should put a
        for-loop in your forward implementation.

        If an error occurs, the process group will be freed and the client will
        have to create a new session.

        The server rank is 0 and the client rank is 1.

        Must be implemented by subclasses.

        Args:
            session_id: a unique uuid for this session
            pg: the ProcessGroup that's configured for the client.
        """
        ...
