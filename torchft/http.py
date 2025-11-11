# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import socket
from http.server import ThreadingHTTPServer


class _IPv6HTTPServer(ThreadingHTTPServer):
    address_family: socket.AddressFamily = socket.AF_INET6
    request_queue_size: int = 1024
