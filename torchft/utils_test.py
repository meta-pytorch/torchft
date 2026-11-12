# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
from unittest import TestCase
from unittest.mock import patch

import torch
from parameterized import parameterized
from torchft.utils import format_host_port, get_hostname, HOSTNAME_ENV, record_event


class TestRecordEvent(TestCase):
    def test_returns_the_event_it_recorded(self) -> None:
        """Regression test: `record_event` used to return None, so the process
        receiving the event had nothing to wait on."""
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        event = record_event()

        self.assertIsNotNone(event, "record_event must return the event it recorded")
        if not isinstance(event, torch.cuda.Event):
            raise AssertionError(f"Expected a CUDA event, got {type(event)}")
        # a recorded interprocess event must be able to produce an IPC handle
        self.assertIsNotNone(event.ipc_handle())
        event.synchronize()
        self.assertTrue(event.query())


class TestHostnameUtils(TestCase):
    def test_get_hostname_defaults_to_socket_gethostname(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_hostname(), socket.gethostname())

    @parameterized.expand(
        [
            ("ipv4_env", None, "10.0.0.12", "10.0.0.12"),
            ("ipv6_env", None, "2001:db8::1", "2001:db8::1"),
            ("bracketed_ipv6_env", None, "[2001:db8::1]", "2001:db8::1"),
            ("whitespace_env", None, "  10.0.0.12  ", "10.0.0.12"),
            ("explicit_overrides_env", "10.0.0.99", "10.0.0.12", "10.0.0.99"),
            ("explicit_bracketed_ipv6", "[::1]", "10.0.0.12", "::1"),
            ("empty_explicit_uses_env", "", "10.0.0.12", "10.0.0.12"),
        ]
    )
    def test_get_hostname_resolution(
        self,
        _name: str,
        explicit: str | None,
        env_val: str,
        expected: str,
    ) -> None:
        with patch.dict(os.environ, {HOSTNAME_ENV: env_val}, clear=True):
            self.assertEqual(get_hostname(explicit), expected)

    @parameterized.expand(
        [
            ("empty_env", ""),
            ("whitespace_env", "   "),
            ("empty_brackets", "[]"),
        ]
    )
    def test_get_hostname_ignores_empty_env(self, _name: str, env_val: str) -> None:
        with patch.dict(os.environ, {HOSTNAME_ENV: env_val}, clear=True):
            self.assertEqual(get_hostname(), socket.gethostname())

    @parameterized.expand(
        [
            ("hostname", "localhost", 29500, "localhost:29500"),
            ("ipv4", "10.4.0.12", 29500, "10.4.0.12:29500"),
            ("ipv6", "2001:db8::1", 29500, "[2001:db8::1]:29500"),
            ("ipv6_loopback", "::1", 8080, "[::1]:8080"),
            ("bracketed_ipv6", "[2001:db8::1]", 29500, "[2001:db8::1]:29500"),
        ]
    )
    def test_format_host_port(
        self, _name: str, host: str, port: int, expected: str
    ) -> None:
        self.assertEqual(format_host_port(host, port), expected)
