# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import time
from typing import List, Sequence

from opentelemetry._logs import set_logger_provider

from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs._internal import LogData
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogExporter,
    LogExporter,
)
from opentelemetry.sdk.resources import Resource

_LOGGER_PROVIDER: dict[str, LoggerProvider] = {}
# Path to the file containing OTEL resource attributes
TORCHFT_OTEL_RESOURCE_ATTRIBUTES_JSON = "TORCHFT_OTEL_RESOURCE_ATTRIBUTES_JSON"


class TeeLogExporter(LogExporter):
    """Exporter that writes to multiple exporters."""

    def __init__(
        self,
        exporters: List[LogExporter],
    ) -> None:
        self._exporters = exporters

    def export(self, batch: Sequence[LogData]) -> None:
        for e in self._exporters:
            e.export(batch)

    def shutdown(self) -> None:
        for e in self._exporters:
            e.shutdown()


def setup_logger(name: str) -> None:
    if os.environ.get("TORCHFT_USE_OTEL", "false") == "false":
        return

    global _LOGGER_PROVIDER

    if name in _LOGGER_PROVIDER:
        return

    torchft_otel_resource_attributes_json = os.environ.get(
        TORCHFT_OTEL_RESOURCE_ATTRIBUTES_JSON
    )

    if torchft_otel_resource_attributes_json is not None:
        with open(torchft_otel_resource_attributes_json) as f:
            attributes = json.loads(f.read())
            resource = Resource.create(attributes=attributes[name])
    else:
        resource = Resource.create()

    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)

    exporter = TeeLogExporter(
        exporters=[
            ConsoleLogExporter(),
            OTLPLogExporter(
                timeout=5,
            ),
        ],
    )
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)

    # Attach OTLP handler to otel logger
    logging.getLogger(name).addHandler(handler)

    _LOGGER_PROVIDER[name] = logger_provider


def shutdown() -> None:
    for logger_provider in _LOGGER_PROVIDER.values():
        logger_provider.shutdown()


# Example usage of the logger
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    setup_logger("torchft_test")

    while True:
        time.sleep(1)
        loggers = [
            logging.getLogger("torchft_test"),
            logging.getLogger("myapp.area1"),
            logging.getLogger(),
        ]

        for i, logger in enumerate(loggers):
            # only this should be picked up by OTEL when using otel logger
            logger.info(
                "Quick zephyrs blow, vexing daft Jim.",
                extra={
                    "test_attr": f"value{i}",
                },
            )
            logger.debug("Jackdaws love my big sphinx of quartz.")

        print("Example done; exiting...")


if __name__ == "__main__":
    main()
