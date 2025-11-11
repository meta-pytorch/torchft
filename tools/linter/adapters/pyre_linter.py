# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Set, TypedDict

logger: logging.Logger = logging.getLogger(__name__)


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: Optional[str]
    line: Optional[int]
    char: Optional[int]
    code: str
    severity: LintSeverity
    name: str
    original: Optional[str]
    replacement: Optional[str]
    description: Optional[str]


class PyreResult(TypedDict):
    line: int
    column: int
    stop_line: int
    stop_column: int
    path: str
    code: int
    name: str
    description: str
    concise_description: str


def run_pyre() -> List[PyreResult]:
    proc = subprocess.run(
        ["pyre", "--output=json", "incremental"],
        capture_output=True,
    )
    return json.loads(proc.stdout)


def check_pyre(
    filenames: Set[str],
) -> List[LintMessage]:
    try:
        results = run_pyre()

        return [
            LintMessage(
                path=result["path"],
                line=result["line"],
                char=result["column"],
                code="pyre",
                severity=LintSeverity.WARNING,
                name=result["name"],
                description=result["description"],
                original=None,
                replacement=None,
            )
            for result in results
        ]
    except Exception as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code="pyre",
                severity=LintSeverity.ADVICE,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checks files with pyre",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(processName)s:%(levelname)s> %(message)s",
        level=(
            logging.NOTSET
            if args.verbose
            else logging.DEBUG
            if len(args.filenames) < 1000
            else logging.INFO
        ),
        stream=sys.stderr,
    )

    lint_messages = check_pyre(set(args.filenames))

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
