# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import subprocess
from enum import Enum
from typing import NamedTuple, Optional


class LintSeverity(str, Enum):
    ERROR = "error"
    ADVICE = "advice"


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


def check_pyrefly() -> list[LintMessage]:
    try:
        proc = subprocess.run(
            [
                "pyrefly",
                "check",
                "--config",
                "pyrefly.toml",
                "--output-format=json",
                "--summary=none",
                "--progress-bar=no",
            ],
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code="PYREFLY",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=str(error),
            )
        ]

    if proc.returncode not in (0, 1):
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code="PYREFLY",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=proc.stderr.strip(),
            )
        ]

    try:
        result, _ = json.JSONDecoder().raw_decode(proc.stdout.lstrip())
        errors = result.get("errors", [])
    except (AttributeError, json.JSONDecodeError, TypeError) as error:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code="PYREFLY",
                severity=LintSeverity.ERROR,
                name="json-parse-error",
                original=None,
                replacement=None,
                description=str(error),
            )
        ]

    return [
        LintMessage(
            path=error["path"],
            line=error["line"],
            char=error["column"],
            code="PYREFLY",
            severity=LintSeverity.ERROR,
            name=error["name"],
            original=None,
            replacement=None,
            description=error["description"],
        )
        for error in errors
    ]


def main() -> None:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("filenames", nargs="+")
    parser.parse_args()

    for lint_message in check_pyrefly():
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
