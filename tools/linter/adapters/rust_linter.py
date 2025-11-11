# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
import subprocess
import sys
from typing import Pattern

import lintrunner_adapters
from lintrunner_adapters import as_posix, LintMessage, LintSeverity, run_command

LINTER_CODE = "RUSTFMT"

SYNTAX_ERROR_ARROW_RE: Pattern[str] = re.compile(
    r"(?m)^( +--> )(.+)(:(?P<line>\d+):(?P<column>\d+))\n"
)

SYNTAX_ERROR_PARSE_RE: Pattern[str] = re.compile(r"(?m)^failed to parse .*\n")


def strip_path_from_error(error: str) -> str:
    # Remove full paths from the description to have deterministic messages.
    error = SYNTAX_ERROR_ARROW_RE.sub("", error, count=1)
    error = SYNTAX_ERROR_PARSE_RE.sub("", error, count=1)
    return error


def check_file(
    filename: str,
    *,
    binary: str,
    config_path: str | None,
) -> list[LintMessage]:
    try:
        with open(filename, "rb") as f:
            original = f.read()
        with open(filename, "rb") as f:
            proc = run_command(
                [
                    binary,
                    "+nightly",
                    "--emit=stdout",
                    "--quiet",
                ]
                + (["--config-path", config_path] if config_path else []),
                stdin=f,
                check=True,
            )
    except (OSError, subprocess.CalledProcessError) as err:
        # https://github.com/rust-lang/rustfmt#running
        # TODO: Fix the syntax error regexp to handle multiple issues and
        # to handle the empty result case.
        if (
            isinstance(err, subprocess.CalledProcessError)
            and err.returncode == 1
            and err.stderr
        ):
            line = None
            char = None
            description = err.stderr.decode("utf-8")
            match = SYNTAX_ERROR_ARROW_RE.search(description)
            if match:
                line = int(match["line"])
                char = int(match["column"])
                description = f"```\n{strip_path_from_error(description)}\n```"
            return [
                LintMessage(
                    path=filename,
                    line=line,
                    char=char,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="parsing-error",
                    original=None,
                    replacement=None,
                    description=description,
                )
            ]

        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,
                        command=" ".join(as_posix(x) for x in err.cmd),
                        stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                        stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                    )
                ),
            )
        ]

    replacement = proc.stdout
    if original == replacement:
        return []

    if proc.stderr.startswith(b"error: "):
        clean_err = strip_path_from_error(proc.stderr.decode("utf-8")).strip()
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.WARNING,
                name="rustfmt-bug",
                original=None,
                replacement=None,
                description=(
                    "Possible rustfmt bug. "
                    f"rustfmt returned error output but didn't fail:\n{clean_err}"
                ),
            )
        ]

    return [
        LintMessage(
            path=filename,
            line=1,
            char=1,
            code=LINTER_CODE,
            severity=LintSeverity.WARNING,
            name="format",
            original=original.decode("utf-8"),
            replacement=replacement.decode("utf-8"),
            description="See https://github.com/rust-lang/rustfmt#tips",
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format rust files with rustfmt. Linter code: {LINTER_CODE}",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--binary",
        required=True,
        default="rustfmt",
        help="rustfmt binary path",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        default=None,
        help="rustfmt config path",
    )

    lintrunner_adapters.add_default_options(parser)
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=(
            logging.NOTSET
            if args.verbose
            else logging.DEBUG
            if len(args.filenames) < 1000
            else logging.INFO
        ),
        stream=sys.stderr,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(
                check_file, x, binary=args.binary, config_path=args.config_path
            ): x
            for x in args.filenames
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    lint_message.display()
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
