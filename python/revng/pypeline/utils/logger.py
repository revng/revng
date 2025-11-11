#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys


class Logger:
    """A simple logger that writes messages to stderr."""

    def __init__(self, module_name: str, debug: bool = False) -> None:
        self.module_name = module_name
        self.debug = debug

    def log(self, message: str) -> None:
        print(f"[{self.module_name}] {message}", file=sys.stderr)

    def debug_log(self, message: str) -> None:
        if self.debug:
            self.log(message)


def get_logger(module_name: str, debug: bool = False) -> Logger:
    """Get a logger for a given module name."""
    return Logger(module_name, debug)


pypeline_logger = get_logger("pypeline")
"""
The pre-initialized logger to use inside pypeline code.
"""
