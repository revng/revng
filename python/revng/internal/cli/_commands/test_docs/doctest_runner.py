#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#


import argparse
import doctest
import importlib.util
import sys

IGNORE_OUTPUT = doctest.register_optionflag("IGNORE_OUTPUT")


class CustomOutputChecker(doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        if optionflags & IGNORE_OUTPUT:
            return True
        return super().check_output(want, got, optionflags)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("file", help="File to test")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("doctest-test", args.file)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    options = 0
    finder = doctest.DocTestFinder(args.verbose, exclude_empty=False)
    runner = doctest.DocTestRunner(CustomOutputChecker(), args.verbose, options)
    for test in finder.find(module, module.__name__):
        runner.run(test)

    return 0 if runner.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
