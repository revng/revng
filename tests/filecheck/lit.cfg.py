#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# lit defines config and then loads this file, so we must prevent
# check-conventions to complain about config not being defined.
# flake8: noqa: F821
# type: ignore

import os
import tempfile

import lit.formats

config.name = "revng-filecheck"
config.test_format = lit.formats.ShTest(True)
# CMake registers each test file explicitly; no suffix-based discovery.
config.suffixes = []
config.standalone_tests = True
config.test_source_root = os.path.dirname(__file__)

# REVNG_BUILD_DIR is exported by CMake via add_test's ENVIRONMENT property.
# Tests use `%root/bin/revng` and `%root/lib/...` so a single substitution
# resolves both the binary and any libraries we need to point at.
build_dir = os.environ.get("REVNG_BUILD_DIR", os.getcwd())
# Each ctest runs lit on one file, so we get parallel lit invocations all
# writing to the same exec root — and the shared .lit_test_times.txt then
# corrupts. Give every invocation its own scratch dir.
config.test_exec_root = tempfile.mkdtemp(prefix="revng-filecheck-")
config.substitutions.append(("%root", build_dir))
