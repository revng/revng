#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# lit defines config and then loads this file, so we must prevent check-conventions to complain about config not being defined
# flake8: noqa: F821
# type: ignore

import lit.formats

config.name = "revng"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".ll"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.my_obj_root

bin_dir = f"{config.my_obj_root}/bin"
config.substitutions.append(("%revngopt", f"{bin_dir}/revng opt "))
