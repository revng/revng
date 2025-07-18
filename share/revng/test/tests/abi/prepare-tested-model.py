#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# This script is used for model preparation for the tests using runtime-abi
# artifacts. It removes all the function not explicitly relevant to the test
# so only `test_X` function plus a couple of helpers (like `main` and `memcpy`)
# remain. After which, it ensures that the prototypes of all the remaining
# functions have their ABI set correctly to work around the limitations of
# ABI detection when importing from dwarf.

import argparse
import sys

import yaml

from revng import model


def main():
    parser = argparse.ArgumentParser(description="Extract and process rev.ng model.")
    parser.add_argument("new_abi")
    parser.add_argument("input_model_path")
    parser.add_argument("output_model_path")
    args = parser.parse_args()

    with open(args.input_model_path, "r") as f:
        binary = yaml.load(f, Loader=model.YamlLoader)

    #
    # Remove any function that is not explicitly relevant to the test.
    #

    def can_drop(function):
        return not (
            function.Name.startswith("test_")
            or function.Name == "main"
            or function.Name == "memcpy"
        )

    for function in [f for f in binary.Functions if can_drop(f)]:
        binary.Functions.remove(function)

    #
    # Ensure the ABI of all remaining functions is what we expect it to be.
    #

    functions = binary.Functions
    function_prototypes_left = [f.Prototype.Definition.id for f in functions if f.Prototype]
    for current_type in binary.TypeDefinitions:
        if (
            current_type.Kind == model.TypeDefinitionKind.CABIFunctionDefinition
            and current_type.key() in function_prototypes_left
        ):
            current_type.ABI = args.new_abi

    with open(args.output_model_path, "w") as f:
        f.write(yaml.dump(binary, Dumper=model.YamlDumper))


if __name__ == "__main__":
    sys.exit(main())
