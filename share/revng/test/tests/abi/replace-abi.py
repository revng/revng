#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

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

    prototypes_to_fix = [str(f.Prototype) for f in binary.Functions if f.Prototype.is_valid()]
    for current_type in binary.Types:
        if current_type.Kind == model.TypeKind.CABIFunctionType:
            reference = binary.get_reference_str(current_type)
            if reference in prototypes_to_fix:
                current_type.ABI = args.new_abi

    with open(args.output_model_path, "w") as f:
        f.write(yaml.dump(binary, Dumper=model.YamlDumper))


if __name__ == "__main__":
    sys.exit(main())
