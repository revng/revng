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
        working_model = yaml.load(f, Loader=model.YamlLoader)

    for working_type in working_model.Types:
        if working_type.Kind == model.TypeKind.CABIFunctionType:
            working_type.ABI = args.new_abi

    with open(args.output_model_path, "w") as f:
        f.write(yaml.dump(working_model, Dumper=model.YamlDumper))


if __name__ == "__main__":
    sys.exit(main())
