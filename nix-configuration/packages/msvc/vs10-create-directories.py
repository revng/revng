#!/usr/bin/env python3

import os
import json
import argparse
import sys
import csv
from collections import defaultdict


def log(message):
    sys.stderr.write(message + "\n")


def post_pipe(name):
    return name.split("|")[1] if "|" in name else name


def visit(tree, entry, prefix, result):
    name = post_pipe(tree[entry][0])

    if ":" in name:
        return

    path = prefix + "/" + name
    result[entry] = path
    for subdir in tree[entry][1]:
        visit(tree, subdir, path, result)


def main():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--create-directories", action="store_true", help="Create directories."
    )
    parser.add_argument("directory", metavar="DIRECTORY")
    parser.add_argument("file", metavar="FILE")
    parser.add_argument("component", metavar="COMPONENT")
    args = parser.parse_args()

    tree = defaultdict(lambda: ["", set()])
    with open(args.directory, "r") as input_file:
        for row in csv.DictReader(input_file):
            tree[row["Directory_Parent"]][1].add(row["Directory"])
            tree[row["Directory"]][0] = row["DefaultDir"]
    directories = {}
    visit(tree, "TARGETDIR", "", directories)

    component_dir = {}
    with open(args.component, "r") as input_file:
        for row in csv.DictReader(input_file):
            dir_name = row["Directory_"]
            if dir_name not in directories:
                log(f"Warning: unknown directory {dir_name}")
            else:
                component_dir[row["Component"]] = directories[dir_name]

    files = {}
    with open(args.file, "r") as input_file:
        for row in csv.DictReader(input_file):
            component = row["Component_"]
            if component not in component_dir:
                log(f"Warning: unknown component {component}")
            else:
                files[row["File"]] = (
                    component_dir[component] + "/" + post_pipe(row["FileName"])
                )

    if args.create_directories:
        for path in sorted([path for _, path in directories.items()]):
            path = "." + path
            if not os.path.isdir(path):
                log(f"Creating {path}")
                os.mkdir(path)

    for name, path in files.items():
        print(f"{name},{path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
