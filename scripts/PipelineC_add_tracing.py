#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pycparser.c_ast import FuncDecl, NodeVisitor, TypeDecl, Typedef
from pycparser.c_generator import CGenerator
from pycparser.c_parser import CParser

# Regex that's used to remove all comments within the source headers
comment_re = re.compile(r"/\*.*?\*/|//([^\n\\]|\\.)*?$", re.DOTALL | re.MULTILINE)
# Regex matching function names in PipelineC
name_re = re.compile("[a-zA-Z0-9_]+$")


@dataclass
class Argument:
    # Name of the argument (e.g. "foo")
    name: str
    # Declaration of the argument (with type, e.g. "int *foo[]")
    decl: str


@dataclass
class Function:
    # Name of the function
    name: str
    # List of arguments
    arguments: List[Argument]
    # Return type
    return_type: str


class Visitor(NodeVisitor):
    def __init__(self):
        super().__init__()
        self.c_gen = CGenerator()
        self.functions: List[Function] = []
        self.types: List[str] = []

    def visit_FuncDecl(self, node: FuncDecl):  # noqa: N802
        name = node.type.declname if isinstance(node.type, TypeDecl) else node.type.type.declname
        return_type = self.c_gen.visit(node.type)
        arguments = []
        if node.args is not None:
            arguments = [self.parse_argument(arg) for arg in node.args.params]
        self.functions.append(Function(name, arguments, return_type))

    def visit_Typedef(self, node: Typedef):  # noqa: N802
        if node.name not in {"bool", "uint8_t", "uint32_t", "uint64_t"}:
            self.types.append(node.name)

    def parse_argument(self, arg: str) -> Argument:
        string: str = self.c_gen.visit(arg)
        name_match = name_re.search(string.removesuffix("[]"))
        assert name_match is not None
        return Argument(name_match[0], string)


def parse_headers(headers: List[Path]):
    # Fixup definitions for types not present in C99
    text = """
typedef struct bool bool;
typedef struct uint8_t uint8_t;
typedef struct uint32_t uint32_t;
typedef struct uint64_t uint64_t;
"""
    for header in headers:
        text += header.read_text()

    # Remove comments and pre-processor directives
    text = comment_re.sub("", text)
    text = "".join(
        ln if not (ln.startswith("#") or ln.startswith("LENGTH_HINT")) else "\n"
        for ln in text.splitlines(True)
    )

    parser = CParser()
    visitor = Visitor()
    visitor.visit(parser.parse(text, "input"))
    return (visitor.functions, visitor.types)


def generate(output: Path, functions: List[Function], types: List[str]):
    # This file exposes a wrapper for each PipelineC function
    # e.g. `void foo(int i) { return wrap<"foo">(_foo, i); }`
    with open(output / "Wrappers.h", "w") as output_file:
        for f in functions:
            args = ", ".join(a.decl for a in f.arguments)
            arg_names = "".join(f", {a.name}" for a in f.arguments)
            # Write the function wrapper
            output_file.write(
                f"{f.return_type} {f.name}({args}) "
                f'{{ return wrap<"{f.name}">(_{f.name}{arg_names}); }};\n'
            )

    # Create an include file that contains the string `registerFunction<Name>(function)`
    # This is to be used with trace-runner-like programs to allow function enumeration
    with open(output / "Functions.inc", "w") as output_file:
        for f in functions:
            output_file.write(f"FUNCTION({f.name})\n")

    with open(output / "Types.inc", "w") as output_file:
        for type_name in types:
            output_file.write(f"TYPE({type_name})\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--header", type=str, action="append", help="Header file to parse")
    parser.add_argument("output", type=str, help="Output File")
    args = parser.parse_args()

    if args.header is None or len(args.header) == 0:
        raise ValueError("At least one header must be specified")

    functions, types = parse_headers([Path(p) for p in args.header])
    generate(
        Path(args.output),
        functions,
        types,
    )


if __name__ == "__main__":
    main()
