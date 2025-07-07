#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import ast
import sys


class NodeVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.bad_annotations = set()

    def visit_AnnAssign(self, node: ast.AnnAssign):  # noqa: N802
        self._check_constant(node.annotation)

    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802
        for arg in self._collect_arguments(node.args):
            self._check_constant(arg.annotation)
        self._check_constant(node.returns)

    @staticmethod
    def _collect_arguments(arguments: ast.arguments) -> list[ast.arg]:
        result: list[ast.arg] = []
        result.extend(arguments.posonlyargs)
        result.extend(arguments.args)
        if arguments.vararg is not None:
            result.append(arguments.vararg)
        result.extend(arguments.kwonlyargs)
        if arguments.kwarg is not None:
            result.append(arguments.kwarg)
        return result

    def _check_constant(self, expr: ast.expr | None):
        if not (expr is not None and isinstance(expr, ast.Constant)):
            return
        if isinstance(expr.value, str):
            # TODO: could be a forward reference, but for now it's most likely
            #       a sign that a caster is missing. If needed in the future
            #       it'd be easy to have a `visit_ClassDef` method that removes
            #       entries if they are present
            self.bad_annotations.add(expr.value)


def main():
    parsed_input = ast.parse(sys.stdin.read())
    node_visitor = NodeVisitor()
    node_visitor.visit(parsed_input)
    if len(node_visitor.bad_annotations) == 0:
        return 0

    for entry in node_visitor.bad_annotations:
        print(f"Found string annotation for {entry}, probably a caster is missing!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
