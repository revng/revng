#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from click import Group

from revng.internal.cli.common import CommandRegistry


# Groups that do not implement anything on their own, they only exist to
# namespace the commands they contain
def setup(registry: CommandRegistry):
    namespaces: list[tuple[tuple[str, ...], Group]] = [
        ((), Group("internal", help="Internal CLI tools for testing purposes", hidden=True)),
        ((), Group("model", help="Model manipulation helpers")),
        (("model",), Group("import", help="Model import helpers")),
        (("model",), Group("export", help="Model export helpers")),
    ]

    for parent, group in namespaces:
        registry.register(parent, group)
