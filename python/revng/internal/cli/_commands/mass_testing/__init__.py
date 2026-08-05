#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import click

from revng.internal.cli.common import CommandRegistry

from .configure import configure
from .generate_report import generate_report
from .run import mass_testing_run


@click.group(name="mass-testing", help="Mass-testing CLI tools")
def mass_testing():
    pass


def setup(registry: CommandRegistry):
    registry.register((), mass_testing)
    registry.register(("mass-testing",), configure)
    registry.register(("mass-testing",), mass_testing_run)
    registry.register(("mass-testing",), generate_report)
