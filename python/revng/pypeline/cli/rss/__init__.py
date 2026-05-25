#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import click

from .relay import relay
from .server import server
from .viewer import viewer


@click.group(help="RSS protocol group")
def rss():
    pass


rss.add_command(server)
rss.add_command(relay)
rss.add_command(viewer)
