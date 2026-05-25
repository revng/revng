#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from dataclasses import dataclass

import click

from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.cli.hypercorn import hypercorn_command, run_hypercorn
from revng.pypeline.rss_server.postgres import PostgresRSSStorage
from revng.pypeline.rss_server.storage import RSSStorage
from revng.pypeline.rss_server.viewer import PDVHTTPServer

DRIVERS: dict[str, type[RSSStorage]] = {
    "postgres": PostgresRSSStorage,
}


@dataclass
class AppMaker:
    production: bool
    storage_driver: str
    connection_string: str

    def __call__(self):
        server = PDVHTTPServer(DRIVERS[self.storage_driver], self.connection_string)
        return server.make_starlette(self.production)


@click.command(help="Start the Remote Storage Server HTTP server")
@hypercorn_command(workers=True)
@click.option(
    "-d",
    "--storage-driver",
    type=click.Choice(DRIVERS),
    default="postgres",
    help="The storage driver to use",
    show_default=True,
)
@click.option(
    "-c",
    "--connection-string",
    default="postgresql://localhost/rss",
    help="The connection string that will be used by the storage class",
    show_default=True,
    envvar="PDV_CONNECTION_STRING",
)
@pass_context
def viewer(ctx: ClickContext, production: bool, storage_driver: str, connection_string: str):
    app_maker = AppMaker(production, storage_driver, connection_string)
    return run_hypercorn(app_maker, ctx.obj.hypercorn_configuration)
