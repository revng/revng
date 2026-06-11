#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
from dataclasses import dataclass

import click

from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.cli.hypercorn import hypercorn_command, run_hypercorn
from revng.pypeline.rss_server.postgres import PostgresRSSStorage
from revng.pypeline.rss_server.server import RSSHTTPServer
from revng.pypeline.rss_server.storage import RSSStorage

DRIVERS: dict[str, type[RSSStorage]] = {
    "postgres": PostgresRSSStorage,
}


@dataclass
class AppMaker:
    production: bool
    storage_driver: str
    connection_string: str
    notification_url: str | None
    notification_psk: str | None
    public_notification_url: str | None

    def __call__(self):
        server = RSSHTTPServer(
            DRIVERS[self.storage_driver],
            self.connection_string,
            self.notification_url,
            self.notification_psk,
            self.public_notification_url,
        )
        return server.make_starlette(self.production)


@click.command(help="Start the Remote Storage Server HTTP server")
@hypercorn_command(workers=True, exclude={"keep-alive"})
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
    envvar="PRSS_CONNECTION_STRING",
)
@click.option("--notification-url", help="URL to send notifications to")
@click.option(
    "--notification-psk",
    help="PSK to use when sending notifications",
    envvar="PRSS_NOTIFICATION_PSK",
)
@click.option(
    "--public-notification-url",
    help="public-facing URL in case the notification url is private",
)
@pass_context
def server(
    ctx: ClickContext,
    production: bool,
    storage_driver: str,
    connection_string: str,
    notification_url: str | None,
    notification_psk: str | None,
    public_notification_url: str | None,
):
    driver = DRIVERS[storage_driver]
    # migrate the DB
    driver.migrate(connection_string)

    # Set the keep alive, this avoids connection being closed over long-running
    # requests e.g. /model/set
    ctx.obj.hypercorn_configuration.keep_alive_timeout = 3600

    # Instantiate the AppMaker and run hypercorn
    app_maker = AppMaker(
        production,
        storage_driver,
        connection_string,
        notification_url,
        notification_psk,
        public_notification_url,
    )

    def background_maker(event: asyncio.Event):
        async def wrapper():
            tasks = driver.background_tasks(connection_string)
            await event.wait()
            for task in tasks:
                task.cancel()

        return wrapper()

    return run_hypercorn(
        app_maker, ctx.obj.hypercorn_configuration, background_maker=background_maker
    )
