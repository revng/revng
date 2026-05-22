#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import click

import revng.pypeline.rss_server.relay as app
from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.cli.hypercorn import hypercorn_command, run_hypercorn


@click.command(help="Start the notification relay HTTP server")
@hypercorn_command(exclude={"bind"})
@click.option(
    "--psk",
    help="The pre-shared key for the /publish endpoint",
    envvar="PYPELINE_RSS_RELAY_PSK",
)
@click.option(
    "--notifications-bind",
    help="Address where notifications will be listened on",
    default="127.0.0.1:8002",
)
@click.option(
    "--publish-bind",
    help="Address where publications will be listened on",
    default="127.0.0.1:8003",
)
@pass_context
def relay(
    ctx: ClickContext,
    production: bool,
    psk: str,
    notifications_bind: str,
    publish_bind: str,
):
    def on_shutdown():
        assert app.shutdown_begun is not None
        app.shutdown_begun.set()

    ctx.obj.hypercorn_configuration.bind = [*{notifications_bind, publish_bind}]

    return run_hypercorn(
        lambda: app.make_starlette(production, notifications_bind, publish_bind, psk),
        ctx.obj.hypercorn_configuration,
        on_shutdown,
    )
