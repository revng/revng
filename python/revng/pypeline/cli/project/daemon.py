#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os

import click
import uvicorn
from uvicorn.main import Server

import revng.pypeline.daemon.app as app
from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.cli.utils import PypeCommand
from revng.pypeline.daemon.daemon import Daemon

# The ASGI spec does not have any facility to report that the shutdown of the
# server has begun. The lifespan protocol sends the `lifespan.shutdown` message
# only when "the server has stopped accepting connections and closed all active
# connections" (cited from the spec).
# This does not work well with long-running websocket connections, because we
# want to know when the server has started shutting down so that we can close
# the sockets. This create a catch-22 where the sockets are waiting for the
# shutdown signal to be closed and the server is waiting for the sockets to
# close to send the shutdown signal.
# Seemingly [1] the only reliable way of fixing this is to monkey-patch the
# `handle_exit` method of `uvicorn.main.Server` so that we can trigger an
# `asyncio.Event` variable and trigger the websockets to shut down.
# Uvicorn has a pending PR [2] that makes this unnecessary but it hasn't been
# merged yet.
#
# [1] https://stackoverflow.com/q/58133694
# [2] https://github.com/Kludex/uvicorn/pull/2242
original_handle_exit = Server.handle_exit


def new_handle_exit(self, *args, **kwargs):
    if not self.should_exit:
        assert app.shutdown_begun is not None
        app.shutdown_begun.set()
    return original_handle_exit(self, *args, **kwargs)


Server.handle_exit = new_handle_exit  # type: ignore[method-assign]

# End of hack to trigger websocket shutdown


@click.command(cls=PypeCommand)
@click.option(
    "--production",
    is_flag=True,
    help="Enable production settings.",
)
@pass_context
def run_daemon(ctx: ClickContext, production, **kwargs):
    """Start the HTTP daemon."""

    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG

    # Setup formatting
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = log_format
    log_config["formatters"]["default"]["fmt"] = log_format

    for uvlogger in log_config["loggers"].values():
        uvlogger["level"] = "INFO"

    if not production:
        os.environ["STARLETTE_DEBUG"] = "1"
        os.environ["REVNG_ORIGINS"] = "*"
        kwargs.setdefault("host", "127.0.0.1")
    else:
        kwargs.setdefault("host", "0.0.0.0")

    daemon = Daemon(
        pipeline=ctx.obj.pipeline,
        storage_provider_url=ctx.obj.storage_provider_url,
        cache_dir=ctx.obj.cache_dir,
        base_directory=ctx.obj.base_directory,
    )

    # Start the uvicorn server
    uvicorn.run(app=app.make_starlette(daemon), **kwargs)


# Inherit all params from the uvicorn cli
for param in uvicorn.main.params:
    # ignore the app param because we will pass it in `run_daemon`
    if param.name == "app":
        continue
    # Add help text that explains production changes
    if param.name == "host":
        param.default = None  # type: ignore [attr-defined]
        param.show_default = False  # type: ignore [attr-defined]
        param.help += (  # type: ignore [attr-defined]
            " Defaults to 0.0.0.0 in production and 127.0.0.1 otherwise."
        )
    run_daemon.params.append(param)
