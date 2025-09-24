#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging

import click
import uvicorn
import yaml

import revng
from revng.pypeline.daemon.app import app
from revng.pypeline.daemon.daemon import Daemon
from revng.pypeline.daemon.lock.local_lock import LocalLock

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="The interface to bind to.",
    show_default=True,
)
@click.option(
    "--port",
    type=int,
    default=5000,
    help="The port to bind to.",
    show_default=True,
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Number of worker processes.",
    show_default=True,
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode.",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload on file changes.",
)
# TODO: expose all other uvicorn parameters from
# https://github.com/Kludex/uvicorn/blob/master/uvicorn/main.py
@click.option(
    "--lock",
    type=click.Choice(["local"]),
    default="local",
    help="The lock type to use.",
    show_default=True,
)
@click.pass_context
def run_daemon(ctx, host, port, debug, reload, workers, lock):
    """Start the HTTP daemon."""
    # Store in the app state so all routes can access them
    lock_ty = {
        "local": LocalLock,
    }[lock]

    with open(ctx.obj["pipeline_path"], "r", encoding="utf-8") as f:
        pipeline_yaml = yaml.safe_load(f.read())

    app.state.daemon = Daemon(
        version=revng.__version__,
        pipeline_yaml=pipeline_yaml,
        pipeline=ctx.obj["pipeline"],
        debug=debug,
        lock=lock_ty(),
        storage_provider_url=ctx.obj["storage_provider"],
        cache_dir=ctx.obj["cache_dir"],
    )

    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG

    # Setup formatting
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = log_format
    log_config["formatters"]["default"]["fmt"] = log_format

    for uvlogger in log_config["loggers"].values():
        uvlogger["level"] = "INFO"

    # Start the uvicorn server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        workers=workers,  # reload doesn't work with multiple workers
        log_config=log_config,
        access_log=True,
    )
