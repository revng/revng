#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import click

import revng.pypeline.daemon.app as app
from revng.pypeline.cli.backend import backend_factory_for
from revng.pypeline.cli.backend.daemon_backend import DaemonBackendFactory
from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.cli.hypercorn import hypercorn_command, run_hypercorn
from revng.pypeline.cli.utils import PypeCommand
from revng.pypeline.daemon.daemon import Daemon


@click.command(cls=PypeCommand)
@hypercorn_command()
@pass_context
def run_daemon(ctx: ClickContext, production: bool):
    """Start the HTTP daemon."""
    backend_factory = backend_factory_for(
        ctx.obj.storage_provider_url,
        pipeline=ctx.obj.pipeline,
        base_directory=ctx.obj.base_directory,
        cache_dir=ctx.obj.cache_dir,
    )
    if isinstance(backend_factory, DaemonBackendFactory):
        raise click.UsageError(
            "The daemon cannot be backed by a `daemon://` URL. Point "
            "--storage-provider at an actual storage provider (e.g. local://)."
        )
    daemon = Daemon(
        pipeline=ctx.obj.pipeline,
        storage_provider_url=ctx.obj.storage_provider_url,
        cache_dir=ctx.obj.cache_dir,
        base_directory=ctx.obj.base_directory,
    )

    def on_shutdown():
        assert app.shutdown_begun is not None
        app.shutdown_begun.set()

    return run_hypercorn(
        lambda: app.make_starlette(production, daemon),
        ctx.obj.hypercorn_configuration,
        on_shutdown,
    )
