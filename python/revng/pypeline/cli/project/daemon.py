#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import socket
from pathlib import Path

import click
from hypercorn.config import Config

import revng.pypeline.daemon.app as app
from revng.pypeline.cli.backend import backend_factory_for
from revng.pypeline.cli.backend.daemon_backend import DaemonBackendFactory
from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.cli.hypercorn import hypercorn_command, run_hypercorn
from revng.pypeline.cli.wrappers import WrappablePypeCommand, exec_wrapper_if_needed
from revng.pypeline.daemon.daemon import Daemon


class SocketInUseException(click.ClickException):
    exit_code = 4

    def __init__(self, path: Path):
        super().__init__(f"unix socket {path!s} is already in use")


def unix_socket_has_server(path: Path) -> bool:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect(str(path))
            return True
        except OSError:
            return False


@click.command(cls=WrappablePypeCommand)
@hypercorn_command()
@click.option(
    "--socket-location-file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="If the server is bound to a unix socket, write its path in the specified file",
)
@exec_wrapper_if_needed
@pass_context
def run_daemon(ctx: ClickContext, production: bool, socket_location_file: Path | None):
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

    hypercorn_config = ctx.obj.hypercorn_configuration
    daemon_socket_path = None
    socket_in_use = False
    # Check if the `--bind` option has been specified by checking if the value
    # is the default one
    if hypercorn_config.bind is Config._bind:
        # Get the default path of the unix socket from the daemon. If the
        # result is none (e.g. model is in-memory) then the hypercorn default
        # of `127.0.0.1:8000` will be used.
        daemon_socket_path = daemon.socket_path()
        if daemon_socket_path is not None:
            socket_in_use = unix_socket_has_server(daemon_socket_path)
            hypercorn_config.bind = [f"unix:{daemon_socket_path!s}"]

    def write_socket_location_file():
        if socket_location_file is not None:
            # Check that bind has only one entry and that it is a unix socket path
            assert len(hypercorn_config.bind) == 1 and hypercorn_config.bind[0].startswith("unix:")
            bind_path = Path(hypercorn_config.bind[0].removeprefix("unix:")).resolve()
            socket_location_file.write_text(str(bind_path))

    if socket_in_use:
        assert daemon_socket_path is not None
        write_socket_location_file()
        raise SocketInUseException(daemon_socket_path)

    if daemon_socket_path is not None and daemon_socket_path.exists():
        daemon_socket_path.unlink()

    def on_startup():
        write_socket_location_file()

    def on_shutdown():
        assert app.shutdown_begun is not None
        app.shutdown_begun.set()
        if daemon_socket_path is not None:
            daemon_socket_path.unlink()

    return run_hypercorn(
        lambda: app.make_starlette(production, daemon, on_startup),
        ctx.obj.hypercorn_configuration,
        on_shutdown,
    )
