#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# This implements the `revng internal daemon-manager` command, which is a
# daemon that handles the spawning of multiple revng daemons from multiple
# clients.
# Each clients registers to the daemon and declares when it's alive by
# periodically sending refreshes. When alive, the client can request an
# instance of `revng project daemon` to be started, if multiple clients request
# the same daemon then successive requests do nothing. The daemon instance is
# terminated once all clients request it to be stopped.

import asyncio
import fcntl
import os
import signal
import sys
from contextlib import contextmanager, suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, sleep
from typing import cast

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

import click
from xdg import xdg_runtime_dir

from revng.internal.cli.common import ClickContext, CommandRegistry, cli_logger, pass_context
from revng.internal.support import check_unix_socket
from revng.pypeline.cli.hypercorn import hypercorn_command, is_bind_default, run_hypercorn

# Maximum time (in seconds) between each client refresh request
CLIENT_REFRESH_TIMEOUT = 60.0
# Grace time (in seconds) after which the daemon will terminate if the number
# of clients reaches 0
CLIENT_TERMINATION_DELAY = 30.0
# Grace time (in seconds) after which a daemon without any clients will be
# terminated
DAEMON_TERMINATION_DELAY = 30.0


class ClientError(Exception):
    def __init__(self, message):
        self.message = message


def cancel_task(task: asyncio.Task | None):
    if task is not None:
        task.cancel()


class Daemon:
    def __init__(self, path: Path):
        # Client IDs which have requested the start of this daemon
        self.clients: set[str] = set()
        # Task that, if not cancelled, will stop the daemon.
        # This task is created when the clients transition from 1 to 0.
        self.termination: asyncio.Task | None = None
        # Path to the project the daemon is running
        self._path = path
        # Process instance of the underlying daemon
        self._process: asyncio.subprocess.Process | None = None
        # Path of the unix socket the daemon is listening to
        self._socket: Path | None = None
        # Event that will resolve when the daemon is ready to accept
        # connections
        self._ready = asyncio.Event()
        # This is set to `True` by the `stop` function
        self._stopping = False
        # Actual task that takes care of running and restarting the daemon in
        # case of crashes
        self._task = asyncio.create_task(self._run())

    async def wait_ready(self) -> Path:
        await self._ready.wait()
        assert self._socket is not None
        return self._socket

    async def stop(self):
        self._stopping = True
        self._signal_process()
        await self._task

    def _signal_process(self):
        process = self._process
        if process is not None and process.returncode is None:
            # The process might have exited between the check and the signal
            with suppress(ProcessLookupError):
                process.send_signal(signal.SIGINT)

    async def _run(self):
        while not self._stopping:
            with NamedTemporaryFile() as temp_file:
                cmd = (
                    "revng",
                    "-C",
                    str(self._path),
                    "project",
                    "daemon",
                    "--socket-location-file",
                    temp_file.name,
                )
                cli_logger.debug_log(f"Starting daemon: {cmd}")
                self._process = await asyncio.create_subprocess_exec(*cmd)

                # Read the socket path from the socket location file, poll the
                # file until the contents are present since there is no other
                # (easy) way to know when the file has been written.
                socket = None
                while True:
                    if self._process.returncode is not None:
                        break

                    location = Path(temp_file.name).read_text().strip()
                    if location != "":
                        socket = Path(location)
                        break

                    await asyncio.sleep(0.05)

                # The daemon could have crashed before writing the socket file,
                # in that case just move to the next iteration of the loop
                if socket is not None:
                    self._socket = socket
                    self._ready.set()

                return_code = await self._process.wait()
                if return_code != 0:
                    cli_logger.debug_log(
                        f"Daemon on {self._path} exited abruptly with {return_code}"
                    )
                self._ready.clear()
                self._process = None


def locked(func):
    async def wrapper(self: "DaemonManager", *args, **kwargs):
        async with self._lock:
            return await func(self, *args, **kwargs)

    return wrapper


class DaemonManager:
    def __init__(self, do_not_shutdown: bool):
        self._do_not_shutdown = do_not_shutdown
        # Lock used to serialize modifications to `_daemons` and
        # `_client_last_refresh`
        self._lock = asyncio.Lock()
        # Map of <project path> -> Daemon instance
        self._daemons: dict[Path, Daemon] = {}
        # Registered clients, mapped to the time of their last refresh
        self._client_last_refresh: dict[str, float] = {}
        # Task that, if not cancelled, will terminate the server.
        # Created when the clients transition from 1 to 0.
        self._termination: asyncio.Task | None = None

    def _check_client_registered(self, client_id: str):
        if client_id not in self._client_last_refresh:
            raise ClientError("client-id is not registered")

    async def start(self, path: Path, client_id: str) -> Path:
        async with self._lock:
            self._check_client_registered(client_id)
            daemon = self._daemons.get(path)
            if daemon is None:
                daemon = Daemon(path)
                self._daemons[path] = daemon

            cancel_task(daemon.termination)
            daemon.termination = None
            daemon.clients.add(client_id)

        return await daemon.wait_ready()

    @locked
    async def stop(self, path: Path, client_id: str) -> bool:
        self._check_client_registered(client_id)
        daemon = self._daemons.get(path)
        if daemon is None or client_id not in daemon.clients:
            return False

        self._drop_client(path, daemon, client_id)
        return True

    @locked
    async def register_client(self, client_id: str):
        if client_id in self._client_last_refresh:
            raise ClientError("client-id is already registered")
        cancel_task(self._termination)
        self._termination = None
        self._client_last_refresh[client_id] = monotonic()

    @locked
    async def refresh_client(self, client_id: str):
        self._check_client_registered(client_id)
        self._client_last_refresh[client_id] = monotonic()

    @locked
    async def unregister_client(self, client_id: str):
        self._check_client_registered(client_id)
        self._unregister_client(client_id)
        self._schedule_termination()

    async def expire_clients(self):
        """Unregister the clients that did not refresh in time."""
        while True:
            async with self._lock:
                now = monotonic()
                expired = [
                    client_id
                    for client_id, refresh in self._client_last_refresh.items()
                    if now - refresh >= CLIENT_REFRESH_TIMEOUT
                ]
                for client_id in expired:
                    cli_logger.debug_log(f"Client {client_id} did not refresh, unregistering")
                    self._unregister_client(client_id)

                self._schedule_termination()
                next_expiry = (
                    min(self._client_last_refresh.values(), default=now) + CLIENT_REFRESH_TIMEOUT
                )

            await asyncio.sleep(max(next_expiry - monotonic(), 0.0))

    def status(self) -> dict:
        return {
            "client-ids": list(self._client_last_refresh),
            "instances": [
                {"path": str(path), "client-ids": sorted(daemon.clients)}
                for path, daemon in self._daemons.items()
            ],
        }

    @locked
    async def shutdown(self):
        cancel_task(self._termination)
        daemons = list(self._daemons.values())
        self._daemons.clear()
        for daemon in daemons:
            cancel_task(daemon.termination)

        await asyncio.gather(*(daemon.stop() for daemon in daemons))

    async def _terminate(self):
        await asyncio.sleep(CLIENT_TERMINATION_DELAY)
        async with self._lock:
            if len(self._client_last_refresh) > 0:
                return

        if not self._do_not_shutdown:
            cli_logger.debug_log("0 clients connected, shutting down")
            os.kill(os.getpid(), signal.SIGINT)

    async def _terminate_daemon(self, path: Path, daemon: Daemon):
        await asyncio.sleep(DAEMON_TERMINATION_DELAY)
        async with self._lock:
            if len(daemon.clients) > 0 or self._daemons.get(path) is not daemon:
                return

            cli_logger.debug_log(f"Daemon on {path} has 0 clients, shutting it down")
            del self._daemons[path]
            await daemon.stop()

    def _unregister_client(self, client_id: str):
        del self._client_last_refresh[client_id]
        for path, daemon in list(self._daemons.items()):
            if client_id in daemon.clients:
                self._drop_client(path, daemon, client_id)

    def _drop_client(self, path: Path, daemon: Daemon, client_id: str):
        daemon.clients.remove(client_id)
        if len(daemon.clients) == 0:
            daemon.termination = asyncio.create_task(self._terminate_daemon(path, daemon))

    def _schedule_termination(self):
        if len(self._client_last_refresh) == 0 and self._termination is None:
            self._termination = asyncio.create_task(self._terminate())


def make_starlette(production: bool, manager: DaemonManager) -> Starlette:
    async def start_endpoint(request: Request):
        body = await request.json()
        socket = await manager.start(Path(body["path"]).resolve(), body["client-id"])
        return JSONResponse({"socket": str(socket)})

    async def stop_endpoint(request: Request):
        body = await request.json()
        if not await manager.stop(Path(body["path"]).resolve(), body["client-id"]):
            return Response(status_code=404)
        return JSONResponse({})

    async def register_client_endpoint(request: Request):
        await manager.register_client((await request.json())["id"])
        return JSONResponse({})

    async def refresh_client_endpoint(request: Request):
        await manager.refresh_client((await request.json())["id"])
        return JSONResponse({})

    async def unregister_client_endpoint(request: Request):
        await manager.unregister_client((await request.json())["id"])
        return JSONResponse({})

    async def status_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(manager.status())

    def client_error_handler(request: Request, exc: Exception):
        return JSONResponse({"message": cast(ClientError, exc).message}, 400)

    return Starlette(
        debug=not production,
        routes=[
            # Register a client ID, this is necessary before any daemon can be
            # started
            Route("/client/register", register_client_endpoint, methods=["POST"]),
            # Refresh a client, this is required so that if a client crashes
            # the daemons it spawned will be cleaned up
            Route("/client/refresh", refresh_client_endpoint, methods=["POST"]),
            # Gracefully unregister the client, implicitly stopping any daemons
            # it has started
            Route("/client/unregister", unregister_client_endpoint, methods=["POST"]),
            # Start a daemon at the specified path, if a daemon was already
            # started at this path then the request will do nothing
            Route("/daemon/start", start_endpoint, methods=["POST"]),
            # Stop a daemon, if other clients started it in the meantime the
            # daemon will continue executing until all clients stopped it
            Route("/daemon/stop", stop_endpoint, methods=["POST"]),
            # Debug endpoint that exposes the overall status of the manager
            Route("/status", status_endpoint, methods=["GET"]),
        ],
        exception_handlers={ClientError: client_error_handler},
    )


def manager_dir() -> Path:
    runtime_dir = xdg_runtime_dir()
    if runtime_dir is not None:
        target = runtime_dir / "revng"
    else:
        target = Path(f"/tmp/revng-runtime-dir-{os.getuid()}")
    target.mkdir(exist_ok=True)
    return target


@contextmanager
def manager_lock():
    path = manager_dir() / "manager.lock"
    with open(path, "wb") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            os.remove(path)


def demonize_manager(socket_path: Path):
    log_path = socket_path.parent / "manager.log"

    with manager_lock():
        if check_unix_socket(socket_path):
            return

        # Delete the socket path, if present
        if socket_path.exists():
            os.unlink(socket_path)

        # Start the manager as a daemon, by double forking
        # TODO: in windows we need to use Popen with `DETACHED_PROCESS`
        # First fork
        pid = os.fork()
        if pid > 0:
            # Only return once the daemon is up
            while not check_unix_socket(socket_path):
                sleep(0.05)
            return
        else:
            # Start a new session
            os.setsid()

            # Second fork
            pid2 = os.fork()
            if pid2 > 0:
                # Close the middle process via `_exit`, this avoids any cleanup
                # from happening (since the parent will take care of that)
                os._exit(0)
            else:
                # Redirect stdout and stderr to the log file, close off stdin
                output_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
                input_fd = os.open(os.devnull, os.O_RDONLY)
                os.dup2(output_fd, sys.stdout.fileno())
                os.dup2(output_fd, sys.stderr.fileno())
                os.dup2(input_fd, sys.stdin.fileno())
                os.close(output_fd)
                os.close(input_fd)
                # Exec into the daemon manager
                os.execve(
                    sys.argv[0],
                    [*sys.argv, "--foreground", "--bind", f"unix:{socket_path!s}"],
                    os.environ,
                )


@click.command(
    name="daemon-manager",
    help="Start a server that manages `revng project daemon` instances",
)
@click.option(
    "--do-not-shutdown",
    is_flag=True,
    help=(
        "Do not shut down the daemon when there aren't any client connected."
        " (implies --foreground)"
    ),
)
@click.option("--foreground", is_flag=True, help="Do not demonize, run in foreground")
@hypercorn_command()
@pass_context
def daemon_manager(ctx: ClickContext, production: bool, do_not_shutdown: bool, foreground: bool):
    if do_not_shutdown:
        foreground = True
    if not foreground and not is_bind_default(ctx.obj.hypercorn_configuration):
        raise click.UsageError("--bind cannot be used without --do-not-shutdown or --foreground")

    if not foreground:
        socket_path = manager_dir() / "manager.sock"
        demonize_manager(socket_path)
        return

    manager = DaemonManager(do_not_shutdown)

    async def background(shutdown_event: asyncio.Event):
        expire_task = asyncio.create_task(manager.expire_clients())
        await shutdown_event.wait()
        await manager.shutdown()
        expire_task.cancel()
        with suppress(asyncio.CancelledError):
            await expire_task

    return run_hypercorn(
        lambda: make_starlette(production, manager),
        ctx.obj.hypercorn_configuration,
        background_maker=background,
    )


def setup(registry: CommandRegistry):
    registry.register(("internal",), daemon_manager)
