#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import asyncio
import multiprocessing
import signal
from functools import partial
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event as EventType
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import click
import hypercorn.__main__ as main_module
import uvloop
from hypercorn.app_wrappers import ASGIWrapper
from hypercorn.asyncio import serve
from hypercorn.asyncio.run import _run, worker_serve
from hypercorn.config import Config, Sockets
from hypercorn.run import _join_exited, run
from hypercorn.utils import check_multiprocess_shutdown_event

from .context import ClickContext

_CAPTURED_ARGS: list[tuple[tuple[str, ...], dict]] = []
_ARGUMENT_WHITELIST = (
    "access-logfile",
    "access-logformat",
    "bind",
    "backlog",
    "graceful-timeout",
    "read-timeout",
    "max-requests",
    "group",
    "keep-alive",
    "log-level",
    "umask",
    "user",
    "websocket-ping-interval",
)
_ARGUMENT_MAPPING = {
    "access_logfile": "accesslog",
    "access_logformat": "access_log_format",
    "log_level": "loglevel",
    "keep_alive": "keep_alive_timeout",
}


class _FakeArgumentParser(argparse.ArgumentParser):
    def add_argument(self, *args, **kwargs):
        _CAPTURED_ARGS.append((args, kwargs))
        super().add_argument(*args, **kwargs)


def _fake_run(config):
    return 0


def _capture_args():
    if len(_CAPTURED_ARGS) > 0:
        return _CAPTURED_ARGS

    # Replace the variables that control parsing and running
    main_module.argparse = SimpleNamespace()
    main_module.argparse.ArgumentParser = _FakeArgumentParser
    main_module.run = _fake_run

    # Trigger the main
    main_module.main(["dummy"])

    # Restore the variables
    main_module.argparse = argparse
    main_module.run = run

    return _CAPTURED_ARGS


def _production_callback(ctx: ClickContext, param: click.Option, value):
    config = ctx.obj.hypercorn_configuration
    if config.bind is Config._bind and value:
        config.bind = ["0.0.0.0:8000"]
    return value


def hypercorn_command(*, workers: bool = False, exclude: set[str] = set()):
    args = _capture_args()
    needed_options = {f"--{name}" for name in _ARGUMENT_WHITELIST if name not in exclude}
    if workers:
        needed_options.add("--workers")
    lists_appended = set()

    def wrapper(func):
        def callback_generator(kwargs):
            def callback(ctx: ClickContext, param: click.Option, value):
                if value is None:
                    return value

                config = ctx.obj.hypercorn_configuration
                config_name = _ARGUMENT_MAPPING.get(param.name, param.name)
                assert param.name is not None
                assert hasattr(config, config_name)
                action = kwargs.get("action")
                if action is not None:
                    assert action == "append"
                    if param.name in lists_appended:
                        getattr(config, config_name).append(value)
                    else:
                        setattr(config, config_name, [value])
                        lists_appended.add(param.name)
                else:
                    setattr(config, config_name, value)
                return value

            return callback

        for arg in args:
            for option in arg[0]:
                if option in needed_options:
                    needed_options.remove(option)
                    break
            else:
                continue

            func = click.option(
                *arg[0],
                callback=callback_generator(arg[1]),
                type=arg[1].get("type"),
                help=arg[1]["help"],
                expose_value=False,
            )(func)

        func = click.option(
            "--production",
            is_flag=True,
            callback=_production_callback,
            help="Enable production settings",
        )(func)

        assert len(needed_options) == 0
        return func

    return wrapper


type AppMaker = Callable[[], Any]
type OnShutdown = Callable[[], None]
type BackgroundMaker = Callable[[asyncio.Event], Awaitable[None]]


def run_hypercorn(
    app_maker: AppMaker,
    config: Config,
    on_shutdown: OnShutdown | None = None,
    background_maker: BackgroundMaker | None = None,
):
    if config.workers in (0, 1):
        return _run_hypercorn_single(app_maker(), config, on_shutdown, background_maker)
    else:
        return _run_hypercorn_multi(app_maker, config, on_shutdown, background_maker)


def _run_hypercorn_single(
    app,
    config: Config,
    on_shutdown: OnShutdown | None,
    background_maker: BackgroundMaker | None,
):
    async def main():
        shutdown_event = asyncio.Event()

        def _signal_handler(*args):
            if on_shutdown is not None:
                on_shutdown()
            shutdown_event.set()

        # Reset the signals first, otherwise `add_signal_handler` does not
        # actually install the handlers
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)

        background_task = None
        if background_maker is not None:
            background_task = asyncio.create_task(background_maker(shutdown_event))

        await serve(app, config, shutdown_trigger=shutdown_event.wait)
        if background_task is not None:
            await background_task

    uvloop.run(main())


def _worker(
    app_maker: AppMaker, config: Config, sockets: Sockets, shutdown_event: EventType
) -> None:
    app = app_maker()
    shutdown_trigger = None
    if shutdown_event is not None:
        shutdown_trigger = partial(check_multiprocess_shutdown_event, shutdown_event, asyncio.sleep)

    _run(
        partial(worker_serve, ASGIWrapper(app), config, sockets=sockets),
        debug=config.debug,
        shutdown_trigger=shutdown_trigger,
        loop_factory=uvloop.new_event_loop,
    )


def _run_hypercorn_multi(
    app_maker: AppMaker,
    config: Config,
    on_shutdown: OnShutdown | None,
    background_maker: BackgroundMaker | None,
):
    # Copied and adapted from hypercorn/run.py:run
    context = multiprocessing.get_context("spawn")

    async def main():
        loop = asyncio.get_running_loop()
        shutdown_event = multiprocessing.Event()
        async_shutdown_event = asyncio.Event()

        def shutdown(*args) -> None:
            shutdown_event.set()
            async_shutdown_event.set()
            if on_shutdown is not None:
                on_shutdown()

        background_task = None
        if background_maker is not None:
            background_task = asyncio.create_task(background_maker(async_shutdown_event))

        sockets = config.create_sockets()
        processes: list[BaseProcess] = []
        while not shutdown_event.is_set():
            # Ignore signals before creating processes, so that only the main
            # thread controls them
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

            for _ in range(config.workers - len(processes)):
                process = context.Process(
                    target=_worker,
                    kwargs={
                        "app_maker": app_maker,
                        "config": config,
                        "shutdown_event": shutdown_event,
                        "sockets": sockets,
                    },
                )
                process.start()
                processes.append(process)

            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            loop.add_signal_handler(signal.SIGINT, shutdown)
            loop.add_signal_handler(signal.SIGTERM, shutdown)

            await asyncio.to_thread(wait, (process.sentinel for process in processes))
            exitcode = _join_exited(processes)
            if exitcode != 0:
                shutdown()

        if background_task is not None:
            await background_task

        for process_to_terminate in processes:
            process_to_terminate.terminate()

        exitcode = _join_exited(processes) if exitcode != 0 else exitcode

        for sock in (*sockets.secure_sockets, *sockets.insecure_sockets):
            sock.close()

    uvloop.run(main())
