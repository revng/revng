#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
import os
import signal
from datetime import timedelta
from importlib import import_module
from typing import List

from starlette.applications import Starlette
from starlette.config import Config
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

from ariadne.asgi import GraphQL
from ariadne.asgi.handlers import GraphQLHTTPHandler, GraphQLTransportWSHandler
from ariadne.contrib.tracing.apollotracing import ApolloTracingExtension

from revng.api import Manager
from revng.api._capi import initialize as capi_initialize
from revng.api._capi import shutdown as capi_shutdown

from .demo_webpage import generate_demo_page
from .event_manager import EventManager
from .schema_generator import SchemaGenerator
from .util import project_workdir

config = Config()
DEBUG = config("STARLETTE_DEBUG", cast=bool, default=False)


def get_middlewares() -> List[Middleware]:
    extra_middlewares_early = parse_middleware_env(os.environ.get("STARLETTE_MIDDLEWARES_EARLY"))
    extra_middlewares_late = parse_middleware_env(os.environ.get("STARLETTE_MIDDLEWARES_LATE"))

    origins: List[str] = []
    if "REVNG_ORIGINS" in os.environ:
        origins = os.environ["REVNG_ORIGINS"].split(",")

    return [
        *extra_middlewares_early,
        Middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
        Middleware(GZipMiddleware, minimum_size=1024),
        *extra_middlewares_late,
    ]


def parse_middleware_env(string: str | None) -> List[Middleware]:
    if string is None:
        return []

    result = []
    for element in string.split(","):
        module_path, attribute_name = element.rsplit(".", 1)
        module = import_module(module_path)
        if not hasattr(module, attribute_name):
            raise ValueError(f"Middleware not found: {element}")
        result.append(getattr(module, attribute_name))

    return result


def make_startlette() -> Starlette:
    capi_initialize(
        signals_to_preserve=(
            # Common terminal signals
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGHUP,
            signal.SIGCHLD,
            # Issued by writing in closed sockets
            signal.SIGPIPE,
            # Used by uvicorn workers
            signal.SIGUSR1,
            signal.SIGUSR2,
            signal.SIGQUIT,
        )
    )
    manager = Manager(project_workdir())
    event_manager = EventManager(manager)
    startup_done = False

    async def status(request):
        if startup_done:
            return PlainTextResponse("OK")
        else:
            return PlainTextResponse("KO", 503)

    def generate_context(request: Request):
        return {
            "manager": manager,
            "event_manager": event_manager,
            "headers": request.headers,
        }

    routes = [
        Route("/", generate_demo_page(manager.workdir, DEBUG), methods=["GET"]),
        Route("/status", status, methods=["GET"]),
        Mount(
            "/graphql",
            GraphQL(
                SchemaGenerator().get_schema(manager),
                context_value=generate_context,
                http_handler=GraphQLHTTPHandler(extensions=[ApolloTracingExtension]),
                websocket_handler=GraphQLTransportWSHandler(
                    connection_init_wait_timeout=timedelta(seconds=5)
                ),
                debug=DEBUG,
            ),
        ),
    ]

    def startup():
        nonlocal startup_done
        startup_done = True

    def shutdown():
        store_result = manager.save()
        if not store_result:
            logging.warning("Failed to store manager's containers")
        manager._manager = None
        capi_shutdown()

    return Starlette(
        debug=DEBUG,
        middleware=get_middlewares(),
        routes=routes,
        on_startup=[startup],
        on_shutdown=[shutdown],
    )


app = make_startlette()
