#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
import os
import signal
from datetime import timedelta
from importlib import import_module
from typing import List, Optional

from starlette.applications import Starlette
from starlette.config import Config
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from ariadne.asgi import GraphQL
from ariadne.asgi.handlers import GraphQLHTTPHandler, GraphQLTransportWSHandler
from ariadne.contrib.tracing.apollotracing import ApolloTracingExtension

from revng.api import Manager
from revng.api._capi import initialize as capi_initialize
from revng.api._capi import shutdown as capi_shutdown

from .demo_webpage import demo_page, production_demo_page
from .event_manager import EventManager
from .schema_generator import SchemaGenerator
from .util import project_workdir

manager: Optional[Manager] = None
event_manager: Optional[EventManager] = None
startup_done = False

config = Config()
DEBUG = config("STARLETTE_DEBUG", cast=bool, default=False)


class ManagerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        assert manager, "Manager not initialized"
        request.scope["manager"] = manager

        # Safety checks
        assert request.scope["manager"] is not None

        return await call_next(request)


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


def startup():
    global manager, event_manager, startup_done
    capi_initialize(signals_to_preserve=(signal.SIGINT, signal.SIGTERM))
    manager = Manager(project_workdir())
    event_manager = EventManager(manager)
    app.mount(
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
    )
    startup_done = True


def shutdown():
    global manager
    if manager is not None:
        store_result = manager.save()
        if not store_result:
            logging.warning("Failed to store manager's containers")
        del manager
    capi_shutdown()


def get_middlewares() -> List[Middleware]:
    extra_middlewares: List[Middleware] = []
    if "STARLETTE_MIDDLEWARES" in os.environ:
        for element in os.environ["STARLETTE_MIDDLEWARES"].split(","):
            module_path, attribute_name = element.split(":", 1)
            module = import_module(module_path)
            if not hasattr(module, attribute_name):
                raise ValueError(f"Middleware not found: {element}")
            extra_middlewares.append(getattr(module, attribute_name))

    return [
        *extra_middlewares,
        Middleware(ManagerMiddleware),
        Middleware(
            CORSMiddleware,
            allow_origins=os.environ["REVNG_ORIGINS"].split(",")
            if "REVNG_ORIGINS" in os.environ
            else [],
            allow_methods=["*"],
        ),
    ]


app = Starlette(
    debug=DEBUG,
    middleware=get_middlewares(),
    on_startup=[startup],
    on_shutdown=[shutdown],
)


app.add_route("/status", status, ["GET"])
if DEBUG:
    app.add_route("/", demo_page, ["GET"])
else:
    app.add_route("/", production_demo_page, ["GET"])
