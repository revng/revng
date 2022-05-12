#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
from pathlib import Path
from typing import Optional

from starlette.applications import Starlette
from starlette.config import Config
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from ariadne.asgi import GraphQL
from ariadne.contrib.tracing.apollotracing import ApolloTracingExtension

from revng.api._capi import initialize as capi_initialize

from .demo_webpage import demo_page, production_demo_page
from .schema import SchemafulManager
from .util import project_workdir

workdir: Path = project_workdir()
manager: Optional[SchemafulManager] = None
startup_done = False

config = Config()
DEBUG = config("STARLETTE_DEBUG", cast=bool, default=False)


class ManagerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        assert manager, "Manager not initialized"

        request.scope["workdir"] = workdir
        request.scope["manager"] = manager

        # Safety checks
        assert request.scope["workdir"] is not None
        assert request.scope["workdir"].exists() and request.scope["workdir"].is_dir()
        assert request.scope["manager"] is not None

        return await call_next(request)


async def status(request):
    if startup_done:
        return PlainTextResponse("OK")
    else:
        return PlainTextResponse("KO", 503)


def startup():
    global manager, startup_done
    capi_initialize()
    manager = SchemafulManager(workdir=str(workdir.resolve()))
    app.mount(
        "/graphql",
        GraphQL(
            manager.schema,
            context_value={"manager": manager, "workdir": workdir},
            extensions=[ApolloTracingExtension],
            debug=DEBUG,
        ),
    )
    startup_done = True


def shutdown():
    if manager is not None:
        store_result = manager.save()
        if not store_result:
            logging.warning("Failed to store manager's containers")


app = Starlette(
    debug=DEBUG,
    middleware=[Middleware(ManagerMiddleware)],
    on_startup=[startup],
    on_shutdown=[shutdown],
)

app.add_route("/status", status, ["GET"])
if DEBUG:
    app.add_route("/", demo_page, ["GET"])
else:
    app.add_route("/", production_demo_page, ["GET"])
