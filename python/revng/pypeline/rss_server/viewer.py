#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import cast

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from revng.pypeline.compression import Compression
from revng.pypeline.utils.registry import get_registry, register_all_subclasses
from revng.pypeline.utils.starlette import get_middlewares, get_project_id

from .storage import LockCheckType, RSSStorage


class EpochMismatchException(Exception):
    pass


def epoch_mismatch_hander(request, exc: EpochMismatchException):
    return Response("Epoch mismatch", 409)


def _parse_accept_encoding(request: Request) -> list[str]:
    string = request.headers.get("Accept-Encoding", "*")
    parts = string.split(",")
    return [part.split(";", 1)[0].strip() for part in parts]


class PDVHTTPServer:
    def __init__(self, storage_class: type[RSSStorage], connection_string: str):
        register_all_subclasses(Compression, use_name=True)
        self.storage_class = storage_class
        self.connection_string = connection_string
        # Placeholder, will be initialized by `initialize`
        self.storage: RSSStorage = cast(RSSStorage, None)

    async def initialize(self):
        assert self.storage is None
        self.storage = await self.storage_class.make_unlocked(self.connection_string)

    async def close(self):
        await self.storage.close()

    async def _get_project_storage(self, request: Request):
        project_id = get_project_id(request.headers)
        assert project_id is not None
        return self.storage.get_project_storage(project_id)

    @asynccontextmanager
    async def _get_locked_project_storage(self, request: Request):
        project_storage = await self._get_project_storage(request)
        async with project_storage.get_locked("", LockCheckType.NONE) as storage:
            yield storage

    async def _check_epoch_parameter(self, request: Request):
        epoch_raw = request.query_params.get("epoch")
        if epoch_raw is None:
            return
        epoch = int(epoch_raw)
        async with self._get_locked_project_storage(request) as storage:
            real_epoch = await storage.get_epoch()

        if epoch != real_epoch:
            raise EpochMismatchException

    async def epoch(self, request: Request) -> Response:
        async with self._get_locked_project_storage(request) as storage:
            epoch = await storage.get_epoch()
        return JSONResponse({"epoch": epoch})

    async def model(self, request: Request) -> Response:
        async with self._get_locked_project_storage(request) as storage:
            model_bytes, epoch = await storage.get_model()
        model_str = model_bytes.decode() if model_bytes is not None else ""
        return JSONResponse({"epoch": epoch, "model": model_str})

    async def pipeline_description(self, request: Request) -> Response:
        await self._check_epoch_parameter(request)
        storage = await self._get_project_storage(request)
        description = await storage.get_pipeline_description()
        return Response(description, media_type="application/x-yaml")

    async def list_objects(self, request: Request) -> Response:
        await self._check_epoch_parameter(request)
        savepoint_id = int(request.query_params["savepoint_id"])
        container_id = request.query_params["container_id"]

        async with self._get_locked_project_storage(request) as storage:
            objects = await storage.list_objects(savepoint_id, container_id)
        return JSONResponse(objects)

    async def get_object(self, request: Request) -> Response:
        await self._check_epoch_parameter(request)
        savepoint_id = int(request.query_params["savepoint_id"])
        container_id = request.query_params["container_id"]
        object_id = request.query_params["object_id"]
        decompress: str | None = request.query_params.get("decompress")

        async with self._get_locked_project_storage(request) as storage:
            contents = await storage.get_object(savepoint_id, container_id, object_id)

        if contents is None:
            return Response(status_code=404)

        # Optimize the case where the `decompress` option is also present in
        # `Accept-Encoding`, if so then the data doesn't have to be
        # decompressed at all, just send it to the browser and and let it
        # decompress it.
        if decompress is None:
            return Response(contents)
        elif decompress in _parse_accept_encoding(request):
            return Response(contents, headers={"Content-Encoding": decompress})
        else:
            compression_cls = get_registry(Compression)[decompress]  # type: ignore[type-abstract]
            compression = compression_cls()  # type: ignore[type-abstract]
            try:
                decompressed_contents = compression.decompress(contents)
                return Response(decompressed_contents)
            except compression_cls.decompression_error:
                return Response("Error while decompressing", status_code=400)

    async def status(self, request: Request):
        return PlainTextResponse("OK")

    # Return a starlette instance with all the routes
    def make_starlette(self, production: bool) -> Starlette:
        routes: list[Route] = [
            Route("/epoch", self.epoch),
            Route("/model", self.model),
            Route("/pipeline-description", self.pipeline_description),
            Route("/list-objects", self.list_objects),
            Route("/object", self.get_object),
            Route("/status", self.status),
        ]
        exception_handlers = {EpochMismatchException: epoch_mismatch_hander}

        @asynccontextmanager
        async def lifespan(app):
            await self.initialize()
            yield
            await self.close()

        return Starlette(
            debug=not production,
            routes=routes,
            exception_handlers=exception_handlers,  # type: ignore
            middleware=get_middlewares(production),
            lifespan=lifespan,
        )
