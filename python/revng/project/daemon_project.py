#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from io import IOBase
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Dict, List, Set, Union
from urllib.parse import urlparse

import requests
from urllib3.response import HTTPResponse

from revng.project.common import ALL_OBJECTS, AllObjects, HTTPError, ProjectError
from revng.project.model import Binary  # type: ignore[attr-defined]
from revng.project.project import Project
from revng.support import TarDictionary


class BufferedReader(IOBase):
    """
    Adapter class that buffers an urllib3 HTTPResponse content in a spooled file
    """

    CHUNK_SIZE = 1 * 1024 * 1024

    def __init__(self, response: HTTPResponse):
        self.response = response
        self.file = SpooledTemporaryFile(max_size=2 * 1024 * 1024)
        self.max_offset = 0
        self.position = 0
        self.end = False

    def close(self):
        self.response.close()
        self.file.close()

    def fileno(self):
        # We do not want to return self.file.fileno(), we want users to use
        # read instead so we can perform buffering gradually
        raise OSError

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""

        if size == -1 and not self.end:
            self._read_internal(-1)

        if size > 0 and self.position + size > self.max_offset:
            self._read_internal(self.position + size - self.max_offset)

        self.file.seek(self.position, os.SEEK_SET)
        result = self.file.read(size)
        self.position += len(result)
        return result

    def _read_internal(self, size: int):
        chunk_size = self.__class__.CHUNK_SIZE
        self.file.seek(self.max_offset)
        if size == -1:
            while (buffer := self.response.read(chunk_size)) != b"":
                self.file.write(buffer)
                self.max_offset += len(buffer)
            self.end = True
        else:
            while size > 0:
                size_to_read = chunk_size if size > chunk_size else size
                buffer = self.response.read(size_to_read)
                if buffer == b"":
                    self.end = True
                    break

                self.file.write(buffer)
                self.max_offset += len(buffer)
                size -= len(buffer)

    def seek(self, offset: int, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.position = offset
            return self.position
        elif whence == os.SEEK_CUR:
            self.position += offset
            return self.position
        else:
            raise ValueError

    def tell(self) -> int:
        return self.position


class DaemonProject(Project):
    """
    This class is used to run revng analysis and artifact through
    the revng daemon via HTTP.
    """

    def __init__(self, url: str):
        self._base_url = urlparse(url)
        self._session = requests.Session()
        self._session.hooks["response"].append(self._response_hook)

        # The constructor calls `_get_pipeline_description`, so we need to
        # initialize our variables first
        super().__init__()
        self._set_model(self._get_model())

    @staticmethod
    def _response_hook(response: requests.Response, *args, **kwargs):
        try:
            response.raise_for_status()
        except requests.RequestException as e:
            raise HTTPError from e

    def upload_binary(self, binary_path: Union[str, Path]) -> str:
        with open(binary_path, "rb") as f:
            response = self._session.post(
                self._join_url("/api/put-file"),
                files={"file": (Path(binary_path).name, f)},
            )
        return response.json()["hash"]

    def _get_artifact_impl(
        self,
        artifact_name: str,
        objects: Union[Set[str], AllObjects],
        configuration: Dict[str, str],
    ) -> Dict[str, bytes]:
        body: dict = {
            "artifact": artifact_name,
            "epoch": self._get_epoch(),
            "format": "tar",
        }
        if objects is not ALL_OBJECTS:
            body["objects"] = list(objects)
        if len(configuration) > 0:
            body["configuration"] = configuration

        response = self._session.post(self._join_url("/api/artifact"), json=body, stream=True)
        # TODO: right now we return the response as a dict, with BufferedReader
        #       we could return things more lazily and leverage pipelining
        #       (e.g. parsing input while the rest of the request is incoming)
        return dict(TarDictionary(BufferedReader(response.raw)))

    def _analyze_impl(
        self,
        analysis_name: str,
        configuration: Dict[str, str] = {},
        containers: Dict[str, List[str]] = {},
    ) -> int:
        body = {
            "analysis": analysis_name,
            "epoch": self._epoch,
            "configuration": configuration,
            "containers": containers,
        }

        if analysis_name not in self._analyses_list and analysis_name not in self._analysis_names:
            raise ProjectError(f"Analysis {analysis_name} does not exist")

        req = self._session.post(self._join_url("/api/analysis"), json=body)
        self._set_model(self._get_model())
        return req.json()["epoch"]

    def _get_pipeline_description(self):
        response = self._session.get(self._join_url("/api/pipeline"))
        return response.json()

    def _get_model(self) -> Binary:
        response = self._session.get(self._join_url("/api/model"))
        data = response.json()
        return Binary.deserialize(data["model"], self)

    def _get_epoch(self) -> int:
        response = self._session.get(self._join_url("/api/epoch"))
        return response.json()["epoch"]

    def _join_url(self, path: str) -> str:
        new_path = os.path.normpath(self._base_url.path + path)
        return self._base_url._replace(path=new_path).geturl()
