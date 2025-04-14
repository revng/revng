#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import re
import sys
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import ParamSpec, Protocol, Sequence, TypeVar, cast
from uuid import uuid4

import boto3
import botocore
import yaml


class Synchronizer(Protocol):
    """A Synchronizer is a class that takes care of saving the manager's
    workdir somewhere other than the local disk."""

    def load(self, path: Path) -> bool:
        ...

    def save(self, path: Path, old_directories: list[Path]) -> bool:
        ...

    def save_exceptions(self) -> tuple[type[Exception], ...]:
        ...

    def get_initial_credentials(self) -> str:
        ...

    def set_credentials(self, credentials: str):
        ...


executor = ThreadPoolExecutor(8)

T = TypeVar("T")
P = ParamSpec("P")


class BotocoreSession(botocore.session.Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.credentials = botocore.credentials.Credentials("", "")

    def get_credentials(self):
        return self.credentials

    def set_credentials(self, access_key: str, secret_key: str):  # type: ignore
        self.credentials.access_key = access_key
        self.credentials.secret_key = secret_key


@dataclass(frozen=True)
class FileData:
    s3_filename: str
    index: int = field(default=0)
    encoding: str = field(default="None")


def gather_bools(future_list: Sequence[Future[bool]], exception_message: str) -> bool:
    """Similar to asyncio.gather for a list of Future[bool]. Will gracefully
    manage multiple exceptions by returning an ExceptionGroup. The returned
    result is True if all the awaited booleans are true and false otherwise."""
    exceptions: list[Exception] = []
    result = True
    for future in future_list:
        try:
            if not future.result():
                result = False
        except Exception as ex:
            exceptions.append(ex)

    if len(exceptions) > 0:
        raise ExceptionGroup(exception_message, exceptions)

    return result


# Url format is:
# s3(s)://<username>:<password>@<region>+<host:port>/<bucket name>/<path>
# WARNING: this regex is not RFC173-compliant, some parts (e.g. username)
#          should be escaped but here are not.
S3_URL = re.compile(
    # Start of the string
    r"\A"
    # s3:// (for HTTP) or s3s (for HTTPS)
    r"^(?P<proto>s3(|s))://"
    # username, followed by a ':'
    r"(?P<username>[^:]*):"
    # password, followed by a '@'
    r"(?P<password>[^@]*)@"
    # region, followed by a '+'
    r"(?P<region>[^+]*)\+"
    # host (including ':port'), followed by a '/'
    r"(?P<host>[^/]*)/"
    # bucket, which is the first path component and is required
    r"(?P<bucket>[^/]+)"
    # path, which includes a leading slash
    r"(?P<path>/.*)?"
    # End of the string
    r"\Z"
)


class S3Synchronizer(Synchronizer):
    def __init__(self, url: str):
        match_obj = S3_URL.match(url)
        if match_obj is None:
            raise ValueError("S3 endpoint invalid")

        self.initial_credentials: str = f'{match_obj["username"]}:{match_obj["password"]}'
        self.botocore_session = BotocoreSession()
        self.botocore_session.set_credentials(match_obj["username"], match_obj["password"])
        config = botocore.config.Config(s3={"addressing_style": "path"})
        session = boto3.session.Session(botocore_session=self.botocore_session)
        proto = "https" if match_obj["proto"] == "s3s" else "http"
        self.client = session.client(
            "s3",
            endpoint_url=f"{proto}://{match_obj['host']}",
            region_name=match_obj["region"],
            config=config,
        )
        self.bucket = match_obj["bucket"]
        if match_obj["path"] is not None:
            self.path = match_obj["path"][1:]
        else:
            self.path = ""

        self.last_file_list: dict[str, FileData] = {}

    def load(self, path: Path) -> bool:
        """Download the contents of S3 to the specified path. Since this method
        is typically called at startup it is not thread-safe."""

        # Check that the provided path is not a file
        assert not path.exists() or path.is_dir()
        path.mkdir(parents=True, exist_ok=True)

        # Download index.yml from S3 and parse it
        index = self._get_index()
        # If the index is empty do nothing
        if len(index) == 0:
            return True

        # Generate the `last_file_list`, with the data retrieved from index.yml
        self.last_file_list = {k: FileData(v) for k, v in index.items()}
        # Download all the files mentioned in index.yml in the specified
        # resume directory
        queue = [
            executor.submit(self._download_file, value, path / key) for key, value in index.items()
        ]
        # Actually wait for all the downloads to finish
        return gather_bools(queue, "Failed to load")

    def save_exceptions(self):
        return (
            # File-read related errors
            OSError,
            # Errors related to S3 facilities (e.g. S3Transfer)
            boto3.exceptions.Boto3Error,
            # Errors related to the connection (e.g. connection timeout)
            botocore.exceptions.BotoCoreError,
            # Error related to S3 (e.g. wrong permissions)
            botocore.exceptions.ClientError,
        )

    def save(self, path: Path, old_directories: list[Path]) -> bool:
        # Check that the provided path is not a file
        assert not path.exists() or path.is_dir()

        # Early exit, if the directory does not exist or file-list isn't there
        if not path.exists() or not (path / "file-list.yml").is_file():
            return True

        # Read and parse the file list
        with open(path / "file-list.yml") as f:
            file_list = yaml.safe_load(f)

        to_upload = set()
        new_file_list = {}
        for file_path, attributes in file_list["Files"].items():
            # The if below decides if a file needs to be uploaded, which
            # happens in one of two conditions:
            # * The path was not seen in the last upload
            # * The path has an index higher than last time
            if (
                file_path not in self.last_file_list
                or self.last_file_list[file_path].index < attributes["Index"]
            ):
                # Create a new filename string, this prepends a UUID to the
                # filename, which preserving the path.
                new_filename = self._generate_filename(file_path)
                to_upload.add(file_path)
                # Generate a new FileData entry with the index and encoding
                # from file-list.yml
                new_file_list[file_path] = FileData(
                    new_filename, attributes["Index"], attributes["Encoding"]
                )
            else:
                new_file_list[file_path] = self.last_file_list[file_path]

        queue = []
        for key in to_upload:
            data = new_file_list[key]
            key_path = self.find_file(key, path, old_directories)
            queue.append(
                executor.submit(self._upload_file, key_path, data.s3_filename, data.encoding)
            )

        if not gather_bools(queue, "Failed to save"):
            return False

        # Generate the `index.yml` file and upload it to S3
        new_index = {k: v.s3_filename for k, v in new_file_list.items()}
        with NamedTemporaryFile("w") as temp_index_file:
            yaml.safe_dump(new_index, temp_index_file)
            result = self._upload_file(Path(temp_index_file.name), "index.yml", "None")
            if not result:
                return False

        # Replace last_file_list with the new one
        self.last_file_list = new_file_list

        return True

    @staticmethod
    def find_file(key: str, path: Path, old_directories: list[Path]) -> Path:
        for base_path in [path, *old_directories]:
            candidate_path = base_path / key
            if candidate_path.is_file():
                return candidate_path
        raise ValueError(f"Could not find key: {key}")

    @staticmethod
    def _generate_filename(filename: str) -> str:
        filename_path = Path(filename)
        return str(filename_path.parent / f"{uuid4()}-{filename_path.name}")

    def get_initial_credentials(self) -> str:
        return self.initial_credentials

    def set_credentials(self, credentials: str):
        access_key, secret_key = credentials.split(":", 1)
        self.botocore_session.set_credentials(access_key, secret_key)

    def _get_key(self, name: str) -> str:
        return os.path.join(self.path, name) if name != "" else self.path

    def _download_file(self, key: str, destination: Path) -> bool:
        destination.parent.mkdir(parents=True, exist_ok=True)
        key = self._get_key(key)
        try:
            self.client.download_file(
                Key=key,
                Bucket=self.bucket,
                Filename=str(destination.resolve()),
            )
            return True
        except self.client.exceptions.NoSuchKey:
            return False

    def _read_file(self, name: str) -> bytes | None:
        key = self._get_key(name)
        try:
            object_request = self.client.get_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.NoSuchKey:
            return None

        return cast(bytes, object_request["Body"].read())

    def _get_index(self) -> dict[str, str]:
        data = self._read_file("index.yml")
        if data is None:
            return {}
        return yaml.safe_load(data.decode("utf-8"))

    def _upload_file(self, source: Path, name: str, encoding: str) -> bool:
        key = self._get_key(name)
        extra_args = {"ACL": "private"}
        if encoding == "Gzip":
            extra_args["ContentEncoding"] = "gzip"
        try:
            self.client.upload_file(
                Filename=str(source.resolve()),
                Key=key,
                Bucket=self.bucket,
                ExtraArgs=extra_args,
            )
            return True
        except boto3.exceptions.S3UploadFailedError as e:
            sys.stderr.write("Failed uploading file\n")
            traceback.print_exception(e, file=sys.stderr)
            sys.stderr.flush()
            return False
