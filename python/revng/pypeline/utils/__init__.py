#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import base64
import hashlib
import os
from collections.abc import Buffer
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Generic, TypeVar

from xdg import xdg_cache_home

T = TypeVar("T")


def cache_directory() -> Path:
    if "PYPELINE_CACHE_DIR" in os.environ:
        return Path(os.environ["PYPELINE_CACHE_DIR"])
    else:
        return xdg_cache_home() / "pypeline"


def is_mime_type_text(mime_type: str) -> bool:
    """
    Returns True if the mime type represents a text file, False otherwise.
    """
    return mime_type.startswith("text/") or mime_type in {
        "application/json",
        "application/xml",
        "application/x-yaml",
    }


def bytes_to_string(data: Buffer, is_text: bool) -> str:
    """Standard way to convert bytes to string in the codebase."""
    if is_text:
        return bytes(data).decode("utf-8")
    else:
        return base64.b64encode(data).decode("utf-8")


def string_to_bytes(data: str, is_text: bool) -> bytes:
    """Inverse of `bytes_to_string`"""
    if is_text:
        return data.encode("utf-8")
    else:
        return base64.b64decode(data)


def crypto_hash(data: bytes | str) -> str:
    """
    Returns the SHA256 hash of the given data in hexadecimal format.
    """
    hasher = hashlib.sha256()
    if isinstance(data, str):
        data = data.encode()
    hasher.update(data)
    return hasher.hexdigest()


class Locked(Generic[T]):
    """
    A generic class that wraps an object with an asyncio.Lock
    to ensure safe concurrent access in an async environment.

    The intended use is:
    ```python
    locked_obj = Locked(your_object)
    async with locked_obj() as obj:
        # Safely access and modify obj here
    ```

    But careful that if the object itself it's immutable (like integers, strings, tuples),
    you cannot modify it in place, making the lock useless.
    ```python
    locked_int = Locked(0)
    async with locked_int() as value:
        value += 1  # This does NOT modify the locked object!

    async with locked_int() as value:
        print(value)  # This will still print 0
    ```
    """

    def __init__(self, obj: T) -> None:
        self._value: T = obj
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[T]:
        """
        Asynchronous context manager that acquires the lock
        and yields the protected object.
        """
        async with self._lock:
            yield self._value
