#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Any, Callable, Generator, Generic, Optional, TypeVar

from ._capi import _api, ffi
from .exceptions import RevngCException
from .utils import make_generator, make_python_string


class ErrorContainer:
    def __init__(self, _error_container: Optional[Any] = None):
        if _error_container is not None:
            self._error_container = _error_container
        else:
            self._error_container = _api.rp_make_error_container()

    def empty(self) -> bool:
        return _api.rp_error_container_is_empty(self._error_container)

    def count(self) -> int:
        return _api.rp_error_container_error_count(self._error_container)

    def get_message(self, index: int) -> Optional[str]:
        result = _api.rp_error_container_get_error_message(self._error_container, index)
        return make_python_string(result) if result != ffi.NULL else None

    def get_last_message(self) -> Optional[str]:
        count = self.count()
        if count > 0:
            return self.get_message(count - 1)
        else:
            return None

    def messages(self) -> Generator[str, None, None]:
        return make_generator(self.count(), self.get_message)

    def check(self):
        if not self.empty():
            raise RevngCException(self.get_last_message())


T = TypeVar("T")


class Expected(Generic[T]):
    def __init__(self, result: Optional[T], error_container: ErrorContainer):
        self.result = result if result != ffi.NULL else None
        self.error_container = error_container

    def __bool__(self):
        return self.error_container.empty()


def run_with_ec(func: Callable[..., T], *args) -> Expected[T]:
    ec = ErrorContainer()
    result = func(*args, ec._error_container)
    return Expected(result, ec)
