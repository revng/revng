#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from collections.abc import Sequence
from itertools import islice
from typing import Any, Generic, Optional, TypeVar, cast, overload

from ._capi import _api, ffi
from .exceptions import RevngDocumentException, RevngSimpleException
from .utils import make_python_string


class DocumentError:
    def __init__(self, error: "Error"):
        if not _api.rp_error_is_document_error(error._error):
            raise ValueError("Error is not a DocumentError")

        # We must keep the original error to avoid destruction
        self.base_error = error
        self._document_error = _api.rp_error_get_document_error(error._error)
        self.reasons = DocumentErrorReasonList(self)

    @property
    def error_type(self) -> str:
        result = _api.rp_document_error_get_error_type(self._document_error)
        return make_python_string(result)

    @property
    def location_type(self) -> str:
        result = _api.rp_document_error_get_location_type(self._document_error)
        return make_python_string(result)


class DocumentErrorReason:
    def __init__(self, document_error: DocumentError, index: int):
        self.document_error = document_error
        self.index = index

    @property
    def message(self):
        result = _api.rp_document_error_get_error_message(
            self.document_error._document_error, self.index
        )
        return make_python_string(result)

    @property
    def location(self):
        result = _api.rp_document_error_get_error_location(
            self.document_error._document_error, self.index
        )
        return make_python_string(result)


class DocumentErrorReasonList(Sequence[DocumentErrorReason]):
    def __init__(self, document_error: "DocumentError"):
        self.document_error = document_error

    def __len__(self) -> int:
        return _api.rp_document_error_reasons_count(self.document_error._document_error)

    @overload
    def __getitem__(self, idx: int) -> DocumentErrorReason:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[DocumentErrorReason]:
        ...

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, int):
            if idx < len(self):
                return DocumentErrorReason(self.document_error, idx)
            else:
                raise IndexError("list index out of range")
        else:
            return list(islice(self, idx.start, idx.stop, idx.step))


class SimpleError:
    def __init__(self, error: "Error"):
        if _api.rp_error_is_document_error(error._error):
            raise ValueError("Error is not a SimpleError")

        # We must keep the original error to avoid destruction
        self.base_error = error
        self._simple_error = _api.rp_error_get_simple_error(error._error)

    @property
    def error_type(self) -> str:
        result = _api.rp_simple_error_get_error_type(self._simple_error)
        return make_python_string(result)

    @property
    def message(self) -> str:
        result = _api.rp_simple_error_get_message(self._simple_error)
        return make_python_string(result)


class Error:
    def __init__(self, _error: Optional[Any] = None):
        if _error is not None:
            self._error = _error
        else:
            self._error = _api.rp_error_create()

    def is_success(self) -> bool:
        return _api.rp_error_is_success(self._error)

    def unwrap(self) -> DocumentError | SimpleError:
        if _api.rp_error_is_document_error(self._error):
            return DocumentError(self)
        else:
            return SimpleError(self)

    def to_exception(self) -> RevngDocumentException | RevngSimpleException:
        unwrapped = self.unwrap()
        if isinstance(unwrapped, DocumentError):
            return RevngDocumentException(unwrapped)
        else:
            return RevngSimpleException(unwrapped)


T = TypeVar("T")


class Expected(Generic[T]):
    def __init__(self, result: Any, error: Error):
        self.result: Optional[T] = result if result != ffi.NULL else None
        self.error = error

    def __bool__(self):
        return self.error.is_success() and self.result is not None

    def unwrap(self) -> T:
        """Returns the wrapped object, unless there's an error, in that case
        an exception is raised"""
        if not bool(self):
            raise self.error.to_exception()
        return cast(T, self.result)
