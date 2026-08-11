#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from io import IOBase
from tempfile import SpooledTemporaryFile

from urllib3.response import HTTPResponse


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
        # self.file.fileno() should not be returned here, the users need to use
        # the read function instead so that buffering can be performed
        # gradually
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
