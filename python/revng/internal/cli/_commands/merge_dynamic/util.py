#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

import os
import stat
from typing import Iterable, List


def file_size(file):
    file.seek(0, 2)
    return file.tell()


def set_executable(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def only(i: Iterable):
    i = list(i)
    assert len(i) == 1
    return i[0]


def first_or_none(i: Iterable):
    i = list(i)
    return i[0] if len(i) != 0 else None


def chunks(chunk_list: List, size):
    for i in range(0, len(chunk_list), size):
        yield chunk_list[i : i + size]


def parse(buffer, struct):
    struct_size = struct.sizeof()
    assert len(buffer) % struct_size == 0
    return list(map(struct.parse, chunks(buffer, struct_size)))


def serialize(structs, struct):
    return b"".join(map(struct.build, structs))
