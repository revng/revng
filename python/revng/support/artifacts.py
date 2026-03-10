#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import IO, Dict, Type, TypeVar, Union

import pycparser

from revng.ptml.parser import PTMLDocument
from revng.ptml.parser import parse as ptml_parse
from revng.ptml.printer import ColorMode, ptml_print
from revng.support import get_llvmcpy

T = TypeVar("T")


class Artifact:
    _CHILDREN: Dict[str, Type[Artifact]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        mimes = getattr(cls, "MIMES", None)
        if mimes is not None:
            Artifact._CHILDREN.update({mime: cls for mime in mimes})

    @classmethod
    def make(cls, data: bytes, mime: str) -> Artifact:
        class_ = cls._CHILDREN.get(mime, cls)
        return class_(data, mime)

    # This class and all its subclasses have the same constructor
    def __init__(self, data: bytes, mime: str):
        # If we're in a subclass check that the mime matches with the MIMES
        # filed in the class
        if type(self) is not Artifact:
            assert mime in self.__class__.MIMES  # type: ignore[attr-defined]
        self._data = data
        self._mime = mime

    # Common methods
    @property
    def raw_data(self) -> bytes:
        return self._data

    def write_to_disk(self, path: Union[str, Path]):
        with open(path, "wb") as f:
            f.write(self._data)


class PTMLArtifact(Artifact):
    MIMES = ("text/plain+ptml", "text/x.c+ptml", "text/x.asm+ptml", "text/x.hexdump+ptml")

    def parse(self) -> PTMLDocument:
        return ptml_parse(self._data)

    def print(  # noqa: A003
        self,
        output: IO[str] = sys.stdout,
        color: ColorMode = ColorMode.Autodetect,
    ):
        ptml_print(self._data, output, color)


class LLVMArtifact(Artifact):
    MIMES = ("application/x.llvm.bc",)

    def module(self, name: str = "module"):
        llvmcpy = get_llvmcpy()
        context = llvmcpy.get_global_context()
        buffer = llvmcpy.create_memory_buffer_with_memory_range_copy(
            self._data, len(self._data), name
        )
        return context.parse_ir(buffer)


class ImageArtifact(Artifact):
    MIMES = ("image/svg",)

    def show(self):
        with NamedTemporaryFile("wb", prefix="image-artifact-") as f:
            f.write(self._data)
            f.flush()
            subprocess.run(["xdg-open", f.name])


class RecompilableArchiveArtifact(Artifact):
    MIMES = ("application/x.recompilable-archive",)

    def parse(self) -> pycparser.c_ast.FileAST:
        with TemporaryDirectory() as temp_dir:
            with tarfile.open(fileobj=BytesIO(self._data), mode="r:*") as tar:
                tar.extractall(path=temp_dir)

            c_file = Path(temp_dir) / "decompiled/functions.c"
            args = ["-E", "-DDISABLE_ATTRIBUTES", "-DDISABLE_FLOAT16"]
            for executable in ("cpp", "clang-cpp", "gcc", "clang"):
                if shutil.which(executable) is not None:
                    processed_text = pycparser.preprocess_file(c_file, executable, args)
                    break
            else:
                raise ValueError("No suitable C preprocessor found")
            return pycparser.CParser().parse(processed_text, str(c_file))
