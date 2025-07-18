#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Mapping
from contextlib import suppress
from functools import cached_property
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import IO, Callable, Dict, Generator, Generic, Optional, Tuple, Type, TypeVar, Union
from typing import cast

import pycparser
import yaml
import zstandard as zstd

from revng.ptml.parser import PTMLDocument
from revng.ptml.parser import parse as ptml_parse
from revng.ptml.printer import ColorMode, ColorPrinter, PlainPrinter, PrinterBackend, ptml_print
from revng.ptml.printer import ptml_print_with_printer
from revng.support import get_llvmcpy

# Get yaml.CSafeLoader if present, fall back to yaml.SafeLoader otherwise
YAMLLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

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
        with open(path, "rb") as f:
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


class _TarMixin(Artifact, Mapping, Generic[T]):
    def __init__(self, data: bytes, mime: str):
        assert mime.endswith("+tar+gz")
        super().__init__(data, mime)
        self._extracted_mime = mime.removesuffix("+tar+gz")
        self._member_function = self.__class__.MEMBER_FUNCTION  # type: ignore[attr-defined]

    @cached_property
    def _keys(self) -> Dict[str, str]:
        with tarfile.open(fileobj=BytesIO(self._data)) as tar_file:
            names = tar_file.getnames()
        return {n.split(".", 1)[0]: n for n in names}

    def __getitem__(self, key: str) -> T:
        if key not in self._keys:
            raise KeyError
        archive_name = self._keys[key]
        with tarfile.open(fileobj=BytesIO(self._data)) as tar_file:
            contents = cast(IO[bytes], tar_file.extractfile(archive_name)).read()
            return self._member_function(contents, self._extracted_mime)

    def __contains__(self, key) -> bool:
        return key in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def items(self) -> Generator[Tuple[str, T], None, None]:  # type: ignore
        with tarfile.open(fileobj=BytesIO(self._data)) as tar_file:
            for member in tar_file.getmembers():
                name = member.name.split(".", 1)[0]
                contents = cast(IO[bytes], tar_file.extractfile(member)).read()
                yield (name, self._member_function(contents, self._extracted_mime))

    def values(self) -> Generator[T, None, None]:  # type: ignore
        for _, value in self.items():
            yield value


# Filter function, if it returns 'False' the key will not be processed
Filter = Callable[[Optional[str]], bool]


class _MappedPTMLMixin:
    def parse(self) -> Dict[str, PTMLDocument]:
        assert isinstance(self, Mapping)
        return {k: v.parse() for k, v in self.items()}

    def print(  # noqa: A003
        self,
        output: IO[str] = sys.stdout,
        color: ColorMode = ColorMode.Autodetect,
        filter_: Optional[Filter] = None,
    ):
        assert isinstance(self, Mapping)
        if filter_ is None:
            filter_ = lambda x: True  # noqa: E731

        printer: PrinterBackend
        if color == ColorMode.Off:
            printer = PlainPrinter(output, indent="  ")
            key_writer = lambda key: output.write(f"{key}: |-\n  ")  # noqa: E731
        else:
            printer = ColorPrinter(output, indent="  ", color=color)
            key_color = printer.key_color()
            key_writer = lambda key: output.write(key_color(f"{key}:") + " |-\n  ")  # noqa: E731
        for key, value in self.items():
            if not filter_(key):
                continue

            key_writer(key)
            ptml_print_with_printer(value.raw_data, printer)
            output.write("\n")


class PTMLTarArtifact(_TarMixin[PTMLArtifact], _MappedPTMLMixin):
    MIMES = ("text/plain+ptml+tar+gz", "text/x.c+ptml+tar+gz", "text/x.asm+ptml+tar+gz")
    MEMBER_FUNCTION = PTMLArtifact


class TarArtifact(_TarMixin[str]):
    MIMES = ("text/x.c+tar+gz",)
    MEMBER_FUNCTION = lambda data, _: data.decode("utf-8")  # noqa; E731


class PTMLYAMLArtifact(Artifact, _MappedPTMLMixin, Mapping):
    MIMES = ("text/plain+ptml+yaml", "text/x.c+ptml+yaml", "text/x.asm+ptml+yaml")

    def __init__(self, data: bytes, mime: str):
        Artifact.__init__(self, data, mime)
        self._member_mime = mime.removesuffix("+yaml")
        self._dict: Dict[str, str] = yaml.load(data, Loader=YAMLLoader)

    def __getitem__(self, key: str) -> PTMLArtifact:
        return PTMLArtifact(self._dict[key].encode("utf-8"), self._member_mime)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)


class LLVMArtifact(Artifact):
    MIMES = ("application/x.llvm.bc+zstd",)

    def module(self, name: str = "module"):
        llvmcpy = get_llvmcpy()
        context = llvmcpy.get_global_context()
        buffer = llvmcpy.create_memory_buffer_with_memory_range_copy(
            self._data, len(self._data), name
        )
        try:
            return context.parse_ir(buffer)
        except llvmcpy.LLVMException as e:
            if self._data[:4] != zstd.FRAME_HEADER:
                raise e
            # If outside of revng parse_ir cannot parse bitcode, decompress it
            # manually via the zstandard library
            with zstd.ZstdDecompressor().stream_reader(BytesIO(self._data)) as stream:
                data = stream.read()
            buffer = llvmcpy.create_memory_buffer_with_memory_range_copy(data, len(data), name)
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


def ptml_artifact_autodetect(
    input_: bytes,
) -> Union[PTMLArtifact, PTMLTarArtifact, PTMLYAMLArtifact]:
    if len(input_) == 0:
        raise ValueError("Input is empty!")

    with suppress(tarfile.ReadError):
        result = PTMLTarArtifact(input_, "text/plain+ptml+tar+gz")
        # Force listing keys to check if it's actually a tar
        list(result.keys())
        return result

    if re.match(rb"\s*<", input_) is None:
        with suppress(yaml.YAMLError):
            return PTMLYAMLArtifact(input_, "text/plain+ptml+yaml")

    return PTMLArtifact(input_, "text/plain+ptml")
