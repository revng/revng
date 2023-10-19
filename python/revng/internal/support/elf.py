#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path
from typing import List, Union

from elftools.elf.dynamic import DynamicSegment
from elftools.elf.elffile import ELFFile


def get_elf_needed(path: str) -> List[str]:
    with open(path, "rb") as elf_file:
        segments = [
            segment
            for segment in ELFFile(elf_file).iter_segments()
            if isinstance(segment, DynamicSegment)
        ]

        if len(segments) != 1:
            return []

        needed = [
            to_string(tag.needed)
            for tag in segments[0].iter_tags()
            if tag.entry.d_tag == "DT_NEEDED"
        ]

        elf_runpath: List[str] = [
            tag.runpath for tag in segments[0].iter_tags() if tag.entry.d_tag == "DT_RUNPATH"
        ]

        assert len(elf_runpath) < 2

        if not elf_runpath:
            return needed

        runpaths = [
            runpath.replace("$ORIGIN", str(Path(path).parent))
            for runpath in elf_runpath[0].split(":")
        ]
        absolute_needed = []
        for lib in needed:
            found = False
            for runpath in runpaths:
                full_path = Path(runpath) / lib
                if full_path.is_file():
                    absolute_needed.append(str(full_path.resolve()))
                    found = True
                    break

            if not found:
                absolute_needed.append(lib)

        return absolute_needed


def is_executable(path: str) -> bool:
    with open(path, "rb") as program:
        return program.read(4) == b"\x7fELF"


def is_dynamic(path: str) -> bool:
    with open(path, "rb") as input_file:
        return (
            len(
                [
                    segment
                    for segment in ELFFile(input_file).iter_segments()
                    if segment.header.p_type == "PT_DYNAMIC"
                ]
            )
            != 0
        )


# Use in case different version of pyelftools might give str or bytes
def to_string(obj: Union[str, bytes]) -> str:
    if isinstance(obj, str):
        return obj
    return obj.decode("utf-8")
