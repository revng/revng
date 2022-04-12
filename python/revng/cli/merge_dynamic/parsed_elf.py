#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import struct
from io import BytesIO

from elftools.elf.elffile import ELFFile

from .util import chunks, first_or_none, only, parse


class ParsedElf:
    def __init__(self, file):
        self.file = file
        self.elf = ELFFile(file)

        self.is_dynamic = len(self.segment_by_type("PT_DYNAMIC")) != 0
        if not self.is_dynamic:
            return

        self.segments = list(self.elf.iter_segments())
        self.sections = list(self.elf.iter_sections())

        self.dynamic = only(self.segment_by_type("PT_DYNAMIC"))
        self.is_rela = (self.has_tag("DT_PLTREL") and self.tag("DT_PLTREL") == 7) or self.has_tag(
            "DT_RELA"
        )

        self.dynstr = self.read_section("DT_STRTAB", "DT_STRSZ")
        if self.is_rela:
            self.reldyn = self.read_section("DT_RELA", "DT_RELASZ")
            self.relstruct = self.elf.structs.Elf_Rela
        else:
            self.reldyn = self.read_section("DT_REL", "DT_RELSZ")
            self.relstruct = self.elf.structs.Elf_Rel
        self.reldyn_relocations = parse(self.reldyn, self.relstruct)

        self.relplt = self.read_section("DT_JMPREL", "DT_PLTRELSZ")
        self.relplt_relocations = parse(self.relplt, self.relstruct)

        # New symbol counting code
        reloc_info_sym = [
            relocation.r_info_sym
            for relocation in (self.reldyn_relocations + self.relplt_relocations)
        ]
        if len(reloc_info_sym) > 0:
            self.symbols_count = max(reloc_info_sym) + 1
        else:
            self.symbols_count = 0

        self.dynsym = self.read_section("DT_SYMTAB", "DT_SYMENT", self.symbols_count)
        self.symbols = parse(self.dynsym, self.elf.structs.Elf_Sym)

        self.gnuversion = self.read_section("DT_VERSYM", scale=self.symbols_count * 2)
        self.gnuversion_indices = self.parse_ints(self.gnuversion, 2)

        self.verneeds = []
        if self.has_tag("DT_VERNEED"):
            verneed_count = self.tag("DT_VERNEEDNUM")
            self.seek_address(self.tag("DT_VERNEED"))
            verneed_position = self.current_position()

            verneed_struct = self.elf.structs.Elf_Verneed
            vernaux_struct = self.elf.structs.Elf_Vernaux

            for _ in range(verneed_count):
                verneed = self.read_struct(verneed_struct)

                vernauxs = []
                vernaux_position = verneed_position + verneed.vn_aux
                self.file.seek(vernaux_position)
                for _ in range(verneed.vn_cnt):
                    vernaux = self.read_struct(vernaux_struct)
                    vernauxs.append(vernaux)

                    vernaux_position += vernaux.vna_next
                    self.file.seek(vernaux_position)

                self.verneeds.append((verneed, vernauxs))

                verneed_position += verneed.vn_next
                self.file.seek(verneed_position)

    def serialize_verneeds(self, verneeds):
        stream = BytesIO()
        verneed_struct = self.elf.structs.Elf_Verneed
        vernaux_struct = self.elf.structs.Elf_Vernaux

        verneed_position = 0
        for verneed in verneeds:
            stream.write(verneed_struct.build(verneed[0]))

            vernaux_position = verneed_position + verneed[0].vn_aux
            for vernaux in verneed[1]:
                stream.seek(vernaux_position)
                stream.write(vernaux_struct.build(vernaux))
                vernaux_position += vernaux.vna_next

            verneed_position += verneed[0].vn_next
            stream.seek(verneed_position)

        return stream.getvalue()

    def parse_ints(self, buffer, size):
        assert len(buffer) % size == 0
        size_map = {1: "B", 2: "H", 4: "I", 8: "Q"}
        if self.elf.little_endian:
            format_str = "<" + size_map[size]
        else:
            format_str = ">" + size_map[size]
        return [struct.unpack(format_str, chunk)[0] for chunk in chunks(buffer, size)]

    def serialize_ints(self, ints, size):
        size_map = {1: "B", 2: "H", 4: "I", 8: "Q"}
        if self.elf.little_endian:
            format_str = "<" + size_map[size]
        else:
            format_str = ">" + size_map[size]
        return b"".join([struct.pack(format_str, number) for number in ints])

    def current_position(self):
        return self.file.tell()

    def read_struct(self, struct_def):
        buffer = self.file.read(struct_def.sizeof())
        assert len(buffer) == struct_def.sizeof()
        return only(parse(buffer, struct_def))

    def read_section(self, address_tag, size_tag=None, scale=1):
        if size_tag is not None:
            if not self.has_tag(size_tag):
                return bytes()
            size = self.tag(size_tag) * scale
        else:
            size = scale

        if self.has_tag(address_tag):
            return self.read_address(self.tag(address_tag), size)
        return bytes()

    def seek_address(self, address):
        self.file.seek(only(self.elf.address_offsets(address)))

    def read_address(self, address, size):
        if size == 0:
            return b""
        self.seek_address(address)
        result = self.file.read(size)
        assert len(result) == size
        return result

    def segment_by_type(self, stype):
        return [segment for segment in self.elf.iter_segments() if segment.header.p_type == stype]

    def dt_by_tag(self, search):
        return [tag for tag in self.dynamic.iter_tags() if tag.entry.d_tag == search]

    def tag(self, tag):
        return only(self.dt_by_tag(tag)).entry.d_val

    def has_tag(self, tag):
        return len(self.dt_by_tag(tag)) != 0

    def segment_by_range(self, address, size):
        return first_or_none(
            [
                segment
                for segment in self.segment_by_type("PT_LOAD")
                if overlaps(
                    address,
                    size,
                    segment.header.p_vaddr,
                    segment.header.p_memsz,
                )
            ]
        )

    def dynamic_size(self):
        return (
            len(self.dynsym)
            + len(self.dynstr)
            + len(self.gnuversion)
            + len(self.reldyn)
            + self.dynamic.header.p_memsz
            + len(self.segments) * self.elf.structs.Elf_Phdr.sizeof()
            + len(self.sections) * self.elf.structs.Elf_Shdr.sizeof()
        )


def overlaps(start1, size1, start2, size2):
    return (start1 + size1) >= start2 and (start2 + size2) >= start1
