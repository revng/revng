#
# This file is distributed under the MIT License. See LICENSE.md for details.
#


# DR_{INIT,FINI}_ARRAY?

import shutil
import struct
from copy import copy

from elftools.elf.constants import P_FLAGS
from elftools.elf.enums import ENUM_RELOC_TYPE_ARM, ENUM_RELOC_TYPE_i386, ENUM_RELOC_TYPE_x64

from .log import log
from .parsed_elf import ParsedElf
from .util import file_size, serialize, set_executable


def rebuild_r_info(relocation, is64):
    if is64:
        relocation.r_info = (relocation.r_info_sym << 32) | relocation.r_info_type
    else:
        relocation.r_info = (relocation.r_info_sym << 8) | relocation.r_info_type


def align(start, alignment):
    return start + (-start % alignment)


def right_pad_align(buf, to, pad_char=b"\x00"):
    assert len(pad_char) == 1, "pad_char must be only one byte long!"
    required_padding = -len(buf) % to
    return buf + pad_char * required_padding


def get_relative_relocation(machine):
    if machine == "EM_X86_64":
        relative_relocation = ENUM_RELOC_TYPE_x64["R_X86_64_RELATIVE"]
    elif machine == "EM_ARM":
        relative_relocation = ENUM_RELOC_TYPE_ARM["R_ARM_RELATIVE"]
    elif machine == "EM_386":
        relative_relocation = ENUM_RELOC_TYPE_i386["R_386_RELATIVE"]
    elif machine == "EM_MIPS":
        # TODO: check
        relative_relocation = 0xFFFFFFFF
    elif machine == "EM_S390":
        relative_relocation = 12  # R_390_RELATIVE
    else:
        raise ValueError(f"Unknown machine: {machine}")

    return relative_relocation


def merge_dynamic(
    to_extend_file,
    source_file,
    output_file,
    base,
    merge_load_segments=False,
):
    to_extend_elf = ParsedElf(to_extend_file)
    source_elf = ParsedElf(source_file)

    # If the original ELF was not dynamic, we don't have to do anything
    if not source_elf.is_dynamic:
        to_extend_elf.file.seek(0)
        shutil.copyfileobj(to_extend_elf.file, output_file)
        if not output_file.isatty():
            set_executable(output_file.fileno())
        return 0

    assert to_extend_elf.is_dynamic
    assert to_extend_elf.elf.header.e_machine == source_elf.elf.header.e_machine

    relocation_offset = 0
    if source_elf.elf.header.e_type == "ET_DYN":
        relocation_offset = base

    # Prepare new .dynstr
    new_dynstr = to_extend_elf.dynstr
    to_extend_dynstr_size = len(new_dynstr)
    assert new_dynstr[-1] == 0, ".dynstr section of the binary to extend is not NULL terminated"
    new_dynstr += source_elf.dynstr

    # TODO: replace this code.
    #  Many sections have alignment requirements.
    #  Because we align dynstr all subsequent sections will be aligned,
    #  as all of them consist of tables with entries of a fixed size.
    #  We really should ensure that all the sections are aligned explicitly
    new_dynstr = right_pad_align(new_dynstr, 0x4)

    to_extend_size = file_size(to_extend_file)
    base_address = min(
        [segment.header.p_vaddr for segment in to_extend_elf.segment_by_type("PT_LOAD")]
    )
    alignment = 0x1000
    estimated_size = align(to_extend_elf.dynamic_size() + source_elf.dynamic_size(), alignment)
    start_address = align(base_address + to_extend_size, alignment)
    matching_segment = to_extend_elf.segment_by_range(
        start_address, estimated_size
    ) or source_elf.segment_by_range(start_address, estimated_size)

    while matching_segment is not None:
        log(
            f"Discarding {hex(start_address)} since overlaps the following segment:\n"
            f"  {matching_segment.header}"
        )
        start_address = align(
            matching_segment.header.p_vaddr + matching_segment.header.p_memsz,
            alignment,
        )
        matching_segment = to_extend_elf.segment_by_range(
            start_address, estimated_size
        ) or source_elf.segment_by_range(start_address, estimated_size)

    padding = start_address - base_address - to_extend_size
    new_dynstr_offset = to_extend_size + padding

    # Prepare new .dynsym
    new_dynsym = to_extend_elf.dynsym
    defined_symbols = []
    for index, symbol in enumerate(to_extend_elf.symbols):
        if symbol.st_shndx != "SHN_UNDEF":
            defined_symbols.append(index)

    dynsym_offset = len(to_extend_elf.symbols) - 1
    new_symbols = list(source_elf.symbols)[1:]
    for index, symbol in enumerate(new_symbols):
        symbol.st_name += to_extend_dynstr_size
        if symbol.st_value != 0:
            symbol.st_value += relocation_offset
        if symbol.st_shndx != "SHN_UNDEF":
            defined_symbols.append(index + dynsym_offset + 1)

    new_dynsym += serialize(new_symbols, source_elf.elf.structs.Elf_Sym)
    new_dynsym_offset = new_dynstr_offset + len(new_dynstr)

    # Prepare new .dynrel
    relative_relocation = get_relative_relocation(to_extend_elf.elf.header.e_machine)
    new_reldyn = to_extend_elf.reldyn
    new_relocations = source_elf.relplt_relocations + source_elf.reldyn_relocations
    for relocation in new_relocations:
        if relocation.r_info_sym != 0:
            relocation.r_info_sym += dynsym_offset
        if relocation.r_info_type == relative_relocation:
            relocation.r_addend += relocation_offset
        relocation.r_offset += relocation_offset
        rebuild_r_info(relocation, to_extend_elf.elf.elfclass == 64)
    new_reldyn += serialize(new_relocations, source_elf.relstruct)
    new_reldyn_offset = new_dynsym_offset + len(new_dynsym)

    # 1. Find the highest version index in to_extend_elf
    version_index_offset = 0
    for verneed in to_extend_elf.verneeds:
        for vernaux in verneed[1]:
            version_index_offset = max(version_index_offset, vernaux.vna_other)
    version_index_offset -= 1

    # 2. Go though all the version indexes of source_elf and, unless they
    #  are 0 or 1, increase them by the previous value
    # 3. Concat .gnu.version
    new_gnuversion_offset = new_reldyn_offset + len(new_reldyn)
    new_gnuversion = to_extend_elf.gnuversion
    new_gnuversion_indices = source_elf.gnuversion_indices[1:]
    for index, value in enumerate(new_gnuversion_indices):
        if value not in (0, 1):
            new_gnuversion_indices[index] += version_index_offset
    new_gnuversion += source_elf.serialize_ints(new_gnuversion_indices, 2)

    # 4. Go through .gnu.version_r and, for each verneed add the string
    #    table offset to the library name.
    # 5. Go through each Vernaux and increment vna_name
    new_verneeds = to_extend_elf.verneeds

    # Find the start position of the last verneed
    position = 0
    for verneed in new_verneeds:
        position += verneed[0].vn_next

    # Update the pointer to the next element of the last verneed to the end
    # of the buffer
    new_verneeds_size = len(to_extend_elf.serialize_verneeds(new_verneeds))
    new_verneeds[-1][0].vn_next = new_verneeds_size - position
    new_gnuversion_r_offset = new_gnuversion_offset + len(new_gnuversion)

    # Fix verneeds and vernaux in source_elf
    for verneed in source_elf.verneeds:
        verneed[0].vn_file += to_extend_dynstr_size
        for vernaux in verneed[1]:
            vernaux.vna_name += to_extend_dynstr_size
            vernaux.vna_other += version_index_offset
    new_verneeds += source_elf.verneeds

    # Explicitly ensure the last entry is marked as such
    new_verneeds[-1][0].vn_next = 0

    new_gnuversion_r = source_elf.serialize_verneeds(new_verneeds)

    # We now build a fake old-style (non-GNU) hash table. Basically we have a
    # single bucketup pointing to the first defined symbol, which in turn will
    # point to the second one and so on, up to the last which has identifier 0
    # and stops the search for a symbol.  Basically, we transformed a hash
    # lookup in a linear search.

    # TODO: implement an actual hash table, possibly GNU

    symbols_count = dynsym_offset + len(new_symbols) + 1

    new_hash = build_dummy_hashtable(
        symbols_count,
        defined_symbols,
        little_endian=to_extend_elf.elf.little_endian,
    )
    new_hash_offset = new_gnuversion_r_offset + len(new_gnuversion_r)

    # Prepare new .dynamic
    new_dynamic_tags = list(to_extend_elf.dynamic.iter_tags())

    libraries = set()

    def to_address(offset):
        return start_address + offset - new_dynstr_offset

    for index, dynamic_tag in enumerate(new_dynamic_tags):
        if dynamic_tag.entry.d_tag == "DT_STRTAB":
            dynamic_tag.entry.d_val = to_address(new_dynstr_offset)
        elif dynamic_tag.entry.d_tag == "DT_STRSZ":
            dynamic_tag.entry.d_val = len(new_dynstr)
        elif dynamic_tag.entry.d_tag in ["DT_REL", "DT_RELA"]:
            dynamic_tag.entry.d_val = to_address(new_reldyn_offset)
        elif dynamic_tag.entry.d_tag in ["DT_RELSZ", "DT_RELASZ"]:
            dynamic_tag.entry.d_val = len(new_reldyn)
        elif dynamic_tag.entry.d_tag == "DT_SYMTAB":
            dynamic_tag.entry.d_val = to_address(new_dynsym_offset)
        elif dynamic_tag.entry.d_tag == "DT_NEEDED":
            libraries.add(dynamic_tag.needed)
        elif dynamic_tag.entry.d_tag == "DT_VERNEED":
            dynamic_tag.entry.d_val = to_address(new_gnuversion_r_offset)
        elif dynamic_tag.entry.d_tag == "DT_VERNEEDNUM":
            dynamic_tag.entry.d_val = len(new_verneeds)
        elif dynamic_tag.entry.d_tag == "DT_VERSYM":
            dynamic_tag.entry.d_val = to_address(new_gnuversion_offset)
        elif dynamic_tag.entry.d_tag == "DT_GNU_HASH":
            dynamic_tag.entry.d_tag = "DT_HASH"
            dynamic_tag.entry.d_val = to_address(new_hash_offset)

    new_dynamic_tags = [dt.entry for dt in new_dynamic_tags]

    new_dynamic = serialize(new_dynamic_tags, source_elf.elf.structs.Elf_Dyn)
    new_dynamic_offset = new_hash_offset + len(new_hash)

    new_section_headers_offset = new_dynamic_offset + len(new_dynamic)
    new_sections = to_extend_elf.sections
    for section in new_sections:
        if section.name == ".dynstr":
            section.header.sh_addr = to_address(new_dynstr_offset)
            section.header.sh_offset = new_dynstr_offset
            section.header.sh_size = len(new_dynstr)
        elif section.name == ".dynsym":
            section.header.sh_addr = to_address(new_dynsym_offset)
            section.header.sh_offset = new_dynsym_offset
            section.header.sh_size = len(new_dynsym)
        elif section.name in [".rela.dyn", ".rel.dyn"]:
            section.header.sh_addr = to_address(new_reldyn_offset)
            section.header.sh_offset = new_reldyn_offset
            section.header.sh_size = len(new_reldyn)
        elif section.name == ".dynamic":
            section.header.sh_addr = to_address(new_dynamic_offset)
            section.header.sh_offset = new_dynamic_offset
            section.header.sh_size = len(new_dynamic)
        elif section.name == ".gnu.version":
            section.header.sh_addr = to_address(new_gnuversion_offset)
            section.header.sh_offset = new_gnuversion_offset
            section.header.sh_size = len(new_gnuversion)
        elif section.name == ".gnu.version_r":
            section.header.sh_addr = to_address(new_gnuversion_r_offset)
            section.header.sh_offset = new_gnuversion_r_offset
            section.header.sh_size = len(new_gnuversion_r)
            section.header.sh_info = len(new_verneeds)
    new_section_headers = serialize(
        [section.header for section in new_sections],
        source_elf.elf.structs.Elf_Shdr,
    )

    # Prepare new program headers
    new_program_headers_offset = new_section_headers_offset + len(new_section_headers)

    segment_header_size = source_elf.elf.structs.Elf_Phdr.sizeof()
    new_segments = [segment.header for segment in to_extend_elf.segments]

    additional_segments = []
    if merge_load_segments:
        # TODO: this assumes the new LOAD segments have 0x1000 alignment
        additional_segments_offset = align(new_program_headers_offset, 0x1000)
        for s in source_elf.segments:
            additional_segment_phdr = copy(s.header)
            if additional_segment_phdr.p_type != "PT_LOAD":
                continue

            additional_segment_content = source_elf.read_address(
                additional_segment_phdr.p_vaddr,
                additional_segment_phdr.p_filesz,
            )
            required_padding = additional_segment_phdr.p_offset % additional_segment_phdr.p_align

            additional_segment_phdr.p_offset = additional_segments_offset + required_padding
            additional_segments_offset = align(
                additional_segments_offset + required_padding + additional_segment_phdr.p_filesz,
                0x1000,
            )

            new_segments.append(additional_segment_phdr)
            additional_segments.append((additional_segment_phdr, additional_segment_content))

    new_program_headers_size = (len(new_segments) + 1) * segment_header_size

    for segment in new_segments:
        if segment.p_type == "PT_DYNAMIC":
            segment.p_filesz = len(new_dynamic)
            segment.p_memsz = len(new_dynamic)
            segment.p_paddr = to_address(new_dynamic_offset)
            segment.p_vaddr = to_address(new_dynamic_offset)
            segment.p_offset = new_dynamic_offset
        elif segment.p_type == "PT_PHDR":
            segment.p_filesz = new_program_headers_size
            segment.p_memsz = new_program_headers_size
            segment.p_paddr = to_address(new_program_headers_offset)
            segment.p_vaddr = to_address(new_program_headers_offset)
            segment.p_offset = new_program_headers_offset

    new_segment_size = new_program_headers_offset + new_program_headers_size - new_dynstr_offset

    if new_segment_size > estimated_size:
        log("Warning: the new segment for dynamic sections is larger than expected:")
        log(f"Expected: {estimated_size}\n  Actual: {new_segment_size}")

    new_segment = source_elf.elf.structs.Elf_Phdr.parse(b"\x00" * segment_header_size)
    new_segment.p_type = "PT_LOAD"
    new_segment.p_offset = new_dynstr_offset
    new_segment.p_flags = P_FLAGS.PF_R | P_FLAGS.PF_W
    new_segment.p_vaddr = start_address
    new_segment.p_paddr = start_address
    new_segment.p_memsz = new_segment_size
    new_segment.p_filesz = new_segment_size
    new_segment.p_align = alignment

    new_segments += [new_segment]

    # Sort LOAD entries in ascending order
    new_segments.sort(key=phdrs_sort_key)
    new_program_headers = serialize(new_segments, source_elf.elf.structs.Elf_Phdr)

    # TODO: Prepare a new PHDR mapping the new DYNAMIC

    # Prepare new ELF header
    new_elf_header = to_extend_elf.elf.header
    new_elf_header.e_phnum = len(new_segments)
    new_elf_header.e_phoff = new_program_headers_offset
    new_elf_header.e_shnum = len(new_sections)
    new_elf_header.e_shoff = new_section_headers_offset
    new_elf_header = to_extend_elf.elf.structs.Elf_Ehdr.build(new_elf_header)

    # Write new ELF header
    output_file.write(new_elf_header)

    # Write rest of the to_extend file
    to_extend_elf.file.seek(len(new_elf_header))
    shutil.copyfileobj(to_extend_elf.file, output_file)

    # Align to page
    output_file.write(b"\x00" * padding)

    # Write new .dynstr
    assert output_file.tell() == new_dynstr_offset
    output_file.write(new_dynstr)

    # Write new .dynsym
    assert output_file.tell() == new_dynsym_offset
    output_file.write(new_dynsym)

    # Write new .rel.dyn
    assert output_file.tell() == new_reldyn_offset
    output_file.write(new_reldyn)

    # Write new .gnu.version
    assert output_file.tell() == new_gnuversion_offset
    output_file.write(new_gnuversion)

    # Write new .gnu.version_r
    assert output_file.tell() == new_gnuversion_r_offset
    output_file.write(new_gnuversion_r)

    # Write new .hash
    assert output_file.tell() == new_hash_offset
    output_file.write(new_hash)

    # Write new .dynamic
    assert output_file.tell() == new_dynamic_offset
    output_file.write(new_dynamic)

    # Write new section headers
    assert output_file.tell() == new_section_headers_offset
    output_file.write(new_section_headers)

    # Write new program headers
    assert output_file.tell() == new_program_headers_offset
    output_file.write(new_program_headers)

    # Write additional segments
    for header, content in additional_segments:
        cur_pos = output_file.tell()
        required_padding = header.p_offset - cur_pos
        output_file.write(b"\x00" * required_padding)
        output_file.write(content)

    if not output_file.isatty():
        set_executable(output_file.fileno())

    return 0


def build_dummy_hashtable(symbols_count, defined_symbols, little_endian=True):
    if little_endian:
        parse32 = "<I"
    else:
        parse32 = ">I"

    chain = [0 for i in range(symbols_count)]
    for i in range(len(defined_symbols) - 1):
        chain[defined_symbols[i]] = defined_symbols[i + 1]
    chain = [struct.pack(parse32, chain[i]) for i in range(len(chain))]
    chain = b"".join(chain)
    return (
        struct.pack(parse32, 1)
        + struct.pack(parse32, symbols_count)
        + (struct.pack(parse32, defined_symbols[0]) if defined_symbols else b"")
        + chain
    )


def phdrs_sort_key(s):
    if s.p_type == "PT_PHDR":
        first_key = 1
    elif s.p_type == "PT_INTERP":
        first_key = 2
    elif s.p_type == "PT_LOAD":
        first_key = 3
    else:
        first_key = 4

    return first_key, s.p_vaddr
