#!/usr/bin/env python3

import struct
import subprocess
import sys
import tempfile

from elftools.elf.elffile import ELFFile

def main():
  arguments = sys.argv[1:]

  if (len(arguments) == 0) or ("--help" in arguments):
    print("Usage: {} COMPILER [ARGS ...]".format(sys.argv[0]))
    return 0

  if "-c" not in arguments:
    arguments.append("-c")
  assert "-o" not in arguments

  with tempfile.NamedTemporaryFile(suffix=".o") as object_file:
    arguments += ["-o", object_file.name]
    subprocess.check_call(arguments)

    elf = ELFFile(object_file)
    symtab = elf.get_section_by_name(".symtab")
    assert symtab is not None

    targets = [(symbol.name,
                symbol.entry.st_size,
                (elf.get_section(symbol.entry.st_shndx).header.sh_offset
                 + symbol.entry.st_value))
               for symbol
               in symtab.iter_symbols()
               if (symbol.name
                   and symbol.entry.st_info.bind == "STB_GLOBAL"
                   and symbol.entry.st_info.type == "STT_OBJECT")]

    stream = elf.stream
    direction = "<" if elf.little_endian else ">"
    size_map = {1: "B", 2: "H", 4: "I", 8: "Q"}
    for name, size, offset in targets:
      stream.seek(offset)
      buffer = stream.read(size)
      assert len(buffer) == size
      value = struct.unpack(direction + size_map[size], buffer)[0]
      print("{},{}".format(name, hex(value)))

  return 0

if __name__ == "__main__":
  sys.exit(main())
