/* This file is distributed under the MIT License. See LICENSE.md for details.
 * Keep the ELF header outside the executable segment.
 */
ENTRY(entry)
PHDRS {
  headers PT_LOAD FILEHDR PHDRS FLAGS(4);
  code PT_LOAD FLAGS(5);
}
SECTIONS {
  . = 0x400000 + SIZEOF_HEADERS;
  .header : { BYTE(0) } :headers
  . = 0x401000;
  .text : { *(.text) } :code
}
