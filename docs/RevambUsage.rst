******
revng
******

---------------------------------------------------------
lift a binary from any architecture to compilable LLVM IR
---------------------------------------------------------

:Author: Alessandro Di Federico <ale+revng@clearmind.me>
:Date:   2016-12-22
:Copyright: MIT
:Version: 0.1
:Manual section: 1
:Manual group: rev.ng

SYNOPSIS
========

    revng [options] [--] INFILE OUTFILE

OPTIONS
=======

`revng` options allow to customize its output. In the following the main ones
are described:

:``-e``, ``--entry``: Tell `revng` to ignore the program entry point (typically
                      the `_start` function) and start to translated code at the
                      specified address (as *decimal* number). Note that this
                      option also disables the harvesting of the global data in
                      search for code pointers (see
                      `/usr/share/doc/revng/GeneratedIRReference.html`). This
                      option is mainly useful to test how `revng` acts on a
                      limited portion of code without translating the whole
                      program. Default: entry point specified by the ELF header
                      (``ElfN_Ehdr.e_entry``).
:``-i``, ``--linking-info``: Path where the CSV containing instructions for the
                             linker on how to position the segment variables
                             (see
                             `/usr/share/doc/revng/FromIRToExecutable.html`).
                             Default: ``OUTFILE.li.csv``.
:``-g``, ``--debug-info``: Type of debug information to associate to the
                           generated module, i.e. the type of *source code* to
                           use in step-by-step debugging of the generated
                           program (e.g., using `gdb`). Possible values are:
                           `none` for no debugging information, `asm` to dump
                           the disassembly of the input code, `ptc` to dump TCG
                           instructions and `ll` to dump the LLVM IR.
                           ``--debug-path`` specifies the location of the
                           output. Default locations are ``OUTFILE.S`` for
                           `asm`, ``OUTFILE.ptc`` for `ptc` and ``OUTFILE``
                           itself for `ll`.
:``-s``, ``--debug-path``: Path where the *debug source* should be saved. See
                           ``--debug-info`` for additional information and
                           default value.
:``-c``, ``--coverage-path``: Path where the list of the address ranges that
                              have been translated should be stored (in CSV
                              form). This option is deprecated in favor of using
                              `revng-dump` and will be removed. Default:
                              ``OUTFILE.coverage.csv``
:``-d``, ``--debug``: Enable verbose debugging info during execution. The
                      argument is a comma-separated list of keywords identifying
                      a type of debug information. This debug information are
                      for development usage and possible values can be
                      identified directly in the source code in the first
                      argument of the ``DBG(...)`` macro.
:``-O``, ``--no-osra``: Disable OSR Analysis. Useful for debugging purposes to
                        evaluate its effectiveness.
:``-L``, ``--no-link``: Do not link in QEMU helper functions. The call will
                        appear as calls to extern functions. This option makes
                        the resulting module non-functioncal but it's useful for
                        analysis-only purposes.
:``-E``, ``--external``: Set the linkage of global variables representing the
                         CPU state (aka CSVs) to `external` instead of
                         `internal`. This degrades performances of the produced
                         binary sensibly but makes debugging the generated code
                         easier.
:``-S``, ``--use-debug-symbols``: If they are available, ELF sections and
                                  function symbols are employed to identify
                                  executable code and function boundaries. This
                                  options is useful evaluate `revng` ignoring
                                  the code issue.
:``-b``, ``--bb-summary``: Output path for the CSV containing statistics about
                           the translated basic blocks. This option is
                           deprecated in favor of using `revng-dump` and will
                           be removed. Default: ``OUTFILE.bbsummary.csv``.
:``-f``, ``--functions-boundaries``: Enable function boundaries detection. This
                                    process currently can be quite expensive and
                                    it's therefore disabled by default.
