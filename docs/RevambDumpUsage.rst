***********
revng-dump
***********

----------------------------------------
extract information from `revng` output
----------------------------------------

:Author: Alessandro Di Federico <ale+revng@clearmind.me>
:Date:   2016-12-22
:Copyright: MIT
:Version: 0.1
:Manual section: 1
:Manual group: rev.ng

SYNOPSIS
========

    revng-dump [options] INFILE

DESCRIPTION
===========

`revng-dump` is a simple tool to extract some high level information from the
IR produced by `revng`.

OPTIONS
=======

Note that all the options specifying a path support the special value ``-``
which indicates ``stdout``. Note also that `revng-dump` expresses the *name of
a basic block* as represented by `revng` in the generated module (typically
``bb.0xaddress`` or ``bb.symbol.0xoffset``.

:``-c``, ``--cfg``: Path where the control-flow graph should be stored. The
                    output will be a CSV file with two columns: `source` and
                    `destination`. Both of them contain the name of a basic
                    block.
:``-n``, ``--noreturn``: Path where the list of the ``noreturn`` basic block
                         should be stored. The output will be a CSV file with a
                         single column `noreturn`, containing the name of the
                         ``noreturn`` basic block.
:``-f``, ``--functions-boundaries``: Path where the list of *function*<->*basic
                                    block* pairs should be stored. The output
                                    will be a CSV file with two column:
                                    `function`, the name of the entry basic
                                    block of the function, and `basicblock`, the
                                    name of a basic block belonging to
                                    `function`.
:``-i``, ``--function-isolation``: Path where to store the LLVM module that is
                                   the result of the function isolation pass on
                                   the input module.
