*********
translate
*********

----------------------------------------------------------------
translate a program from an architecture to another using revng
----------------------------------------------------------------

:Author: Alessandro Di Federico <ale+revng@clearmind.me>
:Date:   2016-12-22
:Copyright: MIT
:Version: 0.1
:Manual section: 1
:Manual group: rev.ng

SYNOPSIS
========

    revng-dump [options] [--] INFILE [revng options]

DESCRIPTION
===========

`translate` is a simple wrapper script which using `revng` to lift an input
program to LLVM IR and then compiling it to x86-64.

In practice, `translate` first invokes `revng` then, depending on the options,
some optimizations are performed using `llc` and or `opt`, and finally the
generated object file is linked against the require libraries and `support.c`.

Options after `INFILE` are forwarded as is to `revng`.

OPTIONS
=======

:``-O0``: Disable optimizations both in the mid-end (`opt`) and the backend
          (`llc`). This is default option.
:``-O1``: Enable backend optimizations (`llc`).
:``-O2``: Enable optimizations both in the mid-end (`opt`) and the backend
          (`llc`).
:``-s``: Skip invoking `revng`, assumes a file named `INFILE.ll` already
         exists. This is useful for optimizing previously generated code.
:``-trace``: Enable tracing support: if the `REVAMB_TRACE_PATH` environment
             variable is set at run-time, the translated program will log all
             the executed program counters into the file specified by the
             environment variable. This effect it obtained by linking the
             translated program against the `support-$ARCH-trace.ll` module
             instead of the `support-$ARCH-normal.ll`. Enabling this option
             introduces a non-negligible slow down in the output program, even
             if `REVAMB_TRACE_PATH` is not specified at run-time.
:``-i``: Optionally apply the function isolation pass before re-compiling the
         program.
