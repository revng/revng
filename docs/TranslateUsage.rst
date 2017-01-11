*********
translate
*********

----------------------------------------------------------------
translate a program from an architecture to another using revamb
----------------------------------------------------------------

:Author: Alessandro Di Federico <ale+revng@clearmind.me>
:Date:   2016-12-22
:Copyright: MIT
:Version: 0.1
:Manual section: 1
:Manual group: rev.ng

SYNOPSIS
========

    revamb-dump [options] [--] INFILE [revamb options]

DESCRIPTION
===========

`translate` is a simple wrapper script which using `revamb` to lift an input
program to LLVM IR and then compiling it to x86-64.

In practice, `translate` first invokes `revamb` then, depending on the options,
some optimizations are performed using `llc` and or `opt`, and finally the
generated object file is linked against the require libraries and `support.c`.

Options after `INFILE` are forwarded as is to `revamb`.

OPTIONS
=======

:``-O0``: Disable optimizations both in the mid-end (`opt`) and the backend
          (`llc`). This is default option.
:``-O1``: Enable backend optimizations (`llc`).
:``-O2``: Enable optimizations both in the mid-end (`opt`) and the backend
          (`llc`).
:``-s``: Skip invoking `revamb`, assumes a file named `INFILE.ll` already
         exists. This is useful for optimizing previously generated code.
