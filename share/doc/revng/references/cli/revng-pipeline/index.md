`revng-pipeline`
================

NAME
----

`revng pipeline` - Low-level (plumbing) commands to run a single pipe or analysis.

These commands enable you to run individual pipes or analyses on plain containers, without having configured a pipeline (`--pipeline` in [`revng-project`](../revng-project/index.md)).

SYNOPSIS
--------

    revng pipeline COMMAND NAME [OPTIONS] [ARGS...] [-- PIPEBOX ARGS...]

DESCRIPTION
-----------

The `revng pipeline` command group is a subcommand of [`revng`](../index.md).

It runs a single pipe or a single analysis in isolation, reading and writing container files on disk.

These are plumbing commands: they operate on raw containers and models rather than on a project, and are meant for debugging and for building higher-level tooling.
Most users should use [`revng project`](../revng-project/index.md) or [`revng quick`](../revng-quick/index.md) instead.

Each command comes in two flavours:

- the default (Python-driven) variant, which works for any registered pipe or analysis and takes explicit input and output file arguments;
- a `-native` variant, which runs only the pipes and analyses that have a native (C++) implementation by executing the corresponding runner in `libexec/revng/` directly, without going through Python.

The `-native` variant is a thin wrapper: it forwards the `--*-objects` selectors and everything after `--` to the native runner. Use it to run a pipe or analysis with minimal overhead, or under a debugger or profiler (`--gdb`, `--valgrind`, `--perf`, ...) without the Python interpreter in the way. The two flavours expose the same set of names; only pipes and analyses with a native implementation appear under the `-native` variant. Each flavour is documented together with its default counterpart.

COMMANDS
--------

[`run-pipe`](run-pipe.md) (and `run-pipe-native`)
: Run a single pipe.

[`run-analysis`](run-analysis.md) (and `run-analysis-native`)
: Run a single analysis.

SEE ALSO
--------

[`revng`](../index.md), [`revng-pipeline-run-pipe`](run-pipe.md), [`revng-pipeline-run-analysis`](run-analysis.md), [`revng-project`](../revng-project/index.md), [`revng-quick`](../revng-quick/index.md)
