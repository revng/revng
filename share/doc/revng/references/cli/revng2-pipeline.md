`revng2-pipeline`
================

NAME
----

`revng2 pipeline` - Low-level (plumbing) commands to run a single pipe or analysis.

These commands enable you to run individual pipes or analyses on plain containers, without having configured a pipeline (`--pipeline` in [`revng2-project`](revng2-project.md)).

SYNOPSIS
--------

    revng2 pipeline COMMAND NAME [OPTIONS] [ARGS...] [-- PIPEBOX ARGS...]

DESCRIPTION
-----------

The `revng2 pipeline` command group is a subcommand of [`revng2`](revng2.md).

It runs a single pipe or a single analysis in isolation, reading and writing container files on disk.

These are plumbing commands: they operate on raw containers and models rather than on a project, and are meant for debugging and for building higher-level tooling.
Most users should use [`revng2 project`](revng2-project.md) or [`revng2 quick`](revng2-quick.md) instead.

Each command comes in two flavours:

- the default (Python-driven) variant, which works for any registered pipe or analysis and takes explicit input and output file arguments;
- a `-native` variant, which runs only the pipes and analyses that have a native (C++) implementation by executing the corresponding runner in `libexec/revng/` directly, without going through Python.

The `-native` variant is a thin wrapper: it forwards the `--*-objects` selectors and everything after `--` to the native runner. Use it to run a pipe or analysis with minimal overhead, or under a debugger or profiler (`--gdb`, `--valgrind`, `--perf`, ...) without the Python interpreter in the way. The two flavours expose the same set of names; only pipes and analyses with a native implementation appear under the `-native` variant. Each flavour is documented together with its default counterpart.

COMMANDS
--------

[`run-pipe`](revng2-pipeline-run-pipe.md) (and `run-pipe-native`)
: Run a single pipe.

[`run-analysis`](revng2-pipeline-run-analysis.md) (and `run-analysis-native`)
: Run a single analysis.

SEE ALSO
--------

[`revng2`](revng2.md), [`revng2-pipeline-run-pipe`](revng2-pipeline-run-pipe.md), [`revng2-pipeline-run-analysis`](revng2-pipeline-run-analysis.md), [`revng2-project`](revng2-project.md), [`revng2-quick`](revng2-quick.md)
