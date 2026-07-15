`revng2-pipeline-run-pipe`
================

NAME
----

`revng2 pipeline run-pipe` - Run a single pipe.

SYNOPSIS
--------

    revng2 pipeline run-pipe PIPE MODEL [READ_INPUT...] [WRITE_OUTPUT...] [OPTIONS] [-- PIPEBOX ARGS...]
    revng2 pipeline run-pipe-native PIPE [OPTIONS] -- NATIVE RUNNER ARGS...

DESCRIPTION
-----------

This command is a subcommand of [`revng2 pipeline`](revng2-pipeline.md).

Runs the single pipe `PIPE` in isolation, reading and writing container files on disk.
Without arguments, it report the list of known pipes.

This is a plumbing command meant for debugging and tooling; most users should use [`revng2 project`](revng2-project.md) or [`revng2 quick`](revng2-quick.md) instead.

`MODEL` is the path to the model file; then, following the pipe's signature, each read container takes an input file argument and each write container takes an output file argument. Run `revng2 pipeline run-pipe --help` to list the available pipes, and `revng2 pipeline run-pipe PIPE --help` to see its exact arguments.

OPTIONS
-------

`--file-storage DIRECTORY`
: Directory holding the input files referenced by the model (for example the original binary), looked up by hash. Defaults to the base directory.

`--dependencies FILE`
: Output dependency data as a tar file, containing a `dependencies.yml` for plain dependencies plus one file per advanced invalidation entry.

`-s, --static-configuration TEXT`
: Static configuration for the pipe, as a YAML document whose schema is pipe-specific.

`-c, --configuration TEXT`
: Configuration for the pipe, as a YAML document whose schema is pipe-specific.

`--format [auto|tar|yaml]` (Python only)
: Select the output format. Defaults to `yaml`, emitting the output container as a YAML dictionary `object -> data`, where `data` is text data if the container's MIME is textual or a Base64 encoding if it is binary. `--tar` emits a plain (uncompressed) tar without any transformation of the data. `auto` emits a lone object raw and otherwise behaves like `yaml`.

`--yaml`
: Shortcut for `--format yaml`.

`--tar`
: Shortcut for `--format tar`.

`--list` (Python only)
: List the available objects for each argument and then exit.

`--<CONTAINER>-objects /id1,/id2,...`
: For each writable container, restrict the objects to produce to the given comma-separated list of ids. If not passed, all objects are requested.

NATIVE VARIANT
--------------

`revng2 pipeline run-pipe-native PIPE` runs the same pipe through its native (C++) implementation, executing `libexec/revng/pypeline-run-pipe` directly without going through Python. Only pipes that have a native implementation are available.

It takes no explicit file arguments: pass the native runner's own arguments after `--`. Use it for minimal-overhead runs, or under a debugger or profiler.

The Python variant supports the [developer wrapper options](revng2-common.md#developer-wrapper-options) and [pipebox arguments](revng2-common.md#pipebox-arguments).

EXAMPLES
--------

Run the `lift` pipe on a model, reading the input binaries container and writing the LLVM root container:

```{bash notest}
revng2 pipeline run-pipe lift revng.yml input.tar output.tar
```

SEE ALSO
--------

[`revng2-pipeline`](revng2-pipeline.md), [`revng2-pipeline-run-analysis`](revng2-pipeline-run-analysis.md), [`revng2-project`](revng2-project.md)
