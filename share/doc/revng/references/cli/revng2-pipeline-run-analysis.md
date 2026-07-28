`revng2-pipeline-run-analysis`
================

NAME
----

`revng2 pipeline run-analysis` - Run a single analysis.

SYNOPSIS
--------

    revng2 pipeline run-analysis ANALYSIS MODEL [CONTAINER...] [OPTIONS] [-- PIPEBOX ARGS...]
    revng2 pipeline run-analysis-native ANALYSIS [OPTIONS] -- NATIVE RUNNER ARGS...

DESCRIPTION
-----------

This command is a subcommand of [`revng2 pipeline`](revng2-pipeline.md).

Runs the single analysis `ANALYSIS` in isolation, reading container files on disk and writing out the modified model.
Run without arguments to list the [available analyses](../analyses.md).

This is a plumbing command meant for debugging and tooling; most users should use [`revng2 project`](revng2-project.md) or [`revng2 quick`](revng2-quick.md) instead.

`MODEL` is the path to the model file, followed by one input file argument per container in the analysis' signature. Run `revng2 pipeline run-analysis --help` to list the available analyses, and `revng2 pipeline run-analysis ANALYSIS --help` to see its exact arguments.

OPTIONS
-------

`-o FILENAME`
: Path to write the resulting model to. Defaults to `-` (standard output).

`-c, --configuration TEXT`
: Configuration for the analysis, as a YAML document whose schema is analysis-specific.

`--list` (Python only)
: List the available objects for each argument and then exit.

`--<CONTAINER>-objects /id1,/id2,...`
: For each container, restrict the objects to use to the given comma-separated list of ids. If not passed, all objects are used.

NATIVE VARIANT
--------------

`revng2 pipeline run-analysis-native ANALYSIS` runs the same analysis through its native (C++) implementation, executing `libexec/revng/pypeline-run-analysis` directly without going through Python. Only analyses that have a native implementation are available.

It takes no explicit file arguments: pass the native runner's own arguments after `--`. Use it for minimal-overhead runs, or under a debugger or profiler.

The python variant supports the [developer wrapper options](revng2-common.md#developer-wrapper-options) and [pipebox arguments](revng2-common.md#pipebox-arguments).

EXAMPLES
--------

Run the `detect-abi` analysis on a model and container, saving the updated model:

```{bash notest}
revng2 pipeline run-analysis detect-abi revng.yml llvm-root.tar -o revng.yml
```

SEE ALSO
--------

[`revng2-pipeline`](revng2-pipeline.md), [`revng2-pipeline-run-pipe`](revng2-pipeline-run-pipe.md), [`revng2-project-analyze`](revng2-project-analyze.md)
