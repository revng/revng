`revng2-quick`
================

NAME
----

`revng2 quick` - One-shot commands operating directly on a binary.

SYNOPSIS
--------

    revng2 quick [OPTIONS] COMMAND BINARY [ARGS...]

DESCRIPTION
-----------

The `revng2 quick` command group is a subcommand of [`revng2`](revng2.md).

It runs analyses and produces artifacts directly from a `BINARY`, without creating a persistent project. Each invocation uses a throwaway temporary storage provider, so nothing is written to disk except the requested output. It is the convenient equivalent of running [`revng2 project init`](revng2-project-init.md) and then an analysis or producing an artifact, in a single command.

For repeated work on the same binary, prefer a persistent project (see [`revng2-project`](revng2-project.md)) so intermediate results are cached and reused.

COMMANDS
--------

[`analyze`](revng2-quick-analyze.md)
: Run analyses on a binary and dump the resulting model.

[`artifact`](revng2-quick-artifact.md)
: Run analyses on a binary and produce an artifact.

OPTIONS
-------

`--pipeline PIPELINE`
: Path to the pipeline file. Defaults to the `PYPELINE_PIPELINE` environment variable if set, otherwise the pipeline shipped with the installation.

SEE ALSO
--------

[`revng2`](revng2.md), [`revng2-quick-analyze`](revng2-quick-analyze.md), [`revng2-quick-artifact`](revng2-quick-artifact.md), [`revng2-project`](revng2-project.md), [`revng2-pipeline`](revng2-pipeline.md)
