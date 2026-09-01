`revng-quick`
================

NAME
----

`revng quick` - One-shot commands operating directly on a binary.

SYNOPSIS
--------

    revng quick [OPTIONS] COMMAND BINARY [ARGS...]

DESCRIPTION
-----------

The `revng quick` command group is a subcommand of [`revng`](../index.md).

It runs analyses and produces artifacts directly from a `BINARY`, without creating a persistent project. Each invocation uses a throwaway temporary storage provider, so nothing is written to disk except the requested output. It is the convenient equivalent of running [`revng project init`](../revng-project/init.md) and then an analysis or producing an artifact, in a single command.

For repeated work on the same binary, prefer a persistent project (see [`revng-project`](../revng-project/index.md)) so intermediate results are cached and reused.

COMMANDS
--------

[`analyze`](analyze.md)
: Run analyses on a binary and dump the resulting model.

[`artifact`](artifact.md)
: Run analyses on a binary and produce an artifact.

OPTIONS
-------

`--pipeline PIPELINE`
: Path to the pipeline file. Defaults to the `PYPELINE_PIPELINE` environment variable if set, otherwise the pipeline shipped with the installation.

SEE ALSO
--------

[`revng`](../index.md), [`revng-quick-analyze`](analyze.md), [`revng-quick-artifact`](artifact.md), [`revng-project`](../revng-project/index.md), [`revng-pipeline`](../revng-pipeline/index.md)
