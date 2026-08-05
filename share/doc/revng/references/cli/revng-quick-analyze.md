`revng-quick-analyze`
================

NAME
----

`revng quick analyze` - Run analyses on a binary and dump the model.

SYNOPSIS
--------

    revng quick analyze BINARY [OPTIONS] [-- PIPEBOX ARGS...]

DESCRIPTION
-----------

This command is a subcommand of [`revng quick`](revng-quick.md).

Runs one or more analyses on `BINARY`, in a throwaway temporary project, and prints out the resulting model.

By default the `initial-auto-analysis` [analysis list](../pipeline.md#analysis-lists) is run.

This is the one-shot equivalent of [`revng project init`](revng-project-init.md) followed by [`revng project analyze`](revng-project-analyze.md), but without persisting anything on disk.

OPTIONS
-------

`-o FILENAME`
: Path to write the resulting model to. Defaults to `-` (standard output).

`--analyses TEXT`
: Run the specified comma-separated list of [analyses](../analyses.md) instead of the default `initial-auto-analysis`.

`--debug RUNNER_CONTEXT`
: Run the command in debug mode, recording each pipeline step under the given directory. See [`revng-common`](revng-common.md#debug).

`--help-full`
: Show the help including the hidden per-pipe and per-analysis `--*-configuration` options, then exit.

This command supports the [developer wrapper options](revng-common.md#developer-wrapper-options) and [pipebox arguments](revng-common.md#pipebox-arguments). The group-level `--pipeline` option is documented in [`revng-quick`](revng-quick.md).

EXAMPLES
--------

Run the default initial auto analysis and dump the model:

```{bash notest}
revng quick analyze /usr/bin/hostname -o revng.yml
```

Run only the ABI detection analysis:

```{bash notest}
revng quick analyze /usr/bin/hostname --analyses detect-abi
```

SEE ALSO
--------

[`revng-quick`](revng-quick.md), [`revng-quick-artifact`](revng-quick-artifact.md), [`revng-project-analyze`](revng-project-analyze.md)
