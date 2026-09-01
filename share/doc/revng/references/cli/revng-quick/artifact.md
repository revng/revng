`revng-quick-artifact`
================

NAME
----

`revng quick artifact` - Run analyses on a binary and produce an artifact.

SYNOPSIS
--------

    revng quick artifact ARTIFACT BINARY [OPTIONS] [-- PIPEBOX ARGS...]

DESCRIPTION
-----------

This command is a subcommand of [`revng quick`](index.md).

Runs the `initial-auto-analysis` [analysis list](../../pipeline.md#analysis-lists) (or the analyses selected with `--analyses`) on `BINARY`, then produces the artifact `ARTIFACT`, in a throwaway temporary project.

`ARTIFACT` is a subcommand: run `revng quick artifact --help` to list the available artifacts.
The documentation describes [what an artifact is](../../../user-manual/key-concepts/artifacts-and-analyses.md#artifacts) and the [available artifacts](../../artifacts.md).

This is the one-shot equivalent of [`revng project init`](../revng-project/init.md) followed by [`revng project artifact`](../revng-project/artifact.md), but without persisting anything on disk.

OPTIONS
-------

`-o FILE`
: Path to write the computed artifact to. If not specified, the result is printed to standard output.

`--analyses TEXT`
: Run the specified comma-separated list of analyses instead of the default `initial-auto-analysis`.

`--format [auto|tar|yaml]`
: Select the output format. Defaults to `auto`, which emits a lone object raw (with no wrapping) and otherwise behaves like `yaml`. With `yaml` the artifact is emitted as a YAML dictionary `object -> data`, where `data` is text data if the artifact's MIME is textual or a Base64 encoding if it is binary. `--tar` emits a plain (uncompressed) tar without any transformation of the data.

`--yaml`
: Shortcut for `--format yaml`.

`--tar`
: Shortcut for `--format tar`.

`--debug RUNNER_CONTEXT`
: Run the command in debug mode, recording each pipeline step under the given directory. See [`revng-common`](../revng-common.md#debug).

`--help-full`
: Show the help including the hidden artifacts and per-pipe `--*-configuration` options, then exit.

This command supports the [developer wrapper options](../revng-common.md#developer-wrapper-options) and [pipebox arguments](../revng-common.md#pipebox-arguments). The group-level `--pipeline` option is documented in [`revng-quick`](index.md).

EXAMPLES
--------

Decompile a binary in a single command, saving the result to `decompiled.c.yml`:

```{bash notest}
revng quick artifact emit-c-as-single-file /usr/bin/hostname -o decompiled.c.yml
```

SEE ALSO
--------

[`revng-quick`](index.md), [`revng-quick-analyze`](analyze.md), [`revng-project-artifact`](../revng-project/artifact.md)
