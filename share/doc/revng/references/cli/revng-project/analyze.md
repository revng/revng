`revng-project-analyze`
================

NAME
----

`revng project analyze` - Run an analysis or an analysis list.

SYNOPSIS
--------

    revng project analyze ANALYSIS [OPTIONS] [-- PIPEBOX ARGS...]

DESCRIPTION
-----------

This command is a subcommand of [`revng project`](index.md).

Runs the [analysis](../../analyses.md) (or [analysis list](../../pipeline.md#analysis-lists)) `ANALYSIS` on the project in the current directory, updating its model in place.

Run without arguments to list all the available analyses.

The documentation describes [what an analysis is](../../../user-manual/key-concepts/artifacts-and-analyses.md#analyses).

The project must have been created with [`revng project init`](init.md) first, so that a `revng.yml` is present in the working directory.

OPTIONS
-------

`-o FILENAME`
: Path to write the resulting model to. If not specified, the model is printed to standard output.

`-c, --configuration PATH`
: Path to a file holding the configuration for the analysis, a YAML document whose schema is analysis-specific. Available only for a single analysis, not for an analysis list.

`--invalidations FILENAME`
: Write invalidation data to the specified file. Its format is described in [Invalidation data](#invalidation-data) below.

`--list`
: List the available objects for each argument and then exit.

`--<ARG>-objects LIST`
: For the incoming container argument `ARG`, the comma-separated list of objects to feed the analysis. Objects use the same syntax accepted by [`revng project artifact`](artifact.md#description) (a location, a bare key, a name, or an automatic name). There is one such option per container argument of the analysis; run with `--help` to see them, or `--list` to see the available objects. If omitted, all the available objects are used.

`--debug RUNNER_CONTEXT`
: Run the command in debug mode, recording each pipeline step under the given directory. See [`revng-common`](../revng-common.md#debug).

`--help-full`
: Show the help including the hidden per-pipe `--*-configuration` options, then exit.

This command supports the [developer wrapper options](../revng-common.md#developer-wrapper-options) and [pipebox arguments](../revng-common.md#pipebox-arguments).

INVALIDATION DATA
-----------------

When `--invalidations FILENAME` is given, a YAML document is written in `FILENAME`, listing the cached objects that the analysis invalidated, i.e., the objects whose previously produced artifacts are now stale and will be recomputed the next time they are requested.

It is a list of entries, one per affected container, each with:

- `savepoint` - name of the savepoint the container belongs to.
- `container` - name of the container within that savepoint.
- `configuration` - hash identifying the pipeline configuration of that container (the value below is the SHA-256 of the empty configuration).
- `objects` - the invalidated object ids, in the usual object-location form.

For example, renaming the function at `0x401000` invalidates it in every savepoint that depends on it:

```yaml
- configuration: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
  container: cfg-map
  objects:
  - /function/0x401000:Code_x86_64
  - /function/0x401c4f:Code_x86_64
  savepoint: cfg-computed
- configuration: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
  container: assembly
  objects:
  - /function/0x401000:Code_x86_64
  - /function/0x401c4f:Code_x86_64
  savepoint: disassemble
```

If the analysis invalidated nothing, the file contains an empty list (`[]`).

EXAMPLES
--------

Apply a patch (a model diff) to the current project, for example renaming a function:

```{bash notest}
revng project analyze apply-diff -c patch.yml
```

where `patch.yml` contains the diff:

```yaml
Changes:
- Path: /Functions/0x401000:Code_x86_64/Name
  Remove: ''
  Add: example_renamed_function
```

Run a single analysis and save the resulting model to a file:

```{bash notest}
revng project analyze detect-abi -o revng.yml
```

SEE ALSO
--------

[`revng-project`](index.md), [`revng-project-artifact`](artifact.md), [`revng-project-init`](init.md)
