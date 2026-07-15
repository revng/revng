`revng2-project-init`
================

NAME
----

`revng2 project init` - Initialize a new project.
Given a binary, it runs the initial auto analyses to start the reverse engineering process.

SYNOPSIS
--------

    revng2 project init [OPTIONS] [BINARY] [-- PIPEBOX ARGS...]

DESCRIPTION
-----------

This command is a subcommand of [`revng2 project`](revng2-project.md).

Creates a new project in the current directory by creating and initializing a `revng.yml`.

If `BINARY` is given, it is imported into the project and, unless `--no-initial-auto-analysis` is passed, the `initial-auto-analysis` [analysis list](../pipeline.md#analysis-lists) is run on it. These automatically import the binary, identify functions, prototypes, data structures and so on.

If a `revng.yml` is already present, the command refuses to overwrite it and exits with an error, unless `--overwrite` is passed.

OPTIONS
-------

`--no-initial-auto-analysis`
: Do not run the `initial-auto-analysis` analysis list after importing `BINARY`.

`--overwrite`
: Overwrite an existing project model instead of failing. Any `revng.yml` already present in the target directory is clobbered.

This command supports the [developer wrapper options](revng2-common.md#developer-wrapper-options), [pipebox arguments](revng2-common.md#pipebox-arguments) and the [project id and token](revng2-common.md#project-id-and-token) options.

EXAMPLES
--------

Create a project from a binary and run the initial auto analysis:

```{bash notest}
revng2 project init /usr/bin/hostname
```

Create an empty project and import the binary without running any analysis:

```{bash notest}
revng2 project init /usr/bin/hostname --no-initial-auto-analysis
```

SEE ALSO
--------

[`revng2-project`](revng2-project.md), [`revng2-project-analyze`](revng2-project-analyze.md), [`revng2-project-artifact`](revng2-project-artifact.md)
