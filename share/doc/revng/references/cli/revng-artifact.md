`revng-artifact`
================

NAME
----

`revng-artifact` - Produce an artifact.

SYNOPSIS
--------

    revng artifact [OPTIONS] ARTIFACT BINARY [TARGET [TARGET [...]]]

DESCRIPTION
-----------

Produces the artifact `ARTIFACT` from `BINARY`.

Run `revng artifact` without arguments to list the available artifacts.
The documentation describes [what an artifact is](../../user-manual/key-concepts/artifacts-and-analyses.md#artifacts) and the [available artifacts](../artifacts.md).

OPTIONS
-------

`-o PATH`
: Store the artifact to `PATH`.
  By default, the artifact is written to standard output.

`--analyze`
: Before producing the artifact, run the [`revng-initial-auto-analysis` analyses list](../pipeline.md#analysis-lists).

`--analyses [ANALYSIS[,ANALYSIS[...]]]`
: Comma-separate list of [analyses](../analyses.md) to run before producing the artifact.

Other important options are documented in [`revng-common`](revng-common.md).

EXAMPLES
--------

A single command to produce the decompiled code saving the result to `decompiled.c`:

```
revng artifact --analyze decompile-to-single-file /usr/bin/hostname -o decompiled.c
```

An equivalent command using `--analyses`:

```
revng artifact --analyses=revng-initial-auto-analysis decompile-to-single-file /usr/bin/hostname
```

An equivalent set of commands using [`revng-analyze`](revng-analyze.md) and `--resume`:

```{bash notest}
revng analyze --resume project-dir/ revng-initial-auto-analysis /usr/bin/hostname
revng artifact --resume project-dir/ decompile-to-single-file /usr/bin/hostname
```

Decompile a program using `mymodel.yml` as the model:

```{bash notest}
revng artifact --model mymodel.yml decompile-to-single-file /usr/bin/hostname
```

SEE ALSO
--------

[`revng-common`](revng-common.md), [`revng-analyze`](revng-analyze.md)
