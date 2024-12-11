`revng-analyze`
================

NAME
----

`revng-analyze` - Run an analysis.

SYNOPSIS
--------

    revng-analyze [options] ANALYSIS BINARY

DESCRIPTION
-----------

Runs the analysis `ANALYSIS` on `BINARY`.

Run `revng analyze` without arguments to list the available analyses.
The documentation describes [what an analysis is](../../user-manual/key-concepts/artifacts-and-analyses.md#analyses), the [available analyses and their options](../analyses.md).

OPTIONS
-------

`-o PATH`
: Instead of dumping the model to standard output, save it to `PATH`.

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

[`revng-common`](revng-common.md), [`revng-artifact`](revng-artifact.md)
