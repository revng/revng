`revng2-project-artifact`
================

NAME
----

`revng2 project artifact` - Produce an artifact.

SYNOPSIS
--------

    revng2 project artifact ARTIFACT [OPTIONS] [OBJECT [OBJECT [...]]]

DESCRIPTION
-----------

Produces the artifact `ARTIFACT`.

If no `OBJECT` is specified then the command will produce all available objects.

Run `revng2 project artifact --help` to list the available artifacts.
The documentation describes [what an artifact is](../../user-manual/key-concepts/artifacts-and-analyses.md#artifacts) and the [available artifacts](../artifacts.md).

OPTIONS
-------

<!-- TODO: use mkdocs-click or something similar to auto-generate the list command line options -->

`-C DIR`
: When running the command, make it as the command was run in the directory `DIR`

`--list`
: List the available objects for this artifact and the quit.

`-o PATH`
: Store the artifact to `PATH`.
  By default, the artifact is written to standard output.

`--format [tar|yaml]` `--tar` `--yaml`
: Select the output format. By default the artifact is emitted as a YAML dictionary `object -> data`, where `data` is text data if the artifact's MIME is textual or a Base64 encoding if it's binary. Tar output emits a plain tar (without compression) without any transformation to the data.

EXAMPLES
--------

A single command to produce the decompiled code saving the result to `decompiled.c`:

```{bash notest}
revng2 project artifact emit-c-as-single-file -o decompiled.c.yml
```

SEE ALSO
--------

[`revng2-project-analyze`](revng2-project-analyze.md)
