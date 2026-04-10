`revng2-project-analyze`
================

NAME
----

`revng2 project analyze` - Run an analysis or analysis list.

SYNOPSIS
--------

    revng2 project analyze ANALYSIS [options]

DESCRIPTION
-----------

Runs the analysis `ANALYSIS`.

Run `revng2 project analyze --help` to list the available analyses.
The documentation describes [what an analysis is](../../user-manual/key-concepts/artifacts-and-analyses.md#analyses), the [available analyses and their options](../analyses.md).

OPTIONS
-------

<!-- TODO: use mkdocs-click or something similar to auto-generate the list command line options -->

`-C DIR`
: When running the command, make it as the command was run in the directory `DIR`

`-o PATH`
: Instead of dumping the model to standard output, save it to `PATH`.

`-c CONFIGURATION`
: Use the specified configuration string when running the analysis.

EXAMPLES
--------

A single command to run the initial auto analysis:

```{bash notest}
revng project analyze initial-auto-analysis
```

SEE ALSO
--------

[`revng2-project-artifact`](revng2-project-artifact.md)
