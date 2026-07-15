`revng2`
================

NAME
----

`revng2` - The rev.ng command line interface.

SYNOPSIS
--------

    revng2 [OPTIONS] COMMAND [ARGS...]

DESCRIPTION
-----------

`revng2` is the top-level entry point of the rev.ng CLI. It groups its functionality into subcommand groups, from high-level porcelain down to low-level plumbing.

COMMANDS
--------

[`project`](revng2-project.md)
: Porcelain commands operating on a persistent project, i.e., a directory holding a `revng.yml` and the input binaries.

[`quick`](revng2-quick.md)
: One-shot commands operating directly on a binary, without a persistent project.

[`pipeline`](revng2-pipeline.md)
: Low-level, plumbing commands to run a single pipe or analysis in isolation.

OPTIONS
-------

These options are accepted by `revng2` itself and, being group options, must appear before the subcommand (for example `revng2 -C DIR project artifact ...`).

`-C, --directory DIRECTORY`
: Run the command as if it had been started in `DIRECTORY`. Defaults to the current directory.

`--pipebox PIPEBOX`
: Path to the pipebox file, the Python module that loads the pipeline's pipes and analyses. Defaults to the `PYPELINE_PIPEBOX` environment variable if set, otherwise the pipebox shipped with rev.ng. This usually does not need to be changed.

`--verbose`
: Enable debug logging for the pypeline related code.

SEE ALSO
--------

[`revng2-project`](revng2-project.md), [`revng2-quick`](revng2-quick.md), [`revng2-pipeline`](revng2-pipeline.md)
