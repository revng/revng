`revng2-project`
================

NAME
----

`revng2 project` - Porcelain commands operating on a persistent project, i.e., a directory containing `revng.yml`.

SYNOPSIS
--------

    revng2 project [OPTIONS] COMMAND [ARGS...]

DESCRIPTION
-----------

The `revng2 project` command group is a subcommand of [`revng2`](revng2.md).

It operates on a *project*: a directory holding a `revng.yml` file. Unlike the one-shot [`revng2 quick`](revng2-quick.md) commands, the state produced by one invocation is preserved and reused by the next one.

A typical workflow is:

```{bash notest}
# Create the revng.yml file and run the initial auto analysis
revng2 project init /usr/bin/hostname

# Obtain an artifact
revng2 project artifact emit-c-as-single-file -o decompiled.c.yml
```

COMMANDS
--------

[`init`](revng2-project-init.md)
: Initialize a new project.

[`analyze`](revng2-project-analyze.md)
: Run an analysis or analysis list.

[`artifact`](revng2-project-artifact.md)
: Produce an artifact.

[`daemon`](revng2-project-daemon.md)
: Start the HTTP daemon serving the project.

OPTIONS
-------

These options are accepted by the `revng2 project` group itself and apply to every subcommand:

`--pipeline PIPELINE`
: Path to the pipeline file. Defaults to the `PYPELINE_PIPELINE` environment variable if set, otherwise the pipeline shipped with the installation.

`--storage-provider URL`
: URL of the storage provider to use, in the form `SCHEME://...`. Defaults to `local://`. Can also be set through the `REVNG_STORAGE_PROVIDER` environment variable, which takes precedence over the default but not over an explicitly passed `--storage-provider`.

The available schemes are:

- `local://` - store the project in the local directory. This is the default.
- `local://?inline` - like `local://`, but keep the cache database in a `.cache` directory inside the project (next to `revng.yml`) instead of under `--cache-dir`. The cache is then co-located with the project and moves with it.
- `temporary://` - like `local://`, but in a throwaway directory that is removed on exit. This is what [`revng2 quick`](revng2-quick.md) uses.
- `memory://` - keep the project in memory; nothing is written to disk.
- `null://` - discard all writes.
- `rss://HOST:PORT/?proto=http|https` - use a Remote Storage Server over HTTP(S).
- `daemon://HOST:PORT` - do not compute anything locally; instead delegate every artifact and analysis to a running daemon (started with [`revng2 project daemon`](revng2-project-daemon.md)) and exchange models and containers with it over HTTP. The daemon owns the project storage, so `init` is unavailable (the daemon manages its own project) and options that need local compute (`--debug`, `--invalidations`) are rejected.

`--cache-dir DIRECTORY`
: Directory to use for caching. Defaults to `$XDG_CACHE_HOME/revng` or `~/.cache/revng`.

SEE ALSO
--------

[`revng2`](revng2.md), [`revng2-project-init`](revng2-project-init.md), [`revng2-project-analyze`](revng2-project-analyze.md), [`revng2-project-artifact`](revng2-project-artifact.md), [`revng2-project-daemon`](revng2-project-daemon.md)
