`revng-project`
================

NAME
----

`revng project` - Porcelain commands operating on a persistent project, i.e., a directory containing `revng.yml`.

SYNOPSIS
--------

    revng project [OPTIONS] COMMAND [ARGS...]

DESCRIPTION
-----------

The `revng project` command group is a subcommand of [`revng`](revng.md).

It operates on a *project*: a directory holding a `revng.yml` file. Unlike the one-shot [`revng quick`](revng-quick.md) commands, the state produced by one invocation is preserved and reused by the next one.

A typical workflow is:

```{bash notest}
# Create the revng.yml file and run the initial auto analysis
revng project init /usr/bin/hostname

# Obtain an artifact
revng project artifact emit-c-as-single-file -o decompiled.c.yml
```

COMMANDS
--------

[`init`](revng-project-init.md)
: Initialize a new project.

[`analyze`](revng-project-analyze.md)
: Run an analysis or analysis list.

[`artifact`](revng-project-artifact.md)
: Produce an artifact.

[`daemon`](revng-project-daemon.md)
: Start the HTTP daemon serving the project.

OPTIONS
-------

These options are accepted by the `revng project` group itself and apply to every subcommand:

`--pipeline PIPELINE`
: Path to the pipeline file. Defaults to the `PYPELINE_PIPELINE` environment variable if set, otherwise the pipeline shipped with the installation.

`--storage-provider URL`
: URL of the storage provider to use, in the form `SCHEME://...`. Defaults to `local://`. Can also be set through the `REVNG_STORAGE_PROVIDER` environment variable, which takes precedence over the default but not over an explicitly passed `--storage-provider`.

The available schemes are:

- `local://` - store the project in the local directory. This is the default.
- `local://?inline` - like `local://`, but keep the cache database in a `.cache` directory inside the project (next to `revng.yml`) instead of under `--cache-dir`. The cache is then co-located with the project and moves with it.
- `temporary://` - like `local://`, but in a throwaway directory that is removed on exit. This is what [`revng quick`](revng-quick.md) uses.
- `memory://` - keep the project in memory; nothing is written to disk.
- `null://` - discard all writes.
- `rss://HOST:PORT/?proto=http|https` - use a Remote Storage Server over HTTP(S).
- `daemon://HOST:PORT` - do not compute anything locally; instead delegate every artifact and analysis to a running daemon (started with [`revng project daemon`](revng-project-daemon.md)) and exchange models and containers with it over HTTP. The daemon owns the project storage, so `init` is unavailable (the daemon manages its own project) and options that need local compute (`--debug`, `--invalidations`) are rejected.

`--cache-dir DIRECTORY`
: Directory to use for caching. Defaults to `$XDG_CACHE_HOME/revng` or `~/.cache/revng`.

SEE ALSO
--------

[`revng`](revng.md), [`revng-project-init`](revng-project-init.md), [`revng-project-analyze`](revng-project-analyze.md), [`revng-project-artifact`](revng-project-artifact.md), [`revng-project-daemon`](revng-project-daemon.md)
