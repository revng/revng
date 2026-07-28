`revng2-project-daemon`
================

NAME
----

`revng2 project daemon` - Start the HTTP daemon.

SYNOPSIS
--------

    revng2 project daemon [OPTIONS] [-- PIPEBOX ARGS...]

DESCRIPTION
-----------

This command is a subcommand of [`revng2 project`](revng2-project.md).

Starts an HTTP daemon serving the project in the current directory. The daemon exposes the analyses and artifacts of the project over HTTP, and is what the rev.ng UI and the Python client (`DaemonProject`) talk to.

The project must already contain a `revng.yml`: create it with [`revng2 project init`](revng2-project-init.md) before starting the daemon, otherwise the API returns an error.

By default the daemon binds to `127.0.0.1:8000`.

OPTIONS
-------

`-b`, `--bind TEXT`
: TCP host/address to bind to. Should be `host:port`, `host`, `unix:path` or `fd://num` (for example `127.0.0.1:5000`, `127.0.0.1`, `unix:/tmp/socket` or `fd://33`).

`--production`
: Enable production settings.

This command supports [pipebox arguments](revng2-common.md#pipebox-arguments).

EXAMPLES
--------

Serve a project on the default address `127.0.0.1:8000`:

```{bash notest}
revng2 project init /usr/bin/hostname
revng2 project daemon
```

Serve a project reachable from other hosts on port 9000:

```{bash notest}
revng2 project daemon --bind 0.0.0.0:9000
```

SEE ALSO
--------

[`revng2-project`](revng2-project.md), [`revng2-project-init`](revng2-project-init.md), [`revng2-project-artifact`](revng2-project-artifact.md)
