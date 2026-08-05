`revng-project-daemon`
================

NAME
----

`revng project daemon` - Start the HTTP daemon.

SYNOPSIS
--------

    revng project daemon [OPTIONS] [-- PIPEBOX ARGS...]

DESCRIPTION
-----------

This command is a subcommand of [`revng project`](revng-project.md).

Starts an HTTP daemon serving the project in the current directory. The daemon exposes the analyses and artifacts of the project over HTTP, and is what the rev.ng UI and the Python client (`DaemonProject`) talk to.

The project must already contain a `revng.yml`: create it with [`revng project init`](revng-project-init.md) before starting the daemon, otherwise the API returns an error.

By default the daemon binds to `127.0.0.1:8000`.

OPTIONS
-------

`-b`, `--bind TEXT`
: TCP host/address to bind to. Should be `host:port`, `host`, `unix:path` or `fd://num` (for example `127.0.0.1:5000`, `127.0.0.1`, `unix:/tmp/socket` or `fd://33`).

`--production`
: Enable production settings.

This command supports [pipebox arguments](revng-common.md#pipebox-arguments).

EXAMPLES
--------

Serve a project on the default address `127.0.0.1:8000`:

```{bash notest}
revng project init /usr/bin/hostname
revng project daemon
```

Serve a project reachable from other hosts on port 9000:

```{bash notest}
revng project daemon --bind 0.0.0.0:9000
```

SEE ALSO
--------

[`revng-project`](revng-project.md), [`revng-project-init`](revng-project-init.md), [`revng-project-artifact`](revng-project-artifact.md)
