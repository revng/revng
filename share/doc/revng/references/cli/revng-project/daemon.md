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

This command is a subcommand of [`revng project`](index.md).

Starts an HTTP daemon serving the project in the current directory. The daemon exposes the analyses and artifacts of the project over HTTP, and is what the rev.ng UI and the Python client (`DaemonProject`) talk to.

The project must already contain a `revng.yml`: create it with [`revng project init`](init.md) before starting the daemon, otherwise the API returns an error.

By default the daemon binds to a unix socket named `revng.sock` in the project's directory.

OPTIONS
-------

`-b`, `--bind TEXT`
: TCP host/address to bind to. Should be `host:port`, `host`, `unix:<path>` or `fd://<num>` (for example `127.0.0.1:5000`, `127.0.0.1`, `unix:/tmp/socket.sock` and `fd://33`).

`--production`
: Enable production settings.

This command supports [pipebox arguments](../revng-common.md#pipebox-arguments).

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

Serve a project via IPv4 and interact with it with the CLI:

```{bash notest}
# On server machine
revng project daemon --bind 172.16.0.1:8000
# On user machine
export REVNG_STORAGE='http://172.16.0.1:8000/'
revng project artifact emit-c
```

Serve a project via unix socket and interact with it:

```{bash notest}
revng project daemon --bind unix:/tmp/revng.sock
export REVNG_STORAGE='http://!unix/tmp/revng.sock'
revng project artifact emit-c
```

SEE ALSO
--------

[`revng-project`](index.md), [`revng-project-init`](init.md), [`revng-project-artifact`](artifact.md)
