`revng-project-agent`
================

NAME
----

`revng project agent` - Run a coding agent against the project's daemon.

SYNOPSIS
--------

    revng project agent [OPTIONS]

DESCRIPTION
-----------

This command is a subcommand of [`revng project`](index.md).

Starts a coding agent on the project in the current directory, so that it can reverse engineer the binary using rev.ng.

Unless `--agent` is given, the agent to run is detected among the supported ones installed on the system. If more than one is found, the command asks which one to use. The choice is remembered and reused in successive invocations.

The agent is provided with instructions on how to interact with rev.ng.

Before starting the agent, the command ensures a [`revng project daemon`](daemon.md) is serving the project: if one is already listening it is reused, otherwise a new one is started and then stopped once the agent exits. This ensures that the agent can execute `revng project` commands against the current project by talking to the daemon.

The project must have been created with [`revng project init`](init.md) first, so that a `revng.yml` is present in the working directory or in one of its parents.

OPTIONS
-------

`--agent [claude|codex|opencode|pi]`
: Use the specified agent, skipping auto-detection. Fails if the agent is not installed.

`--system-prompt-append TEXT`
: Append `TEXT` to the instructions provided to the agent.

`--user-prompt TEXT`
: Start the agent with `TEXT` as the initial user prompt.

EXAMPLES
--------

Create a project from a binary and start an agent on it:

```{bash notest}
revng project init /usr/bin/hostname
revng project agent
```

Start a specific agent with an initial request:

```{bash notest}
revng project agent --agent claude --user-prompt "What is the biggest function in the binary?"
```

SEE ALSO
--------

[`revng-project`](index.md), [`revng-project-daemon`](daemon.md), [`revng-project-analyze`](analyze.md), [`revng-project-artifact`](artifact.md)
