`revng2-project-artifact`
================

NAME
----

`revng2 project artifact` - Produce an artifact.

SYNOPSIS
--------

    revng2 project artifact ARTIFACT [OPTIONS] [OBJECTS] [-- PIPEBOX ARGS...]

DESCRIPTION
-----------

This command is a subcommand of [`revng2 project`](revng2-project.md).

Produces the artifact `ARTIFACT` from the project in the current directory.

Run without arguments to list all the available artifacts.

`OBJECTS` is a single comma-separated list of objects to produce. If omitted, all the available objects are produced. Each object can be given as:

- a full location, e.g. `/function/0x401000:Code_x86_64`;
- a bare key, e.g. `0x401000:Code_x86_64` (interpreted within the artifact's kind);
- its name, e.g. `Sum`;
- its automatic name, e.g. `function_0x401000_Code_x86_64`.

Use `--list` to see the available objects together with their names.

`ARTIFACT` is a subcommand: run `revng2 project artifact --help` to list the available artifacts.
The documentation describes [what an artifact is](../../user-manual/key-concepts/artifacts-and-analyses.md#artifacts) and the [available artifacts](../artifacts.md).

The project must have been created with [`revng2 project init`](revng2-project-init.md) first, so that a `revng.yml` is present in the working directory.

OPTIONS
-------

`--list`
: List the available objects for the artifact and then exit.

`-o FILE`
: Path to write the computed artifact to. If not specified, the result is printed to standard output.

`--format [auto|tar|yaml]`
: Select the output format. Defaults to `auto`, which emits a lone object raw (with no wrapping) and otherwise behaves like `yaml`. With `yaml` the artifact is emitted as a YAML dictionary `object -> data`, where `data` is text data if the artifact's MIME is textual or a Base64 encoding if it is binary. `--tar` emits a plain (uncompressed) tar without any transformation of the data.

`--yaml`
: Shortcut for `--format yaml`.

`--tar`
: Shortcut for `--format tar`.

`--debug RUNNER_CONTEXT`
: Run the command in debug mode, recording each pipeline step under the given directory. See [`revng2-common`](revng2-common.md#debug).

`--help-full`
: Show the help including the hidden artifacts and per-pipe `--*-configuration` options, then exit.

This command supports the [developer wrapper options](revng2-common.md#developer-wrapper-options) and [pipebox arguments](revng2-common.md#pipebox-arguments).

EXAMPLES
--------

Produce the decompiled code, saving the result to `decompiled.c.yml`:

```{bash notest}
revng2 project artifact emit-c-as-single-file -o decompiled.c.yml
```

Disassemble a single function and print it to standard output, addressing it by key or by name:

```{bash notest}
revng2 project artifact disassemble 0x401000:Code_x86_64
revng2 project artifact disassemble Sum
```

SEE ALSO
--------

[`revng2-project`](revng2-project.md), [`revng2-project-analyze`](revng2-project-analyze.md), [`revng2-project-init`](revng2-project-init.md)
