`revng-ptml`
================

NAME
----

`revng ptml` - Inspect and manipulate PTML files.

SYNOPSIS
--------

    revng ptml [OPTIONS] [INPUT]

DESCRIPTION
-----------

Reads [PTML](../ptml.md) (the rich-text markup rev.ng uses for its textual artifacts) and renders or filters it. The input (standard input if omitted) may be a single PTML file, a YAML dictionary mapping object ids to PTML, or a tar of PTML files; the format is auto-detected.

Many artifacts emit PTML, for example `emit-c`, `emit-c-as-single-file`, `disassemble`, `emit-type-and-global-header`, `emit-helper-header` and `emit-single-type-definition`. See the [available artifacts](../artifacts.md).

OPTIONS
-------

`-p, --plain`
: Strip the markup and emit plain text.

`-c, --color`
: Emit ANSI-colored text. Without `-p`/`-c`, the output is colored on a terminal and plain otherwise.

`-f, --filter KEYS`
: Only show the given comma-separated object ids (when the input is a dictionary or tar). May be repeated.

`-e, --extract KEY`
: Emit only the single object with the given id.

`-i, --inplace`
: Strip the markup in place (cannot be used with standard input).

`-o, --output FILE`
: Write to `FILE` instead of standard output.

EXAMPLES
--------

Pretty-print the decompiled C with color:

```{bash notest}
revng project artifact emit-c-as-single-file | revng ptml -c
```

Extract a single function's disassembly from the `disassemble` artifact:

```{bash notest}
revng project artifact disassemble | revng ptml -e /function/0x401000:Code_x86_64
```

Strip a saved artifact to plain text:

```{bash notest}
revng project artifact emit-c-as-single-file -o code.ptml
revng ptml -p code.ptml -o code.c
```

SEE ALSO
--------

[`revng-project-artifact`](revng-project-artifact.md), [`revng-merge-llvm`](revng-merge-llvm.md)
