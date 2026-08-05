`revng-merge-llvm`
================

NAME
----

`revng merge llvm` - Merge multiple LLVM IR files into a single one.

SYNOPSIS
--------

    revng merge llvm [-o OUTPUT] [INPUT...]

DESCRIPTION
-----------

Links multiple LLVM IR modules into a single one. The behavior depends on the number of inputs:

- A single input is interpreted as either a YAML dictionary of Base64-encoded modules, a tar whose members are bitcode files, or a single LLVM module.
- With no input, standard input is read as a single input.
- Multiple inputs are all treated as bitcode files.

This is handy for combining the function-wise LLVM IR produced by artifacts such as `isolate`, `enforce-abi`, `simplify-switch` and `make-segment-ref` (each emits one module per function) into a single module. See the [available artifacts](../artifacts.md).

OPTIONS
-------

`-o, --output OUTPUT`
: Where to write the merged module. Standard output if omitted.

EXAMPLES
--------

Merge the per-function IR of the `isolate` artifact into a single module:

```{bash notest}
revng project artifact isolate | revng merge llvm -o merged.bc
```

SEE ALSO
--------

[`revng-project-artifact`](revng-project-artifact.md), [`revng-ptml`](revng-ptml.md)
