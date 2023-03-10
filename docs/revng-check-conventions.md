# Information about `revng-check-conventions`

`revng-check-conventions` is the tool that's internally used to check code formatting
for the `revng` codebase. Its behavior can be modified via the config file while major
changes are introduce directly into the `revng-check-conventions` script.

## Config

The config can be found in `share/revng/rcc-config.yml`, each element is composed
of the `type` key that refers to a function or class in the `revng-check-conventions`
file and all other keys are arguments of said function/class.

### `matchers`

There is a preliminary phase where all the files are run through a series of
boolean predicates, and will be sorted into `tags` (e.g. `c`, `python` etc.)
Currently `revng-check-conventions` assumes that a file will belong to a single
tag, except for the special tag `all` which has all the files.

Implemented matchers are:
* `suffix_or_shebang`: Will check if a file has suffix in the `suffixes` or have
  the `shebang` string in its first line

* `cmake_filter`: Special matcher for cmake that will match all the files that have
  'cmake' in the name (case-insensitive)

### `write_passes`

As a first step these passes are run, these passes are run first, sequentially for each
tag. These passes can change the file's contents when `revng-check-conventions` is run with
`--force-format` whereas without it they will not change anything (but can still fail if the
formatting does not conform)
Currently it is not possible for a write_pass to operate on the tag `all`

### `read_passes`

These passes are run last and will only read the files and check if there are any style violations.
Thse will be run in parallel as there is a guarantee that no file will be modified by their execution.


## Bypassing checks

In general to bypass checks it's sufficent to add the appropiate annotation of the tool (e.g. flake8)
that caused the violation.

### `set -euo pipefail`

This check, that's part of the builtin `bash-check` can be suppressed with `# rcc-ignore: bash-set-flags`

### `InitRevng`

`Main.cpp` files (tools) must call `InitRevng` to initialize LLVM facilities properly. In case
this is not possible (e.g. the tool calls a library that does `InitRevng`) then the check can
be ignored with `// rcc-ignore: initrevng`
