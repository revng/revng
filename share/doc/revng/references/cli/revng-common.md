`revng-common`
================

NAME
----

`revng-common` - Options and behaviors shared by several `revng` commands.

DESCRIPTION
-----------

This page documents functionality shared by many `revng` subcommands: [pipebox arguments](#pipebox-arguments), the [developer wrapper options](#developer-wrapper-options), the [`--debug`](#debug) mode, and the [project id and token](#project-id-and-token) options.

PIPEBOX ARGUMENTS
-----------------

Every `revng` command that runs the pipeline accepts a trailing `-- PIPEBOX ARGS...`. Everything after `--` is forwarded verbatim to the pipebox `initialize` function, that is, to the native pipeline library, which parses them as global options.

These options are independent of the specific command; they mostly control the logging and the progress reporting of the underlying native execution. Pass `-- --help` to list every available option, including the full set of `--debug-log` loggers.

`--debug-log LOGGER`
: Enable verbose logging for the logger named `LOGGER`. May be repeated to enable several loggers. The set of available loggers is large and build-specific; list it with `-- --help`.

`--trace FILE`
: Write an execution trace of the pipeline tasks to `FILE`, in the Chrome Trace Event Format (a JSON array of begin/end events). The file can be opened in a trace viewer such as `chrome://tracing`, [Perfetto](https://ui.perfetto.dev) or [Speedscope](https://www.speedscope.app).

`--progress`
: Display interactive progress bars on standard error while the pipeline runs.

`--progress-plain`
: Emit progress information on standard error as plain text, without interactive bars. Suitable for non-interactive terminals and log files.

For example, show progress bars while decompiling:

```{bash notest}
revng project artifact emit-c-as-single-file -- --progress
```

or enable a logger and write an execution trace:

```{bash notest}
revng project analyze initial-auto-analysis -- --debug-log LOGGER --trace trace.json
```

DEVELOPER WRAPPER OPTIONS
-------------------------

Several `revng` subcommands accept the following developer wrapper options, which run the underlying process inside the corresponding tool:

`--perf`
: Run the program(s) under perf (for use with hotspot).

`--heaptrack`
: Run the program(s) under `heaptrack`.

`--gdb`
: Run the program(s) under `gdb`.

`--lldb`
: Run the program(s) under `lldb`.

`--valgrind`
: Run the program(s) under `valgrind`.

`--callgrind`
: Run the program(s) under `callgrind`.

`--rr`
: Run the program(s) under `rr`.

`--wrapper WRAPPER`
: Run the program(s) with the specified wrapper command, for instance `--wrapper 'strace -f'`.

DEBUG
-----

Several `revng` subcommands accept a `--debug DIR` option that records every pipeline step under `DIR`, so it can be inspected and re-run in isolation.

`--debug RUNNER_CONTEXT`
: Run the command in debug mode, using the specified directory (created if missing). Where possible, pipes and analyses are run as subcommands with their input and output files stored under that directory, as described below.

With `--debug DIR`, a directory where you can easily reproduce each step of the invocation is created. Each pipe and analysis is run as a separate [`revng pipeline run-pipe`](revng-pipeline-run-pipe.md) / [`run-analysis`](revng-pipeline-run-analysis.md) subcommand, and its inputs and outputs are recorded under `DIR`. `DIR` gets one numbered subdirectory per executed step, in execution order:

```
0000-import-files/
0001-collect-cfg/
0002-process-assembly/
0003-yield-assembly/
```

Each step subdirectory contains a `revng.yml` (the model used as input to the step) and a `run` script that re-runs exactly that step in isolation. The remaining contents differ between pipes and analyses.

A **pipe** produces containers, so its subdirectory also contains:

- `inputs/` - the input containers, one `.tar` per container argument.
- `outputs/` - the containers produced by the step.
- `files/` - the file storage (input files referenced by the model, such as the original binary, addressed by hash); empty when the step needs none.
- `dependencies.tar` - the dependency and invalidation data emitted by the step.

An **analysis** updates the model instead of producing containers, so its subdirectory has no `inputs/`, `outputs/`, `files/` or `dependencies.tar`. Instead it contains:

- `output_revng.yml` - the model produced by the analysis.
- `<container>.tar` - one file per input container argument (placed directly in the subdirectory).

To reproduce or debug a single step without running the whole pipeline, enter its directory and run the script:

```{bash notest}
cd DIR/0002-process-assembly/
./run
```

The `run` script forwards any extra arguments to the underlying `run-pipe`/`run-analysis` command, so you can run the step under a debugger or profiler with a [wrapper option](#developer-wrapper-options), for example `./run --gdb`.

PROJECT ID AND TOKEN
--------------------

The `revng project` subcommands that talk to a storage provider (`init`, `analyze` and `artifact`) accept:

`--project-id TEXT`
: Project id to use for the storage provider.

`--token TEXT`
: Token to pass to the storage provider.

SEE ALSO
--------

[`revng`](revng.md), [`revng-project`](revng-project.md), [`revng-quick`](revng-quick.md), [`revng-pipeline`](revng-pipeline.md)
