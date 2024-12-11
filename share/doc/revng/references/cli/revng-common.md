`revng-common`
==============

NAME
----

`revng-common` - A collection of common option to `revng` commands.

SYNOPSIS
--------

    revng * [options] COMMAND_NAME

DESCRIPTION
-----------

This page documents some options common to all `revng` CLI commands.

OPTIONS
-------

`--resume [PATH]`
: Use `PATH` as the resume directory.
  This option is critical if you want run multiple commands preserving the state from previous runs.
  For instance, if you want to run an analysis and then produce an artifact you can do:

  ```{bash notest}
  revng analyze --resume my-project-directory/ import-binary /usr/bin/hostname -o /dev/null
  revng artifact --resume my-project-directory/ emit-model-header /usr/bin/hostname
  ```

  `PATH` will contain the model and all the intermediate files produced by the various steps.

  By default, the command will create a temporary `--resume` directory and remove it after termination.

`--model PATH`
: Replace the current model with the model from `PATH` before running the command.

`--debug-log LOGGER_NAME`
: Enable the logger `LOGGER_NAME`.
  `--help` will list all the available loggers.

`--save-model [PATH]`
: Run the command and then dump the resulting model to `PATH`.

EXAMPLES
--------

An example usage of `--resume` to run the [`import-binary`](../analyses.md#import-binary-analysis) analysis and then produce the [`emit-model-header`](../artifacts.md#emit-model-header-artifact) artifact:

```{bash notest}
revng analyze --resume project-dir/ import-binary /usr/bin/hostname
revng artifact --resume project-dir/ emit-model-header /usr/bin/hostname
```

Run commands using `mymodel.yml` instead of `--resume`:

```{bash notest}
revng analyze import-binary /usr/bin/hostname --save-model mymodel.yml
revng artifact --model mymodel.yml emit-model-header /usr/bin/hostname
```

Note: it is preferable to use `--resume`.

SEE ALSO
--------

[`revng-artifact`](revng-artifact.md), [`revng-analyze`](revng-analyze.md)
