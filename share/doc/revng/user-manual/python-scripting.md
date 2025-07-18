# Python scripting

This page describes the rev.ng Python scripting capabilities.

## Creating a project

There two project classes to use rev.ng from Python: `CLIProject`, which spawns `revng` as a subprocess for each command, and `DaemonProject`, which interacts with `revng daemon` over the network using the GraphQL API.
Apart from this difference, they offer the same programming interface.

First, import the project of choice:

```python
>>> from revng.project import CLIProject     # for cli
>>> from revng.project import DaemonProject  # for daemon

>>> project = CLIProject()
```

You can provide a path to the resume directory (similar to `revng [command] --resume`).
This useful if you want to persist changes for loading the project again in the future:

```python
>>> resume = "path/to/resume/dir"
>>> project = CLIProject(resume)
```

After that, you can to import a binary and run the [*initial auto analyses*](references/pipeline.md#analysis-lists), we have a method that does just that:

```python
>>> from revng.support import get_example_binary_path
>>> binary_path = get_example_binary_path()
>>> project.import_and_analyze(binary_path)
```

## Producing artifacts

Once you have successfully loaded a binary, you can obtain the available artifacts:

```{python ignoreoutput=2,6,9,13,20}
# Get artifact through the `project`
>>> project.get_artifact("disassemble")

# Pass functions that you want to get artifacts from, the second argument is a
# `list` so you can pass multiple functions
>>> project.get_artifact("disassemble", [project.model.Functions[0]])

# Get artifact through a `function`, this is equal to above
>>> project.model.Functions[0].get_artifact("disassemble")

# Get multiple artifacts at once. Pass `None` if you wish to get the artifact
# for all the targets
>>> project.get_artifacts({
...    "disassemble": [project.model.Functions[0],
...                    project.model.Functions[1]],
...    "decompile": None
... })

# You can also get the artifact for `TypeDefinitions`
>>> project.model.TypeDefinitions[1].get_artifact("emit-type-definitions")
```

You can also `parse` the result with `ptml`:

```{python ignoreoutput=2}
>>> result = project.model.Functions[0].get_artifact("disassemble")
>>> result.parse()
```

LLVM modules can be parsed and explored via the `llvmcpy` python module:

```python
>>> lifted = project.get_artifact("lift")

# Use the parsed IR
>>> for function in lifted.module().iter_functions():
...     for bb in function.iter_basic_blocks():
...         for instruction in bb.iter_instructions():
...             instruction.dump()
```

## Interacting with the model

In order to access [the model](../model/), use `project.model`:

```{python ignoreoutput=1,2}
>>> project.model            # Binary
>>> project.model.Functions  # Functions
```

Beyond inspecting the model, you can change it.
For example rename a function:

```python
>>> project.model.Functions[0].Name = "new_function_name"
```

After you make a change, you need to invoke the `commit` method in order for the changes to be applied (e.g., to see the new name in an artifact):

```python
>>> project.model.Functions[0].Name = "new_function_name"
>>> project.commit()
```

If you want to run a set of predefined analyses, you can run them with:

```python
>>> project.analyses_list("revng-initial-auto-analysis")
```

If you want to run a specific [analysis](../analyses/) instead, you can do that too.

```python
>>> project.analyze("detect-stack-size")
```
