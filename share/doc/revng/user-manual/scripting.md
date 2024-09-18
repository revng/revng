This file introduces the `revng` scripting capabilities. `Revng` scripting allows you to run a subset of `revng` commands with `python`.

Currently we have two modes of operation: `cli` and `daemon`. The `cli` runs the commands via the `cli` (with `subprocess`), the `daemon` spins up the `revng daemon` and runs the commands through it (with `gql`).

First you need to import the profile. **Apart from the the initial import and initialization the interface for both profiles is the same.**

```python
from revng.scripting import CLIProfile     # for cli
from revng.scripting import DaemonProfile  # for daemon
```

Then you must instantiate the `profile` class and get the project
```python
profile = CLIProfile()
project = profile.get_project()
```

You can pass a path to the `resume` dir if you want to persis the changes after restarts
```python
resume = "path/to/resume/dir"
project = profile.get_project(resume)
```

After that you probably want to `import` a binary and run the `initial auto analysis`, we have a
method that does just that
```python
binary = "path/to/binary"
project.import_and_analyze(binary)
```

You can also just import the `binary` if you prefer
```python
binary = "path/to/binary"
project.import_binary(binary)
```

The `model` attributes are composed of objects, you can access them like so
```python
project.model            # Binary
project.model.Functions  # Functions
```

You can, for example rename a `function`
```python
project.model.Functions[0].CustomName = "new_function_name"
```

If you want to persis the changes after the restart, you **need** to explicitly call the `commit` method
```python
project.model.Functions[0].CustomName = "new_function_name"
project.commit()
```

You can run `analysis` or `analysis list`
```python
project.analyze("detect-stack-size")
project.analyze("revng-initial-auto-analysis")
```

You can also get `artifacts`
```python
# Get artifact through the `project`
project.get_artifact("disassemble")

# Pass functions that you want to get artifacts from, the second argument is a `list` so
# you can pass multiple functions
project.get_artifact("disassemble", [project.model.Functions[0]])

# Get artifact through a `function`, this is equal to above
project.model.Functions[0].get_artifact("disassemble")

# Get multiple artifacts at once. Pass an empty `list` if you wish
# to run the analysis on all the targets
result = project.get_artifacts({
    "disassemble": [project.model.Functions[0],
                    project.model.Functions[1]],
    "decompile": []
})

# You can also get the artifact for `typeDefinitions`
project.model.TypeDefinitions[1].get_artifact("emit-type-definitions")
```

You can also `parse` the result with `ptml`
```python
from revng import ptml

result = project.model.Functions[0].get_artifact("disassemble")
ptml.parse(result)
```
