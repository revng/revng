The *model* is a YAML document that contains information about the binary that the user can customize.
You can think about the model as a sort of interchange format for reverse engineering: it is not specific to a binary image format (e.g., ELF) or an architecture.

It includes:

* **Loading Information**: a description of which parts of the input file should be loaded and at which address.
* **The list of functions**: for each function we have an entry address, the layout of its stack frame (where local variables reside), its prototype and more.
* **The type system**: includes all the `struct`s, `enum`s and function prototypes for the binary.

On the other hand, it *does not* include things such as the **control-flow graph**: the control-flow graph is rather complicated to maintain for the user.
<br />For instance, if a user wants to mark a function as `noreturn`, in order to obtain a valid model it'd need to update the CFG of all of its callers.
However, the user might be sometimes interested in providing extra information about the control-flow of a program, which is a job more suitable for rev.ng internals.

The full structure of the model is documented in the [model reference page](../../references/model.md).

The model must be a valid YAML document, but that's not enough.
In order to be consumed by rev.ng, a model needs to be valid.
You can check if a model is valid as follows:

```{bash notest}
$ revng model opt -verify mymodel.yml
```

If the command succeeds, the tool will print the model again.

!!! question "Too much theory?"

    Check out the tutorial on [how to create a model from scratch](../tutorial/model-from-scratch.md)!

## Who uses the model?

The model has a couple of different users:

* **The end user**: the user, as part of his analysis activities, makes changes to the model, e.g., renaming a function, adding arguments, introducing new functions, creating new data structures.
  <br />The user interacts with the model either through the UI or manually or through scripting wrappers that enable easy manipulation of the model.<br />
  We currently provide wrappers for Python and TypeScript.
* **Importers/exporters**: the model is designed to be a sort of interchange format and, as such, it's not strictly rev.ng-specific.
  This means that's possible to implement importers from other formats.
  <br />For example, we provide out of the box importers for the most common binary formats (ELF, PE/COFF, MachO), debug info formats (DWARF and PDB) and other binary analysis tools such as IDA Pro's `.idb`s.
  <br />Users can easily implement new importers by simply manipulating the model (which is a plain YAML file) in their favorite scripting language.
  <br />In the future, we also plan to implement *exporters*, e.g., produce DWARF debug info that enable advanced debugging using information available in the model (e.g., arguments, data structures...) using off-the-shelf debuggers on the target binary.
* **Pipes**: rev.ng provides a set of *pipelines*, composed by *pipes*, that produce *artifacts*. Most pipes, read the model for various reasons.
  For example, there's a pipeline responsible for generating a C header file declaring all the functions in the binary.
  In order to produce this artifact, the pipeline inspects the model and nothing else.
  Other pipelines might inspect the model, other previously generated artifacts and the input binary itself.
* **Analyses**: rev.ng also provides a set of *analyses* that can automatically recover high-level information by analyzing the binary or artifacts produced by a pipeline.
  The final goal of an analysis is to make changes to the model.
  <br />For instance, rev.ng provides an analysis that automatically detects the arguments of a function: when you run such an analysis, it will go through the list of functions in the model, analyze each function and enrich each function that did not initially have a prototype, with arguments and return values.
  To better understand the role of analyses, check out the [artifacts and analysis documentation](artifacts-and-analyses.md).

## Relevant sources

* `include/revng/Model/Binary.h`
