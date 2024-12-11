Now that we are more familiar with the concepts of the model, let's briefly discuss the two main ways that users interact with rev.ng: producing an artifact and running an analysis.

## <a id="artifacts"></a>Artifacts

The rev.ng decompilation pipeline is organized as a tree.
Each node of the tree is known as *step*. Some of these steps exists in order to ease debugging, while others are associated with an *artifact*.
An *artifact* is the output of the step.

For instance, the [`disassemble` artifact](../../references/artifacts.md#disassemble-artifact) produces a set of text files containing the disassembled code of each function.

In order to produce an artifact, use the `revng artifact` CLI tool.

Artifacts can have different *granularity*.
For instance, the [`render-svg-call-graph` artifact](../../references/artifacts.md#render-svg-call-graph-artifact), which represents the call graph of the whole input program, is a single file generated considering the input as a whole.
Other artifacts, such as the previously mentioned `disassemble` artifact, have a *function-wise* granularity.
This means that they contain an "object" for each [function defined in the model](../../references/model.md#Binary.Functions).
Usually, these type of artifacts are in the form of a compressed tar archive.
Alternatively, certain artifacts are a single file which conceptually contains multiple objects. For instance the [`isolate` artifact](../../references/artifacts.md#isolate-artifact) produces a single LLVM IR module that contains multiple LLVM functions, one for each function defined in the model.

Each artifact is associated to a MIME type, check out [their documentation](../../references/mime-types.md) for further information.

Check out the [full list of artifacts](../../references/artifacts.md).

## <a id="analyses"></a>Analyses

While users can manually populate a model file and then use rev.ng to produce artifacts, we also offer tools to automatically populate the model.
These tools are called *analyses*.
An *analysis* takes on some input, such as the input binary itself or some other intermediate artifact, analyzes it and produces changes to the model.
A prime example of an analysis is the [`import-binary`](../../references/analyses.md#import-binary-analysis), which analyzes well-known binary formats (such as ELF, Mach-O and PE/COFF) and debug info (such as DWARF and CodeView) and imports in the model loading instructions, function list, symbol names, data types and so on.

Another interesting analysis that doesn't work on the input binary directly but on an intermediate artifact is the [`detect-abi`](../../references/analyses.md#detect-abi-analysis), which is inspect the results of the `lift` step to detect arguments and return values passed via registers.
Its final results, is to add to the model the prototypes of each analyzed function.

Unlike artifacts, which are designed to be run many times during the lifetime of a project, most analyses are usually run once at the start of project.
In order to maintain a list of analyses that are beneficial to run at the start of a project, rev.ng maintains an [*analysis list*](../../references/pipeline.md#analysis-lists) called `revng-initial-auto-analysis`.

In order to run an analysis, use the `revng analyze` CLI tool.

Check out the [full list of analyses](../../references/analyses.md).
