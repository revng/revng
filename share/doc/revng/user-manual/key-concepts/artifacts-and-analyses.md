Now that we are more familiar with the concepts of the model, let's briefly discuss the two main ways that users interact with rev.ng: producing an artifact and running an analysis.

## <a id="artifacts"></a>Artifacts

*Artifacts* are the main way in which users of rev.ng can get it to produce something, for instance the decompiled C code.

The rev.ng decompilation pipeline is organized as [a tree](../../references/pipeline.md).
Each node of the tree is known as *artifact*.
An *artifact* is an output of the pipeline. Some artifacts are designed to be consumed by the end user (e.g., [`emit-c-as-single-file`](../../references/artifacts.md#emit-c-as-single-file-artifact), the complete decompiled code of the binary), while some other are designed as debug artifacts (e.g., [`enforce-abi`](../../references/artifacts.md#enforce-abi-artifact), an internal LLVM IR artifact).

For instance, the [`disassemble` artifact](../../references/artifacts.md#disassemble-artifact) produces a set of text files containing the disassembled code of each function.

In order to produce an artifact, use the [`revng2-project-artifact`](../../references/cli/revng2-project-artifact.md) CLI tool.

Artifacts can have different *granularity*.
For instance, the [`render-svg-call-graph` artifact](../../references/artifacts.md#render-svg-call-graph-artifact), which represents the call graph of the whole input program, is a single file generated considering the input as a whole.
Other artifacts, such as the previously mentioned `disassemble` artifact, have a *function-wise* granularity.
This means that they contain an "object" for each function defined in the model ([`Binary.Functions`](../../references/model.md#Binary.Functions)).

If there's a single output, `revng2-project-artifact` emits it directly, otherwise it either emits a YAML dictionary with one entry per function (default), or a `.tar` archive with one file per function.

Note that, for performance reasons, rev.ng does not recompute an artifact each time it's requested, but it has a caching feature.
Making changes to the model, automatically invalidates the parts of the cache affected by the change.

Each artifact is associated to a MIME type, you can find the complete list in the [MIME types documentation](../../references/mime-types.md).

The reference also documents the [full list of artifacts](../../references/artifacts.md).

## <a id="analyses"></a>Analyses

While users can manually populate a model file and then use rev.ng to produce artifacts, we also offer tools to automatically populate and refine the model.

These tools are called *analyses*.
An *analysis* takes in some input, such as the input binary itself or some other intermediate artifact, analyzes it and produces changes to the model.

A prime example of an analysis is the [`parse-binary`](../../references/analyses.md#parse-binary-analysis), which analyzes well-known binary formats (such as ELF, Mach-O and PE/COFF) and debug info (such as DWARF and CodeView) and imports in the model loading instructions, function list, symbol names, data types and so on.

Another interesting analysis that doesn't work on the input binary directly but on an intermediate artifact is the [`detect-abi`](../../references/analyses.md#detect-abi-analysis), which inspects the [`lift` artifact](../../references/artifacts.md#lift-artifact) to detect arguments and return values passed via registers.
Its final result is to add to the model the prototypes of each analyzed function.

Unlike artifacts, which are designed to be run many times during the lifetime of a project, most analyses are usually run once at the start of a project.
In order to maintain a list of analyses that are beneficial to run at the start of a project, rev.ng maintains an [*analysis list*](../../references/pipeline.md#analysis-lists) called `initial-auto-analysis`.

In order to run an analysis, use the `revng2 project analyze` CLI tool.

The reference documents the [full list of analyses](../../references/analyses.md).
