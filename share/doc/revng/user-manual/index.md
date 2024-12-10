# User's manual

The user's manual describes concepts that are mostly useful for users.
If you just want to use the UI, most of these are beneficial but not strictly necessary.
On other hand, they are vital to get a better understanding of rev.ng and in particular to approach scripting rev.ng.

The manual discusses the following topics:

1. [Initial setup](initial-setup.md): explains preliminary steps necessary to start using rev.ng, both if you're using a binary distribution or if you prefer to build things from source.
2. Key concepts. This section introduce the following key concepts which are necessary to understand to interact with rev.ng.
    1. [The `MetaAddress`](key-concepts/metaaddress.md): a 64-bit address on steroids used to uniquely identify objects and code in the binary.
    2. [The model](key-concepts/model.md): a brief theoretical introduction to the model, the YAML document where rev.ng stores everything the user can customize, such as function names, prototypes and so on.
       You can think about it as rev.ng's project file.
    3. [Artifacts and analyses](key-concepts/artifacts-and-analyses.md). In rev.ng, you mostly do two thing: either you produce an artifact, for instance the decompiled C code of a function, or you run an analysis, for instance the analysis detecting the arguments of a function.
       This section introduces the concepts of "artifact" and "analysis" in rev.ng.
3. Tutorial: a practical walkthrough to better understand the concepts above and become familiar with rev.ng's command line and scripting interfaces.
    1. [A model from scratch](tutorial/model-from-scratch.md): a practical, step-by-step tutorial on how to build a model from scratch and how to produce artifacts such as disassembly and decompiled code.
    2. [Running analyses](tutorial/running-analyses.md): analyses are the part of rev.ng that automatically infer information about a binary and write them in the model, just as a user could do.
    <br />rev.ng provides analyses to detect function entry points, function prototypes, data structures and more.
