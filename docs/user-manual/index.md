# User's manual

The user's manual describes concepts that are mostly useful for users.
If you just want to use the UI, most of these are benficial but not strictly necessary.
On other hand, they are vital to get a better understanding of rev.ng and in particular to approach scripting rev.ng.

The manual discusses the following topics:

1. [Getting a working environment](working-environment.md): explains preliminary steps necessary to start using rev.ng, both if you're using a binary distribution or if you prefer to build things from source.
2. [The `MetaAddress`](metaaddress.md): a 64-bit address on steroids used to uniquely identify objects and code in the binary.
3. [The model](model.md): a brief theoretical introduction to the model, the YAML document where rev.ng stores everything the user can customize, such as function names, prototypes and so on.
   You can think about it as rev.ng's project file.
4. [Building a model from scratch](model-tutorial.md): a practical, step-by-step tutorial on how to build a model from scratch and how to produce artifacts such as disassembly and decompiled code.
5. [Analyses](analyses.md): analyses are the part of rev.ng that automatically infer information about a binary and write them in the model, just as a user could do.
   <br />rev.ng provides analyses to detect function entry points, function prototypes, data structures and more.
