# User's manual

In the this section, we will introduce:

1. [The `MetaAddress`](metaaddress.md): an 64-bit address on steroids used to uniquely identify objects and code in the binary.
2. [The model](model.md): the YAML document where rev.ng stores all the vital information about the binary.
3. [The analyses](analyses.md): the parts of rev.ng automatically inferring things about a binary and writing them in the model.
   rev.ng provides analyses to detect function entry points, function prototypes and data structures and more.
4. [The artifacts](artifacts.md): things that rev.ng can produce for the user.
   Among other things: the decompiled C code, call graphs, disassembly, rev.ng internal representations and more.

If you just want to use the UI, most of these are benficial but not strictly necessary.
On other hand, they are vital before approaching scripting rev.ng.
