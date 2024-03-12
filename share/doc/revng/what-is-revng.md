# What is rev.ng?

rev.ng stands for *next-generation reverse engineering*.
It is pronounced *revenge* (/ɹɪˈvɛndʒ/).
Its key features are:

* **Decompilation**: rev.ng features a full-fledged decompiler emitting C code.
  The C code we emit is designed to be valid and recompilable without significant warnings.
* **VSCode-based UI**: we provide a UI based on VSCode hiding away all the complexity of rev.ng internals and wrapping everything in a nice, usable UI.
* **Interactivity**: rev.ng is structured to automatically infer as much information as possible from the binary, but lets the user customize it by simply editing a YAML file, aka *the model*.
  Every change to the model automatically invalidates certain things previously produced by rev.ng, which will be automatically recomputed when needed.<br />
  We offer Python and TypeScript wrappers to interact with rev.ng.
* **Focus on data structures**: the most innovative feature of rev.ng is its focus on data structures. Thanks to Data Layout Analysis, we can automatically infer data structures by analyzing how a pointer is used by one or more functions.
* **Compatibility**: rev.ng can import from IDA's `.idb`/`.i64`, Linux/macOS debug information (DWARF) and Windows debug information (PDB).

In terms of design choices, its distinctive characteristics are:

* **Openness**: the entire decompilation pipeline is open source.
* **Based on QEMU**: it employs QEMU to lift assembly code to an architecture-independent representation.
  Note that we never emulate any code, we simply use QEMU as a lifter.
* **Based on LLVM**: all of our analyses are LLVM passes.
  This enables to use existing, powerful analyses and optimizations and ensures scalability.
* **Client/server Architecture**: scripting and automation takes place in your favorite environment by connecting to a GraphQL API over the network or locally, possibly through our wrappers.
