# What is rev.ng?

rev.ng stands for *next-generation reverse engineering*.
It is pronounced *revenge* (/ɹɪˈvɛndʒ/).
Its key features are:

* **Decompilation**: rev.ng features a full-fledged decompiler emitting C code.
  The C code we emit is designed to be recompilable without warnings (TODO: link).
* **VSCode-based UI**: we provide a UI based on VSCode hiding away all the complexity of rev.ng internals and wrapping everything in a nice, usable UI. TODO: link
* **Interactivity**: rev.ng is structured to automatically infer as much information as possible, but lets the user customize it by simply editing a YAML file, aka *the model*.
  Every change to the model automatically invalidates certain things previously produced by rev.ng, which will be automatically recomputed when needed.
  We offer Python and TypeScript wrapper to interact with rev.ng.
* **Focus on data structures**: the most innovative feature of rev.ng is its focus on data structures. Thanks to DLA (TODO: link), we can automatically infer data structures by analyzing how a pointer is used by one or more functions.
  See TODO.
* **Compatibility**: rev.ng can import from IDA's `.idb`/`.i64` (TODO), Linux/macOS debug information (DWARF) and Windows debug information (PDB).

In terms of design choices, its distinctive characteristics are:

* **Openess**: all the decompilation pipeline is open source, except for certain automated analyses.
  See the rev.ng feature comparison table. TODO
* **Based on QEMU**: it employs QEMU to lift assembly code to an architecture-independent representation.
  QEMU supports many architectures, so do we. See the [support matrix](TODO).
  Note that we never emulated any code, we simply use QEMU as a lifter (see TODO).
* **Based on LLVM**: all of our analyses are LLVM passes.
  This enables to use existing powerful analyses and optimizations and ensures scalability.
* **Client/server Architecture**: scripting and automation takes place in your favorite environment by connecting to a GraphQL API over the network or locally, possibly through our wrappers.
