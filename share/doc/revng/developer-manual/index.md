# Developer's manual

The developer's manual describes concepts that are mostly useful for rev.ng developers/advanced users.

The manual discusses the following topics:

1. [QEMU helpers](qemu-helpers.md): an extensive explanation of why QEMU helper functions are needed in the rev.ng lifting process, and how they are used at different stages in the decompilation pipeline.
2. [ABI definition](../references/abi-definition.md): how to declaratively add support for a new ABI, documenting all the available options that control how arguments and return values are distributed across registers and the stack.
