Before we dig any deeper into the model, it is important to understand a concept that's ubiquous in rev.ng: the `MetaAddress`.

You can see the `MetaAddress` as a virtual address on steroids.
It is basically a 64-bit integer plus some additional meta-information useful to capture the fact that the same 64-bit address might represent different things in rev.ng.

For example, in an ARM program, it is in principle possible that the code at a certain address is used both as regular ARM code and as ARM Thumb code. A `MetaAddress` captures this difference and enables us to neatly distinguish the two situations.

The other key situation in which the concept of `MetaAddress` turns out to be very useful is in programs that, at the same address, have different code at different points in time.
This is typical if a program embeds a JIT compiler or if the program is using non-trivial packing techniques.
Thanks to `MetaAddress`, rev.ng is able to analyze such programs as if that code was at different addresses.

## The components of a `MetaAddress`

In practice, a `MetaAddress` is a very simple data structure:

```c title="PlainMetaAddress.h"
typedef struct {
  uint32_t Epoch;
  uint16_t AddressSpace;
  uint16_t Type;
  uint64_t Address;
} PlainMetaAddress;
```

Let's review each field in detail:

* `uint64_t Address`: the absolute address represented as a 64-bits integer;
* `uint16_t Type`: the entry of an `enum` to distinguish what this `MetaAddress` is pointing to and the size of the pointer:

    * `Invalid`: the pointer is not valid;
    * `Generic32`: a 32-bit pointer pointing at raw data (not code);
    * `Generic64`: as before, but the pointer is 64-bits wide;
    * `Code_x86`: a 32-bit pointer to 32-bit x86 code;
    * `Code_x86_64`: a 64-bit pointer to 64-bit x86 code;
    * `Code_arm`: a 32-bit pointer to regular ARM code;
    * `Code_arm_thumb`: a 32-bit pointer to regular ARM Thumb code;
    * `Code_*`: one entry for each other type of code we currently support;

* `uint32_t Epoch`: a progressive value that enables us to distinguish different things at the same address in different points during the execution of the program;
* `uint16_t AddressSpace`: certain architectures can access different type of memories (e.g., a large ROM and a smaller RAM) through different instructions.
  The `AddressSpace` field enables us to distinguish them.
  Note that currently we do not attribute any particular meaning to `AddressSpace`, but in the future we might use them to support multi-binary analysis: every dynamic library you'll feed to rev.ng will be distinguished by a different value of `AddressSpace`.

## Textual representation

A `MetaAddress` has also a string representation:

```
Address:Type[:Epoch[:AddressSpace]]
```

For example:

* `:Invalid`: an invalid `MetaAddress`;
* `0x1000:Code_arm`: a pointer with value `0x1000` targeting regular ARM code;
* `0x1000:Code_arm_thumb`: a pointer with value `0x1000` targeting regular ARM Thumb code;
* `0x3000:Generic32`: a 32-bit pointer with value `0x3000` targeting raw data;
* `0x1000:Code_x86:10:2`: a pointer to 32-bit x86 code at epoch 10 in address space 2;

## Scripting

Our Python library offers a `MetaAddress` `class`:

```python
>>> from revng.model import MetaAddress, MetaAddressType
>>> address = MetaAddress(Address=0x1000, Type=MetaAddressType.Code_arm)
>>> str(address)
'0x1000:Code_arm'
>>> hex(address.Address)
'0x1000'
```

Our TypeScript library offers a `MetaAddress` `class` too:

```typescript
> import { MetaAddress } from "revng-model"
> let address = new MetaAddress("0x1000:Code_arm")
> address.toString()
"0x1000:Code_arm"
> address.address.toString(16)
"1000"
```

These facilities are currently only available within a rev.ng development environment, i.e., an `orc shell`, but they will be generally available in the future.

### Relevant sources

* `include/revng/Runtime/PlainMetaAddress.h`
* `include/revng/Support/MetaAddress.h`
