An empty model is a valid model:

```bash
$ revng model opt /dev/null -verify -Y
---
{}
...
```

Let's get started and try to decompile a very simple program.

```bash
$ printf '\x48\x01\xf7\x48\x89\xf8\xc3\x90' > sum
```

If you disassemble it as raw bytes you get the following:

```bash
$ objdump -D -Mintel,x86-64 -b binary -m i386:x86-64 sum

sum:     file format binary


Disassembly of section .data:

0000000000000000 <.data>:
   0: 48 01 f7              add    rdi,rsi
   3: 48 89 f8              mov    rax,rdi
   6: c3                    ret
   7: 90                    nop
```

Note that this file contains raw x86-64 instructions, it's not a valid program (e.g., an ELF).

Let's now create a simple model that enables us to decompile this simple function.

### Step 1: Loading

The first thing rev.ng needs to know is the architecture and the ABI of the program:

```yaml title="model.yml"
Architecture: x86_64
DefaultABI: SystemV_x86_64
```

Then, we need to describe how to load the program.
Let's pretend we want to load this code at address `0x400000`:

```yaml title="model.yml"
Segments:
  - StartAddress: "0x400000:Generic64"
    VirtualSize: 7
    StartOffset: 0
    FileSize: 7
    IsReadable: true
    IsWriteable: false
    IsExecutable: true
```

The piece of model above tells rev.ng to take 7 bytes from the file and load them at address `0x400000` as `+rx` data (i.e., code).

### Step 2: Function list

Most parts of rev.ng work on a function basis (e.g., we decompile one function at a time).
Let's create an entry in the function list:

```yaml title="model.yml"
Functions:
  - Entry: "0x400000:Code_x86_64"
```

At a minimum, a function is identified by its entry [`MetaAddress`](metaaddress.md).
Note how here the type of the `MetaAddress` is not `Generic64` but `Code_x86_64`, to indicate the type of code we can expect in the function.

### Step 3: Disassembly

At this point, we provided rev.ng enough information to be able to show us the disassembly of our program.

```bash
$ revng artifact disassemble sum --model model.yml | revng ptml
0x400000:Code_x86_64.asm.tar.gz: |-
  _function_0x400000_Code_x86_64:
    400000:    48 01 f7    add rdi, rsi
    400003:    48 89 f8    mov rax, rdi
    400006:    c3          ret
```

The output of the `revng artifact disassemble` command is a `tar.gz` composed by one file for each input function. Each file is an assembly listing decorated using [PTML](../references/ptml.md).
<br />`revng ptml` strips away all this details and outputs a YAML dictionary with one entry for each disassembled function.

For further information on the file types emitted by `revng artifact`, see the [MIME types documentation](../references/mime-types.md).

### Step 4: Defining a function prototype

In order to decompile the function, we need to provide a function prototype.

By looking at the code above, we can see that the registers `rdi` and `rsi` are read at the entry of the function and the result of the computation is stored in `rax`: this function very much looks like a function taking two arguments and returning an integer in the x86-64 SystemV ABI!

Let's create such prototype then.

Specifically, we want to define the prototype for something that in C would look like the following:

```c
uint64_t sum(uint64_t rdi, uint64_t rsi);
```

First of all, let's populate the model type system with a bunch of *primitive types* such as `void`, `uint64_t`, `uint32_t` and so on.
We could write them by hand, but the `add-primitive-types` analysis can help us with that:

```bash
$ revng analyze add-primitive-types /dev/null \
        | grep -vE '^(---|\.\.\.)$' >> model.yml
```

> **Note**: `revng analyze` is used to run *analyses* and automatically populate parts of the model. More on this in the [analyses page](analyses.md).

This will add to the model something similar to the following:

```yaml
Types:
  - Kind: PrimitiveType
    ID: 1288
    PrimitiveKind: Unsigned
    Size: 8
```

Here we are defining a *primitive* type (such as `void`, an integral type or a floating-point type) with size 8 bytes (64 bits) and *kind* `Unsigned`. Basically, an `uint64_t`.

> **Note**: the first 1808 type IDs are currently reserved for `PrimitiveType`s, which have a deterministic ID.
> All other types, can have an arbitrary type ID, but tools usually adopt progressive type IDs.
> <br />In the future, there will be no reserved type IDs since we plan to restructure how `PrimitiveType`s are handled.

Now that we have our `uint64_t`, we can define the function prototype.
In the model type system, it looks like this:

```yaml title="model.yml"
  - Kind: CABIFunctionType
    ABI: SystemV_x86_64
    ID: 1809
    Arguments:
      - Index: 0
        Type:
          UnqualifiedType: "/Types/1288-PrimitiveType"
      - Index: 1
        Type:
          UnqualifiedType: "/Types/1288-PrimitiveType"
    ReturnType:
      UnqualifiedType: "/Types/1288-PrimitiveType"
```

OK, there's a lot here. Let's go through it:

* `Kind: CABIFunctionType`: we're defining a type, specifically a C prototype associated to an ABI;
* `ABI: SystemV_x86_64`: the chosen ABI is the x86-64 SystemV one;
* `ID: ...`: a unique identifier;
* `Arguments`: we have two arguments (index `0` and index `1`);
  * `Type`: a qualified type, i.e., a type plus (optionally) one or more qualifiers such as `const`, pointer (`*`) and so on;
    * `UnqualifiedType: "/Types/PrimitiveType-1288"`: a reference to the actual, unqualified, type; in this case, it's a reference to the *primitive type* with ID 1288, i.e., the `uint64_t` defined above;
* `ReturnType`: again, a reference to `uint64_t`;

At this point, we can associate the function prototype with the previously defined function:

```diff
--- a/model.yml
+++ b/model.yml
@@ -10,6 +10,7 @@
     IsExecutable: true
 Functions:
   - Entry: "0x400000:Code_x86_64"
+    Prototype: "/Types/1809-CABIFunctionType"
 Types:
   - Kind: PrimitiveType
     ID: 1288
```

Basically, we added to our function definition a reference to the prototype we created above.

### Step 5: Decompiling

At this point, we have all the information we need to successfully decompile our example program:

```c
$ revng artifact decompile sum --model model.yml | revng ptml
0x400000:Code_x86_64:
  uint64_t function_0x400000_Code_x86_64(uint64_t unnamed_arg_0, uint64_t unnamed_arg_1) {
      return unnamed_arg_0 + unnamed_arg_1;
  }
```

### Step 6: Renaming

One of the main activities of a reverse engineer is giving things a name, just like Adam in the Genesis.
<br />Let's try to give a name to our function:

```diff
--- a/model.yml
+++ b/model.yml
@@ -11,6 +11,7 @@ Segments:
 Functions:
   - Entry: "0x400000:Code_x86_64"
     Prototype: "/Types/1809-CABIFunctionType"
+    CustomName: Sum
 Types:
   - Kind: PrimitiveType
     ID: 256
```

Almost everything in the model can have a name. Let's add a name to the function arguments:

```diff
--- a/model.yml
+++ b/model.yml
@@ -119,12 +120,14 @@ Types:
   - Kind: CABIFunctionType
     ABI: SystemV_x86_64
     ID: 1809
     Arguments:
       - Index: 0
         Type:
           UnqualifiedType: "/Types/1288-PrimitiveType"
+        CustomName: FirstAddend
       - Index: 1
         Type:
           UnqualifiedType: "/Types/1288-PrimitiveType"
+        CustomName: SecondAddend
     ReturnType:
       UnqualifiedType: "/Types/1288-PrimitiveType"
```

Here's what we get now if we try to decompile again:

```bash
$ revng artifact decompile sum --model model.yml | revng ptml
0x400000:Code_x86_64.c.ptml: |-
  _ABI(SystemV_x86_64)
  uint64_t Sum(uint64_t FirstAddend, uint64_t SecondAddend) {
    return FirstAddend + SecondAddend;
  }

```
