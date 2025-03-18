An empty model is a valid model:

```bash
$ revng model opt /dev/null -verify -Y
---
Version: 3
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

::: tip

    You can check out the [model reference](../../reference/model.md) to see what each field exactly means.

Then, we need to describe how to load the program.
Let's pretend we want to load this code at address `0x400000`.
We can do this by introducing a new [`Segment`](../../references/model.md#segment):

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
Let's create an entry in the [functions list](../../references/model.md#Binary.Functions):

```yaml title="model.yml"
Functions:
  - Entry: "0x400000:Code_x86_64"
```

At a minimum, a function is identified by its entry [`MetaAddress`](../key-concepts/metaaddress.md).
Note how here the type of the `MetaAddress` is not `Generic64` but `Code_x86_64`, to indicate the type of code we can expect in the function.

### Step 3: Disassembly

At this point, we provided rev.ng enough information to be able to show us the disassembly of our program.
Let's produce the [`disassemble` artifact](../../references/artifacts.md#disassemble-artifact) using [`revng-artifact`](../../references/cli/revng-artifact.md).

```bash
$ revng artifact disassemble sum --model model.yml | revng ptml
0x400000:Code_x86_64.asm.tar.gz: |-
  function_0x400000_Code_x86_64:
    400000:    48 01 f7    add rdi, rsi
    400003:    48 89 f8    mov rax, rdi
    400006:    c3          ret
```

The output of the `revng artifact disassemble` command is a `tar.gz` composed by one file for each input function. Each file is an assembly listing decorated using [PTML](../../references/ptml.md).
<br />`revng ptml` strips away all this details and outputs a YAML dictionary with one entry for each disassembled function.

For further information on the file types emitted by `revng artifact`, see the [MIME types documentation](../../references/mime-types.md).

### Step 4: Defining a function prototype

In order to decompile the function, we need to provide a function prototype.

By looking at the code above, we can see that the registers `rdi` and `rsi` are read at the entry of the function and the result of the computation is stored in `rax`: this function very much looks like a function taking two arguments and returning an integer in the x86-64 SystemV ABI!

Let's create such prototype then.

Specifically, we want to define the prototype for something that in C would look like the following:

```c
uint64_t sum(uint64_t rdi, uint64_t rsi);
```
In the model type system, it looks like this:

```yaml title="model.yml"
TypeDefinitions:
  - Kind: CABIFunctionDefinition
    ABI: SystemV_x86_64
    ID: 0
    Arguments:
      - Index: 0
        Type:
          Kind: PrimitiveType
          PrimitiveKind: Unsigned
          Size: 8
      - Index: 1
        Type:
          Kind: PrimitiveType
          PrimitiveKind: Unsigned
          Size: 8
    ReturnType:
      Kind: PrimitiveType
      PrimitiveKind: Unsigned
      Size: 8
```

OK, there's a lot here. Let's go through it line by line:

* [`TypeDefinitions`](../../references/model.md#Binary.TypeDefinitions): begin the part of the model containing type information;
* [`Kind: CABIFunctionDefinition`](../../references/model.md#cabifunctiondefinition): we're defining a type, specifically a C prototype associated to an ABI;
* `ABI: SystemV_x86_64`: the chosen ABI is the x86-64 SystemV one;
* `ID: ...`: a unique identifier. Every type definition must have one. Our tools usually use progressive ones, but that's not necessary: as long as there are no collisions, any integer works.
* `Arguments`: we have two arguments (index `0` and index `1`):
  * `Type`: the type of an argument:
    * `Kind: PrimitiveType`: the "kind" of a type. Supported values include primitives, pointers, arrays and defined types;
    * `PrimitiveKind: Unsigned`: the "kind" of a primitive type (like signed, unsigned, float, and so on);
    * `Size: 8`: the size of a primitive type;
* `ReturnType`: similar to an argument's size;

At this point, we can associate the function prototype with the previously defined function:

```diff
--- a/model.yml
+++ b/model.yml
@@ -10,6 +10,9 @@
     IsExecutable: true
 Functions:
   - Entry: "0x400000:Code_x86_64"
+    Prototype:
+      Kind: DefinedType
+      Definition: "/TypeDefinitions/0-CABIFunctionDefinition"
 TypeDefinitions:
   - Kind: CABIFunctionDefinition
     ABI: SystemV_x86_64
```

Basically, this specifies that the function type we created above is the prototype of this function.

### Step 5: Decompiling

At this point, we have all the information we need to successfully decompile our example program.
To do so, we can ask rev.ng to produce the [`decompile` artifact](../../references/artifacts.md#decompile-artifact):

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
@@ -11,8 +11,9 @@ Segments:
 Functions:
   - Entry: "0x400000:Code_x86_64"
     Prototype:
       Kind: DefinedType
       Definition: "/TypeDefinitions/0-CABIFunctionDefinition"
+    Name: Sum
 TypeDefinitions:
   - Kind: CABIFunctionDefinition
     ABI: SystemV_x86_64
```

Almost everything in the model can have a name. Let's add a name to the function arguments:

```diff
--- a/model.yml
+++ b/model.yml
@@ -119,18 +120,20 @@ TypeDefinitions:
   - Kind: CABIFunctionDefinition
     ABI: SystemV_x86_64
     ID: 0
     Arguments:
       - Index: 0
         Type:
           Kind: PrimitiveType
           PrimitiveKind: Unsigned
           Size: 8
+        Name: FirstAddend
       - Index: 1
         Type:
           Kind: PrimitiveType
           PrimitiveKind: Unsigned
           Size: 8
+        Name: SecondAddend
     ReturnType:
       Kind: PrimitiveType
       PrimitiveKind: Unsigned
       Size: 8
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
