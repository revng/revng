## The Model

TODO: link schema
TODO: mention verify
TODO: link reference
TODO: make simple file with two functions and create a minimal model

The *model* is a YAML document that contains information about the binary that the user might want to customize.
You can think about the model as a sort of interchange format for reverse engineering: it doesn't contain any information that is strictly specific to rev.ng, it describes the binary in an abstract form that can be reused by third party tools too.

It includes:

* **Loading Information**: a description of which parts of the input file should be loaded and at which address; TODO: link model::Segment
* **The list of functions**: for each function we have an entry address, the layout of its stack frame (where local variables reside), its prototype and more;
* **The type system**: includes all the `struct`s, `enum`s and function prototypes for the binary.

On the other hand, it *does not* include the **control-flow graph**: the control-flow graph is rather complicated to maintain for the user.
For instance, if a user wants to mark a function as `noreturn`, in order to obtain a valid model it'd need to update the CFG of all of its callers.
However, the user might be sometimes interested in providing extra information about the control-flow of a program. For instance, suppose that rev.ng has not been able to enumerate all possible destinations of an indirect jump. Through manual analysis, the user could identify the complete list of targets. In the future, we plan to introduce a way to specify *extra edges* in the control flow graph through the model, but pushing on the user the responsiblity of building all of the control-flow graph, is not beneficial.

The model must be a valid YAML document, but that's not enough.
In order to be consumed by rev.ng, a model needs to be valid.
You can check if a model is valid as follows:

<!-- NO-TEST -->
```bash
$ revng model opt mymodel.yml -verify
```

If the command succeeds, the tool will print the model again.
More on `revng model opt` later.

The model has a couple of different users:

* **The user**: the user, as part of his analysis activities, makes changes to the model, e.g., renaming a function, adding arguments, introducing new functions, creating a new data structures.
  The user interacts with the model either through the UI or manually or through scripting wrappers that enables easy maninpulation of the model.
  We currently provide wrappers for Python and TypeScript. TODO: link
* **Importers/exporters**: the model is designed to be a sort of interchange format and, as such, it's not strictly rev.ng-specific.
  This means that's possible to implement importers from other formats. For example, we provide out of the box importers for the most common binary formats (ELF, PE/COFF, MachO), debug info formats (DWARF and PDB) and other static analysis tools such as IDA Pro's `.idb`s. TODO: links
  Users can easily implement new importers by simplying manipulating the model (which is a plain YAML file) in their favorite scripting language.
  In the future, we also plan to implement *exporters*, e.g., produce DWARF debug info that enable advanced debugging using information available in the model (e.g., arguments, data structures...) using off-the-shelf debuggers on the target binary.
* **Pipes**: rev.ng provides a set of *pipelines*, composed by *pipes*, that produce *artifacts*. Most pipes, read the model for various reasons.
  For example, there's a pipeline responsible for generating a C header file declaring all the functions in the binary.
  In order to produce this artifact, the pipeline inspects the model and nothing else.
  Other pipelines might inspect the model, other previously generated artifacts and the input binary itself.
  For a complete list of all the pipelines available in rev.ng see TODO.
  For a complete list of all the pipes available in rev.ng see TODO.
  TODO: how to run
* **Analyses**: rev.ng also provides a set of **analyses** that can automatically recover high-level information by analyzing the binary or artifacts produced by a pipeline.
  The final goal of an analysis is to make changes to the model.
  For instance, rev.ng provides an analysis that automatically detects the arguments of a function: when you run such an analysis, it will go through the list of functions in the model, analyze each function and enriching each function that did not initially have a prototype, with arguments, return values and so on.
  For a complete list of all the analyses available in rev.ng see TODO.
  TODO: how to run

### A model, from scratch

TODO: state + example we have `revng model import`

An empty model is a valid model:

```bash
$ revng model opt /dev/null -verify
---
{}
...
```

`revng model opt` is a tool to *optimize* the model. It is inspired by LLVM's `opt`, the LLVM IR optimizer.
It works as follows:

* it takes a model in input, either as a YAML file or embedded in an LLVM IR file, but we don't care about this second option for now;
* it runs one or more *passes* o the model;
* it prints the modified model to standard output;

`/dev/null` is the input file (an empty file).
The list of passes to run is composed by `-verify` only, which doesn't make any change, it just checks that the model is valid.

OK, let's get started and try to decompile a very simple program.

```bash
$ printf '\x48\x01\xf7\x48\x89\xf8\xc3' > sum
```

If you disassemble it as raw bytes you get the following:

```bash
$ objdump -D -Mintel,x86-64 -b binary -m i386:x86-64 sum

sum:     file format binary


Disassembly of section .data:

0000000000000000 <.data>:
   0:   48 01 f7                add    rdi,rsi
   3:   48 89 f8                mov    rax,rdi
   6:   c3                      ret
```

Let's now create a simple model that enables us to decompile this simple function.

### Step 1: Loading

The first thing rev.ng needs to know is the architecture of the program:

```yaml title="model.yml"
Architecture:    x86_64
DefaultABI:      SystemV_x86_64
```

Then we need to describe how to load the program.
Let's pretend we want to load this code at address `0x400000`:

```yaml title="model.yml"
Segments:
  - StartAddress:    "0x400000:Generic64"
    VirtualSize:     7
    StartOffset:     0
    FileSize:        7
    IsReadable:      true
    IsWriteable:     false
    IsExecutable:    true
```

The piece of model above tells rev.ng to take 7 bytes from the file and load them at address `0x400000` as `+rx` data (i.e., code).

### Step 2: Function list

Most parts of rev.ng work on a function basis (e.g., we decompile one function at a time).
Let's create an entry in the function list:

```yaml title="model.yml"
Functions:
  - Entry:           "0x400000:Code_x86_64"
```

At a minimum, a function is identified by its entry address.
Note how here the type of the `MetaAddress` is not `Generic64` but `Code_x86_64`, to indicate the type of code we can expect in the function.

TODO: not on multi-ISA programs

### Interlude: disassembly

At this point, we provided rev.ng enough information to be able to show us the disassembly of our program.

```bash
$ revng artifact YieldAssembly sum -m model.yml | revng ptml cat | grep -vE '^\s*$'
0x400000:Code_x86_64:
  function_0x400000_Code_x86_64:
    add rdi, rsi
    mov rax, rdi
    ret
```

### Step 3: Defining a function prototype

TODO: in real world we do this using `DetectABI`

In order to decompile the function, we need to provide a function prototype.

By looking at the code above, we can see that the registers `rdi` and `rsi` are read at the entry of the function and the result of the computation is stored in `rax`: this function very much looks like a function taking two arguments and returning an integer in the x86-64 SystemV ABI!

Let's create such prototype then.

First, we create a *primitive type* for an `uint64_t`:

```yaml title="model.yml"
Types:
  - Kind:            PrimitiveType
    ID:              1288
    PrimitiveKind:   Unsigned
    Size:            8
```

Here we are defining a *primitive* type (such as `void`, an integral type or a floating-point type) with size 8 bytes (64 bits) and *kind* `Unsigned`. Basically, an `uint64_t`.

TODO: note about type IDs

Now, we can define the prototype for something that in C would look like the following:

```c
uint64_t sum(uint64_t rdi, uint64_t rsi);
```

In the model type system, it looks like this:

```yaml title="model.yml"
  - Kind:            CABIFunctionType
    ABI:             SystemV_x86_64
    ID:              67426649700812539
    Arguments:
      - Index:           0
        Type:
          UnqualifiedType: "/Types/PrimitiveType-1288"
      - Index:           1
        Type:
          UnqualifiedType: "/Types/PrimitiveType-1288"
    ReturnType:
      UnqualifiedType: "/Types/PrimitiveType-1288"
```

OK, there's a lot here. Let's go through it:

* `Kind: CABIFunctionType`: we're defining a type, specifically a C prototype associated to an ABI;
* `ABI: SystemV_x86_64`: the chosen ABI is the x86-64 SystemV one; TODO: link
* `ID: ...`: a random unique identifier;
* `Arguments`: we have two arguments (index `0` and index `1`);
  * `Type`: a qualified type, i.e., a type plus (optionally) one or more qualifiers such as `const`, pointer (`*`) and so on;
    * `UnqualifiedType: "/Types/PrimitiveType-1288"`: a reference to the actual, unqualified, type; in this case, it's a reference to the *primitive type* with ID 1288, i.e., the `uint64_t` defined above;
* `ReturnType`: again, a reference to `uint64_t`;

At this point, we can associate the function prototype with the previously defined function:

```diff
--- a/model.yml
+++ b/model.yml
@@ -10,6 +10,7 @@
     IsExecutable:    true
 Functions:
   - Entry:           "0x400000:Code_x86_64"
+    Prototype:       "/Types/CABIFunctionType-67426649700812539"
 Types:
   - Kind:            PrimitiveType
     ID:              1288
```

Basically, we added a reference to the prototype we created above to our function definition.

### Decompiling

At this point, we have all the information we need to successfully decompile our example program:


=== "CLI"

    ```c
    revng artifact Disassembly input
    ```

=== "Python"

    ```python
    fetch_artifact("Disassembly")
    ```

=== "TypeScript"

    ```typescript
    fetch_artifact("Disassembly")
    ```

=== "GraphQL"

    ```bash
    curl 'https://' --data='{
      Lift {
        Disassembly
      }
    }'
    ```

```bash
$ revng artifact DecompileToCInYAML sum -m model.yml
```

### Step 4: Naming

One of the main activities of a reverse engineer is giving things a name.
Let's try to give a name to our function:

TODO

Almost everything in the model can have a name:

TODO

### The final model

Here, you can find the final model. Take it and play around with it:

```yaml
```
