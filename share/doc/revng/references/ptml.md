PTML stands for Plain Text Markup Language.
It's an XML format aimed at enriching a plain text file with meta information useful for syntax highlighting, navigation and have anchors to trigger actions.

## Basic usage from the terminal

When dealing with PTML on the terminal you either want to strip all the markup:

```bash
$ echo '<span data-token="asm.register">rax</span>' | orc shell revng ptml
rax
```

You can add `--color` to render the syntax highlighting using terminal coloring.

## Characteristics

* Whitespaces are very relevant. PTML is designed to gracefully degrade into plain text by simply stripping all the XML elements.
* It can be easily embedded into HTML. All the elements are either `<div>` or `<span>`.
* It is used to represent in a uniform way source code in different languages (the *underlying language*).
  For instance, we use it to represent C and various flavor of assembly code.
  If a text editor or a viewer supports PTML, it does not need any further understanding of the underlying language.
* Contains metadata for navigation/highlighting in XML elements/attributes.
* It's easy to emit: just power up your string concatenation with tags.
  An equivalent approach would be to have a separate place for metadata, associated to offset ranges in the original document.
  This approach would be significantly more difficult to emit.
* Potential frontends:
    * Web browsers, using CSS for styling and JS for interactivity.
    * Rich text editors (e.g, [Monaco](https://microsoft.github.io/monaco-editor/))
    * Terminals. `revng ptml` can turn a PTML document into syntax highlighted output on a terminal.
* PTML is *not* designed to have frontends add inline text upon rendering.
  In this way, its bounding box can be easily computed by the emitter (e.g., for graph layout purposes).
  Dropping text is possible, but discouraged.
* Attributes are single-valued, unless stated otherwise.
  In case an attribute needs to represent multiple values, it will do so by concatenating the elements with `,` (U+002C).

Attributes defined in PTML can be organize in the following categories:

1. Syntax attributes: used for syntax highlighting purposes.
2. Navigation attributes: used for navigation purposes (e.g., going from a function call to its definition).
3. Action attributes: used to provide a context for certain *actions* (e.g., `rename`) on a certain portion of text.

## Syntax attributes

### `data-scope`

```xml title="Example"
<div data-scope="asm.basic-block">
```

States that the enclosed portion of text is of a specific type.
`data-scope`s can be nested, for example:

* `<div data-scope="asm.basic-block">` can contain `<div data-scope="asm.instruction">`.
* `<div data-scope="c.scope">` can contain another `<div data-scope="c.scope">`.

The nestability of scopes is dependent on the underlying language.
The main use case for this attribute is folding ranges (e.g., in [VSCode](https://code.visualstudio.com/docs/editor/codebasics#_folding)).

### `data-token`

```xml title="Example"
<span data-token="asm.register">
```

States the type of the enclosed text.
`data-token`s are mutually exclusive, that is, each part of the text can be enclosed in at most one of these attributes.
The main use case for this attribute is syntax highlighting.

### Language references

#### Common

Some `data-token`s are common in all programming languages and, as such, are shared:

* `indentation`: used to indicate that the whitespace inside is indentation
* `comment`: the text inside is a comment (including the language-specific comment marker)

#### ASM

The assembly language defines the following `data-scope`s:

* `asm.function`
* `asm.basic-block`
* `asm.instruction`

```asm title="Example"
printf_core_:                        #                     -    -
  push r15                           #              <-|    |    |
  mov r15, rdi                       #              <-|    |    |
                                     # instructions --|    -    |
                                     #       basic-block --|    |
                                     #                          |
basic_block_at_0x402ac5_Code_x86_64: #                     -    |
  cmp dword ptr [rsp + 0x4], 0x0     #              <-|    |    |
  js 0x402af2                        #              <-|    |    |
                                     # instructions --|    -    |
                                     #       basic-block --|    |
                                     #                          |
basic_block_at_0x402af4_Code_x86_64: #                     -    |
  cmp byte ptr [rbp], 0x0            #              <-|    |    |
  je 0x40321e                        #              <-|    |    |
                                     # instructions --|    -    |
                                     #       basic-block --|    -
                                     #               function --|
```

The assembly language defines the following `data-token`s:

* `asm.label`: the label of a basic block.
* `asm.label-indicator`: terminator for an `asm.label` (usually `:`).
* `asm.mnemonic`: the mnemonic of an assembly instruction (e.g., `mov`).
* `asm.mnemonic-prefix`, `asm.mnemonic-suffix`: in case a mnemonic can be split into a prefix and a suffix, the two respective parts (e.g., the `ne` in `jne`).
* `asm.immediate-value`: used to represent integers.
* `asm.memory-operand`: used to indicate memory operands.
* `asm.register`: the name of a register.
* `asm.helper`: a macro-like function for specifying some information in a more human-readable fashion (e.g., `offset_to` to get the offset of a global).

```asm title="Example"
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv--------- asm.label
#                                    v-------- asm.label-indicator
  basic_block_at_0x402ac5_Code_x86_64:
#   vvv--------------------------------------- asm.mnemonic
#                  vvv---------vvv------------ asm.register
#                 v---------v----------------- asm.memory-operand
#                        vvv------------------ asm.immediate-value
    add dword ptr [rsp + 0x4], ebx

#   v----------------------------------------- asm.mnemonic
#    vv--------------------------------------- asm.mnemonic-suffix
#       vvvvvvvv------------------------------ asm.immediate-value
    jge 0x402af2
#            vvvvvvvvv------------------------ asm.helper
    mov rax, offset_to(some_global)
```

#### C

The C language defines the following `data-scope`s:

* `c.function`: a function, including its prototype.
* `c.function_body`: the body of a function, i.e., the text just after the function declaration and `{` to just before the last `}`.
* `c.scope`: any scope inside a function (e.g., `if`, `switch`).
* `c.struct`: the body of a `struct`.
* `c.union`: the body of a `union`.

Additionally, there are:

* `c.type_declarations_list`: a group of type declarations.
* `c.function_declarations_list`: a group of function declarations.
* `c.dynamic_function_declarations_list`: a group of dynamic function declarations.
* `c.segment_declarations_list`: a group of segment declarations.

```c title="Example"
struct example_struct {/* begin c.struct */
    int x;
    int y;
/* end c.struct */};

union example_union {/* begin c.union */
    float value;
    double bigvalue;
/* end c.union */};

/* begin c.function */ int foo(); /* end c.function */

/* begin c.function */ int foo() {/* begin c.function-body */
    int i = 0;
    for(i; i < 8; i++) {/* begin c.scope */
        if(i % 2 == 0) {/* begin c.scope */
            baz(i);
        /* end c.scope */}
    /* end c.scope */}
    return i;
/* end c.function-body */}/* end c.function */
```

The C language defines the following `data-token`s:

* `c.function`: the name of a function.
* `c.type`: a type name (e.g. `uint8_t` or `struct_1000101`).
* `c.operator`: any unary or binary operator, e.g., `*`, `&`, `->`, `+`, `-`, `&`, `>>`, `>`, `!=`.
* `c.function_parameter`: a function parameter.
* `c.variable`: a local variable.
* `c.field`: the name of a member of a `struct` or a `union`.
* `c.constant`: a constant value, e.g. `1` or `0xDEADBEEF`.
* `c.keyword`: a C reserved keyword, e.g., `const` and `volatile`.
* `c.directive`: any preprocessor directive, e.g., `#include`, `#ifdef`.
* `c.string_literal`: a string literal, e.g., `"DEADBEEF"`.

## Navigation attributes

PTML supports defining and referencing *locations*.

### Locations

Each *location* represent an abstract concept, such as a function or a byte range, which can be used for navigation, e.g., going from a function call to the definition of the called function.
Each location can have zero or more parameters.

We currently implemented the following locations:

* Generic locations
    * `/binary`
    * `/function/<Function_MetaAddress>`
    * `/dynamic-function/<Name_string>`
    * `/type/<Type_Kind>-<Type_ID>`
    * `/segment/<StartAddress_MetaAddress>-<VirtualSize_uint64>`
    * `/struct-field/<Type_Kind>-<Type_ID>/<Offset_uint64>`
    * `/union-field/<Type_Kind>-<Type_ID>/<Index_uint64>`
    * `/enum-entry/<Type_Kind>-<Type_ID>/<Index_uint64>`
    * `/raw-argument/<Type_Kind>-<Type_ID>/<Register>`
    * `/cabi-argument/<Type_Kind>-<Type_ID>/<Index_uint64>`
    * `/return-value/<Type_Kind>-<Type_ID>`
    * `/return-register/<Type_Kind>-<Type_ID>/<Register>`
* Byte-ranges related
    * `/raw-byte/<Start_MetaAddress>`
    * `/raw-byte-range/<Start_MetaAddress>/<End_MetaAddress>`
* Assembly-related:
    * `/basic-block/<Function_MetaAddress>/<BasicBlock_MetaAddress>`
    * `/instruction/<Function_MetaAddress>/<BasicBlock_MetaAddress>/<Instruction_MetaAddress>`
* C-related:
    * `/local-variable/<Function_MetaAddress>/<Name_string>`
    * `/helper-function/<Name_string>`
    * `/helper-struct-type/<Name_string>`
    * `/helper-structs-field/<Name_string>/<FieldName_string>`
    * `/dynamic-function-argument/<Name_string>/<Name_string>`
    * `/artificial-struct/<Type_Kind>-<Type_ID>`

When interacting with revng, there is a description file, which, among other information, states what locations each Kind  *defines*, e.g.:

* `DecompiledC` provides:
    * `/function/<MA>`
* `YieldAssembly` provides:
    * `/function/<MA>`
    * `/basic-block/<MA>/*`
    * `/instruction/<MA>/*/*`
* `ModelHeader` provides:
    * `/types/*`

Some locations directly map to parts of the model, e.g.:

* `/types/$TYPE_ID` -> `/Types/$TYPE_ID`
* `/function/$FUNCTION_ADDRESS` -> `/Functions/$FUNCTION_ADDRESS`
* `/basic-block/...` -> `null`
* `/instruction/...` -> `null`

### Attributes

The following PTML attributes are used to provide navigation via location string:

* `data-location-definition`: represents the *definition* of a location.

    ```xml
    <div data-location-definition="/function/0x1000">...</div>
    <div data-location-definition="/basic-block/0x1000/0x1004">...</div>
    <div data-location-definition="/instruction/0x1000/0x1004/0x1006">...</div>
    ```
    Represents the destination to navigate to starting from a `data-location-reference` (see below).

* `data-location-references`:
  Represents one or more references to a location. Can be multivalued.
  ```xml
  <span data-location-references="/basic-block/0x1000/0x1004,
                                  /basic-block/0x1000/0x1008">
    jmp rax
  </span>
  struct <span data-location-references="/types/0x1123412)">mtype</span> {
    // ...
  }
  ```

## Action attributes

PTML defines a set of *actions*:

* `rename`: rename the current object.
* `comment`: edit the comment of the current object.
* `codeSwitch`: jump to an alternative representation of the current object (e.g. ASM <-> C).
* `editType`: edit the type associated to the current object (e.g., for a function, its prototype).

Each of these actions requires knowledge of how the action is executed correctly, its implementation is optional and left at the discretion of each PTML viewer.
Each location supports a subset of the above actions:

* `function`: supports `rename`, `comment` and `editType`.
* `instruction`: supports  `codeSwitch`.
* `type`: supports `rename`, `comment` and `editType`.
* `struct-field`: supports `rename` and `comment`.
* `union-field`: supports `rename` and `comment`.
* `enum-entry`: supports `rename` and `comment`.
* `cabi-argument`: supports `rename` and `comment`.
* `raw-argument`: supports `rename` and `comment`.
* `return-value`: supports `comment`.
* `return-register`: supports `rename` and `comment`.
* `segment`: supports `rename`, `comment` and `editType`.
* `dynamic-function`: supports `rename`, `comment` and `editType`.

An action is defined by the following attributes:

* `data-action-context-location`: indicates that the contained snippet has the specified context and allows the PTML viewer to activate the supported actions. Nested element can specify different context locations; in this case the viewer should pick the innermost one.
* `data-allowed-actions`: in some cases, the set of possible actions needs to be restricted to a subset of all possible actions, in this case this attribute is used.
