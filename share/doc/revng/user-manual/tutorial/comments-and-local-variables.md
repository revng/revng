In the [previous](model-from-scratch.md) [tutorials](running-initial-auto-analysis.md) we gave names to functions, arguments and types.
Each of those lives at a stable place in the model (a function is keyed by its entry address, a type by its ID), so an edit can simply refer to it by key.

Inside the body of a function there are two more things an analyst usually wants to do:

* attach a **comment** to a statement;
* give a **local variable** a name and a type.

However a statement cannot be easily identify in a robust way: its line number might easily change in the face of a minor change in the model.
The situation for local variables is similar. This is unless they are on the stack, then they are represented a `struct` we can easily edit.

Therefore, we need a more robust way to address them. Specifically, rev.ng addresses a statement by the addresses of the instructions it involves and local variables by the address of the instructions that read/write them.

Those addresses can be collected by looking at the PTML of the decopmiled code but, you rarely have to look for them by hand: the `edit-c-body` analysis reads them for you, so you can annotate a function just by editing its C.

Let's see how it works on a small program:

```c title="calc.c"
#include <stdint.h>

__attribute__((noinline)) int64_t raw(int64_t key) {
  return key * 6364136223846793005ULL;
}

int64_t resolve(int64_t key, int64_t salt) {
  int64_t value = 0;
  if (key != salt)
    value = raw(key) ^ key;
  return value;
}

int main(int argc, char **argv) {
  return (int) resolve(argc, 0);
}
```

Let's compile it and create a project, just like on the [previous page](running-initial-auto-analysis.md):

```bash
$ gcc calc.c -o calc -O1 -fno-stack-protector
$ revng2 project init calc
```

Here is the decompiled `resolve`:

```bash
$ revng2 project artifact emit-c resolve | revng ptml
_ABI(SystemV_x86_64)
generic64_t resolve(generic64_t argument_0, generic64_t argument_1) {
  generic64_t var_0 = 0UL;
  if (argument_0 != argument_1)
    var_0 = raw(argument_0) ^ argument_0;
  return var_0;
}
```

rev.ng recovered the local variable but, lacking any better information, called it `var_0` and gave it the neutral type `generic64_t` (an untyped 64-bit value).

### Editing the body in C

The quickest way to attach these annotations is to edit the decompiled code and let rev.ng import your changes.
The [`edit-c-body` analysis](../../references/analyses.md#edit-c-body-analysis) takes the address of a function and a copy of its C, compares it statement by statement against the code rev.ng emitted, and records the annotations it finds.

It reacts to exactly three kinds of edit, all written as comments:

* a plain comment on the line before a statement becomes a comment attached to that statement;
* a `// RENAME: <name>` comment before a local variable's declaration renames the variable, and before a goto label renames the label;
* a `// RETYPE: <type>` comment before a local variable's declaration changes its type.

Each of these must be on its own line, before the statement it refers to.
A comment placed at the end of a line, after the code, is ignored.

Everything else has to stay as it was: `edit-c-body` does not change the structure of the code, rename functions or edit type definitions (as explained in [Editing types in C](editing-types-in-c.md), that is what [`edit-c-type`](editing-types-in-c.md) is for).

Let's take the C we just emitted and annotate it, commenting the `if` and giving the local variable a name and a type:

```c title="resolve.c"
_ABI(SystemV_x86_64)
generic64_t resolve(generic64_t argument_0, generic64_t argument_1) {
  // RENAME: value
  // RETYPE: int64_t
  generic64_t var_0 = 0UL;
  // only mix the key when it differs from the salt
  if (argument_0 != argument_1)
    var_0 = raw(argument_0) ^ argument_0;
  return var_0;
}
```

We hand this back to `edit-c-body`, together with the address of `resolve`, `0x400828`.
The configuration is a small YAML file with the function address and the edited C, which we assemble (indenting the code two spaces under the `CCode:` block) and pass to `-c`.
We copy the model aside first, so a `diff` afterwards shows exactly what the analysis wrote:

```{bash ignore="^(---|\+\+\+|@@)|TypeDefinitions/[0-9]"}
$ cat > edit.yml << EOF
Function: 0x400828:Code_x86_64
CCode: |
  _ABI(SystemV_x86_64)
  generic64_t resolve(generic64_t argument_0, generic64_t argument_1) {
    // RENAME: value
    // RETYPE: int64_t
    generic64_t var_0 = 0UL;
    // only mix the key when it differs from the salt
    if (argument_0 != argument_1)
      var_0 = raw(argument_0) ^ argument_0;
    return var_0;
  }
EOF
$ cp revng.yml before-edit.yml
$ revng2 project analyze edit-c-body -o /dev/null -c edit.yml
$ diff -u before-edit.yml revng.yml || true
--- before-edit.yml
+++ revng.yml
@@ -159,6 +159,19 @@
     Prototype:
       Kind: DefinedType
       Definition: "/TypeDefinitions/69-CABIFunctionDefinition"
+    Comments:
+      - Index: 0
+        Location:
+          - "0x400830:Code_x86_64"
+        Body: only mix the key when it differs from the salt
+    LocalVariables:
+      - Name: value
+        Type:
+          Kind: PrimitiveType
+          PrimitiveKind: Signed
+          Size: 8
+        Location:
+          - "0x400840:Code_x86_64"
   - Entry: "0x400841:Code_x86_64"
     Name: main
     Prototype:
```

The comment, the name and the type are now part of the model, so they show up in the decompiled code:

```bash
$ revng2 project artifact emit-c resolve | revng ptml | grep -E 'value =|mix the key'
  int64_t value = 0L;
  // only mix the key when it differs from the salt
    value = raw(argument_0) ^ argument_0;
```

### Under the hood

`edit-c-body` is a convenience: it figures out where each annotation belongs and writes it into the model.
Knowing what it produces is useful when you script rev.ng and want to write these entries yourself.

Both kinds of annotation are located the same way, through the *addresses of the machine instructions they refer to*:

* for a **statement**, the addresses of the instructions that make it up;
* for a **local variable**, the addresses of the instructions that use it.

The decompiled code emitted by [`emit-c`](../../references/artifacts.md#emit-c-artifact) is [PTML](../../references/ptml.md), and every fragment of C carries the address of the machine instruction it was lifted from, in a `data-action-context-location` attribute.
For example, the condition of the `if` is emitted like this (simplified, syntax-highlighting markup removed):

```xml
<span data-action-context-location="/instruction/.../0x400830:Code_x86_64"
      data-allowed-actions="codeSwitch,comment">argument_0 != argument_1</span>
```

so the `if` statement is identified by the address `0x400830`, and `var_0`, used only by its assignment, by the address `0x400840`.

The comment we added is a [`StatementComment`](../../references/model.md#statementcomment) in the function's [`Comments`](../../references/model.md#Function.Comments), whose `Location` is the addresses of the statement it attaches to and whose `Body` is the text:

```yaml
Comments:
  - Index:    0
    Location:
      - "0x400830:Code_x86_64"
    Body:     "only mix the key when it differs from the salt"
```

The name and type are a [`LocalVariable`](../../references/model.md#localvariable) in the function's [`LocalVariables`](../../references/model.md#Function.LocalVariables): `Name` renames it, `Type` sets its type (leave it out to keep the inferred one), and `Location` identifies *which* variable through the addresses of the instructions that use it:

```yaml
LocalVariables:
  - Name:     value
    Type:
      Kind:          PrimitiveType
      PrimitiveKind: Signed
      Size:          8
    Location:
      - "0x400840:Code_x86_64"
```

Writing these entries into `revng.yml` directly, instead of going through `edit-c-body`, has exactly the same effect; it is what you would do when scripting rev.ng without the C round-trip.

Goto labels work the same way as local-variable names: a `// RENAME:` before a goto label writes a [`GotoLabel`](../../references/model.md#gotolabel) entry into the function's [`GotoLabels`](../../references/model.md#Function.GotoLabels), located by the address set of the statements it labels (a `GotoLabel` has no `Type`, so `// RETYPE:` does not apply).
