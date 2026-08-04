In the [previous tutorial](running-initial-auto-analysis.md) the initial auto-analysis recovered functions and types for us.
For data structures it can only go so far: it discovers that some memory is accessed as an aggregate and it recovers the fields it sees being used, but it has no way to know what those fields *mean*.
The result is a struct with a generated name and fields called `offset_0`, `offset_8`, and so on.

Giving such a type a proper name, naming its fields and refining their types is something up to the user.
You could edit the model YAML by hand, as we did for functions, but describing a C type in YAML is verbose and error prone.

For this reason, rev.ng offers a more convenient route: it emits the type as a small C header, you edit that header, and it reads your changes back into the model.
The two halves of this round-trip are the [`emit-single-type-definition` artifact](../../references/artifacts.md#emit-single-type-definition-artifact) and the [`edit-c-type` analysis](../../references/analyses.md#edit-c-type-analysis).

Let's see it on a program whose struct the compiler has to pad:

```c title="account.c"
#include <stdint.h>

struct account {
  int32_t id;
  int64_t balance;
  int32_t flags;
};

__attribute__((noinline)) int64_t summary(struct account *a) {
  return a->balance + a->id + a->flags;
}

int main(int argc, char **argv) {
  struct account a = { argc, argc * 100, argc + 1 };
  return (int) summary(&a);
}
```

`balance` is a 64-bit integer and must be 8-byte aligned, so the compiler leaves a four-byte hole between `id` and it.

We compile it *without* debug information, so that rev.ng has to recover the struct on its own, and create a project as usual:

```bash
$ gcc account.c -o account -O1 -fno-stack-protector
$ revng2 project init account
```

Here is the decompiled `summary`:

```{bash ignore="struct_[0-9]+"}
$ revng2 project artifact emit-c summary | revng ptml
_ABI(SystemV_x86_64)
generic64_t summary(struct_58 *argument_0) {
  return argument_0->offset_8 + (int32_t) argument_0->offset_0 + (int32_t) argument_0->offset_16;
}
```

rev.ng, thanks to the [`analyze-data-layout` analysis](../../references/analyses.md#analyze-data-layout-analysis), figured out that `summary` receives a pointer to a `struct` whose fields sit at offsets 0, 8 and 16, but it had no names for any of it: the `struct` is called `struct_58` and its fields `offset_0`, `offset_8` and `offset_16`, after their byte offsets.

### Getting the C for a type

`struct_58` is a generated name; the number is an internal id that changes from run to run, so we read it back out of the decompiled code rather than hard-coding it:

```bash
$ TYPE_ID=$(revng2 project artifact emit-c summary | revng ptml | grep -oE 'struct_[0-9]+' | sort -u | tr -dc 0-9)
```

The [`emit-single-type-definition`](../../references/artifacts.md#emit-single-type-definition-artifact) artifact prints a single type as a C header:

```{bash ignore="struct_[0-9]+"}
$ revng2 project artifact emit-single-type-definition /type-definition/${TYPE_ID}-StructDefinition
struct _PACKED _SIZE(40) struct_58 {
  generic32_t offset_0 _STARTS_AT(0);
  generic64_t offset_8 _STARTS_AT(8);
  generic32_t offset_16 _STARTS_AT(16);
};
```

This is deliberately *not* the same C you get from the regular decompiled header, the [`emit-type-and-global-header` artifact](../../references/artifacts.md#emit-type-and-global-header-artifact).
That one is meant to be recompilable by an ordinary C compiler, so it spells the padding out as explicit fields:

```{bash ignore="struct_[0-9]+"}
$ revng2 project artifact emit-type-and-global-header | revng ptml | grep -A6 "struct_${TYPE_ID} {"
struct _PACKED _SIZE(40) struct_58 {
  generic32_t offset_0;
  uint8_t padding_at_4[4];
  generic64_t offset_8;
  generic32_t offset_16;
  uint8_t padding_at_20[20];
};
```

`emit-single-type-definition` drops those `padding_at_*` fields to stay easy to edit, but adds `_STARTS_AT(N)` to pin a field to byte offset `N`.

### Editing and importing the type

Now we rewrite the header the way we want the struct to look, giving it and its fields real names and concrete types while keeping the `_STARTS_AT` markers untouched.

```c title="account.h"
struct _PACKED _SIZE(40) account {
  int32_t id _STARTS_AT(0);
  int64_t balance _STARTS_AT(8);
  int32_t flags _STARTS_AT(16);
};
```

The [`edit-c-type` analysis](../../references/analyses.md#edit-c-type-analysis) reads its configuration from the file passed to `--configuration`: a small YAML document with two fields, `LocationToEdit`, the type to replace, and `CCode`, the edited header.
We write it out, indenting the header two spaces so it sits inside the `CCode:` block, and hand its path to the analysis:

```bash
$ cat > edit.yml << EOF
LocationToEdit: /type-definition/${TYPE_ID}-StructDefinition
CCode: |
  struct _PACKED _SIZE(40) account {
    int32_t id _STARTS_AT(0);
    int64_t balance _STARTS_AT(8);
    int32_t flags _STARTS_AT(16);
  };
EOF
$ revng2 project analyze edit-c-type -o /dev/null --configuration edit.yml
```

The names and types are now part of the model, so they show up everywhere the struct is used.
Re-emitting `summary` makes the edit concrete: it takes an `account *` and reads `balance`, `id` and `flags` by name instead of the anonymous offsets.

```bash
$ revng2 project artifact emit-c summary | revng ptml
_ABI(SystemV_x86_64)
generic64_t summary(account *argument_0) {
  return *(generic64_t *) &argument_0->balance + (int32_t) *(generic32_t *) &argument_0->id + (int32_t) *(generic32_t *) &argument_0->flags;
}
```

`edit-c-type` works the same way on any type definition: structs, unions, enums and typedefs, along with their fields and, for functions, their prototypes.
It only touches the type you point it at through `LocationToEdit`; to annotate the *body* of a function, giving names to local variables or commenting on statements, see the [next tutorial](comments-and-local-variables.md).
