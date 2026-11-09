The quickest way to see rev.ng at work is to hand it a program and ask for its decompiled code.

Consider the following simple program:

```c title="example.c"
#define WEAK __attribute__((weak))

WEAK int triple(int x) {
  return x * 3;
}

int main(int argc, char *argv[]) {
  return triple(argc);
}
```

Let's compile it:

```bash
$ gcc example.c -o example -O1
```

### Decompiling with `revng quick`

[`revng quick`](../../references/cli/revng-quick/index.md) takes a binary and produces an [artifact](../key-concepts/artifacts-and-analyses.md) from it in a single command.
The artifact we want is [`emit-c-as-single-file`](../../references/artifacts.md#emit-c-as-single-file-artifact), the decompiled C code of the whole program:

```bash
$ revng quick artifact emit-c-as-single-file example \
        | revng ptml \
        | grep -A2 -B1 ' triple(.*{$'
_ABI(SystemV_x86_64)
generic64_t triple(generic64_t argument_0) {
  return (generic32_t) argument_0 * 3U;
}
```

rev.ng emits [PTML](../../references/ptml.md), C code decorated with markup that the UI uses for highlighting and navigation: [`revng ptml`](../../references/cli/revng-ptml.md) strips the markup and leaves plain text.
The full output contains every function of the program, so we use `grep` to pick out `triple`.

A few things to note:

* `x` became `argument_0`: the binary does not record the names of the arguments, so rev.ng makes one up;
* `generic64_t` is a 64-bit value whose kind (integer, pointer and so on) rev.ng could not pin down.

Behind the scenes, `revng quick` loads the binary and runs the *initial auto-analysis*, a [list of analyses](../../references/pipeline.md#analysis-lists) that detect functions, their prototypes, data structures and more.
Everything they find ends up in the [model](../key-concepts/model.md), rev.ng's description of the program, from which the artifact is then produced.
Once the artifact is out, `revng quick` throws the model away.

### Working on a project with `revng project`

`revng quick` is fine for one-off decompilation: it analyses the input binary, emits the desired output and discards everything.
If you want to actually start a project and iterate on it (edit types, names and so on), you have to use [`revng project`](../../references/cli/revng-project/index.md).
Let's create a *project* with [`revng project init`](../../references/cli/revng-project/init.md):

```bash
$ revng project init example
```

This runs the same initial auto-analysis but, instead of throwing the model away, saves it in the current directory as `revng.yml`.
`revng.yml` is the model we mentioned above: it holds everything rev.ng knows about the program, starting with the list of functions, each identified by the [`MetaAddress`](../key-concepts/metaaddress.md) of its entry point:

```{bash ignore="0x[0-9a-f]+:Code_x86_64"}
$ grep -B1 'Name: triple$' revng.yml
  - Entry: "0x400819:Code_x86_64"
    Name: triple
```

[`revng project artifact`](../../references/cli/revng-project/artifact.md) produces artifacts from the project.
Unlike `revng quick`, it builds on the results of the previous commands, and it lets us ask for a single function, by name, from the [`emit-c` artifact](../../references/artifacts.md#emit-c-artifact):

```bash
$ revng project artifact emit-c triple | revng ptml
_ABI(SystemV_x86_64)
generic64_t triple(generic64_t argument_0) {
  return (generic32_t) argument_0 * 3U;
}
```

`revng project artifact` also caches its results, in `$XDG_CACHE_HOME/revng` (`~/.cache/revng` by default, see [`--cache-dir`](../../references/cli/revng-project/index.md)), so asking again for the same artifact is faster: it is not computed again.

Since the project persists, we can now record what we learn.
For instance, let's write down what `triple` does by adding a `Comment` to it in `revng.yml`:

```diff
--- a/revng.yml
+++ b/revng.yml
@@ -159,3 +159,4 @@
     Name: triple
+    Comment: Return three times its argument.
     Prototype:
       Kind: DefinedType
```

The next artifact we ask for reflects the change:

```bash
$ revng project artifact emit-c triple | revng ptml
/// Return three times its argument.
_ABI(SystemV_x86_64)
generic64_t triple(generic64_t argument_0) {
  return (generic32_t) argument_0 * 3U;
}
```

This is the everyday rev.ng workflow: analyses fill in the model, you refine it, and artifacts show the result.
Analyses are not limited to `init`: you can run them on the project at any time with [`revng project analyze`](../../references/cli/revng-project/analyze.md).

Even for a program this small, `revng.yml` is several hundred lines long.
To understand what goes into it, in the [next page](model-from-scratch.md) we will write a model by hand, from scratch.
