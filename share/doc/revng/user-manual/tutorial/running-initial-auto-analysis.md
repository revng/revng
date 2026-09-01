In [the previous page](model-from-scratch.md) we saw how we can build a model from scratch, by hand.

However, we don't want our users to write the model by hand.
Therefore, as mentioned before, we developed a series of analyses which can automatically populate the model for you when you open a new project.

Consider the following simple program:

```c title="example.c"
int main(int argc, char *argv[]) {
  return argc * 3;
}
```

Let's compile it:

```bash
$ gcc example.c -o example -O2
```

We run the [`parse-binary` analysis](../../references/analyses.md#parse-binary-analysis) using [`revng project analyze`](../../references/cli/revng-project/analyze.md) to automatically collect all the loading information available in the ELF headers:

```{bash ignore="^.*(VirtualSize|FileSize):.*[0-9]+$"}
$ mkdir project-dir
$ revng -C project-dir project init example --no-initial-auto-analysis
$ revng -C project-dir project analyze parse-binary -o /dev/null
$ grep -A5 'Segments:' project-dir/revng.yml
Segments:
  - Binary:          "/Binaries/0"
    StartAddress:    "0x400000:Generic64"
    VirtualSize:     1520
    FileSize:        1520
    IsReadable:      true
```

However, the typical workflow does not require the user to manually specify what analyses to run, but there's a set of predefined analyses that should be run on a new project, the *initial autoanalyses*.

```bash
$ revng -C project-dir project analyze initial-auto-analysis -o /dev/null
$ revng -C project-dir project artifact emit-c-as-single-file \
        | revng ptml \
        | grep -A2 -B1 -F ' main('
_ABI(SystemV_x86_64)
generic64_t main(generic64_t argument_0) {
  return argument_0 * 3UL & 0xFFFFFFFFUL;
}
```

The commands above are *stateful*, they build on top of each other storing intermediate results into the directory specified by the `-C` parameter.
The first command runs the set of initial autoanalyses of `revng` and the last one produces the decompiled code.

Alternatively, you can run the `initial-auto-analysis` *and* produce the artifact with a single command, without a persistent project, using [`revng quick`](../../references/cli/revng-quick/index.md):

```bash
$ revng quick artifact emit-c-as-single-file example \
        | revng ptml \
        | grep -A2 -B1 -F ' main('
_ABI(SystemV_x86_64)
generic64_t main(generic64_t argument_0) {
  return argument_0 * 3UL & 0xFFFFFFFFUL;
}
```
