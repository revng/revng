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

We run the [`import-binary` analysis](../../references/analyses.md#import-binary-analysis) using [`revng-analyze`](../../references/cli/revng-analyze.md) to automatically collect all the loading information available in the ELF headers:

```{bash ignore="VirtualSize|FileSize"}
$ revng analyze import-binary example -o example.yml
$ grep -A7 'Segments:' example.yml
Segments:
  - StartAddress:    "0x400000:Generic64"
    VirtualSize:     2528
    StartOffset:     0
    FileSize:        2528
    IsReadable:      true
    IsWriteable:     false
    IsExecutable:    true
```

However, the typical workflow does not require the user to manually specify what analyses to run, but there's a set of predefined analyses that should be run on a new project, the *initial autoanalyses*.

```bash
$ revng analyze \
        --resume=working-directory \
        revng-initial-auto-analysis \
        example \
        -o /dev/null
$ revng artifact \
        --resume=working-directory \
        decompile-to-single-file \
        example \
        | revng ptml | grep -A2 -B1 -F ' main('
_ABI(SystemV_x86_64)
generic64_t main(generic64_t argument_0) {
  return (argument_0 * 3) & 0xFFFFFFFF;
}
```

The commands above are *stateful*, they build on top of each other storing intermediate results into the directory specified by the `--resume` parameter.
The first command runs the set of initial autoanalyses of `revng` and the last one produces the decompiled code.

Alternatively, you can run `revng-initial-auto-analysis` *and* produce the artifact with a single command:

```bash
$ revng artifact \
        --analyze \
        decompile-to-single-file \
        example \
        | revng ptml \
        | grep -A2 -B1 -F ' main('
_ABI(SystemV_x86_64)
generic64_t main(generic64_t argument_0) {
  return (argument_0 * 3) & 0xFFFFFFFF;
}
```
