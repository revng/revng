Running `revng artifact` you will get a list of artifacts that can be produced along with their [MIME types](https://en.wikipedia.org/wiki/Media_type):

```bash
$ revng artifact
USAGE: revng-artifact [options] <artifact> <binary>

<artifact> can be one of:

  lift                        - text/x.llvm.ir
  isolate                     - text/x.llvm.ir
  enforce-abi                 - text/x.llvm.ir
  hexdump                     - text/x.hexdump+ptml
  render-svg-call-graph       - image/svg
  render-svg-call-graph-slice - image/svg
  disassemble                 - text/x.asm+ptml+tar+gz
  render-svg-cfg              - image/svg
  recompile                   - application/x-executable
  recompile-isolated          - application/x-executable
  emit-cfg                    - text/x.yaml
  simplify-switch             - text/x.llvm.ir
  make-segment-ref            - text/x.llvm.ir
  decompile                   - text/x.c+ptml+tar+gz
  decompile-to-single-file    - text/x.c+ptml
  emit-helpers-header         - text/x.c+ptml
  emit-model-header           - text/x.c+ptml
  emit-type-definitions       - text/x.c+tar+gz
  convert-to-mlir             - text/mlir
```

## MIME types

When choosing MIME types for artifacts, we trying to conform as close as possible to [RFC2045](https://datatracker.ietf.org/doc/html/rfc2045) and [RFC6838](https://www.rfc-editor.org/rfc/rfc6838).

There are two fundamental distinctions in MIME types:

* `text/*`: the output can be opened in a text editor;
* everything else: the output needs to be managed by an "external application";

The most common *base* MIME types we use are:

* `application/x-executable`: an executable program.
* `application/x-object`: an object file.
* `text/plain`: a plain text file.
* `text/x.c`: C source code (e.g., decompiled code).
* `text/x.asm`: assembly code.
* `image/svg`: an SVG image.
* `text/x.llvm.ir`: LLVM IR in its textual representation.
* `application/x.llvm.bc`: LLVM IR in its binary representation (also known as bitcode).
* `text/x.hexdump`: an ASCII representation of raw bytes.
* `text/mlir`: MLIR IR in its textual representation.
* `text/x.yaml`: a YAML dictionary, with one key for each function.

MIME types that are not `text/*` or `image/svg` will be transmitted over GraphQL via Base64 encoding

Some of these MIME types can be further wrapped in another format.
To make this explicit, we add suffixes, specifically:

* `$PREFIX+ptml`: `$PREFIX` is wrapped in [PTML](ptml.md).
* `$PREFIX+tar`: `$PREFIX` is wrapped in a `tar` file containing one file for each function.
* `$PREFIX+gzip`: `$PREFIX` is compressed using `gzip`.

For instance: `text/x.c+tar+gz` means that the artifact is GZip-compressed `tar` archive, containing C code for each function in the binary.
While, `text/x.asm+ptml+tar+gz` represents a GZip-compressed `tar` archive containing one file per function, which is in turn assembly code wrapped in PTML.
