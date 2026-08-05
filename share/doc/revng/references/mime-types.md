Each artifact that [`revng project artifact`](cli/revng-project-artifact.md) can produce has an associated [MIME type](https://en.wikipedia.org/wiki/Media_type):

```text
  lift                        - application/x.llvm.bc+zstd
  isolate                     - application/x.llvm.bc+zstd
  enforce-abi                 - application/x.llvm.bc+zstd
  emit-cfg                    - text/yaml+tar+gz
  hexdump                     - text/x.hexdump+ptml
  render-svg-call-graph       - image/svg
  render-svg-call-graph-slice - image/svg
  disassemble                 - text/x.asm+ptml+tar+gz
  render-svg-cfg              - image/svg
  recompile                   - application/x-executable
  recompile-isolated          - application/x-executable
  simplify-switch             - application/x.llvm.bc+zstd
  segregate-stack-accesses    - application/x.llvm.bc+zstd
  emit-c                      - text/x.c+ptml+tar+gz
  emit-c-as-single-file       - text/x.c+ptml
  import-types                - application/x.mlir.bc
  emit-type-and-global-header - text/x.h+ptml
  emit-helper-header          - text/x.h+ptml
  emit-single-type-definition - text/x.c+tar+gz
```

## MIME types

When choosing MIME types for artifacts, we try to conform as close as possible to [RFC2045](https://datatracker.ietf.org/doc/html/rfc2045) and [RFC6838](https://www.rfc-editor.org/rfc/rfc6838).

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
* `application/x.mlir.bc`: MLIR IR in its bytecode representation.
* `text/x.yaml`: a YAML dictionary, with one key for each function.

MIME types that are not `text/*` or `image/svg` will be transmitted over GraphQL via Base64 encoding

Some of these MIME types can be further wrapped in another format.
To make this explicit, we add suffixes, specifically:

* `$PREFIX+ptml`: `$PREFIX` is wrapped in [PTML](ptml.md).
* `$PREFIX+tar`: `$PREFIX` is wrapped in a `tar` file containing one file for each function.
* `$PREFIX+gz`: `$PREFIX` is compressed using `gzip`.
* `$PREFIX+zstd`: `$PREFIX` is compressed using `zstd`.

For instance: `text/x.c+tar+gz` means that the artifact is GZip-compressed `tar` archive, containing C code for each function in the binary.
While, `text/x.asm+ptml+tar+gz` represents a GZip-compressed `tar` archive containing one file per function, which is in turn assembly code wrapped in PTML.
