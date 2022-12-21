## MIME types

* MIMEs describe the content of the container in its entirety (not what's contained within)
* Trying to conform as close as possible to [RFC2045](https://datatracker.ietf.org/doc/html/rfc2045) and [RFC6838](https://www.rfc-editor.org/rfc/rfc6838)
* Types:
    * `text/*`: the output will be opened in a text editor
    * everything else: the output needs to be managed by an "external application"
* Base MIMEs:
    * Built-in MIMEs:
        * `application/x-executable` -> a binary
        * `application/x-object` -> an object file
        * `text/plain` -> Plain text file
    * Custom MIMEs:
        * `text/x.c` -> C
        * `text/x.asm` -> Assembly
        * `text/x.yaml` -> Plain YAML
        * `image/svg` -> svg
        * `text/x.llvm.ir` -> LLVM IR
        * `application/x.llvm.bc` -> LLVM Bitcode

  Mimes that are not `text/*` or `image/svg` will be transmitted over GraphQL via Base64 encoding

* Suffixes (order matters):
    * `+ptml` LHS is wrapped in PTML
    * `+yaml` LHS is wrapped in a YAML dictionary with keys being functions metaaddresses (soon: locations)
