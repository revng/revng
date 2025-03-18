{#- This file is distributed under the MIT License. See LICENSE.md for details. -#}
I have the following decompiled C function:

```c
{{ code }}
```

I need a CSV instructing how to rename functions, arguments and types with meaningful names.

Here's the CSV:

```
old,new
{{ csv -}}
```

Send it back completing the second column with new identifiers. The second column should be an identifier. In case of `struct`s, assume they have a `typedef` (i.e., use `point` instead of `struct point`).
Use `snake_case` names.
Just send the CSV, nothing else.
