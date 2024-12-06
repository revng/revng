#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML
name: FunctionAttribute
doc: |
 Attributes for functions. Can be applied both to functions and call sites.
type: enum
members:
  - name: NoReturn
    doc: The function does not return.
  - name: Inline
    doc: The function must be inlined in callers.
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/FunctionAttribute.h"
#include "revng/Model/Generated/Late/FunctionAttribute.h"
