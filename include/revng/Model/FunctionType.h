#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML
name: FunctionType
type: enum
members:
  - name: Regular
    doc: A function with at least one return instruction
  - name: NoReturn
    doc: A function that never returns
  - name: Fake
    doc: |
      A function with at least one non-proper return instruction. This typically
      represents outlined function prologues.
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/FunctionType.h"
#include "revng/Model/Generated/Late/FunctionType.h"
