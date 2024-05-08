#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

/* TUPLE-TREE-YAML
name: QualifierKind
doc: |
  Enum for identifying different kinds of qualifiers

  Notice that we are choosing to represent pointers and arrays as qualifiers.
  The idea is that a qualifier is something that you can add to a type T to
  obtain another type R, in such a way that if T is fully known also R is fully
  known. In this sense Pointer and Array types are qualified types.
type: enum
members:
  - name: Pointer
  - name: Array
  - name: Const
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/QualifierKind.h"
#include "revng/Model/Generated/Late/QualifierKind.h"
