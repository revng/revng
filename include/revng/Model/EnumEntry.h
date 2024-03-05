#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: EnumEntry
doc: An entry in a model enum, with a name and a value
type: struct
fields:
  - name: Value
    type: uint64_t
  - name: CustomName
    type: Identifier
    optional: true
  - name: OriginalName
    type: string
    optional: true
  - name: Comment
    type: string
    optional: true
key:
  - Value
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/EnumEntry.h"

class model::EnumEntry : public model::generated::EnumEntry {
public:
  using generated::EnumEntry::EnumEntry;

public:
  bool verify(bool Assert = false) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/EnumEntry.h"
