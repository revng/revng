#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: EnumEntry
doc: An entry in an `enum`.
type: struct
fields:
  - name: Value
    doc: |-
      The value associated to this `enum` entry. Has to be unique within the
      `enum`.
    type: uint64_t
  - name: CustomName
    type: Identifier
    optional: true
    doc: |-
      A user-chosen [`Identifier`](#Identifier) for this `enum` entry.
  - name: OriginalName
    type: string
    optional: true
    doc: |-
      The name this `enum` entry had upon import.
      This value can differ from `CustomName`, since `CustomName` needs to
      respect the constraints of an [`Identifier`](#Identifier).
key:
  - Value
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/EnumEntry.h"

class model::EnumEntry : public model::generated::EnumEntry {
public:
  using generated::EnumEntry::EnumEntry;

public:
  // The entry should have a non-empty name, the name should not be a valid
  // alias, and there should not be empty aliases.
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/Model/Generated/Late/EnumEntry.h"
