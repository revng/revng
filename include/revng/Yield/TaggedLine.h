#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/VerifyHelper.h"
#include "revng/Yield/TaggedString.h"

/* TUPLE-TREE-YAML

name: TaggedLine
type: struct
fields:
  - name: Index
    type: uint64_t

  - name: Tags
    sequence:
      type: SortedVector
      elementType: TaggedString

key:
  - Index

TUPLE-TREE-YAML */

#include "revng/Yield/Generated/Early/TaggedLine.h"

namespace yield {

class TaggedLine : public generated::TaggedLine {
public:
  using generated::TaggedLine::TaggedLine;
  template<typename IteratorType>
  TaggedLine(uint64_t Index, IteratorType From, IteratorType To) :
    generated::TaggedLine(Index, SortedVector<TaggedString>(From, To)) {}

public:
  bool verify(model::VerifyHelper &VH) const;
  void dump() const debug_function;

public:
  inline bool verify() const debug_function { return verify(false); }
  inline bool verify(bool Assert) const debug_function {
    model::VerifyHelper VH(Assert);
    return verify(VH);
  }
};

} // namespace yield

#include "revng/Yield/Generated/Late/TaggedLine.h"
