#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <limits>
#include <string>

#include "revng/Model/VerifyHelper.h"
#include "revng/Yield/TagType.h"

/* TUPLE-TREE-YAML

name: Tag
type: struct
fields:
  - name: Type
    type: TagType
  - name: From
    type: uint64_t
  - name: To
    type: uint64_t
key:
  - Type
  - From
  - To

TUPLE-TREE-YAML */

#include "revng/Yield/Generated/Early/Tag.h"

namespace yield {

class Tag : public generated::Tag {
public:
  using generated::Tag::Tag;

  std::strong_ordering operator<=>(const Tag &Another) const {
    if (From() != Another.From())
      return From() <=> Another.From();
    else if (To() != Another.To())
      return Another.To() <=> To(); // reverse order
    else
      return Type() <=> Another.Type();
  }

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

#include "revng/Yield/Generated/Late/Tag.h"
