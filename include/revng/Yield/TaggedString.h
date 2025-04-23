#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <limits>
#include <string>

#include "revng/Yield/TagType.h"

#include "revng/Yield/Generated/Early/TaggedString.h"

namespace model {
class VerifyHelper;
}

namespace yield {

class TaggedString : public generated::TaggedString {
public:
  using generated::TaggedString::TaggedString;
  TaggedString(uint64_t Index, TagType::Values Type, llvm::StringRef Content) :
    generated::TaggedString(Index, Type, Content.str(), {}) {}

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
};

} // namespace yield

#include "revng/Yield/Generated/Late/TaggedString.h"
