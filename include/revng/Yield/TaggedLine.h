#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/TaggedString.h"

#include "revng/Yield/Generated/Early/TaggedLine.h"

namespace model {
class VerifyHelper;
}

namespace yield {

class TaggedLine : public generated::TaggedLine {
public:
  using generated::TaggedLine::TaggedLine;
  template<typename IteratorType>
  TaggedLine(uint64_t Index, IteratorType From, IteratorType To) :
    generated::TaggedLine(Index, SortedVector<TaggedString>(From, To)) {}

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
};

} // namespace yield

#include "revng/Yield/Generated/Late/TaggedLine.h"
