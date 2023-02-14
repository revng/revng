#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "revng/Support/Assert.h"

namespace revng {

///
/// A AccessCounter is optimized std::stack<bool> that can contain up to 8
/// elements.
///
class AccessTracker {
private:
  std::uint8_t Counter = 0;

public:
  bool operator==(const AccessTracker &Other) const = default;

  bool operator!=(const AccessTracker &Other) const = default;

  void clear() { Counter &= ~0x1; }
  void access() { Counter |= 0x1; }
  void push() {
    revng_assert(llvm::countLeadingZeros(Counter) != 0,
                 "More than 8 pushes have been performed");
    Counter = Counter << 1;
  }
  void pop() { Counter = Counter >> 1; }
  bool front() const { return (Counter & 0x1) == 0; }
  bool peak() const { return Counter & 0x1; }
  bool isSet() const { return Counter; }
};

} // namespace revng
