#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
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
  bool IsTracking;

public:
  AccessTracker(bool StartsActive) { IsTracking = StartsActive; }

public:
  bool operator==(const AccessTracker &Other) const = default;

  bool operator!=(const AccessTracker &Other) const = default;

public:
  void clear() {
    Counter &= ~0x1;
    IsTracking = true;
  }
  void access() { Counter |= (0x1 & IsTracking); }
  void push() {
    bool HasLeadingZeroes = llvm::countLeadingZeros(Counter) != 0;
    revng_assert(HasLeadingZeroes, "More than 8 pushes have been performed");
    Counter = Counter << 1;
  }
  void pop() { Counter = Counter >> 1; }
  bool front() const { return (Counter & 0x1) == 0; }
  bool peak() const { return Counter & 0x1; }
  bool isSet() const { return Counter; }
  void stopTracking() { IsTracking = false; }
};

} // namespace revng
