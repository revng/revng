#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

namespace revng {

/// \return Milliseconds since Unix epoch
inline uint64_t getEpochInMilliseconds() {
  namespace sc = std::chrono;
  auto Now = sc::system_clock::now().time_since_epoch();
  return sc::duration_cast<std::chrono::milliseconds>(Now).count();
}
} // namespace revng
