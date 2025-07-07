#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/ObjectID.h"

namespace revng::pypeline {

/// Type used for both incoming and outgoing requests
/// Each index maps to the i-th container passed to the 'run' function
using Request = std::vector<std::vector<const ObjectID *>>;
/// Type representing a path in the model, returned in a set when an analysis
/// runs
using ModelPath = std::string;
/// Type used to return the dependencies of objects produced by a pipe
/// The first index maps to the i-th container of the pipe
using ObjectDependencies = std::vector<
  std::vector<std::pair<ObjectID, ModelPath>>>;

class Buffer {
private:
  llvm::SmallVector<char, 0> Vector;

public:
  template<typename... T>
  Buffer(T... Args) : Vector(Args...) {}

  llvm::SmallVector<char, 0> &get() { return Vector; }

  // Read the contents
  llvm::ArrayRef<const char> ref() const {
    return { Vector.data(), Vector.size() };
  }
};

} // namespace revng::pypeline
