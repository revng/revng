#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/ObjectIDImpl.h"

namespace detail {

using RequestT = std::vector<std::vector<ObjectID *>>;
using ModelPath = std::string;
using ObjectDependencies = std::vector<
  std::vector<std::pair<ObjectID, ModelPath>>>;

class LeakyCharSmallVector : public llvm::SmallVector<char, 0> {
public:
  // Release the malloc-ed block
  // Needed to avoid copies while going back to python
  [[nodiscard]] llvm::ArrayRef<char> release() {
    char *Ptr = static_cast<char *>(BeginX);
    size_t Size = this->Size;
    // This method resets the vector without destroying
    resetToSmall();
    return { Ptr, Size };
  }
};

class Buffer {
private:
  LeakyCharSmallVector Vector;

public:
  template<typename... T>
  Buffer(T... Args) : Vector(Args...) {}

  LeakyCharSmallVector &get() { return Vector; }

  // Read the contents
  llvm::ArrayRef<const char> data() const {
    return { Vector.data(), Vector.size() };
  }

  // Release the stored buffer as a pair of Ptr + Size
  [[nodiscard]] llvm::ArrayRef<char> release() { return Vector.release(); }
};

}; // namespace detail
