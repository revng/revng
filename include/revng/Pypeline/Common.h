#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/ObjectID.h"

#include "nanobind/nanobind.h"

namespace detail {

using RequestT = std::vector<std::vector<ObjectID *>>;
using ModelPath = std::string;
using ObjectDependencies = std::vector<
  std::vector<std::pair<ObjectID, ModelPath>>>;

inline detail::RequestT convertRequests(nanobind::list &List) {
  detail::RequestT Result;
  for (auto It1 = List.begin(); It1 != List.end(); ++It1) {
    nanobind::list ListInner = nanobind::cast<nanobind::list>(*It1);
    std::vector<ObjectID *> Chunk;
    for (auto It2 = ListInner.begin(); It2 != ListInner.end(); ++It2) {
      Chunk.push_back(nanobind::cast<ObjectID *>(*It2));
    }
    Result.push_back(Chunk);
  }
  return Result;
}

template<typename C, size_t I>
inline std::remove_reference_t<C> &
unwrapContainer(nanobind::list &ContainerList) {
  using C_ref_removed = std::remove_reference_t<C>;
  auto It = std::next(ContainerList.begin(), I);
  revng_assert(nanobind::isinstance<C_ref_removed>(*It));
  return *nanobind::cast<C_ref_removed *>(*It);
}

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
