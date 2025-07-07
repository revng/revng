#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "nanobind/nanobind.h"

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/ObjectID.h"

namespace revng::pypeline::helpers::python {

inline Request convertRequests(nanobind::list &List) {
  Request Result;
  for (auto It1 = List.begin(); It1 != List.end(); ++It1) {
    nanobind::list ListInner = nanobind::cast<nanobind::list>(*It1);
    std::vector<const ObjectID *> Chunk;
    for (auto It2 = ListInner.begin(); It2 != ListInner.end(); ++It2) {
      Chunk.push_back(nanobind::cast<ObjectID *>(*It2));
    }
    Result.push_back(Chunk);
  }
  return Result;
}

class ManagedPyBuffer {
private:
  Py_buffer Buffer;

public:
  ManagedPyBuffer() { Buffer.obj = NULL; }
  ~ManagedPyBuffer() {
    if (Buffer.obj != NULL)
      PyBuffer_Release(&Buffer);
  }

  ManagedPyBuffer(const ManagedPyBuffer &) = delete;
  ManagedPyBuffer &operator=(const ManagedPyBuffer &) = delete;
  ManagedPyBuffer(ManagedPyBuffer &&) = default;
  ManagedPyBuffer &operator=(ManagedPyBuffer &&) = default;

  Py_buffer &operator*() { return Buffer; }
  Py_buffer *operator->() { return &Buffer; }
};

} // namespace revng::pypeline::helpers::python
