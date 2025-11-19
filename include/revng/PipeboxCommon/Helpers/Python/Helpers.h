#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "nanobind/nanobind.h"

#include "llvm/ADT/StringRef.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Helpers/Helpers.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/ObjectID.h"

namespace revng::pypeline::helpers {

// Helper struct to unpack containers from a nanobind::list.
// To be used in conjunction with PipeRunner or AnalysisRunner
template<typename C, size_t I>
struct ExtractContainerFromList<C, I, nanobind::list> {
  static C &get(nanobind::list &Containers) {
    return *nanobind::cast<C *>(Containers[I]);
  }
};

namespace python {

inline nanobind::object importObject(llvm::StringRef String) {
  auto [ModulePath, ObjectName] = String.rsplit('.');
  nanobind::module_ Module = nanobind::module_::import_(ModulePath.str()
                                                          .c_str());
  return Module.attr(ObjectName.str().c_str());
}

class ManagedPyBuffer {
private:
  Py_buffer Buffer;

  ManagedPyBuffer() { Buffer.obj = NULL; }

public:
  ~ManagedPyBuffer() {
    if (Buffer.obj != NULL)
      PyBuffer_Release(&Buffer);
  }

  ManagedPyBuffer(const ManagedPyBuffer &) = delete;
  ManagedPyBuffer &operator=(const ManagedPyBuffer &) = delete;

  ManagedPyBuffer(ManagedPyBuffer &&Other) { *this = std::move(Other); }
  ManagedPyBuffer &operator=(ManagedPyBuffer &&Other) {
    if (this == &Other)
      return *this;

    Buffer = std::move(Other.Buffer);
    Other.Buffer.obj = NULL;
    return *this;
  }

  static std::variant<int, ManagedPyBuffer> make(PyObject *Obj, int Flags) {
    ManagedPyBuffer Result;
    int RC = PyObject_GetBuffer(Obj, &Result.Buffer, Flags);
    if (RC != 0)
      return RC;
    return Result;
  }

  const Py_buffer &operator*() { return Buffer; }
  const Py_buffer *operator->() { return &Buffer; }
};

inline Model &convertReadOnlyModel(nanobind::object &TheModel) {
  nanobind::object ReadOnlyModel = importObject("revng.pypeline.model."
                                                "ReadOnlyModel");
  revng_assert(nanobind::isinstance(TheModel, ReadOnlyModel));
  nanobind::object ActualModel = nanobind::getattr(TheModel, "downcast")();
  return *nanobind::cast<Model *>(ActualModel);
}

template<typename T, typename... ArgsT>
inline nanobind::capsule makeCapsule(ArgsT &&...Args) {
  return nanobind::capsule(new T(std::forward<ArgsT>(Args)...),
                           [](void *Ptr) noexcept {
                             delete static_cast<T *>(Ptr);
                           });
}

} // namespace python

} // namespace revng::pypeline::helpers
