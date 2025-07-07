#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Helpers/Python/Utils.h"
#include "revng/Pypeline/ObjectID.h"

namespace revng::pypeline::helpers::python {

template<typename T>
struct ContainerIO {
  static llvm::Error deserialize(T *Handle, nanobind::dict &Data) {
    std::map<const ObjectID *, llvm::ArrayRef<const char>> Input;
    std::vector<ManagedPyBuffer> Buffers;

    for (auto It = Data.begin(); It != Data.end(); It.increment()) {
      ObjectID *First = nanobind::cast<ObjectID *>((*It).first);

      ManagedPyBuffer &Buffer = Buffers.emplace_back();
      int Result = PyObject_GetBuffer((*It).second.ptr(),
                                      &*Buffer,
                                      PyBUF_SIMPLE);
      if (Result != 0) {
        // Return a success because the `PyObject_GetBuffer` already set the
        // correct PyError
        return llvm::Error::success();
      }
      if (Buffer->itemsize != 1 or Buffer->ndim != 1)
        return revng::createError("Invalid buffer shape");

      const char *DataPtr = static_cast<const char *>(Buffer->buf);
      Input[First] = llvm::ArrayRef<const char>(DataPtr, Buffer->len);
    }

    Handle->deserialize(Input);
    return llvm::Error::success();
  }

  static nanobind::dict serialize(T *Handle, nanobind::set Objects) {
    std::vector<const ObjectID *> CppObjects;
    for (auto It = Objects.begin(); It != Objects.end(); ++It)
      CppObjects.push_back(nanobind::cast<ObjectID *>(*It));

    auto Result = Handle->serialize(CppObjects);
    nanobind::dict Return;
    for (auto &Entry : Result) {
      ObjectID KeyCopy = Entry.first;
      nanobind::object Key = nanobind::cast<ObjectID>(std::move(KeyCopy));
      nanobind::object Value = nanobind::cast<Buffer>(std::move(Entry.second));
      Return[Key] = Value;
    }
    return Return;
  };
};

} // namespace revng::pypeline::helpers::python
