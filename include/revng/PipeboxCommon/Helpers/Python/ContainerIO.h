#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Helpers/Python/Helpers.h"
#include "revng/PipeboxCommon/ObjectID.h"

namespace revng::pypeline::helpers::python {

template<IsContainer T>
struct ContainerIO {
  static nanobind::object objects(T &Handle) {
    nanobind::object ObjectSet = importObject("revng.pypeline.object."
                                              "ObjectSet");
    std::set<ObjectID> Objects = Handle.objects();

    nanobind::list Result;
    for (const auto &Object : Objects)
      Result.append(nanobind::cast(Object));

    return ObjectSet(nanobind::cast(T::Kind), nanobind::set(Result));
  }

  static llvm::Error deserialize(T &Handle, nanobind::dict &Data) {
    // Input map that will be fed to the Container's deserialize method
    std::map<const ObjectID *, llvm::ArrayRef<char>> Input;
    // Temporary PyBuffer-s that are needed to read the contents of Data's
    // values
    std::vector<ManagedPyBuffer> Buffers;

    for (const auto &[Key, Value] : Data) {
      ObjectID *First = nanobind::cast<ObjectID *>(Key);

      // Retrieve the value as a PyBuffer, this allows both bytes,
      // revng::pypeline::Buffer-s and any other container implementing the
      // buffer protocol to be used
      revng_assert(PyObject_CheckBuffer(Value.ptr()) == 1);
      auto MaybeBuffer = ManagedPyBuffer::make(Value.ptr(), PyBUF_SIMPLE);
      if (std::holds_alternative<int>(MaybeBuffer)) {
        // Return a success because the `PyObject_GetBuffer` already set the
        // correct PyError
        return llvm::Error::success();
      }
      Buffers.push_back(std::move(std::get<ManagedPyBuffer>(MaybeBuffer)));
      ManagedPyBuffer &Buffer = Buffers.back();

      // Buffer support 2D arrays, we requested PyBUF_SIMPLE which should be
      // equivalent to void[], but check that is 1D anyways
      if (Buffer->itemsize != 1 or Buffer->ndim != 1)
        return revng::createError("Invalid buffer shape");

      // Convert the buffer to ArrayRef
      const char *DataPtr = static_cast<const char *>(Buffer->buf);
      Input[First] = llvm::ArrayRef<char>(DataPtr, Buffer->len);
    }

    Handle.deserialize(Input);
    return llvm::Error::success();
  }

  static nanobind::dict serialize(T &Handle, nanobind::object ObjectSet) {
    nanobind::object ObjectSetCls = importObject("revng.pypeline.object."
                                                 "ObjectSet");
    revng_assert(nanobind::isinstance(ObjectSet, ObjectSetCls));
    nanobind::object Objects = nanobind::getattr(ObjectSet, "objects");

    // Convert the input Objects set into a vector of ObjectIDs
    std::vector<const ObjectID *> CppObjects;
    for (const auto &Object : Objects)
      CppObjects.push_back(nanobind::cast<ObjectID *>(Object));

    std::map<ObjectID, Buffer> Result = Handle.serialize(CppObjects);
    // Manually convert the map above to a Python dict
    nanobind::dict Return;
    for (auto &Entry : Result) {
      // Here the key is copied on purpose because it needs to be moved.
      // We use the special syntax of nanobind::cast to move-construct the
      // python object, so that the amount of memory copying is minimized
      ObjectID KeyCopy = Entry.first;
      nanobind::object Key = nanobind::cast<ObjectID>(std::move(KeyCopy));
      nanobind::object Value = nanobind::cast<Buffer>(std::move(Entry.second));
      Return[Key] = Value;
    }
    return Return;
  };
};

} // namespace revng::pypeline::helpers::python
