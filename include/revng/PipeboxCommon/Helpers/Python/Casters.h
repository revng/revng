#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Helpers/Python/Helpers.h"

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template<>
struct type_caster<revng::pypeline::Request> {
  NB_TYPE_CASTER(revng::pypeline::Request,
                 const_name("list[revng.pypeline.object.ObjectSet]"))

  bool from_python(handle Source, uint8_t, cleanup_list *) {
    using namespace revng::pypeline::helpers::python;
    nanobind::object ObjectSet = importObject("revng.pypeline.object."
                                              "ObjectSet");

    revng_assert(nanobind::isinstance<nanobind::list>(Source));
    for (const auto &OuterElement : Source) {
      revng_assert(nanobind::isinstance(OuterElement, ObjectSet));
      nanobind::object OuterSet = nanobind::getattr(OuterElement, "objects");

      std::vector<const ObjectID *> Chunk;
      for (const auto &Element : OuterSet)
        Chunk.push_back(nanobind::cast<ObjectID *>(Element));

      value.push_back(Chunk);
    }

    return true;
  }

  // from_cpp is not implemented since we don't support returning Request
  // from C++
};

/// This is a caster class, it allows nanobind to automatically convert python
/// strings to `llvm::StringRef`s and back
///
/// Copied from nanobind's caster implementation for std::string_view
/// See the header "nanobind/stl/string_view.h"
template<>
struct type_caster<llvm::StringRef> {
  NB_TYPE_CASTER(llvm::StringRef, const_name("str"))

  bool from_python(handle Source, uint8_t, cleanup_list *) {
    Py_ssize_t Size = 0;
    const char *String = PyUnicode_AsUTF8AndSize(Source.ptr(), &Size);
    if (String == NULL) {
      PyErr_Clear();
      return false;
    }
    value = llvm::StringRef(String, Size);
    return true;
  }

  static handle from_cpp(llvm::StringRef Value, rv_policy, cleanup_list *) {
    return PyUnicode_FromStringAndSize(Value.data(), Value.size());
  }
};

namespace detail {

inline void llvmErrorToPythonException(llvm::Error &&Error) {
  std::string Message = llvm::toString(std::move(Error));
  // TODO: specialize the exception based on the returned llvm::Error
  PyErr_SetString(PyExc_RuntimeError, Message.c_str());
}

} // namespace detail

/// This is a caster class for llvm::Error, it allows converting functions
/// returning it into a thrown python exception
template<>
struct type_caster<llvm::Error> {
  NB_TYPE_CASTER(llvm::Error, const_name("None"))

  // from_python is intentionally not defined because we do not support passing
  // `llvm::Error`s as arguments

  static handle from_cpp(llvm::Error &&Error, rv_policy, cleanup_list *) {
    if (Error) {
      detail::llvmErrorToPythonException(std::move(Error));
      return {};
    }
    return nanobind::none().release();
  }
};

/// This is a caster class for llvm::Expected<T>, it allows converting functions
/// returning it to either the wrapped T or a thrown python exception
/// Inspired by "nanobind/stl/detail/optional.h"
template<typename T>
struct type_caster<llvm::Expected<T>> {
  using Caster = make_caster<T>;
  NB_TYPE_CASTER(llvm::Expected<T>, Caster::Name)

  // from_python is intentionally not defined because we do not support passing
  // `llvm::Expected`s as arguments

  static handle
  from_cpp(llvm::Expected<T> &&Error, rv_policy Policy, cleanup_list *Cleanup) {
    if (not Error) {
      detail::llvmErrorToPythonException(Error.takeError());
      return {};
    }

    return Caster::from_cpp(std::move(Error.get()), Policy, Cleanup);
  }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
