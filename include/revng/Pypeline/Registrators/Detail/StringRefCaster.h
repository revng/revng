#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// This is a caster class, it allows nanobind to automatically convert python
// strings to `llvm::StringRef`s and back

#include "llvm/ADT/StringRef.h"

#include "nanobind/nanobind.h"

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template<>
struct type_caster<llvm::StringRef> {
  NB_TYPE_CASTER(llvm::StringRef, const_name("str"))

  bool from_python(handle Src, uint8_t, cleanup_list *) noexcept {
    Py_ssize_t Size;
    const char *Str = PyUnicode_AsUTF8AndSize(Src.ptr(), &Size);
    if (Str == NULL) {
      PyErr_Clear();
      return false;
    }
    value = llvm::StringRef(Str, Size);
    return true;
  }

  static handle
  from_cpp(std::string_view Value, rv_policy, cleanup_list *) noexcept {
    return PyUnicode_FromStringAndSize(Value.data(), Value.size());
  }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
