#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/CompilationTime.h"

/// Helper class that wraps a std::unique_ptr for C-like objects that have a
/// `free` function. Uses a template parameter for the destructor so that the
/// pointer type can be automatically inferred. A lot of C destructors return a
/// return code, to accommodate that, specify the `OkValue` template parameter;
/// this will be asserted to be equal at destruction.
template<auto &Destructor, auto OkValue = std::monostate{}>
class CUniquePtr {
private:
  using FT = compile_time::FunctionTraits<Destructor>;
  static_assert(std::tuple_size_v<typename FT::Arguments> == 1,
                "Destructor must take a single argument");
  static_assert(std::is_void_v<typename FT::ReturnType>
                  or not std::is_same_v<decltype(OkValue), std::monostate>,
                "Must specify an OkValue if destructor is non-void");

public:
  using pointer = std::tuple_element_t<0, typename FT::Arguments>;
  static_assert(std::is_pointer_v<pointer>);
  using element_type = std::remove_pointer_t<pointer>;

private:
  std::unique_ptr<element_type, void (*)(pointer)> Ptr;

public:
  CUniquePtr() : Ptr(nullptr, &close) {}
  CUniquePtr(element_type *Ptr) : Ptr(Ptr, &close) {}
  CUniquePtr &operator=(element_type *Ptr) {
    this->Ptr = decltype(this->Ptr)(Ptr, &close);
    return *this;
  }

  element_type &operator*() { return *Ptr; }
  pointer operator->() { return &*Ptr; }
  pointer get() { return &*Ptr; }

private:
  static void close(element_type *Ptr) {
    if constexpr (std::is_same_v<typename FT::ReturnType, void>) {
      Destructor(Ptr);
    } else {
      auto RC = Destructor(Ptr);
      revng_assert(RC == OkValue);
    }
  }
};
