#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/CompilationTime.h"

namespace detail {

template<auto &Destructor>
struct CDestructorTraits {
private:
  using FT = compile_time::FunctionTraits<Destructor>;
  static_assert(std::tuple_size_v<typename FT::Arguments> == 1,
                "Destructor must take a single argument");
  using Pointer = std::tuple_element_t<0, typename FT::Arguments>;
  static_assert(std::is_pointer_v<Pointer>,
                "Destructor argument must be a pointer");

public:
  using Type = std::remove_pointer_t<Pointer>;
  using ReturnType = FT::ReturnType;
};

using DefaultOkValueType = std::monostate;

template<auto &Destructor, auto OkValue>
class CDeleter {
private:
  using Traits = CDestructorTraits<Destructor>;
  static_assert(std::is_void_v<typename Traits::ReturnType>
                  or not std::is_same_v<decltype(OkValue), DefaultOkValueType>,
                "Must specify an OkValue if destructor is non-void");

public:
  void operator()(Traits::Type *Ptr) {
    if constexpr (std::is_same_v<typename Traits::ReturnType, void>) {
      Destructor(Ptr);
    } else {
      auto RC = Destructor(Ptr);
      revng_assert(RC == OkValue);
    }
  }
};

} // namespace detail

/// Helper using that specializes a std::unique_ptr for C-like objects that
/// have a `free` function. Uses a template parameter for the destructor so
/// that the pointer type can be automatically inferred. A lot of C destructors
/// return a return code, to accommodate that, specify the `OkValue` template
/// parameter; this will be asserted to be equal at destruction.
/// Examples:
/// ```c++
/// // Without OkValue
/// void zstdCContextFree(ZSTD_CCtx *);
/// CUniquePtr<zstdCContextFree> Ptr(ZSTD_createCCtx());
/// // With OkValue
/// int archive_read_free(struct archive *);
/// CUniquePtr<archive_read_free, ARCHIVE_OK> Ptr(archive_read_new());
/// ```
template<auto &Destructor, auto OkValue = detail::DefaultOkValueType{}>
using CUniquePtr = std::unique_ptr<
  typename detail::CDestructorTraits<Destructor>::Type,
  detail::CDeleter<Destructor, OkValue>>;
